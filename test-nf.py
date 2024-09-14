from pathlib import Path

import torch
from torch import nn
from torch import optim

import numpy as np
from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from zdc_utils import sum_channels_parallel_numpy as sum_channels_parallel
import json
import math
from config import FlowConfig
from dataset import get_dataloader
from flows_utils import prepare_model_name, setup_flow


@torch.no_grad()
def generate(model, config, conditionals):

    conditionals = torch.reshape(conditionals, (-1, config.cond_label_size)).to(config.device)
    samples = model.sample(config.no_imgs_generated, conditionals)

    return samples


def setup_data(particle_file, image_file):
    particles = np.load(particle_file)["arr_0"].astype('float32')
    particles, pdgid, photonsum = particles[..., :-2], particles[..., -2], particles[..., -1].reshape(-1, 1)
    particles = np.hstack((particles, photonsum))
    images = np.load(image_file)["arr_0"]
    images = torch.tensor(images, dtype=torch.float32)
    particles = torch.tensor(particles, dtype=torch.float32)
    return particles, images


def setup_test_photonsum(particle_file):
    particles = np.load(particle_file)["arr_0"].astype('float32')
    particles, pdgid, photonsum = particles[..., :-2], particles[..., -2], particles[..., -1].reshape(-1, 1)
    photonsum = torch.tensor(photonsum, dtype=torch.float32)
    _, photonsum_test = train_test_split(photonsum, test_size=0.2, random_state=42)
    return photonsum_test


def setup_train_particles(particle_file):
    particles = np.load(particle_file)["arr_0"].astype('float32')
    particles, pdgid, photonsum = particles[..., :-2], particles[..., -2], particles[..., -1].reshape(-1, 1)
    particles = np.hstack((particles, photonsum))
    particles = torch.tensor(particles, dtype=torch.float32)
    particles_train, _ = train_test_split(particles, test_size=0.2, random_state=42)
    return particles_train


def prepare_noised_samples(samples, curr_photonsums, config):
    samples = ((torch.sigmoid(samples) - config.alpha) / (1. - 2. * config.alpha))
    samples = samples / (samples.abs().sum(dim=(-1), keepdims=True) + 1e-16)
    samples = samples.cpu().reshape((-1, config.dim * config.dim))
    noised_photonsums = curr_photonsums + config.dim * config.dim * 0.5 * config.noise_mul  # due to the random noise addition
    samples *= noised_photonsums
    return samples


def remove_noise_from_samples(samples, curr_photonsums, config):
    samples_sub = samples - 0.5 * config.noise_mul
    samples_floor = torch.floor(samples).clamp(min=0)
    samples_sub = torch.floor(samples_sub).clamp(min=0)

    samples_sub_recalc = recalc(samples_sub, curr_photonsums)
    samples_floor_recalc = recalc(samples_floor, curr_photonsums)
    samples_floor = samples_floor.reshape((-1, config.dim, config.dim))
    samples_sub = samples_sub.reshape((-1, config.dim, config.dim))
    samples_floor_recalc = samples_floor_recalc.reshape((-1, config.dim, config.dim))
    samples_sub_recalc = samples_sub_recalc.reshape((-1, config.dim, config.dim))

    return samples_sub, samples_sub_recalc, samples_floor, samples_floor_recalc


def recalc(samples_, curr_photonsums):
    samples = samples_.clone()
    samples = samples / (samples.abs().sum(dim=(-1), keepdims=True) + 1e-16)
    samples *= curr_photonsums
    samples = torch.floor(samples)
    samples = samples.clamp(min=0)
    return samples


def update_result_lists(samples_lists, mses, final_ch_gens, imgs, config):
    for i, samples in enumerate(samples_lists):
        mses[i].extend(batch_mse(samples, imgs, config))
        ch_gen = np.array(list(sum_channels_parallel(samples)))
        final_ch_gens[i].append(ch_gen)


def batch_mse(samples, imgs, config):
    return torch.square(samples - imgs).reshape(-1, config.dim * config.dim).sum(dim=(-1)).tolist()


if __name__ == "__main__":
    torch.set_printoptions(threshold=10_000)

    with open(Path(__file__).parent.resolve().as_posix() + "/config.json") as f:
        json_config_obj = json.load(f)
    config = FlowConfig(**json_config_obj)
    config.tail_bound = math.ceil(abs(math.log(config.alpha / (1.0 - config.alpha))))

    print(f"ALPHA: {config.alpha}, tail bound: {config.tail_bound}")

    BNN = 'bnn' if config.bnn_ps else ''
    COM = 'com' if config.cond_label_size == 12 else ''
    particle_file = config.base_dir + config.data_dir_suffix + f"/data_nonrandom_particles_{BNN}{COM}photonsum.npz"
    print(f"Conds data from file: {particle_file}")
    image_file = config.base_dir + config.data_dir_suffix + f"/data_{config.particle}_nonrandom_responses.npz"

    model_name = prepare_model_name(config)
    config.model_name = model_name

    config.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using {}".format(config.device))

    particles, images = setup_data(particle_file, image_file)
    if config.original_ps_scaler:
        particles_train_org = setup_train_particles(
            particle_file=config.base_dir + config.data_dir_suffix + f"/data_nonrandom_particles_{COM}photonsum.npz")
    _, test_dataloader, scaler = get_dataloader(images, particles, config.alpha,
                                                full=False,
                                                apply_logit=False,
                                                with_noise=False,
                                                normalize=False,
                                                device=config.device,
                                                batch_size=config.batch_size,
                                                noise_mul=config.noise_mul,
                                                y_scaler_fit=particles_train_org if config.original_ps_scaler else None)

    flow = setup_flow(config)
    model = flow.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    print(model)
    model_path = config.base_dir + config.models_dir_suffix + f"/{config.model_name}_checkpoint.pt"
    config.model_path = model_path
    print(f'Model path: {model_path}')

    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {total_parameters}")

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.eval()

    final_ch_gen_sub_norecalc, final_ch_gen_sub_recalc, \
    final_ch_gen_floor_norecalc, final_ch_gen_floor_recalc, final_ch_org = [], [], [], [], []
    mse_sub_norecalc, mse_sub_recalc, mse_floor_norecalc, mse_floor_recalc = [], [], [], []
    mses = (mse_sub_norecalc, mse_sub_recalc, mse_floor_norecalc, mse_floor_recalc)
    final_ch_gens = (final_ch_gen_sub_norecalc, final_ch_gen_sub_recalc, final_ch_gen_floor_norecalc,
                     final_ch_gen_floor_recalc)

    if config.save_responses:
        noised_samples = np.empty((0, config.dim, config.dim))
    for i, data in enumerate(test_dataloader):
        imgs = data["img"]
        conds = data["conds"]
        samples = generate(model, config, conds)
        curr_photonsums = (conds[:, -1] * scaler.scale_[-1] + scaler.mean_[-1]).reshape((-1, 1))
        samples = prepare_noised_samples(samples, curr_photonsums, config)
        if config.save_responses:
            noised_samples = np.vstack((noised_samples, samples.reshape((-1, config.dim, config.dim))))

        denoised_samples_lists = remove_noise_from_samples(samples, curr_photonsums, config)
        samples_sub, samples_sub_recalc, samples_floor, samples_floor_recalc = denoised_samples_lists
        imgs = imgs.cpu().reshape((-1, config.dim, config.dim))

        update_result_lists(denoised_samples_lists, mses, final_ch_gens, imgs, config)
        final_ch_org.append(np.array(list(sum_channels_parallel(imgs))))

        print(f"Batch {i} done")

    final_ch_org = np.concatenate(final_ch_org)
    final_ch_gens = (np.concatenate(final_ch_gen) for final_ch_gen in final_ch_gens)

    names = ["sub_norecalc", "sub_recalc", "floor_norecalc", "floor_recalc"]
    for k, final_ch_gen in enumerate(final_ch_gens):
        dists = []
        print(f"Currently processed: {names[k]}")
        for i in range(5):
            dist = wasserstein_distance(final_ch_org[:, i], final_ch_gen[:, i])
            dists.append(dist)
            print(f"Evaluation channel {i}: {dist}")
        print(f"Average wasserstein distance: {np.mean(dists)}\n")

        dists = []
        print(f"Currently processed: {names[k]}")
        for i in range(5):
            dist = mean_absolute_error(final_ch_org[:, i], final_ch_gen[:, i])
            dists.append(dist)
            print(f"Evaluation channel {i}: {dist}")
        print(f"Average MAE distance: {np.mean(dists)}\n")

        mse = np.mean(mses[k])
        print(f"MSE distance: {mse}\n")
        print(f"RMSE distance: {math.sqrt(mse)}\n")

    if config.save_responses:
        data_save_fnm = config.base_dir + config.data_dir_suffix + f"/{config.model_name}_generated_noisedsamples.npz"
        np.savez(data_save_fnm, noised_samples)