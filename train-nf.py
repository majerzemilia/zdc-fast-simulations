from pathlib import Path

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

import numpy as np
import math
import json
from config import FlowConfig
from dataset import setup_center_of_mass_coords, get_dataloader
from flows_utils import prepare_model_name, setup_flow


def train_and_evaluate(model, train_loader, test_loader, optimizer, config):
    best_eval_logprob = float('-inf')
    milestones = [15, 40, 70, 100, 150]
    lr_schedule = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                       milestones=milestones,
                                                       gamma=0.5,
                                                       verbose=True)
    for i in range(config.n_epochs):
        train(model, train_loader, optimizer, i, config)
        with torch.no_grad():
            eval_logprob, _ = evaluate(model, test_loader, i, config)
        if eval_logprob > best_eval_logprob:
            best_eval_logprob = eval_logprob
            save_all(model, optimizer, config)

        lr_schedule.step()


def save_all(model, optimizer, config):
    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()},
               config.model_path)


def train(model, dataloader, optimizer, epoch, config):
    model.train()
    for i, data in enumerate(dataloader):
        logprob = get_logprob(model, data)
        loss = - logprob.mean(0)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % config.log_interval == 0:
            print('epoch {:3d} / {}, step {:4d} / {}; loss {:.4f}'.format(
                epoch + 1, config.n_epochs, i, len(dataloader), loss.item()))


@torch.no_grad()
def evaluate(model, dataloader, epoch, config):
    model.eval()
    loglike = []

    for data in dataloader:
        logprob = get_logprob(model, data)
        loglike.append(logprob)

    logprobs = torch.cat(loglike, dim=0).to(config.device)

    logprob_mean = logprobs.mean(0)
    logprob_std = logprobs.var(0).sqrt() / np.sqrt(len(dataloader.dataset))

    output = 'Evaluate ' + (epoch is not None) * '(epoch {}) -- '.format(epoch + 1) + \
             'logp(x, at E(x)) = {:.3f} +/- {:.3f}'

    print(output.format(logprob_mean, logprob_std))

    return logprob_mean, logprob_std


def get_logprob(model, data):
    x = data["img"]
    conds = data["conds"]

    y = conds.to(config.device)
    x = x.view(x.shape[0], -1).to(config.device)

    logprob = model.log_prob(x, y)
    return logprob


def setup_data(particle_file, image_file, setup_com=False):
    images = np.load(image_file)["arr_0"]
    photons_sums = np.sum(images, axis=(1, 2)).reshape(-1, 1)
    particles = np.load(particle_file)["arr_0"]
    particles, pdgid = particles[..., :-1], particles[..., -1]
    if setup_com:
        coms = setup_center_of_mass_coords(images)
        particles = np.hstack((particles, coms, photons_sums))
    else:
        particles = np.hstack((particles, photons_sums))
    images = torch.tensor(images, dtype=torch.float32)
    particles = torch.tensor(particles, dtype=torch.float32)
    return particles, images


if __name__ == "__main__":
    with open(Path(__file__).parent.resolve().as_posix() + "/config.json") as f:
        json_config_obj = json.load(f)
    config = FlowConfig(**json_config_obj)
    particle_file = config.BASE_DIR + config.DATA_DIR_SUFFIX + f"/data_{config.PARTICLE.lower()}_nonrandom_particles.npz"
    image_file = config.BASE_DIR + config.DATA_DIR_SUFFIX + f"/data_{config.PARTICLE.lower()}_nonrandom_responses.npz"

    config.tail_bound = math.ceil(abs(math.log(config.ALPHA / (1.0 - config.ALPHA))))

    print(f"ALPHA: {config.ALPHA}, tail bound: {config.tail_bound}")

    model_name = prepare_model_name(config)
    config.model_name = model_name

    config.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Using {}".format(config.device))

    particles, images = setup_data(particle_file, image_file, setup_com=True if config.cond_label_size == 12 else False)
    train_dataloader, test_dataloader, _ = get_dataloader(images, particles, config.ALPHA,
                                                               full=False,
                                                               apply_logit=True,
                                                               device=config.device,
                                                               batch_size=config.batch_size,
                                                               with_noise=config.with_noise,
                                                               noise_mul=config.noise_mul)

    flow = setup_flow(config)
    model = flow.to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    print(model)
    model_path = config.BASE_DIR + config.MODELS_DIR_SUFFIX + f"/{config.model_name}_checkpoint.pt"
    config.model_path = model_path
    print(f'Model path: {model_path}')

    total_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {total_parameters}")
    train_and_evaluate(model, train_dataloader, test_dataloader, optimizer, config)
