import torch
from torch import nn
import torch.nn.functional as F

from nflows.flows.base import Flow
from nflows.distributions.normal import StandardNormal
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.transforms.base import CompositeTransform
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.permutations import ReversePermutation
from nflows import transforms, distributions, flows
from layers import InversionLayer, RandomPermutationLayer, BaseContext


def logit(x):
    """ returns logit of input """
    return torch.log(x / (1.0 - x))


def logit_trafo(x, alpha):
    """ implements logit trafo of MAF paper https://arxiv.org/pdf/1705.07057.pdf """
    local_x = alpha + (1. - 2.*alpha) * x
    return logit(local_x)


def inverse_logit(x, alpha, clamp_low=0., clamp_high=1.):
    """ inverts logit_trafo(), clips result if needed """
    return ((torch.sigmoid(x) - alpha) / (1. - 2.*alpha)).clamp_(clamp_low, clamp_high)


def setup_flow(config):
    flow_params_RQS = {'num_blocks': config.n_hidden,  # num of hidden layers per block
                       'use_residual_blocks': config.use_residual,
                       'use_batch_norm': config.batch_norm,
                       'dropout_probability': config.dropout_probability,
                       'activation': getattr(F, config.activation_fn),
                       'random_mask': False,
                       'num_bins': config.n_bins,
                       'tails': 'linear',
                       'tail_bound': config.tail_bound,
                       'min_bin_width': 1e-6,
                       'min_bin_height': 1e-6,
                       'min_derivative': 1e-6}

    flow_blocks = []

    for i in range(config.n_blocks):
        flow_blocks.append(
            transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                **flow_params_RQS,
                features=config.dim * config.dim,
                context_features=config.cond_label_size,
                hidden_features=config.hidden_size
            ))

        if i % 2 == 0:
            flow_blocks.append(InversionLayer([config.dim * config.dim]))
        else:
            flow_blocks.append(RandomPermutationLayer([config.dim * config.dim]))

    del flow_blocks[-1]
    flow_transform = transforms.CompositeTransform(flow_blocks)
    if config.cond_base:
        flow_base_distribution = distributions.ConditionalDiagonalNormal(
            shape=[config.dim * config.dim],
            context_encoder=BaseContext(config.cond_label_size, config.dim * config.dim))
    else:
        flow_base_distribution = distributions.StandardNormal(shape=[config.dim * config.dim])
    flow = flows.Flow(transform=flow_transform, distribution=flow_base_distribution)
    return flow


def prepare_model_name(config):
    model_name = f"NF_CALO_NEUTRON_{config.particle.upper()}_HS{config.hidden_size}_NB{config.n_blocks}"
    if config.n_hidden > 1:
        model_name += f"_NH{config.n_hidden}"
    if config.noise_mul < 1:
        model_name += f"_noise0{str(config.noise_mul)[2:]}"
    if config.n_epochs > 100:
        model_name += f"_E{config.n_epochs}"
    if config.cond_label_size == 12:
        model_name += f"_com"
    if not config.cond_base:
        model_name += f"_stdbase"
    if config.alpha != 1e-6:
        model_name += f"_alpha{str(config.alpha)}"
    return model_name
