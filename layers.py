import torch
from torch import nn
import torch.nn.functional as F
from nflows import transforms, distributions, flows
import numpy as np


class ContextEmbedder(torch.nn.Module):
    """ Small NN to be used for the embedding of the conditionals """
    def __init__(self, input_size, output_size):
        """ input_size: length of context vector
            output_size: length of context vector to be fed to the flow
        """
        super(ContextEmbedder, self).__init__()
        self.layer1 = torch.nn.Linear(input_size, (input_size+output_size)//2)
        self.layer2 = torch.nn.Linear((input_size+output_size)//2, (input_size+output_size)//2)
        self.output = torch.nn.Linear((input_size+output_size)//2, output_size)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        out = self.output(x)
        return out


class BaseContext(torch.nn.Module):
    """ Small NN to map input to mean and width of base_gaussians"""
    def __init__(self, context_size, dimensionality):
        """ context_size: length of context vector
            dimensionality: number of dimensions of base dist.
        """
        super(BaseContext, self).__init__()
        self.layer1 = torch.nn.Linear(context_size, dimensionality)
        self.layer2 = torch.nn.Linear(dimensionality, dimensionality)
        self.output = torch.nn.Linear(dimensionality, 2*dimensionality)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        out = self.output(x)
        return out


class RandomPermutationLayer(transforms.Permutation):
    """ Permutes elements with random, but fixed permutation. Keeps pixel inside layer. """
    def __init__(self, features, dim=1):
        """ features: list of dimensionalities to be permuted"""
        assert isinstance(features, list), ("Input must be a list of integers!")
        permutations = []
        for index, features_entry in enumerate(features):
            current_perm = np.random.permutation(features_entry)
            if index == 0:
                permutations.extend(current_perm)
            else:
                permutations.extend(current_perm + np.cumsum(features)[index-1])
        super().__init__(torch.tensor(permutations), dim)


class InversionLayer(transforms.Permutation):
    """ Inverts the order of the elements in each layer.  Keeps pixel inside layer. """
    def __init__(self, features, dim=1):
        """ features: list of dimensionalities to be inverted"""
        assert isinstance(features, list), ("Input must be a list of integers!")
        permutations = []
        for index, features_entry in enumerate(features):
            current_perm = np.arange(features_entry)[::-1]
            if index == 0:
                permutations.extend(current_perm)
            else:
                permutations.extend(current_perm + np.cumsum(features)[index-1])
        super().__init__(torch.tensor(permutations), dim)
