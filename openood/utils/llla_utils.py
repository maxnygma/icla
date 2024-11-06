import numpy as np

import torch
from torch.nn.utils import parameters_to_vector
from laplace import Laplace


@torch.no_grad()
def swa_update_bn_adapted(loader, model, device=None):
    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.reset_running_stats()
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None

    for input in loader:
        input = input['data']

        if isinstance(input, (list, tuple)):
            input = input[0]
        if device is not None:
            input = input.to(device)

        model(input)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)


class LLLAGMM:
    def __init__(self, list_la, weights: torch.TensorType):
        assert len(list_la) == len(weights)
        assert (torch.sum(weights) > 0.98) and (torch.sum(weights) < 1.02)

        self.list_la = list_la
        self.weights = weights

    def get_probs(self, x):
        probs_gmm = torch.stack([la(x, link_approx='probit').squeeze(0) for la in self.list_la], dim=0)

        scaled_probs = probs_gmm * self.weights.unsqueeze(1).unsqueeze(2).expand(-1, -1, probs_gmm.shape[2]).to(probs_gmm.device)
        scaled_probs = torch.sum(scaled_probs, dim=0)

        return scaled_probs
    
    
    
    

