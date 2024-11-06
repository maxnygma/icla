from typing import Any

import os
import ast
import wandb
import types
import math
import dill
import pickle
import numpy as np
import torch
from torch.nn.utils import parameters_to_vector
import torch.nn as nn
from laplace import Laplace
from laplace.curvature import AsdlGGN, AsdlEF, AsdlHessian
from tqdm import tqdm

from .base_postprocessor import BasePostprocessor
from ..utils import LLLAGMM 


class LLLAPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.args = self.config.postprocessor.postprocessor_args

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        checkpoint_path = self.args.checkpoint_path
        optimize_precision = self.args.optimize_precision
        val_loader = id_loader_dict['val']
        train_loader = id_loader_dict['train']

        print('Loading default model')
        net.load_state_dict({k.replace('module.',''): v for k,v in torch.load(checkpoint_path).items()}, strict=True)

        net.eval()
        la = self.get_laplace(model=net, train_loader=train_loader, val_loader=val_loader, optimize_precision=optimize_precision)

        print(f"LA has been computed. The best epoch is used")

        self.lllagmm = LLLAGMM([la], torch.tensor([1.0]))

        return 

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, calibration=False):
        output = self.lllagmm.get_probs(data)
        conf, pred = torch.max(output, dim=1)

        if calibration:
            return pred, conf, output
        else:
            return pred, conf
    
    def get_laplace(self, model, train_loader, val_loader, optimize_precision=False):
        if self.args.llla_type == 'ef' or self.args.llla_type == 'icla' or self.args.llla_type == 'icla_zero':
            la = Laplace(model=model, likelihood='classification', subset_of_weights='last_layer',
                        hessian_structure='diag', backend=AsdlEF, last_layer_name='fc')
        elif self.args.llla_type == 'ggn':
            la = Laplace(model=model, likelihood='classification', subset_of_weights='last_layer',
                        hessian_structure='diag', backend=AsdlGGN, last_layer_name='fc')
        elif self.args.llla_type == 'k-fac':
            la = Laplace(model=model, likelihood='classification', subset_of_weights='last_layer',
                          hessian_structure='kron', last_layer_name='fc')
        else:  
            raise Exception("Unknown llla_type. Use 'ef', 'ggn', 'k-fac', 'icla' or 'icla_zero'.")

        if self.args.llla_type == 'ef' or self.args.llla_type == 'ggn':
            la.fit = overrided_fit.__get__(la)
            la.fit(train_loader)
        elif self.args.llla_type == 'k-fac':
            la.fit = overrided_fit_kron.__get__(la)
            la.fit(train_loader)
        elif self.args.llla_type == 'icla':
            la.fit = overrided_fit.__get__(la)
            la.fit(train_loader)
        elif self.args.llla_type == 'icla_zero':
            la.model.eval()
            la.mean = parameters_to_vector(la.model.last_layer.parameters())
            la.H = torch.zeros(la.n_params, device='cuda')
        else:  
            raise Exception("Unknown llla_type. Use 'ef', 'ggn', 'k-fac', 'icla' or 'icla_zero'.")

        # Optimize precision
        if optimize_precision:
            print('Optimizing prior precision...')
            la.optimize_prior_precision(method='marglik', val_loader=val_loader, verbose=True)

        if self.args.llla_type == 'icla':
            la.H = torch.zeros(la.n_params, device='cuda')

        return la
    
    
def overrided_fit(self, train_loader, override=True):
    """Fit the local Laplace approximation at the parameters of the model. (OpenOOD-friendly version)

    Parameters
    ----------
    train_loader : torch.data.utils.DataLoader
        each iterate is a training batch (X, y);
        `train_loader.dataset` needs to be set to access \\(N\\), size of the data set
    override : bool, default=True
        whether to initialize H, loss, and n_data again; setting to False is useful for
        online learning settings to accumulate a sequential posterior approximation.
    """
    if override:
        self._init_H()
        self.loss = 0
        self.n_data = 0

    self.model.eval()
    self.mean = parameters_to_vector(self.model.parameters())

    batch = next(iter(train_loader))
    X, _ = batch['data'], batch['label']
    with torch.no_grad():
        try:
            out = self.model(X[:1].to(self._device))
        except (TypeError, AttributeError):
            out = self.model(X.to(self._device))
    self.n_outputs = out.shape[-1]
    setattr(self.model, 'output_size', self.n_outputs)

    N = len(train_loader.dataset)
    for batch in train_loader:
        X, y = batch['data'], batch['label']
        self.model.zero_grad()
        X, y = X.to(self._device), y.to(self._device)
        loss_batch, H_batch = self._curv_closure(X, y, N)
        self.loss += loss_batch
        self.H += H_batch

    self.n_data += N

    # Only for diagonal Hessian approximation!
    self.mean = parameters_to_vector(self.model.last_layer.parameters())


def overrided_fit_kron(self, train_loader, override=True):
    """ Fit K-FAC approximation (OpenOOD-friendly version) """

    if override:
            self.H_facs = None

    if self.H_facs is not None:
        n_data_old = self.n_data
        n_data_new = len(train_loader.dataset)
        self._init_H()  # re-init H non-decomposed
        # discount previous Kronecker factors to sum up properly together with new ones
        self.H_facs = self._rescale_factors(self.H_facs, n_data_old / (n_data_old + n_data_new))

    # Fit
    if override:
        self._init_H()
        self.loss = 0
        self.n_data = 0

    self.model.eval()
    self.mean = parameters_to_vector(self.model.parameters())

    batch = next(iter(train_loader))
    X, _ = batch['data'], batch['label']
    with torch.no_grad():
        try:
            out = self.model(X[:1].to(self._device))
        except (TypeError, AttributeError):
            out = self.model(X.to(self._device))
    self.n_outputs = out.shape[-1]
    setattr(self.model, 'output_size', self.n_outputs)

    N = len(train_loader.dataset)
    for batch in train_loader:
        X, y = batch['data'], batch['label']
        self.model.zero_grad()
        X, y = X.to(self._device), y.to(self._device)
        loss_batch, H_batch = self._curv_closure(X, y, N)
        self.loss += loss_batch
        self.H += H_batch

    self.n_data += N

    if self.H_facs is None:
        self.H_facs = self.H
    else:
        # discount new factors that were computed assuming N = n_data_new
        self.H = self._rescale_factors(self.H, n_data_new / (n_data_new + n_data_old))
        self.H_facs += self.H
    # Decompose to self.H for all required quantities but keep H_facs for further inference
    self.H = self.H_facs.decompose(damping=self.damping)

    

  