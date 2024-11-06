import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from laplace import Laplace
from laplace.curvature import AsdlEF

import openood.utils.comm as comm
from openood.utils import Config

from openood.utils import ASAM

from .lr_scheduler import cosine_annealing


class BaseTrainer:
    def __init__(self, net: nn.Module, train_loader: DataLoader,
                 config: Config) -> None:

        self.net = net
        self.train_loader = train_loader
        self.config = config

        self.optimizer = torch.optim.SGD(
            net.parameters(),
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                config.optimizer.num_epochs * len(train_loader),
                1,
                1e-6 / config.optimizer.lr,
            ),
        )   

        # SAM-like optimizer
        if config.trainer.use_asam:
            self.minimizer = ASAM(optimizer=self.optimizer, model=self.net, rho=0.5, eta=0.01)

        # For Fisher penalty
        if config.trainer.fisher_penalty_lam != 0:
            self._la = Laplace(model=self.net, likelihood='classification', subset_of_weights='last_layer',
                            hessian_structure='diag', backend=AsdlEF, last_layer_name='fc')

    def train_epoch(self, epoch_idx):
        self.net.train()

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(range(1,
                                     len(train_dataiter) + 1),
                               desc='Epoch {:03d}: '.format(epoch_idx),
                               position=0,
                               leave=True,
                               disable=not comm.is_main_process()):
            batch = next(train_dataiter)
            data = batch['data'].cuda()
            target = batch['label'].cuda()

            # forward
            logits_classifier = self.net(data)
            loss = F.cross_entropy(logits_classifier, target)
            
            # Fisher penalty
            if self.config.trainer.fisher_penalty_lam != 0:
                y_hat = torch.randint(0, self.config.dataset.num_classes - 1, target.shape, device='cuda')
                Gs, _ = self._la.backend.gradients(data, y_hat) # gradient w.r.t to parameters
                fisher_penalty = torch.norm(Gs, p=2)
                loss += self.config.trainer.fisher_penalty_lam * fisher_penalty # 0.000001

            # backward
            self.optimizer.zero_grad()
            loss.backward()

            if self.config.trainer.use_asam:
                # Ascent step
                self.minimizer.ascent_step()

                # Descent step
                F.cross_entropy(self.net(data), target).mean().backward()
                self.minimizer.descent_step()
            else:
                self.optimizer.step()
            
            self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2

        # comm.synchronize()

        metrics = {}
        metrics['epoch_idx'] = epoch_idx
        metrics['loss'] = self.save_metrics(loss_avg)
        metrics['lr'] = self.optimizer.param_groups[0]['lr']

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])

        return total_losses_reduced
