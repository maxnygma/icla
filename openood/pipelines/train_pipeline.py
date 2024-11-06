import torch
import os

import openood.utils.comm as comm
from openood.datasets import get_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.recorders import get_recorder
from openood.trainers import get_trainer
from openood.utils import setup_logger

from openood.utils.llla_utils import swa_update_bn_adapted
from openood.utils import seed_everything


class TrainPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        seed_everything(self.config.seed)

        # generate output directory and save the full config file
        setup_logger(self.config)

        # get dataloader
        loader_dict = get_dataloader(self.config)
        train_loader, val_loader = loader_dict['train'], loader_dict['val']
        test_loader = loader_dict['test']

        # init network
        net = get_network(self.config.network)

        # init trainer and evaluator
        trainer = get_trainer(net, train_loader, self.config)
        evaluator = get_evaluator(self.config)

        if comm.is_main_process():
            # init recorder
            recorder = get_recorder(self.config)

            print('Start training...', flush=True)

        dataset_config = self.config.dataset
        for epoch_idx in range(1, self.config.optimizer.num_epochs + 1):
            # When training in distributed mode, sampler's epoch should be updated each epoch:
            # https://github.com/pytorch/pytorch/blob/main/torch/utils/data/distributed.py
            if dataset_config.num_gpus * dataset_config.num_machines > 1:
                if isinstance(train_loader.batch_sampler, torch.utils.data.distributed.DistributedSampler):
                    train_loader.batch_sampler.set_epoch(epoch_idx)
                if isinstance(train_loader.sampler, torch.utils.data.distributed.DistributedSampler):
                    train_loader.sampler.set_epoch(epoch_idx)

            # train and eval the model
            net, train_metrics = trainer.train_epoch(epoch_idx)

            # Update BN stats in case of SWA
            if self.config.trainer.name == 'swa' and epoch_idx == self.config.optimizer.num_epochs:
                # torch.optim.swa_utils.update_bn(train_loader, net)
                swa_update_bn_adapted(train_loader, net, 'cuda')

            val_metrics = evaluator.eval_acc(net, val_loader, None, epoch_idx)
            comm.synchronize()
            if comm.is_main_process():
                # save model and report the result
                recorder.save_model(net, val_metrics)
                recorder.report(train_metrics, val_metrics)

        if comm.is_main_process():
            recorder.summary()
            print(u'\u2500' * 70, flush=True)

            # evaluate on test set
            print('Start testing...', flush=True)

        test_metrics = evaluator.eval_acc(net, test_loader)

        if comm.is_main_process():
            print('\nComplete Evaluation, Last accuracy {:.2f}'.format(
                100.0 * test_metrics['acc']),
                  flush=True)
            print('Completed!', flush=True)
