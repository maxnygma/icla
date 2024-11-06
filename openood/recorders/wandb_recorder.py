import wandb
import torch
from .base_recorder import BaseRecorder


class WandbRecorder(BaseRecorder):
    def __init__(self, config):
        super(WandbRecorder, self).__init__(config)
        # Connect to current sweep worker if already initialized
        if wandb.run is None:
            wandb.init(dir=config.output_dir,
                       project=config.recorder.project,
                       name=config.recorder.experiment,
                       group=config.recorder.group or None)

    def report(self, train_metrics, val_metrics):
        wandb.log({f'train-{k}': v for k, v in train_metrics.items()})
        wandb.log({f'val-{k}': v for k, v in val_metrics.items()})
