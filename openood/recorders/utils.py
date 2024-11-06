from openood.utils import Config

from .base_recorder import BaseRecorder
from .wandb_recorder import WandbRecorder


def get_recorder(config: Config):
    recorders = {
        'base': BaseRecorder,
        'wandb': WandbRecorder,
    }

    return recorders[config.recorder.name](config)
