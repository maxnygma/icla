from openood.pipelines import get_pipeline
from openood.utils import launch, seed_everything, setup_config

import os
import torch

import warnings


def main(config):
    """Main entrance. Config is all you need to provide to run the code. Config
    should be provided in the format of YAML and can be modified with command
    line.

    Example:
        python main.py \
            --config configs/datasets/mnist_datasets.yml \
            configs/train/mnist_baseline.yml \
            --dataset.image_size 32 \
            --network res18

    Note:
        A config file is the minimum requirement.
        You don't need to add "--config_key new_value"
        if you don't have anything to modify.
    """
    if config.seed is not None:
        seed_everything(config.seed)
    pipeline = get_pipeline(config)
    pipeline.run()


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    config = setup_config()

    launch(
        main,
        config.num_gpus,
        num_machines=config.num_machines,
        machine_rank=config.machine_rank,
        dist_url='auto',
        args=(config, ),
    )
