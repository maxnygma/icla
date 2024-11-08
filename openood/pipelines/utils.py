from openood.utils import Config

from .feat_extract_pipeline import FeatExtractPipeline
from .finetune_pipeline import FinetunePipeline
from .test_acc_pipeline import TestAccPipeline
from .test_ood_pipeline import TestOODPipeline
from .train_pipeline import TrainPipeline
from .test_ood_pipeline_aps import TestOODPipelineAPS

def get_pipeline(config: Config):
    pipelines = {
        'train': TrainPipeline,
        'finetune': FinetunePipeline,
        'test_acc': TestAccPipeline,
        'feat_extract': FeatExtractPipeline,
        'test_ood': TestOODPipeline,
        'test_ood_aps': TestOODPipelineAPS,
    }

    return pipelines[config.pipeline.name](config)
