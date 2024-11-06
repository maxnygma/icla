from openood.utils import Config

from .base_evaluator import BaseEvaluator
from .ood_evaluator import OODEvaluator
from .ece_la_evaluator import ECELAEvaluator


def get_evaluator(config: Config):
    evaluators = {
        'base': BaseEvaluator,
        'ood': OODEvaluator,
        'ece_la': ECELAEvaluator
    }
    return evaluators[config.evaluator.name](config)
