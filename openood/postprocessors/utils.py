from openood.utils import Config

from .base_postprocessor import BasePostprocessor
from .llla_postprocessor import LLLAPostprocessor


def get_postprocessor(config: Config):
    postprocessors = {
        'msp': BasePostprocessor,
        'llla': LLLAPostprocessor,
    }

    return postprocessors[config.postprocessor.name](config)
