from openood.utils import Config

from .test_preprocessor import TestStandardPreProcessor


def get_preprocessor(config: Config, split):
    test_preprocessors = {
        'base': TestStandardPreProcessor,
    }

    return test_preprocessors[config.preprocessor.name](config)
