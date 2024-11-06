import torch

from openood.datasets import get_dataloader
from openood.evaluators import get_evaluator
from openood.networks import get_network
from openood.utils import setup_logger


class TestAccPipeline:
    def __init__(self, config) -> None:
        self.config = config

    def run(self):
        # generate output directory and save the full config file
        setup_logger(self.config)

        # get dataloader
        loader_dict = get_dataloader(self.config)
        test_loader = loader_dict['test']

        # init network
        net = get_network(self.config.network)
        net.load_state_dict(torch.load(self.config.network.checkpoint), strict=True)
        net.eval()
        print(f'Loaded {self.config.network.checkpoint}')

        # init evaluator
        evaluator = get_evaluator(self.config)

        # start calculating accuracy
        print('\nStart evaluation...', flush=True)

        # (Visualization, PROLY FOR DELETION)
        test_metrics = evaluator.eval_acc(net, test_loader, visualize_embeddings=self.config.visualize_penum_layer_embeddings)
        # test_metrics = evaluator.eval_acc(net, test_loader, visualize_embeddings=False)
        
        print('\nComplete Evaluation, accuracy {:.2f}%'.format(
            100 * test_metrics['acc']),
              flush=True)
