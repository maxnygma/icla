from typing import Any

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from copy import deepcopy
from torch.optim.swa_utils import AveragedModel


class BasePostprocessor:
    def __init__(self, config):
        self.config = config
        self.checkpoint_name = self.config.network.checkpoint

    def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
        self.net = net
        if self.config.network.swa:
            self.net = AveragedModel(self.net)
            print('SWA is loaded')

        # if not self.config.network.ila_debug:
        self.net.load_state_dict(torch.load(self.checkpoint_name), strict=True)
        self.net.eval()
        print(f'Loaded {self.checkpoint_name}')

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any):
        output = self.net(data)
        score = torch.softmax(output, dim=1)
        conf, pred = torch.max(score, dim=1)
        return pred, conf

    def inference(self, net: nn.Module, data_loader: DataLoader):
        pred_list, conf_list, label_list = [], [], []
        for batch in data_loader:
            data = batch['data'].cuda()
            label = batch['label'].cuda()
            pred, conf = self.postprocess(net, data)
            for idx in range(len(data)):
                pred_list.append(pred[idx].cpu().tolist())
                conf_list.append(conf[idx].cpu().tolist())
                label_list.append(label[idx].cpu().tolist())

        # convert values into numpy array
        pred_list = np.array(pred_list, dtype=int)
        conf_list = np.array(conf_list)
        label_list = np.array(label_list, dtype=int)

        return pred_list, conf_list, label_list
    
    # def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
    #     pass


# class BasePostprocessor:
#     def __init__(self, config):
#         self.config = config
#         self.checkpoint_name = self.config.network.checkpoint

#     def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
#         if self.config.network.swa:
#             _net = deepcopy(net)
#             net = AveragedModel(_net)

#         print(os.listdir('results/c10-swa-base/1'))

#         net.load_state_dict(torch.load(self.checkpoint_name), strict=True)
#         net.eval()

#     @torch.no_grad()
#     def postprocess(self, net: nn.Module, data: Any):
#         output = net(data)
#         score = torch.softmax(output, dim=1)
#         conf, pred = torch.max(score, dim=1)
#         return pred, conf

#     def inference(self, net: nn.Module, data_loader: DataLoader):
#         pred_list, conf_list, label_list = [], [], []
#         for batch in data_loader:
#             data = batch['data'].cuda()
#             label = batch['label'].cuda()
#             pred, conf = self.postprocess(net, data)
#             for idx in range(len(data)):
#                 pred_list.append(pred[idx].cpu().tolist())
#                 conf_list.append(conf[idx].cpu().tolist())
#                 label_list.append(label[idx].cpu().tolist())

#         # convert values into numpy array
#         pred_list = np.array(pred_list, dtype=int)
#         conf_list = np.array(conf_list)
#         label_list = np.array(label_list, dtype=int)

#         return pred_list, conf_list, label_list
    
#     # def setup(self, net: nn.Module, id_loader_dict, ood_loader_dict):
#     #     pass

