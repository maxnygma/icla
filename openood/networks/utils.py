from types import MethodType

import mmcv
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from mmcls.apis import init_model

import openood.utils.comm as comm
from openood.utils import get_config_default

from .resnet18_32x32 import ResNet18_32x32
from .resnet18_64x64 import ResNet18_64x64
from .resnet18_224x224 import ResNet18_224x224
from .resnet18_256x256 import ResNet18_256x256
from .resnet50 import ResNet50
from .wrn import WideResNet


def get_network(network_config):
    num_classes = network_config.num_classes
    num_models = get_config_default(network_config, "num_models", 5)

    discriminator_type = get_config_default(network_config, "discriminator_type", "")
    if discriminator_type == "dice":
        from .discriminator import Discriminator
        discriminator_cls = Discriminator
    elif not discriminator_type:
        discriminator_cls = None
    else:
        raise ValueError(f"Unknown discriminator type: {discriminator_type}")

    if network_config.name == 'resnet18_32x32':
        net = ResNet18_32x32(num_classes=num_classes)

    elif network_config.name == 'resnet18_256x256':
        net = ResNet18_256x256(num_classes=num_classes)

    elif network_config.name == 'resnet18_64x64':
        net = ResNet18_64x64(num_classes=num_classes)

    elif network_config.name == 'resnet18_224x224':
        net = ResNet18_224x224(num_classes=num_classes)

    elif network_config.name == 'resnet50':
        net = ResNet50(num_classes=num_classes)

    elif network_config.name == 'wrn':
        net = WideResNet(depth=28,
                         widen_factor=10,
                         dropRate=0.0,
                         num_classes=num_classes)

    elif network_config.name == 'densenet':
        net = DenseNet3(depth=100,
                        growth_rate=12,
                        reduction=0.5,
                        bottleneck=True,
                        dropRate=0.0,
                        num_classes=num_classes)

    elif network_config.name == 'vit':
        cfg = mmcv.Config.fromfile(network_config.model)
        net = init_model(cfg, network_config.checkpoint, 0)
        net.get_fc = MethodType(
            lambda self: (self.head.layers.head.weight.cpu().numpy(),
                          self.head.layers.head.bias.cpu().numpy()), net)

    else:
        raise Exception('Unexpected Network Architecture!')

    if network_config.num_gpus > 1:
        if type(net) is dict:
            for key, subnet in zip(net.keys(), net.values()):
                net[key] = torch.nn.parallel.DistributedDataParallel(
                    subnet,
                    device_ids=[comm.get_local_rank()],
                    broadcast_buffers=True)
        else:
            net = torch.nn.parallel.DistributedDataParallel(
                net.cuda(),
                device_ids=[comm.get_local_rank()],
                broadcast_buffers=True)

    if network_config.num_gpus > 0:
        if type(net) is dict:
            for subnet in net.values():
                subnet.cuda()
        else:
            net.cuda()
    cudnn.benchmark = True
    return net
