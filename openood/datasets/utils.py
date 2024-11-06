import torch
from numpy import load

from openood.preprocessors.test_preprocessor import TestStandardPreProcessor
from openood.preprocessors.utils import get_preprocessor
from openood.utils import Config, get_config_default

from .feature_dataset import FeatDataset
from .imglist_dataset import ImglistDataset
from .sampler import ShuffledClassBalancedBatchSampler


def make_dataloader(*args, **kwargs):
    loader = torch.utils.data.DataLoader(*args, **kwargs)
    loader._init_params = (args, kwargs)
    return loader


def clone_dataloader(loader):
    """Make a new dataloader with the same set of parameters."""
    args, kwargs = loader._init_params
    return make_dataloader(*args, **kwargs)


def get_dataloader(config: Config):
    # prepare a dataloader dictionary
    dataset_config = config.dataset
    dataloader_dict = {}
    for split in dataset_config.split_names:
        split_config = dataset_config[split]
        preprocessor = get_preprocessor(config, split)
        # weak augmentation for data_aux
        data_aux_preprocessor = TestStandardPreProcessor(config)
        CustomDataset = eval(split_config.dataset_class)
        kwargs = {}
        if get_config_default(split_config, "cache", None) is not None:
            kwargs["cache"] = split_config.cache
        dataset = CustomDataset(name=dataset_config.name + '_' + split,
                                imglist_pth=split_config.imglist_pth,
                                data_dir=split_config.data_dir,
                                num_classes=dataset_config.num_classes,
                                preprocessor=preprocessor,
                                data_aux_preprocessor=data_aux_preprocessor,
                                **kwargs)
        kwargs = {}
        if dataset_config.num_gpus * dataset_config.num_machines > 1:
            kwargs["sampler"] = torch.utils.data.distributed.DistributedSampler(dataset)
            split_config.shuffle = False
        if (split == "train") and (get_config_default(split_config, "samples_per_class", -1) != -1):
            if "sampler" in kwargs:
                raise NotImplementedError("Can't mix multi-node with grouped sampling.")
            if not split_config.shuffle:
                raise ValueError("Grouped sampling requires shuffle.")
            kwargs["batch_sampler"] = ShuffledClassBalancedBatchSampler(dataset,
                                                                        batch_size=split_config.batch_size,
                                                                        samples_per_class=split_config.samples_per_class)
        else:
            kwargs["batch_size"] = split_config.batch_size
            kwargs["shuffle"] = split_config.shuffle

        dataloader = make_dataloader(dataset,
                                     num_workers=dataset_config.num_workers,
                                     **kwargs)

        dataloader_dict[split] = dataloader
    return dataloader_dict


def get_ood_dataloader(config: Config):
    # specify custom dataset class
    ood_config = config.ood_dataset
    CustomDataset = eval(ood_config.dataset_class)
    dataloader_dict = {}
    for split in ood_config.split_names:
        split_config = ood_config[split]
        preprocessor = get_preprocessor(config, split)
        data_aux_preprocessor = TestStandardPreProcessor(config)
        if split == 'val':
            # validation set
            dataset = CustomDataset(
                name=ood_config.name + '_' + split,
                imglist_pth=split_config.imglist_pth,
                data_dir=split_config.data_dir,
                num_classes=ood_config.num_classes,
                preprocessor=preprocessor,
                data_aux_preprocessor=data_aux_preprocessor)
            dataloader = make_dataloader(dataset,
                                         batch_size=ood_config.batch_size,
                                         shuffle=ood_config.shuffle,
                                         num_workers=ood_config.num_workers)
            dataloader_dict[split] = dataloader
        else:
            # dataloaders for csid, nearood, farood
            sub_dataloader_dict = {}
            for dataset_name in split_config.datasets:
                dataset_config = split_config[dataset_name]
                dataset = CustomDataset(
                    name=ood_config.name + '_' + split,
                    imglist_pth=dataset_config.imglist_pth,
                    data_dir=dataset_config.data_dir,
                    num_classes=ood_config.num_classes,
                    preprocessor=preprocessor,
                    data_aux_preprocessor=data_aux_preprocessor)
                dataloader = make_dataloader(dataset,
                                             batch_size=ood_config.batch_size,
                                             shuffle=ood_config.shuffle,
                                             num_workers=ood_config.num_workers)
                sub_dataloader_dict[dataset_name] = dataloader
            dataloader_dict[split] = sub_dataloader_dict

    return dataloader_dict


def get_feature_dataloader(dataset_config: Config):
    # load in the cached feature
    loaded_data = load(dataset_config.feat_path, allow_pickle=True)
    total_feat = torch.from_numpy(loaded_data['feat_list'])
    del loaded_data
    # reshape the vector to fit in to the network
    total_feat.unsqueeze_(-1).unsqueeze_(-1)
    # let's see what we got here should be something like:
    # torch.Size([total_num, channel_size, 1, 1])
    print('Loaded feature size: {}'.format(total_feat.shape))

    split_config = dataset_config['train']

    dataset = FeatDataset(feat=total_feat)
    dataloader = make_dataloader(dataset,
                                 batch_size=split_config.batch_size,
                                 shuffle=split_config.shuffle,
                                 num_workers=dataset_config.num_workers)

    return dataloader
