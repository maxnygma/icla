import ast
import io
import logging
import os

import torch
from PIL import Image, ImageFile

from .base_dataset import BaseDataset

# to fix "OSError: image file is truncated"
ImageFile.LOAD_TRUNCATED_IMAGES = True


class Convert:
    def __init__(self, mode='RGB'):
        self.mode = mode

    def __call__(self, image):
        return image.convert(self.mode)


class ImglistDataset(BaseDataset):
    def __init__(self,
                 name,
                 imglist_pth,
                 data_dir,
                 num_classes,
                 preprocessor,
                 data_aux_preprocessor,
                 filter_classes=None,
                 maxlen=None,
                 dummy_read=False,
                 dummy_size=None,
                 cache=False,
                 **kwargs):
        super(ImglistDataset, self).__init__(**kwargs)

        self.name = name
        with open(imglist_pth) as imgfile:
            self.imglist = imgfile.readlines()
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.preprocessor = preprocessor
        self.transform_image = preprocessor
        self.transform_aux_image = data_aux_preprocessor
        self.maxlen = maxlen
        self.dummy_read = dummy_read
        self.dummy_size = dummy_size
        if dummy_read and dummy_size is None:
            raise ValueError(
                'if dummy_read is True, should provide dummy_size')
        self.cache = cache
        self.images = [None] * len(self)

        # Filter out classes (only for CIFAR-100)
        if filter_classes is None:
            return
        
        print(filter_classes)
        filter_classes = ast.literal_eval(''.join(filter_classes))

        if filter_classes != [0]:
            if name == 'cifar100_train' or name == 'cifar100_val' or name == 'cifar100_test':
                self.imglist = [x for x in self.imglist if int(x.strip('\n').split(' ', 1)[1]) in filter_classes]
                self.num_classes = len(filter_classes) # OVERRIDE IN CONFIG 5 -> 100?

                replacement_dict = self.array_to_dict(filter_classes)
                self.imglist = [' '.join(str(replacement_dict[int(x)]) if x.isdigit() and int(x) in replacement_dict else x for x in s.split()) for s in self.imglist]

                print('Classes are filtered')
            else:
                raise ValueError('Filtering of classes is supported only for CIFAR-100')

    def __len__(self):
        if self.maxlen is None:
            return len(self.imglist)
        else:
            return min(len(self.imglist), self.maxlen)

    def getitem(self, index, only_label=False):
        line = self.imglist[index].strip('\n')
        tokens = line.split(' ', 1)
        image_name, extra_str = tokens[0], tokens[1]
        if self.data_dir != '' and image_name.startswith('/'):
            raise RuntimeError('image_name starts with "/"')
        path = os.path.join(self.data_dir, image_name)
        sample = dict()
        sample['image_name'] = image_name
        kwargs = {'name': self.name, 'path': path, 'tokens': tokens}
        # some preprocessor methods require setup
        self.preprocessor.setup(**kwargs)
        try:
            if not only_label:
                if self.dummy_size is not None:
                    sample['data'] = torch.rand(self.dummy_size)
                else:
                    assert not self.dummy_read
                    if self.images[index] is not None:
                        image = self.images[index]
                    else:
                        image = Image.open(path).convert('RGB')
                        if self.cache:
                            self.images[index] = image
                    sample['data'] = self.transform_image(image)
                    sample['data_aux'] = self.transform_aux_image(image)
            extras = ast.literal_eval(extra_str)
            try:
                for key, value in extras.items():
                    sample[key] = value
                # if you use dic the code below will need ['label']
                sample['label'] = 0
            except AttributeError:
                sample['label'] = int(extra_str)
            # Generate Soft Label
            soft_label = torch.Tensor(self.num_classes)
            if sample['label'] < 0:
                soft_label.fill_(1.0 / self.num_classes)
            else:
                # print(self.num_classes, sample['label'])
                soft_label.fill_(0)
                soft_label[sample['label']] = 1
            sample['soft_label'] = soft_label

        except Exception as e:
            logging.error('[{}] broken'.format(path))
            raise e
        return sample

    # def array_to_dict(self, arr):
    #     res = {}
    #     # Max value in array less than length of array
    #     k = max(i for i in arr if i < len(arr)) if any(i < len(arr) for i in arr) else 0
    #     for i, x in enumerate(sorted(arr)):
    #         if x <= k:
    #             res[x] = i
    #         else:
    #             res[x] = k + i

    #     return res

    def array_to_dict(self, arr):
        res = {}
        for i, k in enumerate(sorted(arr)):
            res[k] = i

        return res