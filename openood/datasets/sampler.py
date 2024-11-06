from collections import defaultdict

import numpy as np
import torch


class BalancedLabelsSampler:
    """Sample labels with probabilities equal to labels frequency."""
    def __init__(self, labels, labels_per_batch, num_batches):
        counts = np.bincount(labels)
        self._probabilities = counts / np.sum(counts)
        self._labels_per_batch = labels_per_batch
        self._num_batches = num_batches

    def __iter__(self):
        for _ in range(self._num_batches):
            batch = np.random.choice(len(self._probabilities), self._labels_per_batch, p=self._probabilities, replace=False)
            yield list(batch)


class ShuffledClassBalancedBatchSampler(torch.utils.data.Sampler):
    """Sampler which extracts balanced number of samples for each class.

    Args:
        data_source: Source dataset. Labels field must be implemented.
        batch_size: Required batch size.
        samples_per_class: Number of samples for each class in the batch.
            Batch size must be a multiple of samples_per_class.
        uniform: If true, sample labels uniformly. If false, sample labels according to frequency.
    """

    def __init__(self, data_source, batch_size, samples_per_class):
        if batch_size > len(data_source):
            raise ValueError("Dataset size {} is too small for batch size {}.".format(
                len(data_source), batch_size))
        if batch_size % samples_per_class != 0:
            raise ValueError("Batch size must be a multiple of samples_per_class, but {} != K * {}.".format(
                batch_size, samples_per_class))

        self._source_len = len(data_source)
        self._batch_size = batch_size
        self._labels_per_batch = self._batch_size // samples_per_class
        self._samples_per_class = samples_per_class
        labels = np.asarray([data_source.getitem(i, only_label=True)["label"] for i in range(len(data_source))])
        self._label_sampler = BalancedLabelsSampler(labels, self._labels_per_batch,
                                                    num_batches=len(self))

        by_label = defaultdict(list)
        for i, label in enumerate(labels):
            by_label[label].append(i)
        self._by_label = list(by_label.values())
        if self._labels_per_batch > len(self._by_label):
            raise ValueError("Can't sample {} classes from dataset with {} classes.".format(
                self._labels_per_batch, len(self._by_label)))

    @property
    def batch_size(self):
        return self._batch_size

    def __iter__(self):
        for labels in self._label_sampler:
            batch = []
            for label in labels:
                batch.extend(np.random.choice(self._by_label[label], size=self._samples_per_class, replace=True))
            yield batch

    def __len__(self):
        return self._source_len // self._batch_size
