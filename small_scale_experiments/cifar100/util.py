import torch

import torchvision
import os
from typing import Any, Callable, Optional, Tuple
import pickle
import numpy as np
from sklearn.utils import shuffle

class AverageMeter(object):
    def __init__(self, name=None, fmt='.6f'):
        fmtstr = f'{{val:{fmt}}} ({{avg:{fmt}}})'
        if name is not None:
            fmtstr = name + ' ' + fmtstr
        self.fmtstr = fmtstr
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

    @property
    def avg(self):
        avg = self.sum / self.count
        if isinstance(avg, torch.Tensor):
            avg = avg.item()
        return avg

    def __str__(self):
        val = self.val
        if isinstance(val, torch.Tensor):
            val = val.item()
        return self.fmtstr.format(val=val, avg=self.avg)

# Datasets for K = 1 
class TwoAugUnsupervisedDataset(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        return self.transform(image), self.transform(image)

    def __len__(self):
        return len(self.dataset)

# Datasets for K = 4
class MultiAugUnsupervisedDataset(torch.utils.data.Dataset):
    r"""Returns two augmentation and no labels."""
    def __init__(self, dataset, transform, Ny):
        self.dataset = dataset
        self.transform = transform
        self.Ny = Ny

    def __getitem__(self, index):
        image, _ = self.dataset[index]
        samples = []
        for _ in range(self.Ny):
            samples.append(self.transform(image))
        return samples

    def __len__(self):
        return len(self.dataset)

# Imbalanced datasets
class Imbalanced_CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(
            self,
            root,
            train=True,
            transform=None,
            target_transform=None,
            download=False,
    ):
        super(Imbalanced_CIFAR100, self).__init__(root, train=train, transform=transform,
                                                  target_transform=target_transform, download=download)

        # sample imbalanced data here
        self.targets = np.array(self.targets)
        sampled_indices = []
        n_class = self.targets.max()+1
        for i in range(n_class):
            indices = np.where(self.targets==i)[0]
            # indices = shuffle(indices, random_state=2021)[:int(((i//10+1.0)/10)*len(indices))]
            indices = shuffle(indices, random_state=2021)[:int(((np.exp(i//10+1.0))/np.exp(10))*len(indices))]
            sampled_indices.append(indices)
        sampled_indices = np.concatenate(sampled_indices)
        self.data = self.data[sampled_indices]
        self.targets = self.targets[sampled_indices].tolist()

        # re-run other init codes
        self._load_meta()
