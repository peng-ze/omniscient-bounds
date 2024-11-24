import numpy as np
import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL.Image import Image
import os
from typing import Iterable
import gc

def optional_to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    return torch.nn.functional.to_tensor(x)

class MyDataset(Dataset):
    normalize = transforms.Normalize(
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
    )
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ])
    mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    def __init__(self, args, _train=True, no_transform=False):
        self.ds = args.dataset

        if self.ds == "mnist":
            if no_transform:
                transform = None
            else:
                transform =  self.mnist_transform
            if args.label_corrupt_prob == 0:
                self.mnist = datasets.MNIST(root=args.data_path, train=_train, download=True, transform=transform)
            else:
                self.mnist = MNISTRandomLabels(root=args.data_path, train=_train, download=True,
                                               transform=transform, corrupt_prob=args.label_corrupt_prob)
            args.num_classes = 10

        if self.ds == "cifar10":
            if not no_transform:
                transform = self.transform 
            else:
                transform = None
            if args.label_corrupt_prob == 0:
                self.cifar10 = datasets.CIFAR10(root=args.data_path, train=_train, download=True, transform=transform)
            else:
                self.cifar10 = CIFAR10RandomLabels(root=args.data_path, train=_train, download=False,
                                                   transform=transform, corrupt_prob=args.label_corrupt_prob)
            args.num_classes = 10


    def __getitem__(self, index):
        if self.ds == "mnist":
            data, target = self.mnist[index]

        # Your transformations here (or set it in CIFAR10)
        if self.ds == "cifar10":
            data, target = self.cifar10[index]
        return data, target, index

    def __len__(self):
        if self.ds == "mnist":
            return len(self.mnist)
        if self.ds == "cifar10":
            return len(self.cifar10)

class InMemoryDataset(Dataset):
    def __init__(self, dataset: Dataset, device=None) -> None:
        super().__init__()
        self.dataset = dataset
        self.load_samples()
        if device is not None:
            self.load_to_device(device)

    def _load_device(self, d, device='cpu'):
        if not isinstance(d, tuple) and not isinstance(d, list):
            return d
        return [e.to(device) if isinstance(e, torch.Tensor) else e for e in d]

    def load_samples(self):
        self.samples = [
            d for d in self.dataset
        ]

    def load_to_device(self, device):
        self.samples = [
            self._load_device(d, device) for d in self.samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]

class AugmentationDataset(Dataset):
    def __init__(self, dataset: Dataset, transform=lambda x:x, target_transform=lambda x:x) -> None:
        super().__init__()
        self.dataset = dataset
        self.transform = transform
        self.target_transform = target_transform 

    def do_transform(self, Z, transform):
        if isinstance(Z, torch.Tensor) or isinstance(Z, int) or isinstance(Z, Image):
            return transform(Z)
        return [self.do_transform(z, transform) for z in Z]

    def __len__(self):  return len(self.dataset)
    
    def __getitem__(self, index):
        data = list(self.dataset[index])
        X = data[0]; Y = data[1]
        return self.do_transform(X, self.transform), self.do_transform(Y, self.target_transform), *data[2:]


class MNISTRandomLabels(datasets.MNIST):
    """CIFAR10 dataset, with support for randomly corrupt labels.
    Params
    ------
    corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
    num_classes: int
    Default 10. The number of classes in the dataset.
    """
    def __init__(self, corrupt_prob=0.0, **kwargs):
        super(MNISTRandomLabels, self).__init__(**kwargs)
        self.n_classes = 10
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        # np.random.seed(12345)
        self.mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, self.mask.sum())
        labels[self.mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]

        self.targets = labels

class CIFAR10RandomLabels(datasets.CIFAR10):
    """CIFAR10 dataset, with support for randomly corrupt labels.
    Params
    ------
    corrupt_prob: float
    Default 0.0. The probability of a label being replaced with
    random label.
    num_classes: int
    Default 10. The number of classes in the dataset.
    """
    def __init__(self, corrupt_prob=0.0, **kwargs):
        super(CIFAR10RandomLabels, self).__init__(**kwargs)
        self.n_classes = 10
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        # np.random.seed(12345)
        self.mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, self.mask.sum())
        labels[self.mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]

        self.targets = labels
        
        
        
class AverageMeter(object):
    #"""Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch, args):
    # """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // 150)) * (0.1 ** (epoch // 225))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def update_hyparam(optimizer, lambda0, lambda1, args):
    # """Sets the learning rate to the initial LR decayed by 10 after 150 and 225 epochs"""
    lr = args.learning_rate * lambda0
    weight_decay = args.weight_decay * lambda1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        param_group['weight_decay'] = weight_decay
        
def weight_init(m):
    '''
    Usage:
        model = Model()
        model.apply(weight_init)
    '''
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        if m.bias is not None:
            init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
        
def checkdir(directory):
            if not os.path.exists(directory):
                os.makedirs(directory)


def clip_grad_norm_(parameters, max_norm: float, norm_type: float = 2.0) -> torch.Tensor:

    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    for p in parameters:
        p.grad.detach().mul_(clip_coef.to(p.grad.device))
    return total_norm

class ClippedLoss(nn.Module):
    def __init__(self, unreduced_loss: nn.Module, clip, reduction='mean') -> None:
        super().__init__()
        self.loss_fn = unreduced_loss
        self.clip = clip
        self.reduction = reduction

    def forward(self, *inputs, **kwargs):
        losses = self.loss_fn(*inputs, **kwargs)
        clipped_losses = losses.clamp(max=self.clip, min=0) if self.clip is not None else losses
        if self.reduction == 'mean':
            res = clipped_losses.mean()
        elif self.reduction == 'sum':
            res = clipped_losses.sum()
        elif self.reduction == 'none':
            res = clipped_losses
        else:
            raise ValueError()
        return res

def ClippedCrossEntropyLoss(
    clip=None,
    weight=None,
    size_average=None,
    ignore_index=-100,
    reduction: str = 'mean',
    label_smoothing: float = 0,
):
    return ClippedLoss(nn.CrossEntropyLoss(weight, size_average, ignore_index, reduce=None, reduction='none', label_smoothing=label_smoothing), clip=clip, reduction=reduction)


def clean_cache():
    gc.collect()
    torch.cuda.empty_cache()


from torch.optim.lr_scheduler import CosineAnnealingLR
import inspect

class UnionDataset(Dataset):
    def __init__(self, *datasets):
        super().__init__()
        self.datasets = datasets

    def __len__(self):
        return sum(len(d) for d in self.datasets)

    def __getitem__(self, index):
        for d in self.datasets:
            if index < len(d):
                return d[index]
            index -= len(d)
        raise IndexError("Index out of range")

class BackupModelParams:
    def __init__(self, model: nn.Module):
        self.model = model
        self.backup = None

    @torch.no_grad()
    def __enter__(self):
        torch.cuda.synchronize()
        self.backup = [p.data.clone().detach() for p in self.model.parameters()]
        torch.cuda.synchronize()
        return self

    @torch.no_grad()
    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        for p, backup_p in zip(self.model.parameters(), self.backup):
            p.data.copy_(backup_p)
        torch.cuda.synchronize()
