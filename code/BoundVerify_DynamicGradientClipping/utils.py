import numpy as np
import torch
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import os

class MyDataset(Dataset):
    def __init__(self, args, _train=True):
        self.ds = args.dataset

        if self.ds == "mnist":
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            if args.label_corrupt_prob == 0:
                self.mnist = datasets.MNIST(root=args.data_path, train=_train, download=False, transform=transform)
            else:
                self.mnist = MNISTRandomLabels(root=args.data_path, train=_train, download=True,
                                               transform=transform, corrupt_prob=args.label_corrupt_prob)

        if self.ds == "cifar10":
            normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                             std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])
            if args.label_corrupt_prob == 0:
                self.cifar10 = datasets.CIFAR10(root=args.data_path, train=_train, download=False, transform=transform)
            else:
                self.cifar10 = CIFAR10RandomLabels(root=args.data_path, train=_train, download=False,
                                                   transform=transform, corrupt_prob=args.label_corrupt_prob)


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
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
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