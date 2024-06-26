from typing import Iterable, Sequence
import torch
from torch import nn, Tensor
import torch.utils
import torch.utils.data
from torch.utils.data.dataloader import _BaseDataLoaderIter, _collate_fn_t, _worker_init_fn_t

class ParallelModel(nn.Module):
    def __init__(self, make_model, k) -> None:
        super().__init__()
        self.models = nn.ModuleList([
            make_model() for _ in range(k)
        ])
        self.copy_parameters()

    @torch.no_grad()
    def copy_parameters(self):
        for m in self.models[1:]:
            for (p0, p) in zip(self.models[0].parameters(), m.parameters()):
                p.copy_(p0)

    def __len__(self):
        return len(self.models)

    def __iter__(self):
        return iter(self.models)

    def forward(self, xs):
        if not isinstance(xs, list):
            return [m(xs) for m in self.models]
        return [m(x) for (x, m) in zip(xs, self.models)]

    # dispersions
    @torch.no_grad()
    def terminal_dispersion(self, delta: Tensor=None, cross_dispersion=False, full_utilization=False):
        """
            Assuming models are intialized the same.
        """
        res = 0
        if delta is None:
            for ps in zip(*[m.parameters() for m in self.models]):
                tensor_ps = torch.stack([p.flatten() for p in ps], dim=0) 
                var = tensor_ps.var(dim=0)
                res += var.sum()
        # elif unbiased:
            # p = torch.stack([torch.cat([p.flatten() for p in m.parameters()]) for m in self.models], dim=0)[:len(delta)]
            # mean = p.mean(dim=0, keepdim=True)
            # unbiased = p.var(dim=0).sum() - 2 * (torch.inner(p, delta).mean() - torch.inner(mean.squeeze(dim=0), delta.mean(dim=0))) + delta.square().sum(dim=-1).mean()
            # biased = (p - mean - delta).square().mean(dim=0).sum()
            # print(unbiased, biased)
            # return unbiased
        else:
            p = torch.stack([torch.cat([p.flatten() for p in m.parameters()]) for m in self.models], dim=0)
            if cross_dispersion:
                if full_utilization:
                    sum = p.sum(dim=0, keepdim=True)
                    mean = (sum - p) / (len(p) - 1) 
                    p = p.unsqueeze(dim=1)
                    mean = mean.unsqueeze(dim=1)
                else:
                    p = p.unflatten(dim=0, sizes=[2, -1])
                    mean = p.mean(dim=1, keepdim=True)[[1, 0]]
            else:
                p = p.unsqueeze(dim=0)
                mean = p.mean(dim=1, keepdim=True)

            delta = delta.reshape(p.shape)
            res = (p - mean - delta).square().flatten(0, 1).mean(dim=0).sum()
        return res

    @torch.no_grad()
    def gradient_dispersion(self):
        """
            E[V_t]
        """
        res = 0 
        for ps in zip(*[m.parameters() for m in self.models]):
            if ps[0].grad is None:
                continue
            grads = torch.stack([p.grad.flatten() for p in ps], dim=0) 
            var = grads.var(dim=0)
            res += var.sum()
        return res


class ParallelLoss(nn.Module):
    def __init__(self, loss_fn: nn.Module, reduction='sum') -> None:
        super().__init__()
        self.loss_fn = loss_fn
        if reduction == 'sum':
            self.reduction = lambda list: torch.stack(list, dim=0).sum(dim=0)
        elif reduction == 'mean':
            self.reduction = lambda list: torch.stack(list,dim=0).mean(dim=0)
        else:
            raise ValueError()

    def forward(self, output, targets, *inputs):
        if isinstance(targets, Tensor):
            assert isinstance(output, list)
            targets = [targets] * len(output)
        return self.reduction([self.loss_fn(*input) for input in zip(output, targets, *inputs)])

def make_parallel_datasets(dataset: torch.utils.data.Dataset, k: int) -> 'list[torch.utils.data.Dataset]':
    return torch.utils.data.random_split(dataset, lengths=[1/k]*k)
    return [
        torch.utils.data.Subset(dataset, torch.randint(0, len(dataset), [int(p*len(dataset))]))
            for _ in range(k)
    ]

class _ParallelDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, iters: 'list[_BaseDataLoaderIter]', loader: torch.utils.data.DataLoader) -> None:
        super().__init__(loader)
        self.iters = iters
    def __next__(self):
        data = [
            next(iter) for iter in self.iters
        ]
        return (
            [tup[i] for tup in data]
                for i in range(len(data[0]))
        )

class ParallelDataloader(torch.utils.data.DataLoader):
    def __init__(self, datasets: 'list[torch.utils.data.Dataset]', *args, **kwargs):
        super().__init__(datasets)
        self.loaders = [torch.utils.data.DataLoader(dataset, *args, **kwargs) for dataset in datasets]

    def __iter__(self):
        return _ParallelDataLoaderIter([iter(loader) for loader in self.loaders], self)

    def get_iter(self, data_field: 'list[int]'):
        return _ParallelDataLoaderIter([iter(loader) for loader in self.loaders], self, data_field=data_field)

    def __len__(self):
        return len(self.loaders[0])

class _SelectedDataFieldDataLoaderIter:
    def __init__(self, iter: _BaseDataLoaderIter, loader: torch.utils.data.DataLoader, data_field: 'list[int]'=None) -> None:
        self.data_field = data_field
        self.iter = iter
    def __next__(self):
        data = next(self.iter)
        if self.data_field is None:
            return data
        data = list(data)
        return (data[i] for i in self.data_field)
class SelectedDataFieldDataLoader:
    def __init__(self, loader: ParallelDataloader, data_field='list[int]'):
        self.loader = loader
        self.data_field = data_field
    
    @property
    def loaders(self): return self.loader.loaders

    def __len__(self): return len(self.loader)

    def __iter__(self):
        return _SelectedDataFieldDataLoaderIter(iter(self.loader), self, self.data_field) 
