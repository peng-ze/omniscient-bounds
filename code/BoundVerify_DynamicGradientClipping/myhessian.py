import torch
from torch import nn, Tensor
from parallel import ParallelModel
from pyhessian import hessian
from typing import Union, Tuple, Iterable
from torch.utils.data import DataLoader
from utils import AverageMeter, ClippedCrossEntropyLoss

def HVP(hessians: 'list[Union[Tuple[Union[hessian], float], float]]', vector: 'list[Tensor]'):
    """
        (sum_i s_i H_i + sum_j C_j I) v
    """
    res: 'list[Tensor]' = [0 for _ in range(len(vector))]
    for h in hessians:
        if not isinstance(h, tuple) and not isinstance(h, list):
            with torch.no_grad():
                res = [res[i] + h * vector[i] for i in range(len(vector))]
            continue
        h, s = h
        hv = h.dataloader_hv_product(vector)
        with torch.no_grad():
            res = [res[i] + s * hv[1][i] for i in range(len(vector))]
    return res

def _inner_product(a: 'list[Tensor]', b: 'list[Tensor]'):
    res = 0
    for (aa, bb) in zip(a, b):
        res = res + torch.inner(aa.flatten(), bb.flatten())
    return res

def iHVP(parallel_model: ParallelModel, hessians:'list[list[Union[Tuple[DataLoader, float], float]]]', b: 'list[list[Tensor]]', epsilon: float, max_iteration=-1, clip=None):
    with torch.no_grad():
        vs = [[torch.zeros_like(p) for p in m] for m in b]
        rs = [[p + 0 for p in m] for m in b]
        ps = [[p + 0 for p in m] for m in b]
        done = [False for _ in range(min(len(parallel_model), len(b)))]
    
    hessians = [[((hessian(parallel_model.models[model_index], ClippedCrossEntropyLoss(clip=clip), dataloader=term[0], cuda=True), term[1]) if isinstance(term, tuple) else term) for term in m] for model_index, m in enumerate(hessians)]

    while True:
        updated = False
        mean_r = AverageMeter() 
        for i in range(len(done)):
            if done[i]:
                continue
            updated = True

            h = hessians[i]
            v, r, p = vs[i], rs[i], ps[i]
            assert torch.is_grad_enabled()
            hp = HVP(h, p) 

            with torch.no_grad():
                alpha =  _inner_product(r, r) / _inner_product(p, hp)
                new_v = [vv + alpha * pp for (vv, pp) in zip(v, p)]
                new_r = [rr - alpha * hhp for (rr, hhp) in zip(r, hp)]
                beta = _inner_product(new_r, new_r) / _inner_product(r, r)
                new_p = [new_rr + beta * pp for (new_rr, pp) in zip(new_r, p)]

                vs[i] = new_v; ps[i] = new_p; rs[i] = new_r

                delta = _inner_product(new_r, new_r) 
                mean_r.update(delta / _inner_product(b[i], b[i]))
                if delta <= epsilon * _inner_product(b[i], b[i]):
                    done[i] = True

        print(mean_r.avg, len(done) - sum(done))
        max_iteration -= 1
        if max_iteration == 0:
            break
        if not updated:
            break

    return vs


