import torch
from torch import nn, Tensor
from abc import abstractmethod

import torch.utils
from parallel import ParallelModel, ParallelDataloader, SelectedDataFieldDataLoader, ParallelLoss
from pyhessian import hessian  
from torch.utils.data import DataLoader
from myhessian import iHVP, HVP, _inner_product
from utils import ClippedCrossEntropyLoss, clean_cache
from tqdm.auto import tqdm
import numpy as np
def average_hessian_trace(clip, model, dataloader: DataLoader):
    trace_sum = 0
    count = 0
    clean_cache()
    for X, Y in tqdm(dataloader, "Computing Hessian trace"):
        model.zero_grad(True)
        batch_mean_trace = np.mean(hessian(model, criterion=ClippedCrossEntropyLoss(clip), data=(X, Y), cuda=True).trace())
        trace_sum = trace_sum + batch_mean_trace * len(Y)
        count += len(Y)
        model.zero_grad(True)
        clean_cache()
    return trace_sum / count


class Bound(nn.Module):
    name: str = None
    def __init__(self) -> None:
        super().__init__()
        self.computed_value = None
    @abstractmethod
    def init(self, parallel_model: ParallelModel):
        pass
    @abstractmethod
    def update(self, parallel_model: ParallelModel, lr: float, *args, **kwargs):
        pass
    @abstractmethod
    def update_after_step(self, parallel_model: ParallelModel, lr: float, *args, **kwargs):
        pass
    @abstractmethod
    def trajectory_term(self, parallel_model: ParallelModel, *args, **kwargs):
        pass
    @abstractmethod
    def compute(self, parallel_model: ParallelModel, *args, **kwargs):
        pass
    
    def forget(self, C: float, parallel_model: ParallelModel, parallel_training_data_loader: ParallelDataloader, val_data_loader: DataLoader, val2_data_loader: DataLoader):
        return float(self.trajectory_term(parallel_model)), 0, None, {}

class GradientDispersionBound(Bound):
    name = "gradient_dispersion"
    def __init__(self) -> None:
        super().__init__()
        self._n_iter = nn.Parameter(torch.zeros([1], dtype=torch.int), requires_grad=False)
        self._gradient_dispersion = nn.Parameter(torch.zeros([1]), requires_grad=False) 
        self.params: list[Tensor] = None

    @property
    def n_iter(self):
        return self._n_iter.data
    @property
    def gradient_dispersion(self):
        return self._gradient_dispersion.data
    

    @torch.no_grad()
    def flattened_param(self, model: nn.Module):
        res = []
        for p in model.parameters():
            res.append(p.detach().flatten())
        return torch.cat(res).flatten()

    def copy_param(self, parallel_model: ParallelModel):
        res: list[Tensor] = []
        for m in parallel_model:
            res.append(self.flattened_param(m))
        return res

    @torch.no_grad()
    def init(self, parallel_model: ParallelModel):
        self.params = self.copy_param(parallel_model)

    @torch.no_grad()
    def step_gradient_dispersion(self, parallel_model: ParallelModel):
        new = self.copy_param(parallel_model) 

        delta = torch.stack([n - o for n, o in zip(new, self.params)])
        res = delta.var(dim=0).sum()

        return res, new
        

    @torch.no_grad()
    def update_after_step(self, parallel_model: ParallelModel, lr: float, *args, **kwargs):
        step_dispersion, new_params = self.step_gradient_dispersion(parallel_model) 
        self._gradient_dispersion.data += step_dispersion
        self._n_iter.data += 1

        self.params = new_params
    
    @torch.no_grad()
    def trajectory_term(self, parallel_model: ParallelModel, *args, **kwargs):
        return float(self.gradient_dispersion)
    
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

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.synchronize()
        for p, backup_p in zip(self.model.parameters(), self.backup):
            p.data.copy_(backup_p)
        torch.cuda.synchronize()

class TerminalDispersionBound(Bound):
    def __init__(self, clip=None, flatness=True, cross_dispersion=False, full_utilization=False, traj_reweight=1.0, tolerance=1e-2) -> None:
        super().__init__()
        self._n_iter = nn.Parameter(torch.zeros([1], dtype=torch.int), requires_grad=False)
        self.clip = clip
        self.flatness = flatness
        self.cross_dispersion = cross_dispersion
        self.full_utilization = full_utilization
        self.traj_reweight = traj_reweight
        self.tolerance = tolerance

    @property
    def n_iter(self):
        return self._n_iter.data

    @property
    def name(self):
        res = "terminal_dispersion" 
        if self.flatness:
            res = res + "+flatness"
            if self.cross_dispersion:
                res = res + "+cross_dispersion"
                if self.full_utilization:
                    res = res + "_full_utilization"
            else:
                res = res + "_possibly_biased"
            if self.traj_reweight != 1.0:
                res = res + f"+reweight{self.traj_reweight:.3f}"
        return res

    @torch.no_grad()
    def update(self, parallel_model: ParallelModel, lr: float, *args, **kwargs):
        self._n_iter.data += 1
    @torch.no_grad()
    def trajectory_term(self, parallel_model: ParallelModel, delta=None, *args, **kwargs):
        term_disp = parallel_model.terminal_dispersion(delta, self.cross_dispersion, self.full_utilization)
        if self.n_iter == 0:
            if term_disp == 0:
                return 0
            else:
                return float('inf')
        return float(term_disp / self.n_iter)
    
    def gradients(self, parallel_model: ParallelModel, data_loader: ParallelDataloader):
        parallel_loss = ParallelLoss(loss_fn=ClippedCrossEntropyLoss(clip=self.clip)) 
        parallel_model.zero_grad(True)
        device = next(parallel_model.parameters()).device
        for data in data_loader:
            data = tuple(data)
            X = data[0]; Y = data[1]
            if isinstance(X, Tensor):
                X, Y = X.to(device), Y.to(device)
            else:
                X = [x.to(device) for x in X]
                Y = [y.to(device) for y in Y]
            output = parallel_model(X)
            loss = parallel_loss(output, Y)
            loss.backward()

        res = [
            [p.grad / len(data_loader) if p.grad is not None else torch.zeros_like(p) for p in m.parameters() ] for m in parallel_model.models
        ]

        parallel_model.zero_grad(True)

        return res


    def surrogate_forget(self, C: Tensor, nu: 'list[list[Tensor]]', parallel_model: ParallelModel, parallel_training_data_loader: ParallelDataloader, val_dataloader: DataLoader):


        # selected_validation_loader = SelectedDataFieldDataLoader(val_dataloader, [0, 1])
        clean_cache()

        grad_empirical = self.gradients(parallel_model, parallel_training_data_loader)
        grad_population = self.gradients(parallel_model, val_dataloader) 
        diff_grad = [[
            grad_empirical[index_model][index_param] 
                - grad_population[index_model][index_param] 
            # torch.zeros_like(grad_empirical[index_model][index_param])
                    for index_param in range(len(grad_empirical[0]))] for index_model in range(len(grad_empirical))] 
        Delta = iHVP(
            parallel_model,
            [[
                (SelectedDataFieldDataLoader(parallel_training_data_loader.loaders[i], data_field=[0, 1]), 1), 
               # (selected_ing_loader, -1), 
                2 * C[i] * self.traj_reweight 
            ] for i in range(len(parallel_model))],
            [[
                - 2 * C[index_model] * self.traj_reweight * nu[index_model][index_param] - diff_grad[index_model][index_param]  
                for index_param in range(len(nu[0]))] 
            for index_model in range(len(nu))],
            self.tolerance,
            clip=self.clip
        )

        return Delta

    @torch.no_grad()
    def loss(self, model: nn.Module, loader: DataLoader):
        criterion = ClippedCrossEntropyLoss(self.clip)
        device = next(model.parameters()).device
        loss_sum = 0
        count = 0
        for data in loader:
            data = list(data)
            X, Y = data[0].to(device), data[1].to(device)
            output = model(X) 
            loss = criterion(output, Y)
            loss_sum = loss_sum + loss * len(Y)
            count += len(Y)

        return loss_sum / count #! DO NOT average over batchwise losses, since the last batch can be smaller. Averaging over batchwise losses thus twists the empirical and the population distribution. Since the bound is tight, such error will make the estimated bound smaller than the generalization gap 



    def gamma(self, delta: 'list[Tensor]', model: nn.Module, loader: DataLoader, trace=True):
        torch.cuda.synchronize()
        loss0 = self.loss(model, loader)
        torch.cuda.synchronize()
        with BackupModelParams(model):
            with torch.no_grad():
                for (d, p) in zip(delta, model.parameters()):
                    p.data += d
            torch.cuda.synchronize()
            loss_delta = self.loss(model, loader)
            if trace:
                hessian_traces = average_hessian_trace(self.clip, model, SelectedDataFieldDataLoader(loader, [0, 1]))
                hessian_trace = torch.tensor(hessian_traces).mean().to(device=next(model.parameters()).device)
                model.zero_grad(True)
            else:
                hessian_trace = None
        return loss_delta - loss0, hessian_trace



    def punishment(self, Delta: 'list[list[Tensor]]', parallel_model: ParallelModel, parallel_training_data_loader: ParallelDataloader, val2_data_loader: DataLoader):
        clean_cache()
        torch.cuda.synchronize()
        hessian_traces = []
        delta_losses = []
        for delta, model, empirical_loader in zip(tqdm(Delta, f"punishing ({self.traj_reweight})"), parallel_model.models, parallel_training_data_loader.loaders):
            empirical_delta_loss, hessian_trace = self.gamma(delta, model, empirical_loader)
            population_delta_loss,  population_hessian_trace = self.gamma(delta, model, val2_data_loader, True)
            # population_hessian_trace = 0
            hessian_traces.append((hessian_trace - population_hessian_trace).abs())
            delta_losses.append((empirical_delta_loss - population_delta_loss).abs())
            clean_cache()
        return torch.stack(delta_losses).mean(),  torch.stack(hessian_traces).mean()

    @torch.no_grad()
    def get_nu(self, parallel_model: ParallelModel):
        def get_mean(parallel_model):
            params = [[p for p in m.parameters()] for m in parallel_model] 
            mean = [torch.stack([params[i_model][i_param] for i_model in range(len(parallel_model))], dim=0).mean(dim=0) for i_param in range(len(params[0]))]
            return mean, torch.cat([m.flatten() for m in mean])
        def _get_nu(parallel_model, mean):
            params = [[p for p in m.parameters()] for m in parallel_model] 
            nu = [[p - mean[i_param] for i_param, p in enumerate(m)] for m in params]
            return nu


        # if self.unbiased:
            # mean, tensor_mean = get_mean(parallel_model.models[-self.trajectories_for_opt:])
            # _, tensor_mean_prime = get_mean(parallel_model.models[:-self.trajectories_for_opt])
            # print((tensor_mean - tensor_mean_prime).norm())
            # nu = _get_nu(parallel_model.models[:-self.trajectories_for_opt], mean)
        if self.cross_dispersion:
            if self.full_utilization:
                nu = []
                for i in range(len(parallel_model)):
                    others = [m for j, m in enumerate(parallel_model) if j != i]
                    mean, _ = get_mean(others)
                    nu = nu + _get_nu([parallel_model.models[i]], mean)
            else:
                l = len(parallel_model)
                mean_0, _ = get_mean(parallel_model.models[:l//2])
                mean_1, _ = get_mean(parallel_model.models[l//2:])
                nu_0 = _get_nu(parallel_model.models[:l//2], mean_1)
                nu_1 = _get_nu(parallel_model.models[l//2:], mean_0)
                nu = nu_0 + nu_1
        else:
            mean, _ = get_mean(parallel_model)
            nu = _get_nu(parallel_model, mean)

        return nu



    def forget(self, C: float, parallel_model: ParallelModel, parallel_training_data_loader: ParallelDataloader, val_data_loader: DataLoader, val2_data_loader: DataLoader):
        if not self.flatness:
            return self.trajectory_term(parallel_model), 0, None, {}
        nu = self.get_nu(parallel_model)
        Delta = self.surrogate_forget(C, nu, parallel_model, parallel_training_data_loader, val_data_loader)
        # Delta = nu

        with torch.no_grad():
            tensor_delta = torch.stack([torch.cat([p.flatten() for p in m]) for m in Delta], dim=0)
        return float(self.trajectory_term(parallel_model, tensor_delta)), *self.punishment(Delta, parallel_model, parallel_training_data_loader, val2_data_loader), {
            'delta_norm': tensor_delta.norm(dim=-1).mean().item()
        }





        
