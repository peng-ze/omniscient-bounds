import os
import logging
import datetime
import numpy as np
import torch
from torch import Tensor
import torch.amp
import torch.nn as nn
import torch.utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torch.distributions import Normal
from utils import *
from FitsubGaussian import gaussian_net
import cmd_args
import time
import random
from tqdm.auto import tqdm
from collections import OrderedDict
from parallel import ParallelModel, ParallelLoss, make_parallel_datasets, ParallelDataloader, SelectedDataFieldDataLoader
from bounds import GradientDispersionBound, TerminalDispersionBound, Bound, average_hessian_trace

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.autograd.set_detect_anomaly(True)


def accuracy(output: Tensor, targets: Tensor):
    return (output.argmax(dim=-1) == targets).float().mean()

def subset(datasets, k):
    if isinstance(datasets, list):
        return [subset(d, k) for d in datasets]
    indices = torch.randperm(len(datasets))[:int(len(datasets) * k)]
    return Subset(datasets, indices) 


class RunModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        import math
        self.traindataset = None
        self.train_loader, self.train_test_loader, self.val_loader, self.val2_loader, self.test_loader, self.model = self.get_data_model(args, shuffle_train=True)
        self.loss_clip = math.log(args.num_classes) * args.loss_upperbound
        self.train_loss_clip = (math.log(args.num_classes) * args.train_loss_upperbound) if args.train_loss_upperbound is not None else None

        self.criterion = ParallelLoss(ClippedCrossEntropyLoss(clip=self.train_loss_clip))
        self.accuracy = ParallelLoss(accuracy, reduction='mean')
        self.val_criterion = ParallelLoss(ClippedCrossEntropyLoss(clip=self.loss_clip), reduction='mean')
        if args.optimizer.lower() == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(), torch.tensor(args.learning_rate), # !!! wrap lr in tensor so to avoid recompilation, see https://pytorch.org/tutorials/recipes/compiling_optimizer_lr_scheduler.html
                momentum=args.momentum,
                weight_decay=args.weight_decay
            )
        elif args.optimizer.lower() == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(), torch.tensor(args.learning_rate),
                weight_decay=args.weight_decay
            )

        if args.scheduler is not None:
            if args.scheduler.lower() == 'cosine':
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=self.args.epochs-self.args.warmup_epochs, eta_min=args.learning_rate/10, verbose=True)
            else:
                raise NotImplemented()
        else:
            self.scheduler = None
        
        if args.warmup_epochs is not None:
            self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1/args.warmup_epochs, end_factor=1.0, total_iters=args.warmup_epochs-1, last_epoch=-1, verbose=True)
        else:
            self.warmup_scheduler = None

        self.grad_norm = 999999
        self.losses_all = [] 
        self.sigma = 0
        self.n_iter = 0
        # self.gradient_norm = []
        # self.gradient_variance = []
        # self.mi = 0

        self.bounds: 'list[Bound]' = nn.ModuleList([
            GradientDispersionBound(), 
            TerminalDispersionBound(clip=self.loss_clip, flatness=False), 
            # TerminalDispersionBound(clip=self.loss_clip),
            # TerminalDispersionBound(clip=self.loss_clip, cross_dispersion=True),
            # TerminalDispersionBound(clip=self.loss_clip, unbiased=True, trajectories_for_opt=args.trajectories_for_optimization)
        ] + ([
                TerminalDispersionBound(clip=self.loss_clip, cross_dispersion=True, full_utilization=True, traj_reweight=w, tolerance=self.args.tolerance) for w in args.traj_reweight
        ] if not self.args.existing_bounds_only else [])
        )

        self.scaler = torch.cuda.amp.GradScaler() if self.args.amp else None

    def get_data_model(self, args, shuffle_train=True):
        basic_transform = MyDataset.transform if args.dataset == 'cifar10' else MyDataset.mnist_transform
        if args.dataset == "mnist":
            from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet
        if args.dataset == "cifar10":
            from archs.cifar10 import AlexNet, LeNet5, fc1, vgg, resnet, densenet, vit

        if 'cifar' in args.dataset:
            if args.arch == 'vit':
                train_transform, test_transform, basic_transform = vit.vit_transform()
            else:
                train_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(32),
                    MyDataset.transform
                ]) 
                test_transform = None 
        else:
            train_transform = MyDataset.mnist_transform  
            test_transform = None

        assert test_transform is None

        train_to_train, train_to_val = torch.utils.data.random_split(MyDataset(args, _train=True, no_transform=True), [1 - args.train_to_val, args.train_to_val])        

        train_to_train, _ = torch.utils.data.random_split(train_to_train, [args.training_data_usage, 1-args.training_data_usage])

        plain_data = make_parallel_datasets(InMemoryDataset(train_to_train), args.k)
        self.unaugmented_train_dataset = [AugmentationDataset(d, transform=basic_transform) for d in plain_data] 
        self.traindataset = [AugmentationDataset(d, transform=train_transform) for d in plain_data]

        train_test_dataset = self.unaugmented_train_dataset 

        valdataset, val2dataset, testdataset = torch.utils.data.random_split(AugmentationDataset(InMemoryDataset(UnionDataset(MyDataset(args, _train=False, no_transform=True), train_to_val)), transform=basic_transform), [args.validation_usage, args.validation_usage, 1 - 2 * args.validation_usage],)

        train_loader = ParallelDataloader(self.traindataset, batch_size=args.batch_size, shuffle=shuffle_train, num_workers=8, pin_memory=True, persistent_workers=True)
        train_test_loader = ParallelDataloader(subset(train_test_dataset, args.data_usage_for_bounds), batch_size=args.batch_size_for_validation, num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(subset(testdataset, args.data_usage_for_bounds), batch_size=args.batch_size_for_validation, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        val_loader = DataLoader(subset(valdataset, args.data_usage_for_bounds), batch_size=args.batch_size_for_validation, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        val2_loader = DataLoader(subset(val2dataset, args.data_usage_for_bounds), batch_size=args.batch_size_for_validation, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)


        def make_model():
            if args.arch == 'fc1':
                model = fc1.fc1(width=args.width, depth=args.depth, bias=not (args.weight_scaling is not None and args.weight_scaling != 1.0)) # bias must be specially dealt with when weights are scaled
                # Weight Initialization
                if args.fixinit:
                    print("loading...")
                    if args.dataset == "mnist":
                        model.load_state_dict(torch.load('./init/fc1/fc1.pth'))
                    else:
                        model.load_state_dict(torch.load('./init/fc1/fc1_cf10.pth'))
                else:
                    model.apply(weight_init)
            if args.arch == 'lenet':
                model = LeNet5.LeNet5()
                # if args.dataset == "mnist":
                #     model.load_state_dict(torch.load('./init/lenet5/lenet5.pth'))
                # else:
                #     model.load_state_dict(torch.load('./init/lenet5/lenet5_cf10.pth'))
                model.apply(weight_init)
            if args.arch == 'alexnet':
                model = AlexNet.AlexNet()
                if args.fixinit:
                    print("loading...")
                    if args.dataset == "mnist":
                        model.load_state_dict(torch.load('./init/alexnet/alexnet.pth'))
                    else:
                        model.load_state_dict(torch.load('./init/alexnet/alexnet_cf10.pth'))
                else:
                    model.apply(weight_init)
            if args.arch == 'resnet':
                model = resnet.resnet18(width=args.width, depth=args.depth)
                model.apply(weight_init)
            if args.arch == 'vgg':
                model = vgg.vgg11()
                model.apply(weight_init)
            if args.arch == 'vit':
                model = vit.vit_small(width=args.width, dropout=args.dropout)
            return model

        model = ParallelModel(make_model, k=args.k).to(device)
        self.n_sample = len(self.unaugmented_train_dataset[0])

        return train_loader, train_test_loader, val_loader, val2_loader, test_loader, model
    
    def should_compute_bound(self, epoch):

        return (self.args.bound_freq is not None and (epoch + 1) % self.args.bound_freq == 0) or epoch + 1 == self.epochs

    def train_model(self, args, start_epoch=None, epochs=None):
        torch.backends.cudnn.benchmark = True
        start_epoch = start_epoch or 0
        epochs = epochs or args.epochs
        self.epochs = epochs
        tr_losses = []
        tr_acces = []
        ts_losses = []
        ts_acces = []
        logging.info('  '.join([f'{bound.name}.traj: {bound.trajectory_term(self.model):.3f}' for bound in self.bounds]))

        for b in self.bounds:
            b.init(self.model)

        for self.epoch in range(start_epoch, epochs):
            t = time.time()
            # if args.ad_lr:
            #     adjust_learning_rate(optimizer, epoch, args)
            # train for one epoch
            self.train_epoch(args, self.train_loader)

            if (self.epoch + 1) % args.test_freq == 0 or self.epoch == epochs - 1 or self.should_compute_bound(self.epoch):
                tr_loss, train_acc = self.validate_test(self.train_test_loader, self.model, args)
                # evaluate on validation set
                ts_loss, test_acc = self.validate_test(self.test_loader, self.model, args)

                tr_losses.append(tr_loss)
                tr_acces.append(train_acc)
                ts_losses.append(ts_loss)
                ts_acces.append(test_acc)


                logging.info('%03d: L-tr: %.4f  L-ts: %.4f  gap: %.4f | Acc-train: %.2f Acc-test: %.2f Error-test: %.2f '
                            '| ' + '  '.join([f'{bound.name}.traj: {bound.trajectory_term(self.model):.3e}' for bound in self.bounds]) + ' | Time: %2.1f s ',
                            self.epoch, tr_loss, ts_loss, ts_loss - tr_loss, train_acc, test_acc, 100-test_acc,
                            (time.time() - t))

                if self.epoch >= 190 and (train_acc < 70):
                    logging.info("Hopless. Exiting...")
                    exit(1)

            if self.should_compute_bound(self.epoch):
                self.log_bound(self.epoch)

            if args.early_stop and self.epoch > 0:
                if tr_loss <= 0.0005:
                    break
                if args.label_corrupt_prob > 0:
                    if train_acc >= 99.995:
                        break
        
        return tr_losses, tr_acces, ts_losses, ts_acces

    def train_epoch(self, args, train_loader):
        self.model.train()
        self.train_epoch_loop(train_loader)
        
        if self.args.warmup_epochs is not None and self.epoch < self.args.warmup_epochs:
            if self.warmup_scheduler is not None:
                self.warmup_scheduler.step()
        else:
            if self.scheduler is not None:
                self.scheduler.step()

    def clip_gradient(self, model: ParallelModel):
        for m in model:
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)

    # @torch.compile 
    def train_epoch_loop(self, train_loader):
        for batch_idx, data in enumerate(tqdm(train_loader, f"Epoch {self.epoch}")):
            inputs, labels, idx = data
            self.train_batch(inputs, labels)


    def train_batch(self, imgs, targets):
        self.model.train()

        self.model.zero_grad(True)
        self.optimizer.zero_grad(True)

        if not isinstance(imgs, Tensor):
            imgs = [d.to(device) for d in imgs]
            targets = [d.to(device) for d in targets]
        else:
            imgs, targets = imgs.to(device), targets.to(device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.scaler is not None):
            output = self.model(imgs)
            train_loss = self.criterion(output, targets)

        if self.scaler is not None:
            self.scaler.scale(train_loss).backward()
            self.scaler.unscale_(self.optimizer)
        else:
            train_loss.backward()

        for bound in self.bounds:
            bound.update(self.model, lr=self.optimizer.param_groups[0]['lr'])
        if self.args.gradient_clipping:
            self.clip_gradient(self.model)

        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        
        for bound in self.bounds:
            bound.update_after_step(self.model, lr=self.optimizer.param_groups[0]['lr'])

        with torch.no_grad():
            loss_fn = ClippedCrossEntropyLoss(clip=self.loss_clip, reduction='none')
            self.losses_all.append(loss_fn(torch.cat(output), torch.cat(targets)).flatten())
        self.n_iter += 1
        self.model.zero_grad(True)
        self.optimizer.zero_grad(True)

    @torch.no_grad()
    def scale_weight(self, model: nn.Module):
        print("scaling weights by", self.args.weight_scaling)
        for p in model.parameters():
            p.data.mul_(self.args.weight_scaling)


    def to_eval(self, model: nn.Module):
        model.eval()
        if self.args.weight_scaling is not None and self.args.weight_scaling != 1.0:
            self.scale_weight(model)


    def validate_test(self, val_loader, model, args):
        with BackupModelParams(model):
            self.to_eval(model)
            test_loss = AverageMeter()
            accuracy = AverageMeter()
            with torch.no_grad():
                for data, target, _ in val_loader:
                    if isinstance(data, Tensor):
                        data, target = data.to(device), target.to(device)
                        n = len(data)
                    else:
                        data = [d.to(device) for d in data]
                        target = [t.to(device) for t in target]
                        n = len(data[0])
                    output = model(data)
                    loss = self.val_criterion(output, target)
                    test_loss.update(loss.item(), n)
                    accuracy.update(self.accuracy(output, target).item(), n)
            return test_loss.avg, accuracy.avg * 100

    def compute_hessian(self, args):
        train_loader = ParallelDataloader(self.unaugmented_train_dataset, batch_size=args.batch_size_for_validation, shuffle=True)
        self.model.eval()

        hessian_traces: list[float] = []
        empirical_hessian_traces: list[float] = []
        for (model, loader) in zip(self.model.models, train_loader.loaders): 
            clean_cache()
            empirical_trace = average_hessian_trace(self.loss_clip, model, SelectedDataFieldDataLoader(loader, [0,1]))
            if not self.args.no_population_Hessian:
                population_trace = average_hessian_trace(self.loss_clip, model, SelectedDataFieldDataLoader(self.val2_loader, [0,1]))
            else:
                population_trace = 0
            trace =  empirical_trace - population_trace
            hessian_traces.append(float(trace))
            empirical_hessian_traces.append(float(empirical_trace))
            clean_cache()
        return hessian_traces, empirical_hessian_traces

    def compute_bound(self, args):
        with BackupModelParams(self.model):
            self.to_eval(self.model)
            clean_cache()
            if args.proxy:
                std_fit = self.fit_subGaussian()
                std_proxy = std_fit
            else:
                device = next(self.model.parameters()).device
                std_proxy = torch.tensor([self.loss_clip / 2], device=device) 
            hessian_traces, empirical_hessian_traces = self.compute_hessian(args)
            results = []
            for bound in self.bounds:
                hessian_term = 1 / 2 * sum(hessian_traces) / len(hessian_traces)
                res = OrderedDict()
                res['name'] = bound.name
                res['hessian_term_prior'] = float(self.n_iter * hessian_term)
                print(res['hessian_term_prior'])
                C = [3/2 * (std_proxy**2 / self.n_sample * h).abs()**(1/3) for h in empirical_hessian_traces]
                trajectory_term, punishment, new_hessian_trace, extra_info = bound.forget(C, self.model, self.train_test_loader, self.val_loader, self.val2_loader)
                if new_hessian_trace is not None:
                    hessian_term = new_hessian_trace / 2
                A = (std_proxy**2 / self.n_sample * trajectory_term).sqrt()
                B = self.n_iter * abs(hessian_term)
                best_sigma = (A / (2 * B))**(1/3)
                bound_value = 3 * ((A/2)**(2/3)) * (B ** (1/3)) + punishment 
                bound.computed_value = bound_value

                res['C'] = float(torch.tensor(C, device=A.device).mean())
                res['trajectory_term_vanilla'] = float(A)
                res['flatness_term_vanilla'] = float(B)
                res['best_sigma'] = float(best_sigma)
                res['trajectory_term'] = float(A / best_sigma)
                res['flatness_term'] = float(best_sigma ** 2 * B)

                res['punishment'] = float(punishment)
                for k, v in extra_info.items():
                    res[k] = v
                res['bound'] = bound_value
                results.append(res)
            return results

    def fit_subGaussian(self):
        losses_all = torch.cat(self.losses_all).flatten()
        train_x = 0.5
        # create model
        model_ = gaussian_net(5).to(device)
        optimizer_ = torch.optim.Adam(params=model_.parameters(), lr=3e-4)

        total_iters = 10000
        for i in range(total_iters):
            dist, _, _ = model_.forward(torch.ones(1, 1).to(device) * train_x)
            likelihood = dist.log_prob(losses_all)
            loss = (-likelihood).sum()
            optimizer_.zero_grad(True)
            loss.backward()
            optimizer_.step()
        _, mean_proxy, std_proxy = model_.forward(torch.ones(1, 1).to(device) * train_x)
        print("Variance proxy: %.4f" % std_proxy.square().item(), "Mean proxy: %.4f" % mean_proxy.item())
        return std_proxy.item()

    def log_bound(self, epoch):
        assert self.args.resume is None
        results = self.compute_bound(self.args)
        self.model.zero_grad(True)
        for res in results:
            logging.info( f'Bound at Epoch {epoch} ' + 
                '  '.join([f'{k}: {v}' for k, v in res.items()])
            )

def has_full_run(path: str, epoch: int):
    if os.path.isfile(path):
        with open(path, 'r') as file:
            lines = file.readlines()
            for line in reversed(lines):
                if f'INFO:root:Bound at Epoch {epoch-1} name: terminal_dispersion+flatness+cross_dispersion_full_utilization+reweight1000000000' in line:
                    return True
                if 'Hop' in line and 'less' in line and 'Exiting' in line:
                    return True
        return False
    for file in os.listdir(path):
        if has_full_run(os.path.join(path, file), epoch):
            return True
    return False



def setup_logging(args):
    exp_dir = os.path.join('runs', args.exp_name)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    if has_full_run(exp_dir, args.epochs) and args.dont_repeat:
        return exp_dir, None
    id = args.resume if args.resume is not None else datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + f'-seed{args.seed}'
    log_fn = os.path.join(exp_dir, "LOG.{0}.txt".format(id))
    logging.basicConfig(filename=log_fn, filemode='a', level=logging.DEBUG)
    # also log into console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    print(f'Logging into {exp_dir}/{id}...')

    return exp_dir, id


def main():
    args = cmd_args.parse_args()
    exp_dir, id = setup_logging(args)
    if id is None and args.dont_repeat:
        print(exp_dir)
        print("There is a full run for this parameter. Skip due to '--dont-repeat'")
        return
    args.exp_dir = exp_dir
    args.id = id
    seed = args.seed
    torch.manual_seed(seed)  # cpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    runmodel = RunModel(args).to(device)
    logging.info(f'Model: {args.arch}   Dataset: {args.dataset}  lr: {args.learning_rate}   batch size: {args.batch_size}   gradient cliiping: {args.gradient_clipping}  '
                f' Corrupt level: {args.label_corrupt_prob}  width: {args.width}' 
                f' Trajectory samples: {args.k}  Seed: {seed} Traectory Term Reweight: {args.traj_reweight} '
                f' Loss Bound (training): {args.train_loss_upperbound}  Loss Bound (evaluation): {args.loss_upperbound} '
                )
    logging.info('Number of parameters: %d', sum([p.data.nelement() for p in runmodel.model.parameters()]) // args.k)

    tr_losses, tr_acces, ts_losses, ts_acces = runmodel.train_model(args)

if __name__ == '__main__':
    main()
