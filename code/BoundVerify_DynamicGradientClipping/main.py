import os
import logging
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal
# import matplotlib.pyplot as plt
from pyhessian import hessian  # Hessian computation
from utils import *
from FitsubGaussian import gaussian_net
import cmd_args
import time
import random
import copy
from tqdm.auto import tqdm
from collections import OrderedDict
# from backpack import extend, backpack
# from backpack.extensions import Variance, BatchGrad, BatchL2Grad
from parallel import ParallelModel, ParallelLoss, make_parallel_datasets, ParallelDataloader
from bounds import GradientDispersionBound, TerminalDispersionBound, Bound

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.autograd.set_detect_anomaly(True)


def accuracy(output: Tensor, targets: Tensor):
    return (output.argmax(dim=-1) == targets).float().mean()

    

class RunModel:
    def __init__(self, args):
        import math
        self.traindataset = None
        self.train_loader, self.test_loader, self.model = self.get_data_model(args, shuffle_train=True)
        self.loss_clip = math.log(args.num_classes) * 2

        self.criterion = ParallelLoss(ClippedCrossEntropyLoss(clip=self.loss_clip))
        self.accuracy = ParallelLoss(accuracy, reduction='mean')
        self.val_criterion = ParallelLoss(ClippedCrossEntropyLoss(clip=self.loss_clip), reduction='mean')
        self.optimizer = torch.optim.SGD(self.model.parameters(), args.learning_rate,
                                         momentum=args.momentum,
                                         weight_decay=args.weight_decay)
        self.grad_norm = 999999
        self.losses_all = [] 
        self.sigma = 0
        self.n_iter = 0
        # self.gradient_norm = []
        # self.gradient_variance = []
        # self.mi = 0
        self.clip = args.clip

        self.bounds: 'list[Bound]' = [
            GradientDispersionBound(), 
            TerminalDispersionBound(clip=self.clip, flatness=False), 
            # TerminalDispersionBound(clip=self.loss_clip),
            # TerminalDispersionBound(clip=self.loss_clip, cross_dispersion=True),
            TerminalDispersionBound(clip=self.loss_clip, cross_dispersion=True, full_utilization=True),
            # TerminalDispersionBound(clip=self.loss_clip, unbiased=True, trajectories_for_opt=args.trajectories_for_optimization)
        ]


    def get_data_model(self, args, shuffle_train=True):
        self.traindataset = make_parallel_datasets(InMemoryDataset(MyDataset(args, _train=True)), args.k, args.p)
        testdataset = InMemoryDataset(MyDataset(args, _train=False))

        train_loader = ParallelDataloader(self.traindataset, batch_size=args.batch_size, shuffle=shuffle_train, num_workers=4, pin_memory=True, persistent_workers=True)
        test_loader = DataLoader(testdataset, batch_size=512, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
        if args.dataset == "mnist":
            from archs.mnist import AlexNet, LeNet5, fc1, vgg, resnet
        if args.dataset == "cifar10":
            from archs.cifar10 import AlexNet, LeNet5, fc1, vgg, resnet, densenet

        def make_model():
            if args.arch == 'fc1':
                model = fc1.fc1()
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
                model = resnet.resnet18()
                model.apply(weight_init)
            if args.arch == 'vgg':
                model = vgg.vgg11()
                model.apply(weight_init)
            return model

        model = ParallelModel(make_model, k=args.k).to(device)
        self.n_sample = len(self.traindataset[0])

        return train_loader, test_loader, model

    def train_model(self, args, start_epoch=None, epochs=None):
        torch.backends.cudnn.benchmark = True
        start_epoch = start_epoch or 0
        epochs = epochs or args.epochs
        tr_losses = []
        tr_acces = []
        ts_losses = []
        ts_acces = []
        logging.info('  '.join([f'{bound.name}.traj: {bound.trajectory_term(self.model):.3f}' for bound in self.bounds]))

        for self.epoch in range(start_epoch, epochs):
            t = time.time()
            # if args.ad_lr:
            #     adjust_learning_rate(optimizer, epoch, args)
            # train for one epoch
            self.train_epoch(args, self.train_loader)

            if (self.epoch + 1) % args.test_freq == 0 or self.epoch == epochs - 1:
                tr_loss, train_acc = self.validate_test(self.train_loader, self.model, args)
                # evaluate on validation set
                ts_loss, test_acc = self.validate_test(self.test_loader, self.model, args)

                tr_losses.append(tr_loss)
                tr_acces.append(train_acc)
                ts_losses.append(ts_loss)
                ts_acces.append(test_acc)

                logging.info('%03d: L-tr: %.3f  L-ts: %.3f  gap: %.3f | Acc-train: %.2f Acc-test: %.2f Error-test: %.2f '
                            '| ' + '  '.join([f'{bound.name}.traj: {bound.trajectory_term(self.model):.3e}' for bound in self.bounds]) + ' | Time: %2.1f s ',
                            self.epoch, tr_loss, ts_loss, ts_loss - tr_loss, train_acc, test_acc, 100-test_acc,
                            (time.time() - t))

            if args.early_stop and self.epoch > 0:
                if tr_loss <= 0.0005:
                    break
                if args.label_corrupt_prob > 0:
                    if train_acc >= 99.995:
                        break
        return tr_losses, tr_acces, ts_losses, ts_acces

    def train_epoch(self, args, train_loader):
        # """Train for one epoch on the training set"""
        # losses = AverageMeter()
        # norm_mean = AverageMeter()
        # variance_mean = AverageMeter()
        # batch_mean = AverageMeter()
        self.model.train()
        for batch_idx, data in enumerate(tqdm(train_loader)):
            inputs, labels, idx = data
            # loss, norm, variance, batch_norm = 
            self.train_batch(inputs, labels, idx, args)
            # losses.update(loss.item(), inputs.size(0))
            # norm_mean.update(norm, inputs.size(0))
            # variance_mean.update(variance, inputs.size(0))
            # batch_mean.update(batch_norm)
        # return losses.avg, norm_mean.avg, variance_mean.avg, batch_mean.avg

    def train_batch(self, imgs, targets, idx, args):

        self.model.zero_grad()
        self.optimizer.zero_grad()

        if not isinstance(imgs, Tensor):
            imgs = [d.to(device) for d in imgs]
            targets = [d.to(device) for d in targets]
        else:
            imgs, targets = imgs.to(device), targets.to(device)
        output = self.model(imgs)
        train_loss = self.criterion(output, targets)
        train_loss.backward()
        # sample_grad_norm = torch.zeros(imgs.shape[0])
        # batch_gradient_norm = torch.tensor(0.0)
        # var = 0
        # for param in self.model.parameters():
            # var = var + ((imgs.shape[0]**2) * param.variance).sum()
            # sample_grad_norm += param.batch_l2.cpu() * (imgs.shape[0] ** 2)
            # batch_gradient_norm += param.grad.data.norm(2).square().cpu()
        # max_grad_norm = batch_gradient_norm.sqrt()
        if self.clip > 0 and self.epoch > args.clip_start:
            raise NotImplemented()
            if self.grad_norm < max_grad_norm:
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
                clip_grad_norm_(self.model.parameters(), self.clip)
                if args.stra == 1:
                    self.grad_norm = max_grad_norm
                self.clip = args.clip_factor * self.grad_norm
            else:
                self.grad_norm = max_grad_norm
        for bound in self.bounds:
            bound.update(self.model, lr=self.optimizer.param_groups[0]['lr'])
        self.optimizer.step()
        # self.sample_mi[idx] += var.cpu()
        # self.mi += var.item()
        # self.sample_mi[range(self.n_sample) != idx] *= 1

        with torch.no_grad():
            loss_fn = ClippedCrossEntropyLoss(clip=self.loss_clip, reduction='none')
            self.losses_all.append(loss_fn(torch.cat(output), torch.cat(targets)).flatten())
        self.n_iter += 1
        # self.gradient_norm.append(max_grad_norm.item())
        # self.gradient_variance.append(var.item())
        # return train_loss, sample_grad_norm.sqrt().mean().item(),  var.item(), max_grad_norm.item()
        self.model.zero_grad()
        self.optimizer.zero_grad()

    def validate_test(self, val_loader, model, args):
        model.eval()
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
        train_loader = ParallelDataloader(self.traindataset, batch_size=self.n_sample//10, shuffle=True)
        # self.model.eval()

        hessian_traces: list[float] = []
        for (model, loader) in zip(self.model.models, train_loader.loaders): 
            inputs, targets, _ = next(iter(loader))
            inputs, targets = inputs.to(device), targets.to(device)
            hessian_comp = hessian(model, ClippedCrossEntropyLoss(clip=self.loss_clip), data=(inputs, targets), cuda=True)
            trace = hessian_comp.trace()
            hessian_traces.append(float(np.mean(trace)))
        return hessian_traces 

    def compute_bound(self, args):
        self.model.train()
        if args.proxy:
            std_fit = self.fit_subGaussian()
            std_proxy = std_fit
        else:
            device = next(self.model.parameters()).device
            std_proxy = torch.tensor([self.loss_clip / 2], device=device) 
        hessian_traces = self.compute_hessian(args)
        hessian_term = 1 / 2 * sum(hessian_traces) / len(hessian_traces)
        results = []
        for bound in self.bounds:
            C = [3/2 * (std_proxy**2 / self.n_sample * h)**(1/3) for h in hessian_traces]
            trajectory_term, punishment, new_hessian_trace, extra_info = bound.forget(C, self.model, self.train_loader, self.test_loader)
            if new_hessian_trace is not None:
                hessian_term = new_hessian_trace / 2
            A = (std_proxy**2 / self.n_sample * trajectory_term).sqrt()
            B = self.n_iter * hessian_term
            best_sigma = (A / (2 * B))**(1/3)
            bound_value = 3 * ((A/2)**(2/3)) * (B ** (1/3)) + punishment 
            bound.computed_value = bound_value

            res = OrderedDict()
            res['name'] = bound.name
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
            optimizer_.zero_grad()
            loss.backward()
            optimizer_.step()
        _, mean_proxy, std_proxy = model_.forward(torch.ones(1, 1).to(device) * train_x)
        print("Variance proxy: %.4f" % std_proxy.square().item(), "Mean proxy: %.4f" % mean_proxy.item())
        return std_proxy.item()



def setup_logging(args):
    import datetime
    exp_dir = os.path.join('runs', args.exp_name)
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    log_fn = os.path.join(exp_dir, "LOG.{0}.txt".format(datetime.date.today().strftime("%y%m%d")))
    logging.basicConfig(filename=log_fn, filemode='w', level=logging.DEBUG)
    # also log into console
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    print('Logging into %s...' % exp_dir)


def main():
    args = cmd_args.parse_args()
    setup_logging(args)
    seed = args.seed
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    runmodel = RunModel(args)
    logging.info(f'Model: {args.arch}   Dataset: {args.dataset}  lr: {args.learning_rate}   batch size: {args.batch_size} '
                 f' Corrupt level: {args.label_corrupt_prob}  width: {args.width} Clip factor: {args.clip_factor} Clip start: {args.clip_start} Clip stratagy: {args.stra} ' 
                 f' Trajectory samples: {args.k}  Seed: {seed}')
    logging.info('Number of parameters: %d', sum([p.data.nelement() for p in runmodel.model.parameters()]) // args.k)

    tr_losses, tr_acces, ts_losses, ts_acces = runmodel.train_model(args)

    results = runmodel.compute_bound(args)
    for res in results:
        logging.info(
            '  '.join([f'{k}: {v}' for k, v in res.items()])
        )

    # plt.title('Loss')
    # plt.plot(np.arange(len(tr_losses)), tr_losses, color='green', linewidth=2.0, linestyle='-', label='Train')
    # plt.plot(np.arange(len(ts_losses)), ts_losses, color='blue', linewidth=2.0, linestyle='-', label='Test')
    # plt.legend(loc='best')
    # plt.savefig(f"{os.getcwd()}/plots/Bound/Loss_{args.dataset}{args.label_corrupt_prob}_"
                # f"{args.arch}_{args.batch_size}_{args.learning_rate}.png", dpi=1200)
    # plt.close()
    # plt.figure()
    # plt.plot(np.arange(len(tr_acces)), tr_acces, color='green', linewidth=3.0, linestyle='-', label='Train')
    # plt.plot(np.arange(len(ts_acces)), ts_acces, color='blue', linewidth=3.0, linestyle='-', label='Test')
    # plt.title('Acc')
    # plt.legend(loc='best')
    # plt.savefig(f"{os.getcwd()}/plots/Bound/Acc_{args.dataset}{args.label_corrupt_prob}"
                # f"_{args.arch}_{args.batch_size}_{args.learning_rate}.png", dpi=1200)
    # plt.close()

    # plt.figure()
    # plt.plot(np.arange(len(runmodel.gradient_norm)), runmodel.gradient_norm, color='green', linewidth=3.0,
             # linestyle='-', label='Grad_Norm')
    # plt.title('Gradient Norm')
    # plt.legend(loc='best')
    # plt.savefig(f"{os.getcwd()}/plots/Bound/GradNorm_{args.dataset}{args.label_corrupt_prob}"
                # f"_{args.arch}_{args.batch_size}_{args.learning_rate}.png", dpi=1200)
    # plt.close()
    # plt.figure()
    # plt.plot(np.arange(len(runmodel.gradient_variance)), runmodel.gradient_variance, color='red', linewidth=3.0,
             # linestyle='-', label='Grad_Var')
    # plt.title('Gradient Variance')
    # plt.legend(loc='best')
    # plt.savefig(f"{os.getcwd()}/plots/Bound/GradVar_{args.dataset}{args.label_corrupt_prob}"
                # f"_{args.arch}_{args.batch_size}_{args.learning_rate}.png", dpi=1200)
    # plt.close()
            

if __name__ == '__main__':
    main()