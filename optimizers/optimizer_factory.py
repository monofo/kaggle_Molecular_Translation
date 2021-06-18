import torch
from timm.optim import (AdamP, AdamW, Nadam, NovoGrad, NvNovoGrad, RAdam,
                        RMSpropTF)
from timm.optim.lookahead import Lookahead
from torch import optim as optim
from torch.optim.lr_scheduler import (CosineAnnealingLR, ReduceLROnPlateau,
                                      StepLR)
from warmup_scheduler import GradualWarmupScheduler




class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]


import math

# Third party libraries
from torch.optim import lr_scheduler


class WarmRestart(lr_scheduler.CosineAnnealingLR):
    """This class implements Stochastic Gradient Descent with Warm Restarts(SGDR): https://arxiv.org/abs/1608.03983.
    Set the learning rate of each parameter group using a cosine annealing schedule,
    When last_epoch=-1, sets initial lr as lr.
    This can't support scheduler.step(epoch). please keep epoch=None.
    """

    def __init__(self, optimizer, T_max=10, T_mult=2, eta_min=0, last_epoch=-1):
        """implements SGDR
        Parameters:
        ----------
        T_max : int
            Maximum number of epochs.
        T_mult : int
            Multiplicative factor of T_max.
        eta_min : int
            Minimum learning rate. Default: 0.
        last_epoch : int
            The index of last epoch. Default: -1.
        """
        self.T_mult = T_mult
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch == self.T_max:
            self.last_epoch = 0
            self.T_max *= self.T_mult
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]


def warm_restart(scheduler, T_mult=2):
    """warm restart policy
    Parameters:
    ----------
    T_mult: int
        default is 2, Stochastic Gradient Descent with Warm Restarts(SGDR): https://arxiv.org/abs/1608.03983.
    Examples:
    --------
    >>> # some other operations(note the order of operations)
    >>> scheduler.step()
    >>> scheduler = warm_restart(scheduler, T_mult=2)
    >>> optimizer.step()
    """
    if scheduler.last_epoch == scheduler.T_max:
        scheduler.last_epoch = -1
        scheduler.T_max *= T_mult
    return scheduler


def get_optimizer(model, optimizer_name, optimizer_params, scheduler_name, scheduler_params, n_epochs):
    opt_lower = optimizer_name.lower()

    opt_look_ahed = optimizer_params["lookahead"]
    if opt_lower == 'sgd':
        optimizer = optim.SGD(
            model.parameters(), lr=optimizer_params["lr"], momentum=optimizer_params["momentum"], weight_decay=optimizer_params["weight_decay"], nesterov=True)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(
            model.parameters(), lr=optimizer_params["lr"], betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    elif opt_lower == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=optimizer_params["lr"], weight_decay=optimizer_params["weight_decay"], eps=optimizer_params["opt_eps"])
    elif opt_lower == 'nadam':
        optimizer = torch.optim.Nadam(
            model.parameters(), lr=optimizer_params["lr"], weight_decay=optimizer_params["weight_decay"], eps=optimizer_params["opt_eps"])
    elif opt_lower == 'radam':
        optimizer = RAdam(
            model.parameters(), lr=optimizer_params["lr"], weight_decay=optimizer_params["weight_decay"], eps=optimizer_params["opt_eps"])
    elif opt_lower == "adabelief":
        optimizer = AdaBelief(
            model.parameters(),  lr=optimizer_params["lr"], eps=1e-8, weight_decay=optimizer_params["weight_decay"]
        )

    elif opt_lower == "adamp":
        optimizer = AdamP(
            model.parameters(), lr=optimizer_params["lr"], weight_decay=optimizer_params["weight_decay"]
        )
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    if opt_look_ahed:
        optimizer = Lookahead(optimizer, alpha=0.5, k=5)

    if scheduler_name == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            eta_min=scheduler_params["eta_min"],
            T_0=scheduler_params["T_0"],
            T_mult=scheduler_params["T_multi"],
            )
    elif scheduler_name == "WarmRestart":
        scheduler = WarmRestart(optimizer, T_max=scheduler_params["T_max"], T_mult=scheduler_params["T_mul"], eta_min=scheduler_params["eta_min"])
    elif scheduler_name == "MultiStepLR":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
                                    optimizer, milestones=scheduler_params["schedule"], gamma=scheduler_params["gamma"])
    if scheduler_params["warmup_factor"] > 0:
        scheduler = GradualWarmupSchedulerV2(optimizer, multiplier=scheduler_params["warmup_factor"], total_epoch=1, after_scheduler=scheduler)

    return optimizer, scheduler