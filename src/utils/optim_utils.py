import warnings

import numpy as np
import torch
import math
from .helpfuns import *
from .dist_utills import *
from PIL import ImageFile
from torch.optim.lr_scheduler import _LRScheduler

# For truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

def cospace(start, stop, num):
    steps = np.arange(num)
    return stop + 0.5 * (start - stop) * (1 + np.cos(np.pi * steps / len(steps)))

def cosine_scheduler(base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

class CosineSchedulerWithWarmup:
    """
    Combination of the CosineAnnealingLR and LinearWarmUp schedulers.
    It starts with a linear warm-up phase where the learning rate gradually increases, and then it follows a cosine annealing schedule to slowly decrease the learning rate over time.
    This scheduler is often used in deep learning models to provide a warm-up phase followed by a smooth learning rate decay.
    It helps with stabilizing the training process at the beginning and fine-tuning the learning rate later on.
    """
    def __init__(
        self, base_value, final_value, iters, warmup_iters=0, warmup_init_val=None
    ):
        if warmup_init_val is None:
            warmup_init_val = base_value
        self.base_value = base_value
        self.final_value = final_value
        self.iters = iters
        self.warmup_iters = warmup_iters
        self.warmup_init_val = warmup_init_val

        warmup_sch = np.linspace(warmup_init_val, base_value, warmup_iters)
        core_sch = cospace(base_value, final_value, iters - warmup_iters)
        self.scheduler = np.concatenate((warmup_sch, core_sch))

        if not self.scheduler.size:
            self.scheduler = np.array([base_value])

    def __len__(self):
        return len(self.scheduler)

    def __call__(self, it):
        if it < len(self):
            return self.scheduler[it]
        else:
            warnings.warn(
                "Iteration number exceeds scheduler's def - Proceeding with last value",
                UserWarning,
            )
            return self.scheduler[-1]

class MixedLRScheduler:
    """
    Useful in situations where you have different parameter groups or layers in your model that require different learning rates.
    It allows you to specify different learning rates for each parameter group, such as the base model and the additional layers, enabling you to fine-tune different parts of the model with different learning rates.
    This scheduler is commonly used in transfer learning scenarios or when you have specific requirements for different parts of your model.
    """
    __name__ = "MixedLRScheduler"

    def __init__(self, schedulers, scheduler_types, steps_per_epoch):
        """
        Expectes a list of schedulers and a list of scheduler types with the same order
        The init function will update the schedulers' iterations etc with high priority for warmups
        The step will make a step (based on iteration!) with high priority for warmups
        """
        self.iteration_based = ["LinearWarmup", "OneCycleLR"]
        self.epoch_based = ["MultiStepLR"]
        self.other_types = ["ReduceLROnPlateau", "CosineAnnealingLR"]
        self.wanrup_based = ["LinearWarmup"]
        self.accepted_types = self.iteration_based + self.epoch_based + self.other_types
        self.schedulers = schedulers
        self.scheduler_types = scheduler_types
        self.steps_per_epoch = steps_per_epoch
        self.iter = 0

        self.warmup_iters = [
            self.schedulers[sc].warmup_iters
            for sc, sctype in enumerate(self.scheduler_types)
            if sctype in self.wanrup_based
        ]
        self.warmup_iters = max(self.warmup_iters) if self.warmup_iters else 0

    def reset_iters():
        self.iter = 0

    def step(self, val_acc=None, val_loss=None):
        self.iter += 1
        for stype, sch in zip(self.scheduler_types, self.schedulers):
            if stype in self.iteration_based:
                sch.step()
            elif stype in self.epoch_based:
                if (self.iter + 1) % self.steps_per_epoch == 0:
                    sch.step()
            elif stype == "ReduceLROnPlateau":
                if (self.iter + 1) % self.steps_per_epoch == 0:
                    if sch.mode == "min":
                        sch.step(val_loss)
                    else:
                        sch.step(val_acc)
            elif stype == "CosineAnnealingLR":
                if self.iter > self.warmup_iters:
                    sch.step()
            elif stype is None:
                pass
            else:
                raise ValueError(f"{stype} is not a supported scheduler")

class LinearWarmup(_LRScheduler):
    """
    Gradually increases the learning rate from an initial value to a target value over a specified number of warm-up steps or epochs.
    It is often used in scenarios where you want to stabilize the training process at the beginning by gradually increasing the learning rate.
    This can be helpful when training large models or dealing with complex datasets.
    LinearWarmUp is often combined with other learning rate schedulers, such as CosineAnnealingLR or ReduceLROnPlateau, to fine-tune the learning rate during training.
    """
    __name__ = "LinearWarmup"

    def __init__(
        self,
        optimizer,
        max_lr,
        warmup_iters=0,
        warmup_epochs=0,
        eta_min=1e-8,
        last_epoch=-1,
        verbose=False,
        steps_per_epoch=None,
    ):
        if warmup_iters and warmup_epochs:
            print("\033[93m Found nonzero arguments for warmup_iters and warmup_epochs \033[0m")
            print("\033[93m Using warmup_epochs instead of warmup_iters \033[0m")
            warmup_iters = steps_per_epoch * warmup_epochs
        if not warmup_iters and not warmup_epochs:
            print("\033[93m No warmup period found but LinearWarmup is used \033[0m")
            warmup_iters = 1
        else:
            if warmup_epochs and steps_per_epoch is None:
                raise TypeError("LinearWarmup with warmup_epochs settings must include steps_per_epoch")
            elif warmup_epochs and steps_per_epoch is not None:
                warmup_iters = steps_per_epoch * warmup_epochs

        self.warmup_iters = warmup_iters
        self.eta_min = eta_min
        self.max_lr = max_lr
        for group in optimizer.param_groups:
            group["lr"] = self.eta_min
        super(LinearWarmup, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == -1:
            return [self.eta_min for group in self.optimizer.param_groups]
        elif self.last_epoch > self.warmup_iters:
            return [group["lr"] for group in self.optimizer.param_groups]
        else:
            return [
                group["lr"] + (1 / self.warmup_iters) * (self.max_lr - self.eta_min)
                for group in self.optimizer.param_groups
            ]

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )

        if self.last_epoch == -1:
            return [self.eta_min for group in self.optimizer.param_groups]
        elif self.last_epoch > self.warmup_iters:
            return [group["lr"] for group in self.optimizer.param_groups]
        else:
            return [
                group["lr"] + (1 / self.warmup_iters) * (self.max_lr - self.eta_min)
                for group in self.optimizer.param_groups
            ]

class LARS(torch.optim.Optimizer):
    """
    Almost copy-paste from https://github.com/facebookresearch/barlowtwins/blob/main/main.py
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=None, lars_adaptation_filter=None):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim != 1:
                    dp = dp.add(p, alpha=g['weight_decay'])

                if p.ndim != 1:
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])