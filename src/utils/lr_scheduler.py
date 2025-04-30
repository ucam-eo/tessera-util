# src/utils/lr_scheduler.py

import math
import torch
from torch.optim import Optimizer

class LARS(Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                dp = p.grad

                if not group['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=group['weight_decay'])

                # LARS adaptation
                if not group['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0,
                                    torch.where(update_norm > 0,
                                                (group['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(group['momentum']).add_(dp)

                p.add_(mu, alpha=-group['lr'])

def adjust_learning_rate(optimizer, step, total_steps, base_lr, warmup_ratio, plateau_ratio):
    """
    使用warmup + 余弦退火调整学习率：
      - 前warmup_ratio部分线性warmup；
      - 中间保持base_lr；
      - 后面使用余弦退火。
    """
    warmup_steps = int(warmup_ratio * total_steps)
    plateau_steps = int(plateau_ratio * total_steps) + warmup_steps
    if step < warmup_steps:
        lr = base_lr * (step / warmup_steps)
    elif step < plateau_steps:
        lr = base_lr
    else:
        progress = (step - plateau_steps) / (total_steps - plateau_steps)
        # 余弦退火的最低学习率为 base_lr/100
        min_lr_ratio = 0.01
        lr = base_lr * ((1 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress)) + min_lr_ratio)
    # optimizer有两个param group
    optimizer.param_groups[0]['lr'] = lr * 0.2
    optimizer.param_groups[1]['lr'] = lr * 0.0048
