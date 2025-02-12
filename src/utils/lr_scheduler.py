# src/utils/lr_scheduler.py

import math

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
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    # 假设optimizer有两个param group
    optimizer.param_groups[0]['lr'] = lr * 0.2
    optimizer.param_groups[1]['lr'] = lr * 0.0048
