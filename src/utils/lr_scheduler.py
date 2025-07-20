# src/utils/lr_scheduler.py

import math
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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


class ValidationBasedLRScheduler:
    """
    Learning rate scheduler that combines linear warmup with validation-based reduction.
    Reduces learning rate when validation accuracy plateaus.
    """
    def __init__(self, 
                 base_lr, 
                 total_steps, 
                 warmup_ratio=0.1, 
                 patience=5, 
                 reduction_factor=0.5, 
                 min_lr=1e-6,
                 weight_factor=0.2,
                 bias_factor=0.0048):
        """
        Initialize the scheduler.
        
        Args:
            base_lr: Initial learning rate
            total_steps: Total training steps
            warmup_ratio: Proportion of steps to use for warmup
            patience: Number of validation checks to wait before reducing LR
            reduction_factor: Factor to multiply LR by when reducing (e.g., 0.5 for 50%)
            min_lr: Minimum learning rate
            weight_factor: Multiplier for weight parameters LR
            bias_factor: Multiplier for bias parameters LR
        """
        self.base_lr = base_lr
        self.current_lr = base_lr
        self.total_steps = total_steps
        self.warmup_ratio = warmup_ratio
        self.warmup_steps = int(warmup_ratio * total_steps)
        self.patience = patience
        self.reduction_factor = reduction_factor
        self.min_lr = min_lr
        self.weight_factor = weight_factor
        self.bias_factor = bias_factor
        
        # Validation history tracking
        self.val_history = []
        self.best_val_acc = 0.0
        self.stagnant_count = 0
        self.lr_reductions = 0
        
    def adjust_learning_rate(self, optimizer, step):
        """
        Adjust learning rate based on current step, applying warmup if necessary.
        
        Args:
            optimizer: The optimizer to update
            step: Current training step
            
        Returns:
            Current learning rate
        """
        # Apply warmup if in warmup phase
        if step < self.warmup_steps:
            lr = self.current_lr * (step / self.warmup_steps)
        else:
            # After warmup, use the current base LR (which may have been reduced)
            lr = self.current_lr
        
        # Apply different scaling to weight and bias parameters
        optimizer.param_groups[0]['lr'] = lr * self.weight_factor  # weights
        optimizer.param_groups[1]['lr'] = lr * self.bias_factor    # biases
        
        return lr
    
    def check_validation_plateau(self, val_acc):
        """
        Check if validation accuracy has plateaued and reduce learning rate if needed.
        
        Args:
            val_acc: Current validation accuracy
            
        Returns:
            True if learning rate was reduced, False otherwise
        """
        # Track validation accuracy history
        self.val_history.append(val_acc)
        
        # Update best validation accuracy
        improved = False
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.stagnant_count = 0
            improved = True
        else:
            self.stagnant_count += 1
            
        # If we've been stagnant for 'patience' validations, reduce learning rate
        if self.stagnant_count >= self.patience:
            old_lr = self.current_lr
            self.current_lr = max(self.current_lr * self.reduction_factor, self.min_lr)
            
            # Only reduce if it would make a meaningful difference
            if self.current_lr < old_lr * 0.95:  # Ensure it's at least 5% lower
                self.lr_reductions += 1
                self.stagnant_count = 0  # Reset the counter
                logging.info(f"Reducing learning rate from {old_lr:.6f} to {self.current_lr:.6f} due to validation accuracy plateau")
                return True
                
        return improved
        
    def get_status(self):
        """Return a string with the scheduler's current status"""
        return (f"Stagnant count: {self.stagnant_count}/{self.patience}, "
                f"LR reductions: {self.lr_reductions}")