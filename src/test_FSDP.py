#!/usr/bin/env python
"""
This script demonstrates training an MNIST model using Fully Sharded Data Parallel (FSDP)
with PyTorch on Intel GPUs (XPU). It shows how to set up distributed training on XPU devices,
suppress specific warnings, and verify that FSDP sharding is active.
"""

import os
# 设置环境变量，降低 PyTorch C++ 端日志级别，抑制重复的警告信息
os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"

import sys
import contextlib
import argparse
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 导入 Intel® Extension for PyTorch 及 oneCCL Bindings for PyTorch
import intel_extension_for_pytorch
import oneccl_bindings_for_pytorch

from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

# 定义上下文管理器，临时抑制标准错误输出（用于隐藏 FSDP 包装时产生的警告）
@contextlib.contextmanager
def suppress_stderr():
    old_stderr = sys.stderr
    try:
        sys.stderr = open(os.devnull, 'w')
        yield
    finally:
        sys.stderr.close()
        sys.stderr = old_stderr

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # 使用 Intel oneCCL 后端初始化进程组（适用于 XPU 分布式训练）
    dist.init_process_group("ccl", rank=rank, world_size=world_size)
    print(f"Rank {rank}: Process group initialized", flush=True)

def cleanup():
    dist.destroy_process_group()
    print("Cleanup done", flush=True)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=None):
    model.train()
    ddp_loss = torch.zeros(2, device=f"xpu:{rank}")
    if sampler:
        sampler.set_epoch(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(f"xpu:{rank}"), target.to(f"xpu:{rank}")
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target, reduction='sum')
        loss.backward()
        optimizer.step()
        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(data)
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        print(f"Train Epoch: {epoch} \tLoss: {ddp_loss[0] / ddp_loss[1]:.6f}", flush=True)

def test(model, rank, world_size, test_loader):
    model.eval()
    ddp_loss = torch.zeros(3, device=f"xpu:{rank}")
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(f"xpu:{rank}"), target.to(f"xpu:{rank}")
            output = model(data)
            ddp_loss[0] += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            ddp_loss[1] += pred.eq(target.view_as(pred)).sum().item()
            ddp_loss[2] += len(data)
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    if rank == 0:
        test_loss = ddp_loss[0] / ddp_loss[2]
        print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {int(ddp_loss[1])}/{int(ddp_loss[2])} ({100. * ddp_loss[1] / ddp_loss[2]:.2f}%)", flush=True)

def fsdp_main(rank, world_size, args):
    print(f"Process started, rank: {rank}", flush=True)
    setup(rank, world_size)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 加载 MNIST 数据集（支持下载或本地加载）
    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)
    print(f"Rank {rank}: Datasets loaded", flush=True)
    
    sampler1 = DistributedSampler(dataset1, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(dataset2, rank=rank, num_replicas=world_size)
    
    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
    extra_kwargs = {'num_workers': 2, 'pin_memory': True, 'shuffle': False}
    train_kwargs.update(extra_kwargs)
    test_kwargs.update(extra_kwargs)
    
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)
    print(f"Rank {rank}: DataLoaders created", flush=True)
    
    torch.xpu.set_device(f"xpu:{rank}")
    # 使用 XPU Event 计时
    init_start_event = torch.xpu.Event(enable_timing=True)
    init_end_event = torch.xpu.Event(enable_timing=True)
    
    model = Net().to(f"xpu:{rank}")
    print(f"Rank {rank}: Model created and moved to XPU", flush=True)
    # 定义自动包装策略，这里将 min_num_params 设为 1，强制对子模块包装
    auto_wrap_policy = functools.partial(size_based_auto_wrap_policy, min_num_params=1)
    with suppress_stderr():
        model = FSDP(model, device_id=f"xpu:{rank}", auto_wrap_policy=auto_wrap_policy)
    print(f"Rank {rank}: Model wrapped in FSDP with auto_wrap_policy", flush=True)
    
    # 诊断 FSDP 是否进行了参数 sharding
    full_param_count = sum(p.numel() for p in model.module.parameters())
    local_param_count = sum(p.numel() for p in model.parameters())
    if local_param_count < full_param_count:
        print(f"Rank {rank}: FSDP sharding is active: local parameters count ({local_param_count}) < full model parameters count ({full_param_count})", flush=True)
    else:
        print(f"Rank {rank}: WARNING: FSDP sharding might not be active, local parameters count equals full model parameters count ({local_param_count})", flush=True)
    
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    
    init_start_event.record()
    print(f"Rank {rank}: Starting training loop", flush=True)
    for epoch in range(1, args.epochs + 1):
        train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        test(model, rank, world_size, test_loader)
        scheduler.step()
    init_end_event.record()
    
    if rank == 0:
        elapsed_time_sec = init_start_event.elapsed_time(init_end_event) / 1000.0
        print(f"Rank {rank}: XPU event elapsed time: {elapsed_time_sec:.2f} sec", flush=True)
        print(f"Rank {rank}: Final model: {model}", flush=True)
    
    if args.save_model:
        dist.barrier()
        states = model.state_dict()
        if rank == 0:
            torch.save(states, "mnist_cnn.pt")
    
    cleanup()
    print(f"Rank {rank}: Exiting process", flush=True)

if __name__ == "__main__":
    print("FSDP MNIST example for Intel GPUs", flush=True)
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example with FSDP on Intel GPUs')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR', help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M', help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-xpu', action='store_true', default=False, help='disables XPU training')
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    
    # 自动检测 Intel GPU 数量（XPU）
    WORLD_SIZE = torch.xpu.device_count()
    if WORLD_SIZE == 0:
        print("No Intel GPUs (XPU) found. Exiting.", flush=True)
        exit(1)
    else:
        print(f"Detected {WORLD_SIZE} Intel GPUs (XPU)", flush=True)
    
    mp.spawn(fsdp_main, args=(WORLD_SIZE, args), nprocs=WORLD_SIZE, join=True)
