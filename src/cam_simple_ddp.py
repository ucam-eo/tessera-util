import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from socket import gethostname

import intel_extension_for_pytorch as ipex
import oneccl_bindings_for_pytorch
import time

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(42, 3)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        output = F.log_softmax(x, dim=1)
        return output

# Prepare environment variables
init_method   = 'tcp://' + os.environ["MASTER_ADDR"] + ':' + os.environ["MASTER_PORT"]
rank          = int(os.environ.get("PMI_RANK", -1))
world_size    = int(os.environ["WORLD_SIZE"])
# gpus_per_node = int(os.environ["SLURM_GPUS_ON_NODE"])
gpus_per_node = int(os.environ.get("SLURM_NTASKS_PER_NODE", 1))

print(f"gpus_per_node: {gpus_per_node}, torch.xpu.device_count(): {torch.xpu.device_count()}")
print(
    f"Hello from rank {rank} of {world_size} on {gethostname()} where there are "
    f"{gpus_per_node} allocated GPUs per node.",
    flush=True
)

# Initialize the process group using the CCL backend

print(
    f"Rank = {rank}, sees torch.xpu.device_count() = {torch.xpu.device_count()}, "
    f"SLURM_GPUS_ON_NODE = {os.environ.get('SLURM_GPUS_ON_NODE')}, "
    f"ZE_FLAT_DEVICE_HIERARCHY = {os.environ.get('ZE_FLAT_DEVICE_HIERARCHY')}"
)

dist.init_process_group(backend="ccl", init_method=init_method, rank=rank, world_size=world_size)
if rank == 0:
    print(f"Group initialized? {dist.is_initialized()}", flush=True)

# Determine local rank for the current process
local_rank = rank - gpus_per_node * (rank // gpus_per_node)

# Print out device information for debugging
print(f"{gethostname()}: torch.xpu.device_count() = {torch.xpu.device_count()}", flush=True)
current_xpu = f"xpu:{local_rank}"
print(f"Current xpu: {current_xpu}")

# Set the current device
torch.xpu.set_device(current_xpu)

# get device info
print(f"get_device_name: {torch.xpu.get_device_name()}")
print(f"get_device_properties: {torch.xpu.get_device_properties(current_xpu)}")
print(f"current_device: {torch.xpu.current_device()}")
print(f"is_available: {torch.xpu.is_available()}")

# Build model, move to device, wrap with DDP
model = Net().to(current_xpu)
ddp_model = DDP(model, device_ids=[local_rank])
# ddp_model = DDP(model)

# Create a simple optimizer
optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)

# Training loop: 10 steps of forward/backward
ddp_model.train()
for step in range(10):
    # Generate random input and label
    data = torch.rand(1, 42).to(current_xpu)
    target = torch.randint(0, 3, (1,)).to(current_xpu)
    
    # Forward pass
    output = ddp_model(data)
    loss = F.nll_loss(output, target)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(
        f"host: {gethostname()}, rank: {rank}, step: {step}, loss: {loss.item()}, time: {time.time()}",
        flush=True
    )

# Optionally do an evaluation pass after training
ddp_model.eval()
with torch.no_grad():
    data = torch.rand(1, 42).to(current_xpu)
    output = ddp_model(data)
    # print(f"host: {gethostname()}, rank: {rank}, eval output: {output}, time: {time.time()}", flush=True)
    print(f"host: {gethostname()}, rank: {rank}, eval output: {output.detach().cpu().numpy()}, time: {time.time()}", flush=True)

# Clean up process group
dist.destroy_process_group()
