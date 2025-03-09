#!/bin/bash

module purge
module load default-dawn
module load dawn-env/2024-12-29
module load intel-oneapi-mpi/2021.14.1/oneapi/6qxeyc5c
module load oneapi-level-zero/1.14/gcc/3tn5pfua
module load intel-oneapi-compilers/2025.0.4/gcc/umo7dwbo

# 激活 Anaconda/Miniconda 环境
source ~/rds/hpc-work/Softwares/anaconda3/bin/activate btfm-my

source /usr/local/dawn/software/external/intel-oneapi/2024.0/compiler/latest/env/vars.sh
source /usr/local/dawn/software/external/intel-oneapi/2024.0/dnnl/latest/env/vars.sh
source /usr/local/dawn/software/external/intel-oneapi/2024.0/mkl/latest/env/vars.sh
source /usr/local/dawn/software/external/intel-oneapi/2024.0/ccl/latest/env/vars.sh
source /usr/local/dawn/software/external/intel-oneapi/2024.0/tbb/latest/env/vars.sh
source /usr/local/dawn/software/external/intel-oneapi/2024.0/mpi/latest/env/vars.sh

# 显示 Python 和 mpirun 的路径
which python
which mpirun

# 设置环境变量
# export ZE_FLAT_DEVICE_HIERARCHY=FLAT
# export ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7

echo "Running on node: $(hostname)"
echo "Allocated GPUs:"
sycl-ls

# oneCCL环境变量
export I_MPI_OFFLOAD=1