#!/bin/bash -l
#SBATCH --job-name=mm-btfm
#SBATCH --partition=pvc         # 需要换成你所在集群可用的分区
#SBATCH --account=climate-dawn-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:intel:4
#SBATCH --mem=1000G
#SBATCH --time=36:00:00

#SBATCH --output=/home/zf281/rds/hpc-work/Files/btfm4rs/log/btfm_%j.out
#SBATCH --error=/home/zf281/rds/hpc-work/Files/btfm4rs/log/btfm_%j.err

#######################请在根目录提交，不要在src目录下提交#######################
#######################请在根目录提交，不要在src目录下提交#######################
#######################请在根目录提交，不要在src目录下提交#######################

module purge
module load default-dawn
module load dawn-env/2024-12-29
module load intel-oneapi-mpi/2021.14.1/oneapi/6qxeyc5c
module load oneapi-level-zero/1.14/gcc/3tn5pfua
module load intel-oneapi-compilers/2025.0.4/gcc/umo7dwbo

# 如果你使用 Anaconda/Miniconda, 激活你的环境:
source ~/rds/hpc-work/Softwares/anaconda3/bin/activate btfm-my
# source /usr/local/dawn/software/external/intel-oneapi/2024.0/compiler/latest/env/vars.sh
# source /usr/local/dawn/software/external/intel-oneapi/2024.0/dnnl/latest/env/vars.sh
# source /usr/local/dawn/software/external/intel-oneapi/2024.0/mkl/latest/env/vars.sh
# source /usr/local/dawn/software/external/intel-oneapi/2024.0/ccl/latest/env/vars.sh
# source /usr/local/dawn/software/external/intel-oneapi/2024.0/tbb/latest/env/vars.sh
# source /usr/local/dawn/software/external/intel-oneapi/2024.0/mpi/latest/env/vars.sh

which python
which mpirun

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7

export I_MPI_FABRICS=shm

echo "Running on node: $(hostname)"
echo "Allocated GPUs:"
sycl-ls

ulimit -n 65536  # 或者更高，具体看系统允许的最大值

# oneCCL环境变量
export I_MPI_OFFLOAD=1
export I_MPI_DEBUG=5
export MASTER_PORT=29500  # 或者你喜欢的端口
# 其他可选的OpenMP环境变量:
# export OMP_NUM_THREADS=4
# export KMP_AFFINITY=granularity=fine,compact,1,0
# export KMP_BLOCKTIME=1

max_retries=1
count=0

while ! mpirun -n 8 python src/train_multi_xpu.py; do

    count=$((count + 1))
    echo "mpirun failed. Attempt $count of $max_retries. Retrying in 5 seconds..."
    if [ "$count" -ge "$max_retries" ]; then
        echo "mpirun failed after $max_retries attempts. Exiting."
        exit 1
    fi
    sleep 5
done

echo "mpirun succeeded!"