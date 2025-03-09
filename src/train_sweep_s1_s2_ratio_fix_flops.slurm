#!/bin/bash -l
#SBATCH --job-name=btfm-training
#SBATCH --partition=pvc
#SBATCH --account=climate-dawn-gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:intel:1          # 每个任务只用一个 Intel GPU
#SBATCH --ntasks=1                 # 每个任务1个进程
#SBATCH --cpus-per-task=18         # 假设每个节点有52个CPU，则每任务分配13个
#SBATCH --mem=200G
#SBATCH --time=36:00:00
#SBATCH --output=log/sweep_s1_s2_ratio_%A_%a.log
#SBATCH --error=log/sweep_s1_s2_ratio_%A_%a.log
#SBATCH --array=0-3                # 每个节点上启动4个任务

# 加载模块及环境
module purge
module load default-dawn
module load dawn-env/2024-12-29
module load intelpython-conda
module load intel-oneapi-mpi/2021.14.1/oneapi/6qxeyc5c
module load oneapi-level-zero/1.14/gcc/3tn5pfua
module load intel-oneapi-compilers/2025.0.4/gcc/umo7dwbo

source /usr/local/dawn/software/external/intel-oneapi/2024.0/compiler/latest/env/vars.sh
source /usr/local/dawn/software/external/intel-oneapi/2024.0/dnnl/latest/env/vars.sh
source /usr/local/dawn/software/external/intel-oneapi/2024.0/mkl/latest/env/vars.sh
source /usr/local/dawn/software/external/intel-oneapi/2024.0/ccl/latest/env/vars.sh
source /usr/local/dawn/software/external/intel-oneapi/2024.0/tbb/latest/env/vars.sh
source /usr/local/dawn/software/external/intel-oneapi/2024.0/mpi/latest/env/vars.sh

source ~/rds/hpc-work/Softwares/anaconda3/bin/activate btfm-my
which python

echo "Running on node: $(hostname)"
echo "Task ID: $SLURM_ARRAY_TASK_ID"

export I_MPI_OFFLOAD=1

export LD_LIBRARY_PATH=/home/zf281/rds/hpc-work/Softwares/anaconda3/envs/btfm-my/lib:$LD_LIBRARY_PATH

# 运行代码，每个任务只启动一个 agent（这里 count 设置为1，可根据需要修改）
python src/train_sweep_s1_s2_ratio_fix_flops.py --mode agent