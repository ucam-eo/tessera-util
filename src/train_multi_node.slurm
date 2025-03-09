#!/bin/bash -l
#SBATCH --job-name=ddp-torch     # create a short name for your job
#SBATCH --partition=pvc          #
#SBATCH --account=climate-dawn-gpu    #
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=8      # total number of tasks per node
#SBATCH --cpus-per-task=12       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=1000G              # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of allocated gpus per node
#SBATCH --time=36:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=log/btfm-train-%x-%j.out  # set the output file name in which to save stdout and stderr
#SBATCH --error=log/btfm-train-%x-%j.err   # set the error file name in which to save compute-node stderr
#SBATCH --exclusive              # exclusive node access

########################
# 1. Load Environment
########################
module purge
module load default-dawn
module load oneapi-level-zero
module load intel-oneapi-ccl
module load intel-oneapi-compilers
module load intel-oneapi-dnn
module load intel-oneapi-dpct
module load intel-oneapi-dpl
module load intel-oneapi-inspector
module load intel-oneapi-mkl
module load intel-oneapi-mpi
module load intel-oneapi-tbb
module load intelpython-conda

# conda activate pytorch-gpu
source ~/rds/hpc-work/Softwares/anaconda3/bin/activate btfm-my

echo "==== Step: Python / mpirun version check ===="
echo "Which python: $(which python)"
echo "Python version: $(python --version)"
echo "Which mpirun: $(which mpirun)"
mpirun --version

# 一些针对Intel GPU可能需要的环境变量
export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7
# export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
echo "Running on node: $(hostname)"
echo "Allocated GPUs:"
sycl-ls

########################
# 2. Print job environment information
########################
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "Node list: $SLURM_JOB_NODELIST"
echo "SLURM_PROCID=$SLURM_PROCID, SLURM_NTASKS=$SLURM_NTASKS"
echo "Print environment variables (some may be large):"
env | sort

########################
# 3. Set Intel MPI/OFI related environment variables
########################
export CCL_ZE_SHM_ENABLE=1
export CCL_ZE_IPC_EXCHANGE=sockets
export ZES_ENABLE_SYSMAN=1

export I_MPI_OFFLOAD=1
export I_MPI_OFFLOAD_SYMMETRIC=0
export FI_PSM3_RDMA_MODE=1
export PSM3_RDMA=2
export PSM3_IDENTIFY=1

# export FI_PROVIDER=verbs
# export I_MPI_FABRICS=shm:dapl
# export I_MPI_OFI_PROVIDER=verbs
# export I_MPI_DAPL_PROVIDER=ofa-v2-mlx5_0-1u
export FI_PROVIDER=verbs
export I_MPI_FABRICS=shm:tcp
export I_MPI_OFI_PROVIDER=verbs


export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=off
export I_MPI_DEBUG=6
export CCL_LOG_LEVEL=info

########################
# 4. Generate hostfile
########################
scontrol show hostnames $SLURM_JOB_NODELIST > hostfile_intel_mpi_train
echo "Generated hostfile:"
cat hostfile_intel_mpi_train

########################
# 5. Set MASTER_ADDR/PORT
########################
export MASTER_PORT=$(( (SLURM_JOB_ID % 10000) + 10000 ))
export MASTER_ADDR=$(head -n1 hostfile_intel_mpi_train)

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"

########################
# 6. Launch
########################
echo "==== Step: mpirun start ===="

echo "WORLD_SIZE=$WORLD_SIZE"
if [ "$ZE_FLAT_DEVICE_HIERARCHY" = "FLAT" ]; then
    export SLURM_GPUS_ON_NODE=$(($SLURM_GPUS_ON_NODE * 2))
fi
echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

# 额外：向日志中打印一下当前节点上文件系统信息，以防IO卡住
echo "==== Disk usage in current directory ===="
du -sh . | head -n 1
echo "==== Free memory info (first lines) ===="
free -h | head -n 2

# 启动训练
mpiexec -n $WORLD_SIZE -ppn $SLURM_GPUS_ON_NODE -f hostfile_intel_mpi_train \
    -genv MASTER_ADDR $MASTER_ADDR \
    -genv MASTER_PORT $MASTER_PORT \
    python src/train_multi_node.py

echo "==== Step: mpirun done ===="
