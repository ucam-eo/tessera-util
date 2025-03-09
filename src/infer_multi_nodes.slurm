#!/bin/bash -l
#SBATCH --job-name=ddp-torch     # create a short name for your job
#SBATCH --partition=pvc
#SBATCH --account=climate-dawn-gpu
#SBATCH --nodes=4                # node count
#SBATCH --ntasks-per-node=8      # total number of tasks per node
#SBATCH --cpus-per-task=12        # cpu-cores per task
#SBATCH --mem=1000G                # total memory per node
#SBATCH --gres=gpu:4             # number of allocated gpus per node
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=log/btfm-infer-%x-%j.out  # set the output file name in which to save stdout and stderr
#SBATCH --error=log/btfm-infer-%x-%j.err   # set the error file name in which to save compute-node stderr
#SBATCH --exclusive              # exclusive node access

########################
# 1. Load Environment
########################
module purge
module load default-dawn
module load intelpython-conda
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

# conda activate btfm-dawn
conda activate pytorch-gpu

echo "==== Step: Python / mpirun version check ===="
echo "Which python: $(which python)"
echo "Python version: $(python --version)"
echo "Which mpirun: $(which mpirun)"
mpirun --version

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7

echo "Running on node: $(hostname)"
echo "Allocated GPUs:"
sycl-ls

########################
# 2. Print job environment information
########################
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "Node list: $SLURM_JOB_NODELIST"
echo "SLURM_PROCID=$SLURM_PROCID, SLURM_NTASKS=$SLURM_NTASKS"
env | sort

########################
# 3. Set Intel MPI/OFI
########################
# export CCL_ZE_IPC_EXCHANGE=drmfd
export CCL_ZE_SHM_ENABLE=1          # Enable shared memory for IPC
export CCL_ZE_IPC_EXCHANGE=sockets  # Use sockets instead of drmfd
export ZES_ENABLE_SYSMAN=1          # Ensure system management interface is available

export I_MPI_OFFLOAD=1
export I_MPI_OFFLOAD_SYMMETRIC=0
export FI_PSM3_RDMA_MODE=1
# export PSM3_RDMA=1
export PSM3_RDMA=2          # Force RDMA mode
export PSM3_IDENTIFY=1      # Enable device identification

# Use shm + verbs (for Mellanox IB)
# export I_MPI_FABRICS=shm:ofi
# export I_MPI_OFI_PROVIDER=verbs
# export I_MPI_OFI_PROVIDER=mlx
export FI_PROVIDER=verbs
export I_MPI_FABRICS=shm:dapl
export I_MPI_OFI_PROVIDER=verbs
export I_MPI_DAPL_PROVIDER=ofa-v2-mlx5_0-1u

# Let Intel MPI ignore Slurm process placement restrictions
export I_MPI_JOB_RESPECT_PROCESS_PLACEMENT=off

# Enable more debug output
export I_MPI_DEBUG=6
# export FI_LOG_LEVEL=debug
export CCL_LOG_LEVEL=info

########################
# 4. Generate hostfile
########################
scontrol show hostnames $SLURM_JOB_NODELIST > hostfile_intel_mpi
echo "Generated hostfile:"
cat hostfile_intel_mpi

########################
# 5. Set MASTER_ADDR/PORT
########################
export MASTER_PORT=$(( (SLURM_JOB_ID % 10000) + 10000 ))
export MASTER_ADDR=$(head -n1 hostfile_intel_mpi)

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

# 这里假设新的推理脚本infer_multi_xpu.py放在同目录或指定路径
# 并且需要通过 --config 传入配置文件的路径
mpiexec -n $WORLD_SIZE -ppn $SLURM_GPUS_ON_NODE -f hostfile_intel_mpi \
    -genv MASTER_ADDR $MASTER_ADDR \
    -genv MASTER_PORT $MASTER_PORT \
    python src/infer_multi_nodes.py \
       --config configs/infer_config.py

echo "==== Step: mpirun done ===="7