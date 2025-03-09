#!/bin/bash -l
#SBATCH --job-name=ddp-torch     # create a short name for your job
#SBATCH --partition=pvc          #
#SBATCH --account=climate-dawn-gpu    #
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=8      # total number of tasks per node
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of allocated gpus per node
#SBATCH --time=00:30:00          # total run time limit (HH:MM:SS)
#SBATCH --exclusive              # exclusive node access

########################
# 1. Load Environment
########################
module purge
module load default-dawn
# module load oneapi-level-zero/1.14/gcc/3tn5pfua
# module load intel-oneapi-compilers/2025.0.4/gcc/umo7dwbo

# source /usr/local/dawn/software/external/intel-oneapi/2024.0/setvars.sh
# source /usr/local/dawn/software/external/intel-oneapi/2024.0/compiler/latest/env/vars.sh
# source /usr/local/dawn/software/external/intel-oneapi/2024.0/mpi/latest/env/vars.sh
# source /usr/local/dawn/software/external/intel-oneapi/2024.0/dnnl/latest/env/vars.sh
# source /usr/local/dawn/software/external/intel-oneapi/2024.0/mkl/latest/env/vars.sh
# source /usr/local/dawn/software/external/intel-oneapi/2024.0/ccl/latest/env/vars.sh
# source /usr/local/dawn/software/external/intel-oneapi/2024.0/tbb/latest/env/vars.sh

module load oneapi-level-zero
module load intelpython-conda
module load intel-oneapi-ccl
module load intel-oneapi-compilers
module load intel-oneapi-dnn
module load intel-oneapi-dpct
module load intel-oneapi-dpl
module load intel-oneapi-inspector
module load intel-oneapi-mkl
module load intel-oneapi-mpi
module load intel-oneapi-tbb

conda activate pytorch-gpu
# source ~/rds/hpc-work/Softwares/anaconda3/bin/activate btfm-my
# source ~/rds/hpc-work/Softwares/anaconda3/bin/activate btfm-copy
# conda activate btfm-my
# conda activate btfm-copy
# conda activate btfm-dawn

echo "==== Step: Python / mpirun version check ===="
echo "Which python: $(which python)"
echo "Python version: $(python --version)"
echo "Which mpirun: $(which mpirun)"
mpirun --version

export ZE_FLAT_DEVICE_HIERARCHY=FLAT
export ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7
# export ZE_AFFINITY_MASK=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
# export ZE_AFFINITY_MASK=0.0,0.1,1.0,1.1,2.0,2.1,3.0,3.1

# export ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE
# export ZE_AFFINITY_MASK=0,1,2,3

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
# export FI_PROVIDER=verbs
# export I_MPI_FABRICS=shm:dapl
# export I_MPI_OFI_PROVIDER=verbs
# export I_MPI_DAPL_PROVIDER=ofa-v2-mlx5_0-1u

export FI_PROVIDER=verbs
export I_MPI_FABRICS=shm:tcp
export I_MPI_OFI_PROVIDER=verbs

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

# srun python cam_simple_ddp.py

echo "WORLD_SIZE=$WORLD_SIZE"
# echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
# Need to check ZE_FLAT_DEVICE_HIERARCHY, if it's FLAT, then SLURM_GPUS_ON_NODE needs to be multiplied by 2
if [ "$ZE_FLAT_DEVICE_HIERARCHY" = "FLAT" ]; then
    export SLURM_GPUS_ON_NODE=$(($SLURM_GPUS_ON_NODE * 2))
fi
echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
echo "SLURM_NTASKS_PER_NODE=$SLURM_NTASKS_PER_NODE"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "MASTER_PORT=$MASTER_PORT"

# mpiexec -n $WORLD_SIZE -ppn $SLURM_GPUS_ON_NODE -f hostfile_intel_mpi \
mpiexec -n $WORLD_SIZE -ppn $SLURM_NTASKS_PER_NODE -f hostfile_intel_mpi \
    -genv MASTER_ADDR $MASTER_ADDR \
    -genv MASTER_PORT $MASTER_PORT \
    python src/cam_simple_ddp.py

echo "==== Step: mpirun done ===="
