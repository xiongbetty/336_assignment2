#!/bin/bash

#SBATCH --partition=a2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=00:05:00
#SBATCH --output=multi_%j.out
#SBATCH --error=multi_%j.err

#SBATCH --nodes=2


eval "$(conda shell.bash hook)"
conda activate cs336_systems

# Get the master address and port
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((10000 + $(echo -n $SLURM_JOBID | tail -c 4)))

export BACKEND=$1
export DEVICE=$2
export DATA_SIZE=$3

echo "MASTER_PORT: ${MASTER_PORT}"
echo "MASTER_ADDR: ${MASTER_ADDR}"
echo "Running benchmark with 1 processes per node"

# Run the Python script using srun and pass the environment variables
srun --export=ALL python distributed_multi.py