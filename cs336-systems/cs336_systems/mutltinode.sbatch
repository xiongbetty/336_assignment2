#!/bin/bash

#SBATCH --partition=a2
#SBATCH --cpus-per-task=4
#SBATCH --nodes=2
#SBATCH --mem=16G
#SBATCH --time=00:05:00
#SBATCH --output=multi_%j.out
#SBATCH --error=multi_%j.err

# Set the number of tasks per node (manually set this value)
ntasks_per_node=1

# Set the number of GPUs per node equal to the number of tasks per node
#SBATCH --gpus-per-node=$ntasks_per_node

# Set the number of tasks per node
#SBATCH --ntasks-per-node=$ntasks_per_node

eval "$(conda shell.bash hook)"
conda activate cs336_systems

# Get the master address and port
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=$((10000 + $(echo -n $SLURM_JOBID | tail -c 4)))

echo "MASTER_PORT: ${MASTER_PORT}"
echo "MASTER_ADDR: ${MASTER_ADDR}"

# Set the total number of tasks
export SLURM_NTASKS_PER_NODE=$ntasks_per_node
export SLURM_NTASKS=$((ntasks_per_node * 2))  # Multiply by 2 since we have 2 nodes

# Export the required variables
export SLURM_PROCID=$SLURM_PROCID
export SLURM_LOCALID=$SLURM_LOCALID

echo "Running benchmark with $ntasks_per_node processes per node"

# Run the Python script using srun and pass the environment variables
srun --export=ALL python distributed_multi.py