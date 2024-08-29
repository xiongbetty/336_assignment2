#!/usr/bin/env python3

import os
import torch
import timeit
from datetime import timedelta

import torch.distributed as dist
import torch.multiprocessing as mp


def setup(backend):
    # These variables are set via srun
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_world_size = int(os.environ["SLURM_NTASKS_PER_NODE"])

    if backend == "nccl":
        torch.cuda.set_device(local_rank)

    # MASTER_ADDR and MASTER_PORT should have been set in our sbatch script,
    # so we make sure that's the case.
    assert os.environ["MASTER_ADDR"]
    assert os.environ["MASTER_PORT"]

    # Default timeout is 30 minutes. Reducing the timeout here, so the job fails quicker if there's
    # a communication problem between nodes.
    timeout = timedelta(seconds=60)
    dist.init_process_group(backend, rank=rank, world_size=world_size, timeout=timeout)

    return rank, world_size, local_rank, local_world_size


def benchmark_all_reduce(backend, device, data_size):
    rank, world_size, local_rank, local_world_size = setup(backend)
    data = torch.randn(data_size // 4, dtype=torch.float32, device=device)

    if device == "cuda":
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)

    # Warm-up steps
    for _ in range(5):
        dist.all_reduce(data)

    if device == "cuda":
        torch.cuda.synchronize()

    start_time = timeit.default_timer()
    dist.all_reduce(data, async_op=False)

    if device == "cuda":
        torch.cuda.synchronize()

    end_time = timeit.default_timer()
    elapsed_time = end_time - start_time

    # Gather results from all ranks
    times = [None] * world_size
    dist.all_gather_object(times, elapsed_time)

    if rank == 0:
        avg_time = sum(times) / world_size * 1000
        print(f"Backend: {backend}, Device: {device}, Data Size: {data_size / (1024 * 1024):.2f} MB, Processes: {world_size}, Processes per node: {local_world_size}, Average Time: {avg_time:.3f} seconds")


if __name__ == "__main__":
    backend = os.environ["BACKEND"]
    device = os.environ["DEVICE"]
    data_size = int(os.environ["DATA_SIZE"])
    
    benchmark_all_reduce(backend, device, data_size)