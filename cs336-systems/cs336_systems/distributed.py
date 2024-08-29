#!/usr/bin/env python3

import os
import torch
import timeit
import torch.distributed as dist
import torch.multiprocessing as mp


def distributed_demo(rank, world_size):
    setup(rank, world_size)
    data = torch.randint(0, 10, (3,))
    print(f"rank {rank} data (before all-reduce): {data}")
    dist.all_reduce(data, async_op=False)
    print(f"rank {rank} data (after all-reduce): {data}")


def setup(rank, world_size, backend, device):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    if device == "cuda":
        torch.cuda.set_device(rank)


def benchmark_all_reduce(rank, world_size, backend, device, data_size):
    setup(rank, world_size, backend, device)
    data = torch.randn(data_size // 4, dtype=torch.float32, device=device)

    # Warm-up steps
    for _ in range(5):
        dist.all_reduce(data)

    if device == "cuda":
        torch.cuda.synchronize()

    start_time = timeit.default_timer()
    dist.all_reduce(data)

    if device == "cuda":
        torch.cuda.synchronize()

    end_time = timeit.default_timer()
    elapsed_time = end_time - start_time

    # Gather results from all ranks
    times = [None] * world_size
    dist.all_gather_object(times, elapsed_time)

    if rank == 0:
        avg_time = sum(times) / world_size * 1000
        print(f"Backend: {backend}, Device: {device}, Data Size: {data_size / (1024 * 1024):.2f} MB, Processes: {world_size}, Average Time: {avg_time:.5f} milliseconds")


if __name__ == "__main__":
    data_sizes = [512 * 1024, 1 * 1024 * 1024, 10 * 1024 * 1024, 50 * 1024 * 1024, 100 * 1024 * 1024, 500 * 1024 * 1024, 1024 * 1024 * 1024]
    world_sizes = [2, 4, 6]
    backend_device_combinations = [
        ("gloo", "cpu"),
        ("gloo", "cuda"),
        ("nccl", "cuda")
    ]

    for backend, device in backend_device_combinations:
        if device == "cuda" and not torch.cuda.is_available():
            continue

        for data_size in data_sizes:
            for world_size in world_sizes:
                mp.spawn(benchmark_all_reduce, args=(world_size, backend, device, data_size), nprocs=world_size, join=True)