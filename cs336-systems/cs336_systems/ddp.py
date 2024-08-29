#!/usr/bin/env python3

import os
import timeit
import math
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn as nn


# CLASSES

class DDP(nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module
        self.handles = []
        self.world_size = dist.get_world_size()

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        def hook(param):
            param.grad = param.grad / self.world_size
            handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
            self.handles.append(handle)

        for param in self.module.parameters():
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(hook)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()


class DDPBucket(torch.nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super().__init__()
        self.module = module
        self.bucket_size_mb = bucket_size_mb
        self.buckets = []
        self.handles = []
        self.world_size = dist.get_world_size()
        self.param_to_bucket = {}
        self.processed_buckets = set()

        for param in self.module.parameters():
            dist.broadcast(param.data, src=0)

        # Allocate parameters to buckets
        bucket = []
        bucket_size = 0
        for param in reversed(list(self.module.parameters())):
            if param.requires_grad:
                if bucket_size + param.numel() * param.element_size() > self.bucket_size_mb * 1024 * 1024:
                    if bucket:
                        self.buckets.append(bucket)
                        for p in bucket:
                            self.param_to_bucket[p] = len(self.buckets) - 1
                        bucket = []
                        bucket_size = 0
                bucket.append(param)
                bucket_size += param.numel() * param.element_size()
        if bucket:
            self.buckets.append(bucket)
            for p in bucket:
                self.param_to_bucket[p] = len(self.buckets) - 1
                
        # Register hooks for each bucket
        for bucket in self.buckets:
            self._register_hooks(bucket)

    def _register_hooks(self, bucket):
        def hook(param):
            bucket_idx = self.param_to_bucket[param]
            if bucket_idx not in self.processed_buckets:
                if all(p.grad is not None for p in bucket):
                    for p in bucket:
                        p.grad = p.grad / self.world_size
                        handle = dist.all_reduce(p.grad, op=dist.ReduceOp.SUM, async_op=True)
                        self.handles.append(handle)
                    self.processed_buckets.add(bucket_idx)

        for param in bucket:
            if param.requires_grad:
                param.register_post_accumulate_grad_hook(hook)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        self.processed_buckets.clear()
        self.buckets.clear()


# FUNCTIONS

def ddp_main(rank: int, world_size: int, data: torch.Tensor, num_layers: int, num_steps: int, device):
    setup(rank, world_size, device)
    # print(f"Rank: {rank}, Data shape: {data.shape}")
    
    # Get the slice of data for this rank
    batch_size = data.size(0) // world_size
    num_dim = data.size(1)
    start_index = rank * batch_size
    end_index = start_index + batch_size
    data = data[start_index:end_index].to(f"cuda:{rank}")
    
    # Create MLP: # gelu(gelu(x @ params[0]) @ params[1]) ...
    params = [get_init_params(num_dim, num_dim, rank) for i in range(num_layers)]
    # print("initial params")
    # print(params)
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    
    # Warm up
    for _ in range(5):
        # Forward pass
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()  # Loss function is average squared magnitude
        
        # Backward pass
        loss.backward()
        
        # Sync gradients across workers (NEW!)
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)
        
        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

    # Benchmarking
    torch.cuda.synchronize()
    start_time = timeit.default_timer()

    list_time_gradients = []

    for _ in range(num_steps):
        # Forward pass
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()  # Loss function is average squared magnitude
        
        # Backward pass
        loss.backward()
        
        # Sync gradients across workers (NEW!)
        torch.cuda.synchronize()
        start_time_gradients = timeit.default_timer()

        # for param in params:
        #     dist.all_reduce(tensor=flat_grads, op=dist.ReduceOp.AVG, async_op=False)

        # Flatten all parameter gradients into a single tensor
        flat_grads = torch._utils._flatten_dense_tensors(tuple(param.grad for param in params))

        dist.all_reduce(tensor=flat_grads, op=dist.ReduceOp.AVG, async_op=False)
        unflat_grads = torch._utils._unflatten_dense_tensors(flat_grads, tuple(param.grad for param in params))
        for param, grad in zip(params, unflat_grads):
            param.grad = grad

        # Update parameters
        optimizer.step()

        torch.cuda.synchronize()
        end_time_gradients = timeit.default_timer()
        elapsed_time_gradients = end_time_gradients - start_time_gradients
        times_gradients = [None] * world_size
        dist.all_gather_object(times_gradients, elapsed_time_gradients)
        if rank == 0:
            avg_time_gradients = sum(times_gradients) / world_size
            list_time_gradients.append(avg_time_gradients)
        
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    end_time = timeit.default_timer()
    elapsed_time = (end_time - start_time) * 1000 / num_steps
    times = [None] * world_size
    dist.all_gather_object(times, elapsed_time)

    if rank == 0:
        avg_time = sum(times) / world_size
        print("Single-node")
        print(f"Time per training step (milliseonds): {avg_time}")
        print(f"Time for gradients (milliseonds): {sum(list_time_gradients) / num_steps}")

    # print("updated params")
    # print(params)
    cleanup()


def ddp_main_multinode(data: torch.Tensor, num_layers: int, num_steps: int, device):
    rank, world_size, local_rank, local_world_size = setup_multinode()
    # print(f"Rank: {rank}, Data shape: {data.shape}")

    # if device == "cuda":
    #     device = f"cuda:{local_rank}"
    #     torch.cuda.set_device(local_rank)
    
    # Get the slice of data for this rank
    batch_size = data.size(0) // world_size
    num_dim = data.size(1)
    start_index = rank * batch_size
    end_index = start_index + batch_size
    data = data[start_index:end_index].to(f"cuda:{local_rank}")
    
    # Create MLP: # gelu(gelu(x @ params[0]) @ params[1]) ...
    params = [get_init_params(num_dim, num_dim, local_rank) for i in range(num_layers)]
    # print("initial params")
    # print(params)
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    
    # Warm up
    for _ in range(5):
        # Forward pass
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()  # Loss function is average squared magnitude
        
        # Backward pass
        loss.backward()
        
        # Sync gradients across workers (NEW!)
        for param in params:
            dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)
        
        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

    # Benchmarking
    torch.cuda.synchronize()
    start_time = timeit.default_timer()

    list_time_gradients = []

    for _ in range(num_steps):
        # Forward pass
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()  # Loss function is average squared magnitude
        
        # Backward pass
        loss.backward()
        
        # Sync gradients across workers (NEW!)
        torch.cuda.synchronize()
        start_time_gradients = timeit.default_timer()
        
        # for param in params:
        #     dist.all_reduce(tensor=param.grad, op=dist.ReduceOp.AVG, async_op=False)

        # Flatten all parameter gradients into a single tensor
        flat_grads = torch._utils._flatten_dense_tensors(tuple(param.grad for param in params))

        dist.all_reduce(tensor=flat_grads, op=dist.ReduceOp.AVG, async_op=False)
        unflat_grads = torch._utils._unflatten_dense_tensors(flat_grads, tuple(param.grad for param in params))
        for param, grad in zip(params, unflat_grads):
            param.grad = grad

        # Update parameters
        optimizer.step()

        torch.cuda.synchronize()
        end_time_gradients = timeit.default_timer()
        elapsed_time_gradients = end_time_gradients - start_time_gradients
        times_gradients = [None] * world_size
        dist.all_gather_object(times_gradients, elapsed_time_gradients)
        if rank == 0:
            avg_time_gradients = sum(times_gradients) / world_size
            list_time_gradients.append(avg_time_gradients)
        
        optimizer.zero_grad()
    
    torch.cuda.synchronize()
    end_time = timeit.default_timer()
    elapsed_time = (end_time - start_time) * 1000 / num_steps
    times = [None] * world_size
    dist.all_gather_object(times, elapsed_time)

    if rank == 0:
        avg_time = sum(times) / world_size
        print("Multinode")
        print(f"Time per training step (milliseonds): {avg_time}")
        print(f"Time for gradients (milliseonds): {sum(list_time_gradients) / num_steps}")

    # print("updated params")
    # print(params)
    cleanup()


def single_process_main(data: torch.Tensor, num_layers: int, num_steps: int):
    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Move data to the device
    data = data.to(device)
    
    # Create MLP on the device
    # gelu(gelu(x @ params[0]) @ params[1]) ...
    params = [get_init_params(data.size(1), data.size(1), 0).to(device) for i in range(num_layers)]
    print("initial CPU params")
    print(params)
    optimizer = torch.optim.AdamW(params, lr=1e-3)
    
    for _ in range(num_steps):
        # Forward pass
        x = data
        for param in params:
            x = x @ param
            x = F.gelu(x)
        loss = x.square().mean()  # Loss function is average squared magnitude
        
        # Backward pass
        loss.backward()
        
        # Update parameters
        optimizer.step()
        optimizer.zero_grad()

    print("updated CPU params")
    print(params)


def setup(rank: int, world_size: int, device):
    # This is where master lives (rank 0)
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "15623"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    if device == "cuda":
        torch.cuda.set_device(rank)


def setup_multinode():
    # These variables are set via srun
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_world_size = int(os.environ["SLURM_NTASKS_PER_NODE"])

    print(torch.cuda.is_available())
    print(rank, local_rank, world_size, local_world_size)
    torch.cuda.set_device(local_rank)

    # MASTER_ADDR and MASTER_PORT should have been set in our sbatch script,
    # so we make sure that's the case.
    assert os.environ["MASTER_ADDR"]
    assert os.environ["MASTER_PORT"]

    # Default timeout is 30 minutes. Reducing the timeout here, so the job fails quicker if there's
    # a communication problem between nodes.
    timeout = timedelta(seconds=60)
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=timeout)

    return rank, world_size, local_rank, local_world_size


def cleanup():
    dist.destroy_process_group()


def get_init_params(num_inputs: int, num_outputs: int, rank: int) -> nn.Parameter:
    torch.random.manual_seed(0)  # For reproducibility
    return nn.Parameter(torch.randn(num_inputs, num_outputs, device=f"cuda:{rank}") / math.sqrt(num_outputs))


def summarize_tensor(tensor: torch.Tensor) -> str:
    return "x".join(map(str, tensor.shape)) + "[" + str(round(tensor.view(-1)[0].item(), 4)) + "...]"


if __name__ == "__main__":
    # if torch.cuda.is_available():
    #     device = "cuda"
    # world_size = 2
    # num_layers = 3
    # num_steps = 10
    # data = torch.randn(1000, 100)  # Random data for training
    
    # # Single-process training
    # # single_process_main(data, num_layers, num_steps)
    
    # # DDP training
    # mp.spawn(
    #     ddp_main,
    #     args=(world_size, data, num_layers, num_steps, device),
    #     nprocs=world_size,
    #     join=True,
    # )

    # multinode
    device = "cuda"
    data = torch.randn(1000, 100)  # Random data for training
    num_layers = 3
    num_steps = 10

    ddp_main_multinode(data, num_layers, num_steps, device)