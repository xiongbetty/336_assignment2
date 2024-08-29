#!/usr/bin/env python3

import argparse
import timeit
import numpy as np
from typing import List
from contextlib import nullcontext

import torch
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
import torch.distributed as dist
import torch.multiprocessing as mp

from cs336_basics.train import AdamW, cross_entropy, data_loading
from cs336_basics.model import TransformerLM, RMSNorm
from cs336_systems.kernel import TritonRMSNormFunction
from cs336_systems.ddp import DDP, DDPBucket
from tests.common import _cleanup_process_group, _setup_process_group


vocab_size = 10000
context_length = 128
batch_size = 16


def run_benchmarking(rank, world_size, args):
    device = _setup_process_group(rank=rank, world_size=world_size, backend="gloo")
    dist.barrier()
    benchmarking(args, device)
    _cleanup_process_group()


def benchmarking(args, device):
    # Initialize model
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        weights=None,
        device=device,
    ).to(device)

    # model = DDP(model)
    model = DDPBucket(model, 500)
    # bucket size = [5, 10, 50, 100, 500]
    optimizer = AdamW(model.parameters())

    # Generate a random batch of data
    random_data = np.random.choice(10000, 1000, replace=True)
    inputs, targets = data_loading(dataset=random_data,
                                   batch_size=batch_size,
                                   context_length=context_length,
                                   device=device) 

    model.train()  # Set model to training model

    # Run warm-up steps
    for _ in range(args.warm_up_steps):
        # print(inputs)
        # print(targets)
        logits = model(inputs)
        # print(logits)
        logits_reshaped = logits.view(-1, vocab_size)
        targets_reshaped = targets.view(-1)
        loss = cross_entropy(logits_reshaped, targets_reshaped)
        if args.backward_pass:
            loss.backward()

    # Benchmarking
    times: List[float] = []
    backward_times: List[float] = []
    step_times: List[float] = []
    for _ in range(args.measurement_steps):
        torch.cuda.synchronize()
        start_time = timeit.default_timer()

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.mixed_precision):
            logits = model(inputs)
            logits_reshaped = logits.view(-1, vocab_size)
            targets_reshaped = targets.view(-1)
            loss = cross_entropy(logits_reshaped, targets_reshaped)

        torch.cuda.synchronize()
        forward_time = timeit.default_timer()
        times.append((forward_time - start_time) * 1000)  # Time in ms

        if args.backward_pass:
            loss.backward()
            model.finish_gradient_synchronization()
            optimizer.step()
            torch.cuda.synchronize()
            backward_time = timeit.default_timer()
            backward_times.append((backward_time - forward_time) * 1000)  # Time in ms
            step_times.append((backward_time - start_time) * 1000)  # Time in ms
            

    # Print results
    print(args.d_model, args.d_ff, args.num_layers, args.num_heads)
    print("Forward pass times:", times)
    if args.backward_pass:
        print("Backward pass times:", backward_times)
        print("Step times:", step_times)

    if len(times) > 0:
        print("Average forward pass time:", np.mean(times))
        # print("Standard deviation of forward pass time:", np.std(times))
    if len(backward_times) > 0:
        print("Average backward pass time:", np.mean(backward_times))
        # print("Standard deviation of backward pass time:", np.std(backward_times))
    if len(step_times) > 0:
        print("Average step time:", np.mean(step_times))
        # print("Standard deviation of step time:", np.std(step_times))

def run_step(model, inputs, targets, optimizer, enable_backward, compiled=False):
    if compiled:
        model = torch.compile(model)
    with record_function('forward_pass'):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.mixed_precision):
            logits = model(inputs)
            logits_reshaped = logits.view(-1, vocab_size)
            targets_reshaped = targets.view(-1)
            loss = cross_entropy(logits_reshaped, targets_reshaped)

    if enable_backward:
        with record_function('backward_pass'):
            loss.backward() 
        with record_function('optimizer'):
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


def profiling(warm_up_steps, n_steps, enable_backward):
    # Initialize model
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        weights=None,
        device=args.device,
    ).to(args.device)

    # Initialize optimizer
    optimizer = AdamW(model.parameters())

    # Generate a random batch of data
    random_data = np.random.choice(10000, 1000, replace=True)
    inputs, targets = data_loading(dataset=random_data,
                                   batch_size=batch_size,
                                   context_length=context_length,
                                   device=args.device) 

    model.train()  # Set model to training model

    # Run warm-up steps
    for _ in range(warm_up_steps):
        # print(inputs)
        # print(targets)
        logits = model.forward(inputs)
        # print(logits)
        logits_reshaped = logits.view(-1, vocab_size)
        targets_reshaped = targets.view(-1)
        loss = cross_entropy(logits_reshaped, targets_reshaped)
        loss.backward()

    with profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ], experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
    ) as prof:
        for _ in range(n_steps):
            run_step(model, inputs, targets, optimizer, enable_backward)
            prof.step()
    prof.export_stacks("lm_profiler_stacks.txt", "self_cuda_time_total")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=50))


def layer_norm_benchmarking(
        args,
        input_rows=50000,
        input_dims=[1024, 2048, 4096, 8192],
        num_forward_passes = 1000,
        ):
    # Create random inputs
    x = torch.randn(input_rows, max(input_dims)).to(args.device)
    weights = {d: torch.randn(d).to(args.device) for d in input_dims}
    biases = {d: torch.randn(d).to(args.device) for d in input_dims}
    dy = torch.randn_like(x)

    # Run warm-up steps
    for _ in range(args.warm_up_steps):
        rms_norm = RMSNorm(max(input_dims)).to(args.device)
        _ = rms_norm(x)

    # Benchmark RMSNorm
    print("Benchmarking RMSNorm:")
    times_rmsnorm: List[float] = []
    for d_model in input_dims:
        x_partial = x[:, :d_model]  # Slice x to match current dimension
        rms_norm = RMSNorm(d_model).to(args.device)
        rms_norm.weight.data = weights[d_model]

        if args.backward_pass:
            rms_norm.train()
            x_partial.requires_grad = True

        for _ in range(num_forward_passes):
            rms_norm.zero_grad()
            if args.backward_pass:
                x_partial.grad = None
            torch.cuda.synchronize()
            start_time = timeit.default_timer()
            result = rms_norm(x_partial)

            if args.backward_pass:
                result.backward(dy[:, :d_model])

            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            times_rmsnorm.append((end_time - start_time) * 1000)
        print(f"Hidden Dimension: {d_model}, Time (ms): {np.mean(times_rmsnorm)}")

    # Benchmark LayerNorm
    print("\nBenchmarking LayerNorm:")
    times_layernorm: List[float] = []
    for d_model in input_dims:
        x_partial = x[:, :d_model]  # Slice x to match current dimension
        layer_norm = nn.LayerNorm(d_model, elementwise_affine=True).to(args.device)
        layer_norm.weight.data = weights[d_model]
        layer_norm.bias.data = biases[d_model] 

        if args.backward_pass:
            layer_norm.train()
            x_partial.requires_grad = True

        for _ in range(num_forward_passes):
            layer_norm.zero_grad()
            if args.backward_pass:
                x_partial.grad = None

            torch.cuda.synchronize()
            start_time = timeit.default_timer()
            result = layer_norm(x_partial)

            if args.backward_pass:
                result.backward(dy[:, :d_model])

            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            times_layernorm.append((end_time - start_time) * 1000)
        print(f"Hidden Dimension: {d_model}, Time (ms): {np.mean(times_layernorm)}")

    # Benchmark Triton RMSNorm
    print("\nBenchmarking Triton RMSNorm:")
    times_triton: List[float] = []
    for d_model in input_dims:
        x_partial = x[:, :d_model].contiguous()  # Slice x to match current dimension
        
        if args.backward_pass:
            x_partial.requires_grad = True

        for _ in range(num_forward_passes):
            if args.backward_pass:
                x_partial.grad = None

            torch.cuda.synchronize()
            start_time = timeit.default_timer()
            result = TritonRMSNormFunction.apply(x_partial, weights[d_model])

            if args.backward_pass:
                result.backward(dy[:, :d_model])

            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            times_triton.append((end_time - start_time) * 1000)
        print(f"Hidden Dimension: {d_model}, Time (ms): {np.mean(times_triton)}")

    print("\nBenchmarking compiled RMSNorm:")
    times_compiled: List[float] = []
    for d_model in input_dims:
        x_partial = x[:, :d_model]  # Slice x to match current dimension
        rms_norm = RMSNorm(d_model)
        compiled = torch.compile(rms_norm).to(args.device)
        compiled.weight.data = weights[d_model]

        if args.backward_pass:
            compiled.train()
            x_partial.requires_grad = True

        for _ in range(num_forward_passes):
            compiled.zero_grad()
            if args.backward_pass:
                x_partial.grad = None
            torch.cuda.synchronize()
            start_time = timeit.default_timer()
            result = compiled(x_partial)

            if args.backward_pass:
                result.backward(dy[:, :d_model])

            torch.cuda.synchronize()
            end_time = timeit.default_timer()
            times_compiled.append((end_time - start_time) * 1000)
        print(f"Hidden Dimension: {d_model}, Time (ms): {np.mean(times_compiled)}")


def memory_profile(args):
    # Initialize model
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        weights=None,
        device=args.device,
    ).to(args.device)

    if args.compiled_version:
        model = torch.compile(model)

    # Initialize optimizer
    optimizer = AdamW(model.parameters())

    # Generate a random batch of data
    random_data = np.random.choice(10000, 1000, replace=True)
    inputs, targets = data_loading(dataset=random_data,
                                   batch_size=batch_size,
                                   context_length=context_length,
                                   device=args.device) 

    model.train()  # Set model to training model

    # Run warm-up steps
    for _ in range(args.warm_up_steps):
        # print(inputs)
        # print(targets)
        logits = model(inputs)
        # print(logits)
        logits_reshaped = logits.view(-1, vocab_size)
        targets_reshaped = targets.view(-1)
        loss = cross_entropy(logits_reshaped, targets_reshaped)
        loss.backward()

    # Start recording memory history.
    torch.cuda.memory._record_memory_history(max_entries=1000000)
    n_steps = 3
    
    with profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=n_steps),
        experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for _ in range(n_steps):
            run_step(model, inputs, targets, optimizer, args.backward_pass, args.compiled_version)
            prof.step()
        # Save a graphical timeline of memory usage.
        prof.export_memory_timeline("timeline.html", device=inputs.device)
    
    # Save a pickle file to be loaded by PyTorch's online tool.
    torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
    # Stop recording history.
    torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Transformer LM')
    parser.add_argument('--d_model', type=int, default=512, help='Model dimension')
    parser.add_argument('--num_layers', type=int, default=6, help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--d_ff', type=int, default=2048, help='Feedforward dimension')
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu", help='Device (cuda or cpu)')
    parser.add_argument('--warm_up_steps', type=int, default=5, help='Number of warm-up steps')
    parser.add_argument('--measurement_steps', type=int, default=5, help='Number of measurement steps')
    parser.add_argument('--backward_pass', action='store_true', help='Perform backward pass as well')
    parser.add_argument('--mixed_precision', action='store_true', help='Enable mixed precision training for the forward pass')
    parser.add_argument('--compiled_version', action='store_true', help='Enable Pytorch compiler')
    args = parser.parse_args()

    # benchmarking(args)
    # profiling(
    #     warm_up_steps=args.warm_up_steps,
    #     n_steps=args.measurement_steps,
    #     enable_backward=args.backward_pass,
    # )
    # layer_norm_benchmarking(args)
    # memory_profile(args)


    world_size = 2
    mp.spawn(
        run_benchmarking,
        args=(world_size, args),
        nprocs=world_size,
        join=True,
    )
