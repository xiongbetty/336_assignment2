#!/usr/bin/env python3

import torch
import triton
import triton.language as tl
import os

# os.environ["TRITON_INTERPRET"] = "1"


# CLASSES

class RMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-5):
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        mean_square = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + eps)
        out_features = x / mean_square * weight
        return out_features

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        eps = ctx.eps
        grad_g = compute_grad_g(grad_output, x, weight, eps=eps)
        grad_x = compute_grad_x(grad_output, x, weight, eps=eps)
        return grad_x, grad_g


class TritonRMSNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        # Remember x and weight for the backward pass, when we
        # only receive the gradient wrt. the output tensor, and
        # need to compute the gradients wrt. x and weight.
        ctx.save_for_backward(x, weight)

        x_arg = x.reshape(-1, x.shape[-1])
        M, H = x_arg.shape

        assert len(weight.shape) == 1 and weight.shape[0] == H, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        ctx.BLOCK_SIZE = triton.next_power_of_2(H)
        y = torch.empty_like(x)
        rms = torch.empty(M, device=x.device)

        # Save the mean tensor in ctx for use in the backward pass
        ctx.rms = rms

        # Launch our kernel with n instances
        n_rows = M
        rmsnorm_fwd[(n_rows,)](
            x_arg,
            weight,
            x_arg.stride(0),
            y.reshape(-1, y.shape[-1]),
            H,
            num_warps=16,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
            rms_ptr=rms,
        )
        
        return y.reshape_as(x)


    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        x_arg = x.reshape(-1, x.shape[-1])
        M, H = x_arg.shape
        # M, H = x.shape[-2], x.shape[-1]
        
        # Allocate output tensors.
        grad_x = torch.empty_like(x)
        grad_weight = torch.zeros_like(weight)

        # Reshape grad_out to match the shape of x_arg in the forward pass.
        grad_out_arg = grad_out.reshape(-1, grad_out.shape[-1])

        # Launch our kernel with n instances.
        n_rows = M
        rmsnorm_bwd[(n_rows,)](
            grad_out_arg,
            grad_x.reshape(-1, grad_x.shape[-1]),
            grad_weight,
            x_arg,
            weight,
            ctx.rms,
            x_arg.stride(0),
            H,
            BLOCK_SIZE=ctx.BLOCK_SIZE,
        )

        return grad_x.reshape_as(x), grad_weight
    

# FUNCTIONS

@triton.jit
def rmsnorm_fwd(
    x_ptr: tl.pointer_type,
    weight_ptr: tl.pointer_type,
    x_row_stride: tl.uint32,
    output_ptr: tl.pointer_type,
    H: tl.uint32,
    BLOCK_SIZE: tl.constexpr,
    rms_ptr: tl.pointer_type,
    eps=1e-5,
):
    # Each instance will compute the rmsnorm x.
    row_idx = tl.program_id(0)
    # Pointer to the first entry of the row.
    row_start_ptr = x_ptr + row_idx * x_row_stride
    output_ptr = output_ptr + row_idx * x_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)

    # Compute mean_square.
    mean_square = 0
    _mean_square = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    # Pointers to the entries for computation.
    x_ptrs = row_start_ptr + offsets
    weight_ptrs = weight_ptr + offsets

    # Load the data from x given the pointers to its entries,
    # using a mask since BLOCK_SIZE may be > H.
    mask = offsets < H
    row = tl.load(x_ptrs, mask=mask, other=0).to(tl.float32)
    weight = tl.load(weight_ptrs, mask=mask, other=0).to(tl.float32)

    # Compute.
    _mean_square = row * row
    mean_square = tl.sum(_mean_square, axis=0) / H + eps
    rms = tl.sqrt(mean_square)

    # Write mean_square
    tl.store(rms_ptr + row_idx, rms)

    # Write back output.
    output = row / rms * weight
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def rmsnorm_bwd(
    grad_output_ptr: tl.pointer_type,
    grad_x_ptr: tl.pointer_type,
    grad_weight_ptr: tl.pointer_type,
    x_ptr: tl.pointer_type,
    weight_ptr: tl.pointer_type,
    rms_ptr: tl.pointer_type,
    x_row_stride: tl.uint32,
    H: tl.uint32,
    BLOCK_SIZE: tl.constexpr,
    eps=1e-5,
):
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * x_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)

    x_ptrs = row_start_ptr + offsets
    weight_ptrs = weight_ptr + offsets
    grad_x_ptrs = grad_x_ptr + row_idx * x_row_stride + offsets
    grad_weight_ptrs = grad_weight_ptr + offsets
    grad_output_ptrs = grad_output_ptr + row_idx * x_row_stride + offsets

    mask = offsets < H
    x_row = tl.load(x_ptrs, mask=mask, other=0).to(tl.float32)
    weight = tl.load(weight_ptrs, mask=mask, other=0).to(tl.float32)

    # Gradient with respect to the output of RMSNorm at row_idx.
    grad_output = tl.load(grad_output_ptrs, mask=mask, other=0).to(tl.float32)

    # Load mean square value for the current row.
    denominator = tl.load(rms_ptr + row_idx)

    # Compute gradient with respect to the current row of x.
    grad_x_row = (grad_output * weight) / denominator
    dot_product = tl.sum(x_row * grad_output * weight)
    grad_x_row -= (dot_product * x_row) / (H * denominator * denominator * denominator)

    # Write the gradient to grad_x_ptr.
    tl.store(grad_x_ptrs, grad_x_row, mask=mask)

    # Compute gradient with respect to the weight vector.
    grad_weight_row = (x_row * grad_output) / denominator

    # Write the gradient to grad_weight_ptr.
    tl.atomic_add(grad_weight_ptrs, grad_weight_row, mask=mask)


def compute_grad_g(
        grad_y: torch.Tensor, 
        x: torch.Tensor, 
        g: torch.Tensor,
        eps=1e-5,
    ):
    H = x.shape[-1]
    x_norm_squared = torch.sum(x ** 2, dim=-1, keepdim=True)
    denominator = torch.sqrt(x_norm_squared / H + eps)
    grad_g = torch.sum(x * grad_y / denominator, dim=tuple(range(x.ndim - 1)))
    return grad_g


def compute_grad_x(
        grad_y: torch.Tensor, 
        x: torch.Tensor, 
        g: torch.Tensor,
        eps=1e-5,
    ):
    H = x.shape[-1]
    x_norm_squared = torch.sum(x ** 2, dim=-1, keepdim=True)
    denominator = torch.sqrt(x_norm_squared / H + eps)

    term1 = (grad_y * g) / denominator

    z = x * g
    dot_product = torch.sum(grad_y * z, dim=-1, keepdim=True)
    term2 = (dot_product * x) / (H * denominator ** 3)

    grad_x = term1 - term2

    return grad_x