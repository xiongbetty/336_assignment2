#!/usr/bin/env python3

import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer

from collections.abc import Callable, Iterable 
from typing import Optional, BinaryIO, IO

import math
import random
import numpy as np

import os


## CLASSES

class SGD(Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None): 
        loss = None if closure is None else closure() 
        for group in self.param_groups:
            lr = group["lr"] # Get the learning rate. 
            for p in group["params"]:
                if p.grad is None: 
                    continue
                state = self.state[p] # Get state associated with p.
                t = state.get("t", 0) # Get iteration number from the state, or initial value. 
                grad = p.grad.data # Get the gradient of loss with respect to p.
                p.data -= lr / math.sqrt(t + 1) * grad # Update weight tensor in-place. 
                state["t"] = t + 1 # Increment iteration number.
        return loss


class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(AdamW, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # iterate over groups of parameters
        for group in self.param_groups:
            for p in group["params"]:
                # check if gradients are computed, if none, skip
                if p.grad.data == None:
                    continue
                else:
                    grad = p.grad.data

                state = self.state[p]

                if len(state) == 0:
                    # initialize step and first and second moment vectors to zeros
                    state["step"] = 0
                    state["first_moment_vector"] = torch.zeros_like(p.data)
                    state["second_moment_vector"] = torch.zeros_like(p.data)

                first_moment_vector = state["first_moment_vector"]
                second_moment_vector = state["second_moment_vector"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # update moment estimtes
                first_moment_vector = beta1 * first_moment_vector + grad * (1.0 - beta1)
                second_moment_vector = beta2 * second_moment_vector + (grad ** 2) * (1.0 - beta2)

                # compute adjusted lr for time step
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]

                step_size = group["lr"] * (math.sqrt(bias_correction2)) / bias_correction1

                # update parameters
                denom = second_moment_vector.sqrt() + group["eps"]
                p.data = p.data - step_size * first_moment_vector / denom

                # compute the weight decay term
                if group["weight_decay"] != 0:
                    p.data -= group["lr"] * group["weight_decay"] * p.data

                state["first_moment_vector"] = first_moment_vector
                state["second_moment_vector"] = second_moment_vector

        return loss


## FUNCTIONS

def cross_entropy(logits: torch.FloatTensor, targets: torch.LongTensor):
    # subtract the largest element for numerical stability
    max_logits = torch.max(logits, dim=-1, keepdim=True).values
    logits -= max_logits

    # compute log softmax
    log_softmax_logits = logits - torch.log(torch.sum(torch.exp(logits), dim=-1, keepdim=True))

    # gather the log probabilities corresponding to the target indices
    log_probs = torch.gather(log_softmax_logits, dim=-1, index=targets.unsqueeze(-1))

    # compute the negative log likelihood
    neg_log_likelihood = -log_probs.squeeze(-1)

    # average across the batch dimensions
    avg_loss = torch.mean(neg_log_likelihood)

    return avg_loss


def learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int
):
    # warm-up
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate

    # cosine annealing
    elif it <= cosine_cycle_iters:
        cosine_portion = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi *cosine_portion))

    # post-annealing
    else:
        return min_learning_rate


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps=1e-8):
    for param in parameters:
        if param.grad is not None:
            param_l2 = param.grad.data.norm(2).item() ** 2
            if param_l2 > max_l2_norm:
                factor = max_l2_norm / (param_l2 + eps)
                param.grad.data *= factor


def data_loading(dataset: np.ndarray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # sample random start indices
    start_indices = random.sample(range(len(dataset) - context_length), batch_size)
    sampled_input_sequences = [dataset[i : i + context_length] for i in start_indices]
    next_token_targets = [dataset[i + 1 : i + context_length + 1] for i in start_indices]
    # sampled_input_sequences_tensor = torch.tensor(np.array(sampled_input_sequences)).to(device)
    # next_token_targets_tensor = torch.tensor(np.array(next_token_targets)).to(device)
    sampled_input_sequences_tensor = torch.from_numpy(np.array(sampled_input_sequences).astype(np.int64)).to(device)
    next_token_targets_tensor = torch.from_numpy(np.array(next_token_targets).astype(np.int64)).to(device)

    return sampled_input_sequences_tensor.long(), next_token_targets_tensor.long()


def save_checkpoint(
    model: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    iteration: int, 
    out: str | os.PathLike | BinaryIO | IO[bytes]
):
    checkpoint = {
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict()
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes], 
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
):
    checkpoint = torch.load(src)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint["iteration"]


def main():
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    learning_rates = [1e1, 1e2, 1e3]
    for learning_rate in learning_rates:
        print("loss for learning rate: " + str(learning_rate))
        opt = SGD([weights], lr=learning_rate)
        
        for t in range(10):
            opt.zero_grad() # Reset the gradients for all learnable parameters. 
            loss = (weights**2).mean() # Compute a scalar loss value. 
            print(loss.cpu().item())
            loss.backward() # Run backward pass, which computes gradients. 
            opt.step() # Run optimizer step.

if __name__ == "__main__":
    main()