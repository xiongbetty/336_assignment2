#!/usr/bin/env python3

from torch.optim.optimizer import Optimizer
import torch.distributed as dist
from typing import Any, Type


class ShardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        defaults = kwargs.pop("defaults", {})
        if params is None:
            params_list = []
        else:
            params_list = list(params)
        if not params_list:
            raise ValueError("Optimizer got an empty parameter list.")

        self.optimizer = optimizer_cls(params_list, **kwargs)
        super(ShardedOptimizer, self).__init__(params_list, defaults)

    def step(self, closure=None, **kwargs):
        for param_group in self.param_groups:
            # Synchronize parameter gradients across all ranks
            for param in param_group["params"]:
                if param.grad is not None:
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= self.world_size

            # Perform optimizer step on the responsible rank
            if self.rank == 0:
                self.optimizer.step(closure, **kwargs)

            # Broadcast updated parameters from the responsible rank to all other ranks
            for param in param_group["params"]:
                dist.broadcast(param.data, src=0)

    
    def add_param_group(self, param_group):
        super(ShardedOptimizer, self).add_param_group(param_group)