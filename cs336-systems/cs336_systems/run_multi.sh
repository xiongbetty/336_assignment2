#!/bin/bash

nodes=2
backends_devices=(
    # "gloo cpu"
    "gloo cuda"
    "nccl cuda"
)
data_sizes=(
    $((512 * 1024))
    $((1 * 1024 * 1024))
    $((10 * 1024 * 1024))
    $((50 * 1024 * 1024))
    $((100 * 1024 * 1024))
    $((500 * 1024 * 1024))
    $((1024 * 1024 * 1024))
)
processes_per_node_multi=(3)

for ppn in "${processes_per_node_multi[@]}"; do
    for backend_device in "${backends_devices[@]}"; do
        backend=$(echo $backend_device | cut -d' ' -f1)
        device=$(echo $backend_device | cut -d' ' -f2)

        for data_size in "${data_sizes[@]}"; do
            if [ $device == "cuda" ]; then
                sbatch --ntasks-per-node=$ppn --gpus-per-node=$ppn distributed_multi.sbatch $backend $device $data_size
            else
                sbatch --ntasks-per-node=$ppn distributed_multi.sbatch $backend $device $data_size        
            fi
        done
    done
done