#!/bin/bash

echo "Hostname: $(hostname -s)"
echo "Node Rank ${SLURM_PROCID}"

# prepare environment
source ${VIRTUAL_ENV}/bin/activate

# Define these env variables to run ML models on cuda and gpu workers properly.
# without these, tensorflow or jax will not detect any GPU cards.
# we point to the specific cuda and cudnn versions available on the cluster.

export PATH="${VIRTUAL_ENV}/bin:/pkgs/cuda-11.3/bin:$PATH"

export LD_LIBRARY_PATH="/scratch/ssd001/pkgs/cudnn-11.4-v8.2.4.15/lib64:/scratch/ssd001/pkgs/cuda-11.3/targets/x86_64-linux/lib"

export XLA_FLAGS='--xla_gpu_simplify_all_fp_conversions --xla_gpu_all_reduce_combine_threshold_bytes=136314880 ${XLA_FLAGS}'

echo "Using Python from: $(which python)"
