#!/bin/bash

echo "Hostname: $(hostname -s)"
echo "Node Rank ${SLURM_PROCID}"

# prepare environment
source ${HOME}/codes/PromptEngineering/t5x-env/bin/activate

# Define these env variables to run ML models on cuda and gpu workers properly.
# without these, tensorflow or jax will not detect any GPU cards.
# we point to the specific cuda and cudnn versions available on the cluster.

export PATH="${HOME}/codes/PromptEngineering/t5x-env/bin:/pkgs/cuda-11.3/bin:$PATH"

export LD_LIBRARY_PATH="/scratch/ssd001/pkgs/cudnn-11.4-v8.2.4.15/lib64:/scratch/ssd001/pkgs/cuda-11.3/targets/x86_64-linux/lib"

export XLA_FLAGS='--xla_gpu_simplify_all_fp_conversions --xla_gpu_all_reduce_combine_threshold_bytes=136314880 ${XLA_FLAGS}'

echo "Using Python from: $(which python)"

# Directory to find the input gin file.
PROJECT_DIR=${HOME}"/codes/PromptEngineering/src/reference_implementations/t5x"

# Directory where the t5x is cloned.
T5X_DIR=${HOME}"/codes/PromptEngineering/t5x-env/lib/python3.9/site-packages"

TFDS_DATA_DIR="/scratch/ssd004/scratch/snajafi/data_temp/t5x-exps/data"

MODEL_DIR="/scratch/ssd004/scratch/snajafi/data_temp/t5x-exps/model"

python ${T5X_DIR}/t5x/train.py \
    --gin_search_paths=${PROJECT_DIR} \
    --gin_file="base_wmt_train.gin" \
    --gin.MODEL_DIR=\"${MODEL_DIR}\" \
    --tfds_data_dir=${TFDS_DATA_DIR} \
    --multiprocess_gpu \
    --coordinator_address "${MASTER_ADDR}:${MASTER_PORT}" \
    --process_count=${SLURM_NTASKS} \
    --process_index=${SLURM_PROCID}
