#!/bin/bash

bash ../setup_gpu_worker.sh

# Directory to find the input gin file.
PROJECT_DIR="."

# Directory where the t5x is cloned.
T5X_DIR="`python3 -m prompt_tuning.scripts.find_module t5x`/.."

TFDS_DATA_DIR="/scratch/ssd004/scratch/snajafi/data_temp/t5x-exps/data"

MODEL_DIR="/scratch/ssd004/scratch/snajafi/data_temp/t5x-exps/model"

python -m t5x/train.py \
    --gin_search_paths=${PROJECT_DIR} \
    --gin_file="base_wmt_train.gin" \
    --gin.MODEL_DIR=\"${MODEL_DIR}\" \
    --tfds_data_dir=${TFDS_DATA_DIR} \
    --multiprocess_gpu \
    --coordinator_address "${MASTER_ADDR}:${MASTER_PORT}" \
    --process_count=${SLURM_NTASKS} \
    --process_index=${SLURM_PROCID}
