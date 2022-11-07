#!/bin/bash

# For reading key=value arguments
for ARGUMENT in "$@"
do
	KEY=$(echo $ARGUMENT | cut -f1 -d=)
	KEY_LENGTH=${#KEY}
	VALUE="${ARGUMENT:$KEY_LENGTH+1}"
	export "$KEY"="$VALUE"
done

# Current directory to find the local gin config file.
PROJECT_DIR=$( dirname -- "$0"; )

source ${PROJECT_DIR}/../setup_gpu_worker.sh

# Directory where the t5x is cloned.
T5X_DIR="`python3 -m prompt_tuning.scripts.find_module t5x`/.."

TFDS_DATA_DIR=$DATA_DIR
MODEL_DIR=$MODEL_DIR

python -m t5x.train \
    --gin_search_paths=${PROJECT_DIR} \
    --gin_file="base_wmt_train.gin" \
    --gin.MODEL_DIR=\"${MODEL_DIR}\" \
    --tfds_data_dir=${TFDS_DATA_DIR} \
    --multiprocess_gpu \
    --coordinator_address "${MASTER_ADDR}:${MASTER_PORT}" \
    --process_count=${SLURM_NTASKS} \
    --process_index=${SLURM_PROCID}
