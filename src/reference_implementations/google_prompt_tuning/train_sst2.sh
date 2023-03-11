#!/bin/bash

# Adapted the original training script to run on vector's cluster with GPUs.
# Original scripts: https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/scripts

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

# We source to keep the internal env variables defined.
source ${PROJECT_DIR}/../setup_gpu_worker.sh

TFDS_DATA_DIR=$DATA_DIR
MODEL_DIRECTORY=$MODEL_DIR

T5X_DIR="`python3 -m src.find_module t5x`"
FLAXFORMER_DIR="`python3 -m src.find_module flaxformer`"
PROMPT_DIR="`python3 -m src.find_module prompt_tuning`"
echo "Searching for gin configs in:"
echo "- ${T5X_DIR}"
echo "- ${FLAXFORMER_DIR}"
echo "- ${PROMPT_DIR}"
echo "============================="

# Remember that soft-prompt paper fine-tunes T5 on prefix LM  which is extra to T5 of huggingface.
# T5x does extra LM adaptation step over the span reconstruction pre-training of normal T5s.
PRETRAINED_MODEL="/scratch/ssd004/scratch/demerson/t5_1_1_lm100k_base_checkpoint_1100000/"
PROMPT_INIT_FILE="/scratch/ssd004/scratch/demerson/prompt_initialization/prompt.npy"

# `TRAIN_STEPS` should include pre-training steps, e.g., if pre-trained ckpt
# has 1M steps, TRAIN_STEPS = 1.1M will perform 0.1M prompt-tuning steps.

python -m t5x.train \
  --gin_search_paths="${PROJECT_DIR},${T5X_DIR},${FLAXFORMER_DIR},${PROMPT_DIR}" \
  --gin_file="prompt_tuning/configs/models/t5_1_1_base_prompt.gin" \
  --gin_file="prompt_tuning/configs/prompts/from_class_labels_numpy.gin" \
  --gin_file="${PROJECT_DIR}/prompt_finetune.gin" \
  --gin.EMBEDDING_FILE="'${PROMPT_INIT_FILE}'" \
  --gin.CLASS_LABELS="['positive', 'negative']" \
  --gin.MODEL_DIR="'${MODEL_DIRECTORY}'" \
  --gin.MIXTURE_OR_TASK_NAME="'taskless_glue_sst2_v200_examples'" \
  --gin.MIXTURE_OR_TASK_MODULE="'prompt_tuning.data.glue'" \
  --gin.TASK_FEATURE_LENGTHS="{'inputs': 512, 'targets': 8}" \
  --gin.INITIAL_CHECKPOINT_PATH="'${PRETRAINED_MODEL}'" \
  --gin.TRAIN_STEPS="1_150_000" \
  --gin.BATCH_SIZE="64" \
  --gin.PjitPartitioner.num_partitions=1 \
  --gin.USE_CACHED_TASKS="False" \
  --tfds_data_dir=${TFDS_DATA_DIR} \
  --multiprocess_gpu \
  --coordinator_address "${MASTER_ADDR}:${MASTER_PORT}" \
  --process_count=${SLURM_NTASKS} \
  --process_index=${SLURM_PROCID}
