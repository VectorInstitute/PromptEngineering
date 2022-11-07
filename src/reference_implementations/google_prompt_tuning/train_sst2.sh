#!/usr/bin/env bash

# Adapted the original training script to run on vector's cluster with GPUs.
# Original scripts: https://github.com/google-research/prompt-tuning/tree/main/prompt_tuning/scripts

bash ../setup_gpu_worker.sh

TFDS_DATA_DIR="/scratch/ssd004/scratch/snajafi/data_temp/google-prompt-sst2-exps/data"
MODEL_DIR="/scratch/ssd004/scratch/snajafi/data_temp/google-prompt-sst2-exps/model"

T5X_DIR="`python3 -m prompt_tuning.scripts.find_module t5x`/.."
FLAXFORMER_DIR="`python3 -m prompt_tuning.scripts.find_module flaxformer`/.."
PROMPT_DIR="`python3 -m prompt_tuning.scripts.find_module prompt_tuning`/.."
echo "Searching for gin configs in:"
echo "- ${T5X_DIR}"
echo "- ${FLAXFORMER_DIR}"
echo "- ${PROMPT_DIR}"
echo "============================="
PRETRAINED_MODEL="gs://t5-data/pretrained_models/t5x/t5_1_1_lm100k_base/checkpoint_1100000"

python -m t5x.train \
  --gin_search_paths="${T5X_DIR},${FLAXFORMER_DIR},${PROMPT_DIR}" \
  --gin_file="prompt_tuning/configs/models/t5_1_1_base_prompt.gin" \
  --gin_file="prompt_tuning/configs/prompts/from_class_labels.gin" \
  --gin_file="prompt_tuning/configs/runs/prompt_finetune.gin" \
  --gin.CLASS_LABELS="['positive', 'negative']" \
  --gin.MODEL_DIR="'${MODEL_DIR}'" \
  --gin.MIXTURE_OR_TASK_NAME="'taskless_glue_sst2_v200_examples'" \
  --gin.MIXTURE_OR_TASK_MODULE="'prompt_tuning.data.glue'" \
  --gin.TASK_FEATURE_LENGTHS="{'inputs': 512, 'targets': 8}" \
  --gin.INITIAL_CHECKPOINT_PATH="'${PRETRAINED_MODEL}'" \
  --gin.TRAIN_STEPS="10_000" \
  --gin.USE_CACHED_TASKS="False" \
  --tfds_data_dir=${TFDS_DATA_DIR} \
  --multiprocess_gpu \
  --coordinator_address "${MASTER_ADDR}:${MASTER_PORT}" \
  --process_count=${SLURM_NTASKS} \
  --process_index=${SLURM_PROCID}
