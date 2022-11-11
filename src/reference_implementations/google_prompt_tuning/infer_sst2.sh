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

OUTPUT_DIRECTORY=$DATA_DIR

T5X_DIR="`python3 -m src.find_module t5x`/.."
FLAXFORMER_DIR="`python3 -m src.find_module flaxformer`/.."
PROMPT_DIR="`python3 -m src.find_module prompt_tuning`/.."
echo "Searching for gin configs in:"
echo "- ${T5X_DIR}"
echo "- ${FLAXFORMER_DIR}"
echo "- ${PROMPT_DIR}"
echo "============================="

# The trained checkpoint.
PRETRAINED_MODEL="/scratch/ssd004/scratch/snajafi/data_temp/google-prompt-tuning-exps/model/checkpoint_1144000"

python -m t5x.infer \
  --gin_search_paths="${PROJECT_DIR},${T5X_DIR},${FLAXFORMER_DIR},${PROMPT_DIR}" \
  --gin_file="prompt_tuning/configs/models/t5_1_1_base_prompt.gin" \
  --gin_file="infer.gin" \
  --gin.INFER_OUTPUT_DIR="'${OUTPUT_DIRECTORY}'" \
  --gin.MIXTURE_OR_TASK_NAME="'example_binary_sentiment_analysis'" \
  --gin.MIXTURE_OR_TASK_MODULE="'src.reference_implementations.google_prompt_tuning.sentiment_task'" \
  --gin.TASK_FEATURE_LENGTHS="{'inputs': 512, 'targets': 8}" \
  --gin.CHECKPOINT_PATH="'${PRETRAINED_MODEL}'" \
  --gin.BATCH_SIZE="64" \
  --gin.USE_CACHED_TASKS="False"
