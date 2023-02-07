#!/bin/bash

PROJECT_DIR=$( dirname -- "$0"; )

# We source to keep the internal env variables defined.
source ${PROJECT_DIR}/../setup_gpu_worker.sh

model_path=/scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semeval-no-finetune/with_instruction
mkdir -p ${model_path}

python -m src.reference_implementations.prompt_zoo.trainer \
    --batch_size 16 \
    --task_name semeval \
    --test_file ${PROJECT_DIR}/../../../resources/datasets/2018-Valence-oc-En-dev.txt \
    --t5_exp_type no_finetune \
    --model_path ${model_path} \
    --source_max_length 64 \
    --decoder_max_length 16 \
    --prediction_file ${model_path}/test_sentiment.csv \
    --with_instructions True \
    --t5_pretrained_model google/t5-large-lm-adapt


model_path=/scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semeval-no-finetune/no_instruction
mkdir -p ${model_path}

python -m src.reference_implementations.prompt_zoo.trainer \
    --batch_size 16 \
    --task_name semeval \
    --test_file ${PROJECT_DIR}/../../../resources/datasets/2018-Valence-oc-En-dev.txt \
    --t5_exp_type no_finetune \
    --model_path ${model_path} \
    --source_max_length 64 \
    --decoder_max_length 16 \
    --prediction_file ${model_path}/test_sentiment.csv \
    --with_instructions False \
    --t5_pretrained_model google/t5-large-lm-adapt
