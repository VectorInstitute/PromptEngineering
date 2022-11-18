#!/bin/bash

PROJECT_DIR=$( dirname -- "$0"; )

# We source to keep the internal env variables defined.
source ${PROJECT_DIR}/../setup_gpu_worker.sh


python -m src.reference_implementations.prompt_zoo.trainer \
    --batch_size 8 \
    --mode train \
    --task_name semeval_3_class_sentiment \
    --train_file ${PROJECT_DIR}/../../../resources/datasets/2018-Valence-oc-En-train.txt \
    --t5_exp_type all_finetune \
    --gpu True \
    --model_path /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/all_finetune \
    --learning_rate 0.0005 \
    --max_epochs 10 \
    --training_steps 10000000 \
    --steps_per_checkpoint 20 \
    --source_max_length 128 \
    --decoder_max_length 16
