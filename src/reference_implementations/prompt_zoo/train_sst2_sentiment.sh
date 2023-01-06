#!/bin/bash

# For reading key=value arguments
for ARGUMENT in "$@"
do
	KEY=$(echo $ARGUMENT | cut -f1 -d=)
	KEY_LENGTH=${#KEY}
	VALUE="${ARGUMENT:$KEY_LENGTH+1}"
	export "$KEY"="$VALUE"
done

PROJECT_DIR=$( dirname -- "$0"; )
LEARN_RATE=${LR}
EXPERIMENT_TYPE=${EXP_TYPE}
WITH_INSTRUCTIONS=${WITH_INST}

# We source to keep the internal env variables defined.
source ${PROJECT_DIR}/../setup_gpu_worker.sh

model_path=/scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/sst2-v2/${EXPERIMENT_TYPE}_${LEARN_RATE}_${WITH_INSTRUCTIONS}
mkdir -p ${model_path}

python -m src.reference_implementations.prompt_zoo.trainer \
    --batch_size 16 \
    --mode train \
    --task_name sst2 \
    --train_file train \
    --dev_file validation \
    --t5_exp_type ${EXPERIMENT_TYPE} \
    --gpu True \
    --model_path ${model_path} \
    --learning_rate ${LEARN_RATE} \
    --max_epochs 3 \
    --training_steps 10000000 \
    --steps_per_checkpoint 400 \
    --source_max_length 180 \
    --decoder_max_length 16 \
    --prediction_file ${model_path}/dev_sentiment.csv \
    --with_instructions ${WITH_INSTRUCTIONS} \
    --prompt_length 50 \
    --weight_decay_rate 0.00001
