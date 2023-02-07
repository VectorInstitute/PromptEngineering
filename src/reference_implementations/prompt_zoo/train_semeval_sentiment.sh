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
PROM_LEN=${LEN}

# We source to keep the internal env variables defined.
source ${PROJECT_DIR}/../setup_gpu_worker.sh

model_path=/scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semeval-v2/${PROM_LEN}_${EXPERIMENT_TYPE}_${LEARN_RATE}
mkdir -p ${model_path}

python -m src.reference_implementations.prompt_zoo.trainer \
    --batch_size 16 \
    --mode train \
    --task_name semeval \
    --train_file ${PROJECT_DIR}/../../../resources/datasets/2018-Valence-oc-En-train.txt \
    --dev_file ${PROJECT_DIR}/../../../resources/datasets/2018-Valence-oc-En-dev.txt \
    --test_file ${PROJECT_DIR}/../../../resources/datasets/2018-Valence-oc-En-dev.txt \
    --t5_exp_type ${EXPERIMENT_TYPE} \
    --model_path ${model_path} \
    --learning_rate ${LEARN_RATE} \
    --max_epochs 30 \
    --training_steps 10000000 \
    --steps_per_checkpoint 10 \
    --source_max_length 64 \
    --decoder_max_length 16 \
    --prediction_file ${model_path}/dev_sentiment.csv \
    --with_instructions False \
    --prompt_length ${PROM_LEN} \
    --weight_decay_rate 0.00001 \
    --t5_pretrained_model google/t5-large-lm-adapt
