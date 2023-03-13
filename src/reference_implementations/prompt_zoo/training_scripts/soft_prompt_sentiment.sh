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

# We source to keep the internal env variables defined.
source ${PROJECT_DIR}/../../setup_gpu_worker.sh

if [ "${TASK}" = "semeval" ]; then
    python -m src.reference_implementations.prompt_zoo.trainer \
        --train_batch_size 16 \
        --eval_batch_size 128 \
        --mode train \
        --task_name ${TASK} \
        --train_file ${PROJECT_DIR}/../../../../resources/datasets/2018-Valence-oc-En-train.txt \
        --dev_file ${PROJECT_DIR}/../../../../resources/datasets/2018-Valence-oc-En-dev.txt \
        --t5_exp_type ${EXP_TYPE} \
        --model_path ${MODEL_PATH} \
        --max_epochs 30 \
        --training_steps 1000000 \
        --learning_rate ${LR} \
        --steps_per_checkpoint 10 \
        --source_max_length 64 \
        --decoder_max_length 16 \
        --instruction_type no_instruction \
        --weight_decay_rate 0.00001 \
        --prompt_length ${LEN} \
        --t5_pretrained_model google/t5-large-lm-adapt

elif [ "${TASK}" = "sst2" ]; then
    python -m src.reference_implementations.prompt_zoo.trainer \
        --train_batch_size 32 \
        --eval_batch_size 128 \
        --mode train \
        --task_name ${TASK} \
        --t5_exp_type ${EXP_TYPE} \
        --source_max_length 64 \
        --decoder_max_length 16 \
        --train_file train \
        --dev_file validation \
        --model_path ${MODEL_PATH} \
        --instruction_type no_instruction \
        --t5_pretrained_model google/t5-large-lm-adapt \
        --max_epochs 2 \
        --training_steps 1000000 \
        --learning_rate ${LR} \
        --steps_per_checkpoint 10 \
        --prompt_length ${LEN} \
        --weight_decay_rate 0.00001 \

fi
