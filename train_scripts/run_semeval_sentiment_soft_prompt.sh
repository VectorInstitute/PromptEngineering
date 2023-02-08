#!/bin/bash

rates=(0.1 0.25 0.5)

for i in ${!rates[@]};
do
	rate=${rates[$i]}
    sbatch  src/reference_implementations/run_singlenode_prompt.slrm \
            src/reference_implementations/prompt_zoo/train_semeval_sentiment.sh \
            ./torch-prompt-tuning-exps-logs \
            soft_prompt_finetune \
            ${rate} \
            100
done
