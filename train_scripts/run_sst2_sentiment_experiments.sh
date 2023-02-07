#!/bin/bash

rates=(0.3 0.1 0.001 0.01 0.0005 0.005)
exps=(soft_prompt_finetune all_finetune classifier_finetune input_finetune output_finetune)

for i in ${!rates[@]};
do
	rate=${rates[$i]}
    for j in ${!exps[@]};
    do
        exp=${exps[$j]}
        sbatch src/reference_implementations/run_singlenode_prompt.slrm \
            src/reference_implementations/prompt_zoo/train_sst2_sentiment.sh \
            ./torch-prompt-tuning-exps-logs \
            ${exp} \
            ${rate} \
            100
    done
done
