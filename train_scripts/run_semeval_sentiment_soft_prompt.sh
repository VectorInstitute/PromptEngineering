#!/bin/bash

rates=(0.25 0.1 0.5)
exps=(soft_prompt_finetune)

for i in ${!rates[@]};
do
	rate=${rates[$i]}
    for j in ${!exps[@]};
    do
        exp=${exps[$j]}
        sbatch src/reference_implementations/run_singlenode_prompt.slrm \
            src/reference_implementations/prompt_zoo/train_semeval_sentiment.sh \
            ./torch-prompt-tuning-exps-logs \
            ${exp} \
            ${rate} \
            True \
            100
    done
done
