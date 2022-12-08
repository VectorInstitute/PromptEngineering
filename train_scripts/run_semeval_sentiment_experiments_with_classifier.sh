#!/bin/bash

rates=(0.005)

exps=(soft_prompt_classifier_finetune classifier_finetune)

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
            False
    done
done
