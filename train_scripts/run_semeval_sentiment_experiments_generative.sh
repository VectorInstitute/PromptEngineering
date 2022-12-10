#!/bin/bash

rates=(0.0005 0.00005 0.01 0.0001 0.005)

exps=(all_finetune input_finetune output_finetune soft_prompt_finetune classifier_finetune soft_prompt_classifier_finetune)

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
            True
    done
done
