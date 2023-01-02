#!/bin/bash

rates=(0.3 0.5)

exps=(100)

for i in ${!rates[@]};
do
	rate=${rates[$i]}
    for j in ${!exps[@]};
    do
        exp=${exps[$j]}
        sbatch src/reference_implementations/run_singlenode_prompt.slrm \
            src/reference_implementations/prompt_zoo/train_semeval_sentiment.sh \
            ./torch-prompt-tuning-exps-logs \
            soft_prompt_finetune \
            ${rate} \
            True \
            ${exp}
    done
done
