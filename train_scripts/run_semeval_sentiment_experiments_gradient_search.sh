#!/bin/bash

rates=(0.5)
exps=(gradient_search)

for i in ${!rates[@]};
do
    rate=${rates[$i]}
    for j in ${!exps[@]};
    do
        exp=${exps[$j]}
        sbatch src/reference_implementations/run_singlenode_prompt.slrm \
            src/reference_implementations/prompt_zoo/gradient_search_semeval_sentiment.sh \
            ./torch-prompt-tuning-exps-logs \
            ${exp} \
            ${rate} \
            True \
            25

    done
done
