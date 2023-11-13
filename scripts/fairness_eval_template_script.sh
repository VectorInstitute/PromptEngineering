#!/bin/bash

# for model in "opt-125m" "opt-350m" "roberta-large" "roberta-base"
for model in "opt-350m"
do
    # for idx in {0..14} 
    # for idx in 0 4 8 11 13
    # for idx in 0 3 9 10 12
    # for idx in 0 3 8 9 10
    for idx in 4 7 9 10 12
    do
        for template in "regard" "NS-prompts" "amazon"
        do
            echo $model $idx $template
            python3 src/reference_implementations/fairness_measurement/fairness_eval_template.py $model $idx $template
        done
    done
done