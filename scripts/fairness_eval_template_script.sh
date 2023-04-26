#!/bin/bash

for model in "opt-125m" "opt-350m" "roberta-large"
do
    for idx in {0..9} 
    do
        for template in "regard" "NS-prompts" "amazon"
        do
            echo $model $idx $template
            python3 src/reference_implementations/fairness_measurement/fairness_eval_template.py $model $idx $template
        done
    done
done