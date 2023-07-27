#!/bin/bash

for model in "llama-7b"
do
    for template in "regard" "NS-prompts" "amazon"
    do
        echo $model $template
        python3 src/reference_implementations/fairness_measurement/fairness_eval_template_llama.py $model 0 $template
    done
done