#!/bin/bash

for model in "llama-7b"
do
    for dataset in "jacobthebanana/sst5-mapped-extreme"
    do
        echo $model $dataset
        python3 src/reference_implementations/fairness_measurement/fairness_eval_dataset_llama.py $model 0 $dataset
    done
done