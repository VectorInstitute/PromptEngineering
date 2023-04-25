#!/bin/bash

for i in {0..9} 
do
    python3 src/reference_implementations/fairness_measurement/fairness_eval_template.py $i 
done
