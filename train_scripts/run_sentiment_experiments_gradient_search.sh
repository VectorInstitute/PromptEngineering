#!/bin/bash

method_path="/scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/gradient_search"
semeval_path="${method_path}/semval"
sst2_path="${method_path}/sst2"

mkdir -p ${method_path}
mkdir -p ${semeval_path}
mkdir -p ${sst2_path}

# send job for semeval
sbatch  src/reference_implementations/run_singlenode_prompt.slrm \
        src/reference_implementations/prompt_zoo/gradient_search_sentiment.sh \
        ./torch-prompt-tuning-exps-logs \
        gradient_search \
        semeval \
        ${semeval_path}

# send job for sst2
sbatch  src/reference_implementations/run_singlenode_prompt.slrm \
        src/reference_implementations/prompt_zoo/gradient_search_sentiment.sh \
        ./torch-prompt-tuning-exps-logs \
        gradient_search \
        sst2 \
        ${sst2_path}
