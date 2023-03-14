#!/bin/bash

# This is the main directory where the model checkpoints will be saved.
# if you are working under a different username on the vector's cluster,
# please create a specific directory under your username: /scratch/ssd004/scratch/username/
# the following directory is created under the username snajafi.
method_path="/scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/gradient_search"


# different tasks will have their own directories.
semeval_path="${method_path}/semval"
sst2_path="${method_path}/sst2"

# create the directories if they don't exist.
mkdir -p ${method_path}
mkdir -p ${semeval_path}
mkdir -p ${sst2_path}

# send job for semeval
sbatch  src/reference_implementations/run_singlenode_prompt.slrm \
        src/reference_implementations/prompt_zoo/training_scripts/gradient_search_sentiment.sh \
        ./torch-prompt-tuning-exps-logs \
        gradient_search \
        semeval \
        ${semeval_path}

# send job for sst2
sbatch  src/reference_implementations/run_singlenode_prompt.slrm \
        src/reference_implementations/prompt_zoo/training_scripts/gradient_search_sentiment.sh \
        ./torch-prompt-tuning-exps-logs \
        gradient_search \
        sst2 \
        ${sst2_path}
