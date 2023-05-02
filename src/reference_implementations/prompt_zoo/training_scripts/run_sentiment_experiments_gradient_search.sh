#!/bin/bash

# This is the main directory where the model checkpoints will be saved.
# if you are working under a different directory on the vector's cluster,
# you may want to change the following method_path. The following
# directory is created under your username.
method_path="/scratch/ssd004/scratch/${USER}/gradient_search_long_training"


# different tasks will have their own directories.
semeval_path="${method_path}/semeval"
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
