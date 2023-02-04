#!/bin/bash

# experiments with/without QA instructions for the semeval classification without fine-tuning anything.
sbatch  src/reference_implementations/run_singlenode_prompt.slrm \
        src/reference_implementations/prompt_zoo/no_finetune_semeval.sh \
        ./torch-prompt-tuning-exps-logs
