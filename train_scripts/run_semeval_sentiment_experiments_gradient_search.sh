#!/bin/bash

sbatch  src/reference_implementations/run_singlenode_prompt.slrm \
        src/reference_implementations/prompt_zoo/gradient_search_semeval_sentiment.sh \
        ./torch-prompt-tuning-exps-logs \
        gradient_search \
