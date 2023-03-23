#!/bin/bash

PROJECT_DIR=$( dirname -- "$0"; )

# We source to keep the internal env variables defined.
source ${PROJECT_DIR}/../../setup_gpu_worker.sh

python -m src.reference_implementations.hugging_face_basics.hf_fine_tuning_examples.roberta_fine_tune_script
