#!/bin/bash

# For reading key=value arguments
for ARGUMENT in "$@"
do
	KEY=$(echo $ARGUMENT | cut -f1 -d=)
	KEY_LENGTH=${#KEY}
	VALUE="${ARGUMENT:$KEY_LENGTH+1}"
	export "$KEY"="$VALUE"
done

# Current directory to find the local gin config file.
PROJECT_DIR=$( dirname -- "$0"; )

echo ${PROJECT_DIR}

# We source to keep the internal env variables defined.
source ${PROJECT_DIR}/../setup_gpu_worker.sh

# Directory where the t5x is cloned.
T5X_DIR="`python3 -m src.find_module t5x`"

echo ${T5X_DIR}

TFDS_DATA_DIR=$DATA_DIR
MODEL_DIR=$MODEL_DIR
