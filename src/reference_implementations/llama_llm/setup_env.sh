#!/bin/bash

# exit if anything fails
set -e
echo "Setting up LLaMA env..."

# setup dir structure
ROOT_DIR=$(pwd)
echo "Root dir: ${ROOT_DIR}"

LOG_DIR="${ROOT_DIR}/logs"
mkdir -pv "${LOG_DIR}"

# setup env and packages
virtualenv --python="/pkgs/python-3.9.10/bin/python3.9" "${ROOT_DIR}/llama_env"

source "${ROOT_DIR}/llama_env/bin/activate"
git clone https://github.com/facebookresearch/llama.git
cd llama
pip install -r requirements.txt
pip install -e .

echo "Setup finished!"
