#!/bin/bash

# install main module
module load python/3.9.10
python -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -e .[dev]

# install google soft-prompt
bash src/reference_implementations/google_soft_prompt/install_module.sh
