#!/bin/bash

module load jax0.2.24-cuda11.0-python3.8_jupyter
python -m venv env
source env/bin/activate
pip install -r requirements.txt
pip install -e .[dev]
