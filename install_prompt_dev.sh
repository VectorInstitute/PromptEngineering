#!/bin/bash

module load jax0.2.24-cuda11.0-python3.8_jupyter
python -m venv env
source env/bin/activate
./env/bin/python -m pip install --upgrade pip
./env/bin/python -m pip install -e .[dev]
