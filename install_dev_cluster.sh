#!/bin/bash

# Install main module
module load python/3.9.10

python -m venv env

source env/bin/activate

pip install --upgrade pip

# install tensorflow for python 3.9.10
pip install https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-2.10.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

# install torch for python 3.9.10 and cuda 11.3
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# Installs the jax wheel compatible with Cuda >= 11.1 and cudnn >= 8.0.5
pip install "jax[cuda11_cudnn805]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

pip install -e .[dev]

# install t5x
bash src/reference_implementations/t5x/install_module.sh
