#!/bin/bash
# This causes script to stop if any command returns non-0 error code
set -e

PROJECT_PATH=$1
ENV_PATH="${PROJECT_PATH}/google_prompt_tuning-env"

echo "-> Setting up environment at: ${ENV_PATH}"

# Create base virtualenv
virtualenv --python="/pkgs/python-3.9.10/bin/python3.9" "${ENV_PATH}"
source "${ENV_PATH}/bin/activate"
pip install --upgrade pip

# These do not persist, must set them each time you source env
export PATH="${VIRTUAL_ENV}/bin:/pkgs/cuda-11.3/bin:$PATH"
export LD_LIBRARY_PATH="/scratch/ssd001/pkgs/cudnn-11.4-v8.2.4.15/lib64:/scratch/ssd001/pkgs/cuda-11.3/targets/x86_64-linux/lib"

# Installs correct jax version for our cuda+cudnn+drivers
pip install https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.4+cuda11.cudnn82-cp39-cp39-manylinux2014_x86_64.whl

cd "${PROJECT_PATH}"
git clone https://github.com/google-research/prompt-tuning
cd prompt-tuning
pip install .
# Downgrade flax due to issue with propmt-tuning lib
pip install flax==0.5.1
pip install datasets

# Directly print commands to run or be placed in `my_env/bin/activate`
cat << EOF
********RUN THESE COMMANDS********
export PATH="${VIRTUAL_ENV}/bin:/pkgs/cuda-11.3/bin:$PATH"
export LD_LIBRARY_PATH="/scratch/ssd001/pkgs/cudnn-11.4-v8.2.4.15/lib64:/scratch/ssd001/pkgs/cuda-11.3/targets/x86_64-linux/lib"
source ${ENV_PATH}/bin/activate
python -c "import jax; print(f'JAX version: {jax.__version__}'); print(jax.devices()); print(jax.numpy.ones(3))"
**********************************
Place these in your env_name/bin/activate file to prevent headaches!
Output should look like:
JAX version: 0.4.5
[StreamExecutorGpuDevice(id=0, process_index=0, slice_index=0)]
[1. 1. 1.]
EOF
