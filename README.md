# Prompt Engineering Laboratory

This repository holds all of the code associated with the project considering prompt engineering for large language models. This includes work around reference implementations, demo notebooks.

The static code checker runs on python3.9

All reference implementations are housed in `src/reference_implementations/`. Datasets to be used are placed in `resources/datasets`. There is also some information about JAX for those curious in learning more about that framework and the implementation of prompt tuning by Google (This is included but not supported in the prompt engineering lab).

This repository is organized as follows

## Prompt Tuning Reference Implementaitons

Automatic Prompt Tuning Methods are implemented under `src/reference_implementations/prompt_zoo`. Currently supported methods include:
* Prompt Tuning
* Gradient-based Discrete search
* GrIPS
* P-tuning.
There are also several alternatives to prompt optimization implemented, including full model tuning and partial fine-tuning.

For more information about using and running the prompt tuning experiments using the T5 language model, please see the corresponding .md file.
[README.md](/src/reference_implementations/prompt_zoo/README.md) describes the steps to setup the environment and access gpus in the vector's cluster for the experiments around different prompt techniques.

## Prompting OPT-175B, Galactica, or Other Large Language Models On the Cluster

These reference implementations are housed in `src/reference_implementations/prompting_vector_llms/`

This folder houses notebooks and implementations of prompting large language models hosted on Vector's compute cluster. There are notebooks for demonstrating various prompted downstream tasks, the affects of prompts on tasks like Aspect-Based Sentiment Analysis and text classification, along with prompt ensembling.

## Fairness in language models

These reference implementations reside in `src/reference_implementations/measuring_fairness/`

This folder contains implementations for measuring fairness for models that have been fine-tuning or prompted to complete a sentiment classification task. There is also an implementation of measuring fairness for LLMs like OPT-175B or Galactica accessed on Vector's cluster.

## Launching an interactive session on a GPU node

From any of the v-login nodes, run the following. This will reserve an A40 GPU and provide you a terminal to run commands on that node.

```bash
srun --gres=gpu:1 -c 8 --mem 16G -p a40 --pty bash
```

## Installing dependencies

*Note*: The following instructions are for anyone who would like to create there own python virtual environment to run experiments. If you would just like to run the code you can use one of our pre-built virtual environents by simply running, for example,

```bash
source /ssd003/projects/aieng/public/prompt_engineering/bin/activate
```
The above environment is for the `prompt_zoo` examples. Other required environments are discussed in the relevant folders.

If you are using the pre-built environments *do not* modify it, as it will affect all users of the venv. To install your own environment that you can manipulate, follow the instructions below.

### Virtualenv installing on macOS

If you wish to install the package in macOS for local development, you should call the following script to install `python3.9` on macOS and then setup the virtual env for the module you want to install. This approach only installs the ML libraries (`pytorch`, `tensorflow`, `jax`) for the CPU. If you also want to install the package in the editable mode with all the development requirements, you should use the flag `DEV=true` when you run the script, otherwise use the flag `DEV=false`.
```bash
bash setup.sh OS=mac ENV_NAME=env_name DEV=true
```

### Virtualenv installing on Vector's Cluster
You can call `setup.sh` with the `OS=vcluster` flag. This installs python in the linux cluster of Vector and installs the ML libraries for the GPU cards.
```bash
bash setup.sh OS=vcluster ENV_NAME=env_name DEV=true
```

The `setup.sh` script takes an *ENV_NAME* argument value of `prompt_torch`. The value `prompt_torch` should be used for our `prompt_zoo`

## Using Pre-commit Hooks (for developing in this repository)
To check your code at commit time
```
pre-commit install
```

You can also get pre-commit to fix your code
```
pre-commit run
```

## A note on disk space

Many of the experiments in this respository will end up writing to your scratch directory. An example path is:
```
/scratch/ssd004/scratch/snajafi/
```
where `snajafi` is replaced with your cluster username. This directory has a maximum capacity of 50GB. If you run multiple hyperparameter sweeps, you may fill this directory with model checkpoints. If this directory fills, it may interrupt your jobs or cause them to fail. Please be cognizant of the space and clean up old runs if you begin to fill the directory.
