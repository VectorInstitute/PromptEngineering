# Prompt Engineering Laboratory

This repository holds all of the code associated with the project considering prompt engineering for large language models. This includes work around reference implementations and demo notebooks.

The static code checker and all implementations run on python3.9

All reference implementations are housed in `src/reference_implementations/`. Datasets to be used are placed in `resources/datasets` or in the relevant resources subfolder for the implementation. There is also some information about JAX for those curious in learning more about that framework and the implementation of prompt tuning by Google (This is included but not supported in the prompt engineering lab due to issues with their current implementation).

## A Few Tips for Getting Started

1. As part of your cluster account, you have been allocated a scratch folder where checkpoints, training artifacts, and other files will be stored. It should be located at the path:

     `/scratch/ssd004/scratch/<cluster_username>`

    If you don't see this path, please let your facilitator know, and we will ensure that it exists.

2. If you are running any experiments in `prompt_zoo/`, it is best to use an A40 GPU. This can be achieved by following the instructions in `src/reference_implementations/prompt_zoo/README.md`.

    __Note__: Using JupyterHub to directly access a GPU is limited to T4V2 GPUs, which are generally insufficient for running `prompt_zoo` experiments.

3. We have __two__ pre-constructed environments for running experiments. They are __not__ interchangeable.

    * `/ssd003/projects/aieng/public/prompt_zoo` is used to run the experiments in the `prompt_zoo` directory __only__.

    * `/ssd003/projects/aieng/public/prompt_engineering` is used to run all of the other code in this repository.

4. We have provided some exploration guidance in the markdown `Exploration_Guide.md`. This guide provides some suggestions for exploration for each hands-on session based on the concepts covered in preceding lectures.

    __Note__: This guide is simply a suggestion. You should feel free to explore whatever is most interesting to you.

Below is a brief description of the contents of each folder in the reference implementations directory. In addition, each directory has at least a few readmes with more in-depth discussions. Finally, many of the notebooks are heavily documented

This repository is organized as follows

## Prompt Tuning Reference Implementaitons

Automatic Prompt Tuning Methods are implemented under `src/reference_implementations/prompt_zoo/`. Currently supported methods are:
* [Prompt Tuning](https://aclanthology.org/2021.emnlp-main.243.pdf)
* [Gradient-based Discrete search (AutoPrompt)](https://arxiv.org/pdf/2010.15980.pdf)
* [GrIPS](https://arxiv.org/abs/2203.07281)

There are also several alternatives to prompt optimization implemented, including full model tuning and partial fine-tuning (classifier layer, input layer).

For more information about using and running the prompt tuning experiments using the T5 language model, please see [README.md](/src/reference_implementations/prompt_zoo/README.md). The README describes the steps to source the environment and access gpus on Vector's cluster for the experiments around different prompt techniques.

## Prompting OPT-175B, LLaMA, or Other Large Language Models On the Cluster

These reference implementations are housed in `src/reference_implementations/prompting_vector_llms/`

This folder contains notebooks and implementations of prompting large language models hosted on Vector's compute cluster. There are notebooks for demonstrating various prompted downstream tasks, the affects of prompts on tasks like aspect-based sentiment analysis, text classification, summarization, and translation, along with prompt ensembling, activation fine-tuning, and experimenting with whether discrete prompts are transferable across architectures.

## Fairness in language models

These reference implementations reside in `src/reference_implementations/fairness_measurement/`

This folder contains implementations for measuring fairness for languagle models. There is an implementation that assesses fairness through fine-tuning or prompting to complete a sentiment classification task. We also consider LLM performance on the [CrowS-Pairs](https://aclanthology.org/2020.emnlp-main.154/) task and the [BBQ](https://aclanthology.org/2022.findings-acl.165/) task as a means of probing model bias and fairness.

## Hugging Face Basics

These reference implmenetations are in `src/reference_implementations/hugging_face_basics/`.

The reference implementations here are of two kinds. The first is a collection of examples of using HuggingFace for basic ML tasks. The second is a discussion of some important metrics associated with NLP, specifically generative NLP.

## LLaMA Language Model

In the folder `src/reference_implementations/llama_llm/`, we have scripts that facilitate using one of the new large language models known as [LLaMA](https://ai.facebook.com/blog/large-language-model-llama-meta-ai/). The language model has been trained for much longer than the traditional LLMs and, while much smaller than OPT-175, can demonstrate equivalent or better performance.

## T5x and Google Prompt Tuning (Currently Broken)

These implementations exist in `src/reference_implementations/t5x` and `src/reference_implementations/google_prompt_tuning`, respectively. These folders contain scripts for fine-tuning a JAX implementation of T5 and using prompt tuning in JAX for T5 as well. These folders offer a good idea as to how you might use JAX to perform large model training and prompt tuning. __However, they are not fully supported by this laboratory because their implementation is currently broken on the Google side of the repositories.__

## Accessing Compute Resources

### Launching an interactive session on a GPU node and connecting VSCode Server/Tunnel.

From any of the v-login nodes, run the following. This will reserve an A40 GPU and provide you a terminal to run commands on that node.

```bash
srun --gres=gpu:1 -c 8 --mem 16G -p a40 --pty bash
```

Note that `-p a40` requests an a40 gpu. You can also access smaller `t4v2` and `rtx6000` gpus this way. The `-c 8` requests 8 supporting CPUs and `--mem 16G` request 16 GB of cpu memory.

### Virtual Environments

As mentioned above, we offer two pre-built environments for running different parts of the code.

* `/ssd003/projects/aieng/public/prompt_zoo` is used to run the experiments in the `prompt_zoo` directory __only__.

* `/ssd003/projects/aieng/public/prompt_engineering` is used to run all of the other code in this repository.

Before starting a notebook or running code, you should source one of these two environments with the command:
```bash
source /ssd003/projects/aieng/public/prompt_zoo/bin/activate
```
or
```bash
source /ssd003/projects/aieng/public/prompt_engineering/bin/activate
```

When using the pre-built environments, you are not allowed to pip install to them. If you would like to setup your own environments, see section `Installing Custom Dependencies`.

### Starting a Notebook from a GPU Node.

Once an interactive session has been started, we can run a jupyter notebook on the gpu node.

We start the notebook on the example port `8888`: If the port `8888` is taken, try another random port between 1024 and 65000.
Also note the URL output by the command to be used later. (ex. http://127.0.0.1:8888/?token=7ba0ba5c3e9f5668f92518e4c5e723fea8b69aca065b4d57)

```bash
jupyter notebook --ip 0.0.0.0 --port 8888
```

Using a new terminal window from our personal laptop, we need to create an ssh tunnel to that specific port of the gpu node:
Note that `gpu001` is the name of the gpu we reserved at the beginnging. Remember that the port needs to be the same as your jupyter notebook port above.
```bash
ssh username@v.vectorinstitute.ai -L 8888:gpu001:8888
```

Keep the new connection alive by starting a tmux session in the new local terminal:
```bash
tmux
```

Now we can access the notebooks using our local browser. Copy the URL given by the jupyter notebook server into your local webbrowser:
```bash
(Example Token)
http://127.0.0.1:8888/?token=7ba0ba5c3e9f5668f92518e4c5e723fea8b69aca065b4d57
```

You should now be able to navigate to the notebooks and run them.

**Don't close the local terminal windows in your personal laptop!**

### Connecting a VSCode Server and Tunnel on that GPU Node

Rather than working through hosted Jupyter Notebooks, you can also connect directly to a VS Code instance on the GPU. After the cluster has fulfilled your request for a GPU session, run the following to set up a VSCode Server on the GPU node.

This command downloads and saves VSCode in your home folder on the cluster. You need to do this only once:
```bash
cd ~/

curl -Lk 'https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64' --output vscode_cli.tar.gz

tar -xf vscode_cli.tar.gz
rm vscode_cli.tar.gz
```

Please verify the beginning of the command prompt and make sure that you are running this command from a __GPU node__ (e.g., `user@gpu001`) and not the login node (`user@v`). After that, you can spin up a tunnel to the GPU node using the following command:

```bash
~/code tunnel
```

You will be prompted to authenticate via Github. On the first run, you might also need to review Microsoft's terms of services. After that, you can access the tunnel through your browser. If you've logged into Github on your VSCode desktop app, you can also connect from there by installing the extension `ms-vscode.remote-server`, pressing Shift-Command-P (Shift-Control-P), and entering `Remote-Tunnels: Connect to Tunnel`.

Note that you will need to keep the SSH connection running while using the tunnel. After you are done with the work, stop your session by pressing Control-C.

## Installing Custom Dependencies

__Note__: The following instructions are for anyone who would like to create their own python virtual environment to run experiments in `prompt_zoo`. If you would just like to run the code you can use one of our pre-built virtual environments by following the instructions in Section `Virtual Environments`, above. Instructions for creating environments for experiments outside of `prompt_zoo` are contained in the relevant subfolders.

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

## A note on disk space

Many of the experiments, especially in `prompt_zoo`, in this respository will end up writing to your scratch directory. An example path is:
```
/scratch/ssd004/scratch/snajafi/
```
where `snajafi` is replaced with your cluster username. This directory has a maximum capacity of 50GB. If you run multiple hyperparameter sweeps, you may fill this directory with model checkpoints. If this directory fills, it may interrupt your jobs or cause them to fail. Please be cognizant of the space and clean up old runs if you begin to fill the directory.

## Using Pre-commit Hooks (for developing in this repository)

To check your code at commit time
```
pre-commit install
```

You can also get pre-commit to fix your code
```
pre-commit run
```
