# Prompt Engineering
This repository holds all of the code associated with the project considering prompt engineering for large language models. This includes work around reference implementations, demo notebooks, and fairness and bias evaluation.

The static code checker runs on python3.9

# Installing dependencies

## Installing on macOS
If you wish to install the package in macOS for local development, you should call the following script to install `python3.9` on macOS and then setup the virtual env for the module you want to install. This approach only installs the ML libraries (`pytorch`, `tensorflow`, `jax`) for the CPU. If you also want to install the package in the editable mode with all the development requirements, you should use the flag `DEV=true` when you run the script, otherwise use the flag `DEV=false`.
```
bash setup.sh OS=mac ENV_NAME=t5x DEV=true
```

## Installing on Vector's Cluster
You can call `setup.sh` with the `OS=vcluster` flag. This installs python in the linux cluster of Vector and installs the ML libraries for the GPU cards.
```
bash setup.sh OS=vcluster ENV_NAME=t5x DEV=true
```

### Using Pre-commit Hooks
To check your code at commit time
```
pre-commit install
```

You can also get pre-commit to fix your code
```
pre-commit run
```
# Running T5x
We currently support running T5x only on the vector's cluster.
Make sure you install the cluster dependencies via the following command:
```
bash setup.sh OS=vcluster ENV_NAME=t5x DEV=false
```

To start training a translation model using T5x, we submit the following slurm job.
```
source t5x-env/bin/activate
sbatch src/reference_implementations/run_multinode_2_2.slrm \
       <path/to/a_t5x_run_script.sh> \
       <path/to/a_t5x_log/dir> \
       <path/to/a_model_save/dir> \
       <path/to/a_data_save/dir>
```

For example:
```
source t5x-env/bin/activate
sbatch src/reference_implementations/run_multinode_2_2.slrm \
       src/reference_implementations/t5x/run_t5x.sh \
       ./t5x-exps-logs \
       /scratch/ssd004/scratch/snajafi/data_temp/t5x-exps/model \
       /scratch/ssd004/scratch/snajafi/data_temp/t5x-exps/data
```

Then, you can monitor the training status using `tensorboard` by specifying the directory used for saving models:
```
tensorboard --logdir=/scratch/ssd004/scratch/snajafi/data_temp/t5x-exps/model
```

# Using google-prompt-tuning
The following lines outline the steps to train the google's soft-prompt code based on t5x for the binary sentiment analysis task on the vector's cluster.

Make sure you install the cluster dependencies via the following command:
```
bash setup.sh OS=vcluster ENV_NAME=google_prompt_tuning DEV=true
```

Then submit the following slurm job for training prompts for binary sentiment analysis.
```
source google_prompt_tuning-env/bin/activate
sbatch src/reference_implementations/run_multinode_2_2.slrm \
       <path/to/a_t5x_run_script.sh> \
       <path/to/a_t5x_log/dir> \
       <path/to/a_model_save/dir> \
       <path/to/a_data_save/dir>
```

For example:
```
source google_prompt_tuning-env/bin/activate
sbatch src/reference_implementations/run_multinode_2_2.slrm \
       src/reference_implementations/google_prompt_tuning/train_sst2.sh \
       ./google-prompt-tuning-exps-logs \
       /scratch/ssd004/scratch/snajafi/data_temp/google-prompt-tuning-exps/model \
       /scratch/ssd004/scratch/snajafi/data_temp/google-prompt-tuning-exps/data
```

Then, you can monitor the training status using `tensorboard` by specifying the directory used for saving models:
```
tensorboard --logdir=/scratch/ssd004/scratch/snajafi/data_temp/google-prompt-tuning-exps/model
```

Inference:
```
source google_prompt_tuning-env/bin/activate
sbatch src/reference_implementations/run_multinode_2_2.slrm \
       src/reference_implementations/google_prompt_tuning/infer_sst2.sh \
       ./google-prompt-tuning-infer-exps-logs \
       /scratch/ssd004/scratch/snajafi/data_temp/google-prompt-tuning-exps/model/checkpoint_1144000 \
       /scratch/ssd004/scratch/snajafi/data_temp/google-prompt-tuning-exps/data/infer
```
