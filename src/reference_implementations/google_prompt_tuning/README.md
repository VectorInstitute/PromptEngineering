# Using google-prompt-tuning (Not Fully Supported During Prompt Engineering Lab)

# WARNING

__The current implementations of Google Prompt-Tuning and T5X are at odds with each other and no longer work together. We have filed a bug report with the repository, but until it is fixed the code here will simply serve as an example that doesn't fully run. Luckily, we have our own PyTorch implemnetation in `prompt_zoo`!__

For the interested reader, Flax has removed optim in favor of optax in its newest versions above 0.5.3. This means that in order to run the code in this repository, one needs to downgrade below Flax 0.6. However, if you do that with Jax 0.4.5 or even with Jax 0.3.25 + jax.config.update('jax_array', True), the code cannot save a model checkpoint due to msgpack being unable to serialize the jax arrays.

## Introduction

The following lines outline the steps to train the google's soft-prompt code based on t5x for the binary sentiment analysis task on the vector's cluster.

## Virtual Environment

There are two options for utilizing a virtual env to run the code in this director.
1) You can source our pre-built environment from `/ssd003/projects/aieng/public/prompt_engineering_google_prompt_tuning` with the command
```bash
source /ssd003/projects/aieng/public/prompt_engineering_google_prompt_tuning/bin/activate
```
If you are using the pre-built environments *do not* modify it, as it will affect all users of the venv. To install your own environment that you can manipulate, follow the instructions below.

2) You can create your own environment using the `src/reference_implementations/google_prompt_tuning/google_prompt_tuning_env_script.sh` script it's usage is
```bash
bash <path/to/google_prompt_tuning_env_script.sh <path/to>/PromptEngineering/
```
where `<path/to>/PromptEngineering/` is the path to the top level of the git repository.

*Note*: The command above will take a few moments to run

then you activate your environment in the top level of the repository as
```bash
source google_prompt_tuning-env/bin/activate
```

## Dataset Preparation Notes

Before we can train the Prompt tuning model, we need to download the dataset into our data directory. We've already done that in a place on our cluster, but if you would like to download it locally for yourself, you can do so with
```
tfds build glue --data_dir=/scratch/ssd004/scratch/username/path/to/download_dir
```
This download process will take quite a bit of time. The location of our pre-downloaded version of the dataset is below.
```
/ssd003/projects/aieng/public/prompt_engineering_datasets/google_prompt_sst2_dataset/
```
This path should be used to replace
```
<path/to/a_data_save/dir>
```
in the commands below.

## Running Training

Then submit the following slurm job for training prompts for binary sentiment analysis.
```bash
sbatch src/reference_implementations/run_multinode_2_2.slrm \
       <path/to/a_t5x_run_script.sh> \
       <path/to/a_t5x_log/dir> \
       <path/to/a_model_save/dir> \
       <path/to/a_data_save/dir>
```

For example:
```bash
sbatch src/reference_implementations/run_multinode_2_2.slrm \
       src/reference_implementations/google_prompt_tuning/train_sst2.sh \
       ./google-prompt-tuning-exps-logs \
       /scratch/ssd004/scratch/snajafi/data_temp/google-prompt-tuning-exps/model \
       /scratch/ssd004/scratch/snajafi/data_temp/google-prompt-tuning-exps/data
```

## Tensorboard

Then, you can monitor the training status using `tensorboard` by specifying the directory used for saving models:
```
tensorboard --logdir=/scratch/ssd004/scratch/snajafi/data_temp/google-prompt-tuning-exps/model --bind_all
```

*Note* that `snajafi` is a cluster username. Make sure that you replace that part of any paths with your own cluster username.

## Running Inference

*NOTE*: In order to perfrom inference, you must change `PROMPT_FILE` in `src/reference_implementations/google_prompt_tuning/infer_sst2.sh` to a proper checkpoint that you would like to load.

The following commands read the input file `src/reference_implementations/google_prompt_tuning/resources/example_input_sentences.csv` and store the predictions at `/scratch/ssd004/scratch/snajafi/data_temp/google-prompt-tuning-exps/model/inference_eval/`:
```bash
sbatch src/reference_implementations/run_multinode_2_2.slrm \
       src/reference_implementations/google_prompt_tuning/infer_sst2.sh \
       ./google-prompt-tuning-infer-exps-logs \
       /scratch/ssd004/scratch/snajafi/data_temp/google-prompt-tuning-exps/model \
       /scratch/ssd004/scratch/snajafi/data_temp/google-prompt-tuning-exps/data/infer
```
