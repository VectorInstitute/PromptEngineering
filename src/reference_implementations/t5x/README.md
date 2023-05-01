# Running T5x (Not Fully Supported During Prompt Engineering Lab)

We support running T5x only on the vector's cluster.

## Virtual Environment

To install your own environment that you can manipulate, follow the instructions below.

You can create your own environment using the `src/reference_implementations/t5x/t5x_env_script.sh` script it's usage is
```bash
bash <path/to/script/t5x_env_script.sh <path/to>/PromptEngineering/
```
where `<path/to>/PromptEngineering/` is the path to the top level of the git repository.

*Note*: The command above will take a few moments to run

then you activate your environment in the top level of the repository as
```
source t5x-env/bin/activate
```

## Dataset Preparation Notes

Before we can train the T5x model, we need to download the dataset into our data directory. We've already done that in a place on our cluster, but if you would like to download it locally for yourself, you can do so with
```
tfds build wmt_t2t_translate --data_dir=/scratch/ssd004/scratch/username/path/to/download_dir
```
This download process will take quite a bit of time. The location of our pre-downloaded version of the dataset is below.
```
/ssd003/projects/aieng/public/prompt_engineering_datasets/t5x_translation_dataset/
```
This path should be used to replace
```
<path/to/a_data_save/dir>
```
in the commands below.

## Model training commands

To start training a translation model using T5x, we submit the following slurm job.  Then run the training job through the slurm scheduler with sbatch. The structure of the command is below
```

sbatch src/reference_implementations/run_multinode_2_2.slrm \
       <path/to/a_t5x_run_script.sh> \
       <path/to/a_t5x_log/dir> \
       <path/to/a_model_save/dir> \
       <path/to/a_data_save/dir>
```

For example:
```
sbatch src/reference_implementations/run_multinode_2_2.slrm \
       src/reference_implementations/t5x/train_t5x.sh \
       ./t5x-exps-logs \
       /scratch/ssd004/scratch/snajafi/data_temp/t5x-exps/model \
       /ssd003/projects/aieng/public/prompt_engineering_datasets/t5x_translation_dataset/
```
*Note* that `snajafi` is a cluster username. Make sure that you replace that part of any paths with your own cluster username. The configuration for the t5x model is found here `src/reference_implementations/t5x/base_wmt_train.gin`. The model is trained on the `wmt_t2t_ende_v003` dataset, held by Google. It is a small English to German translation datasets.

## Tensorboard

Then, you can monitor the training status using `tensorboard` by specifying the directory used for saving models:
```
tensorboard --logdir=/scratch/ssd004/scratch/snajafi/data_temp/t5x-exps/model --bind_all
```

The tensorboard command will finish and stall in the terminal you're working with. Now, in another terminal window, create an ssh tunnel to the port from tensorboard (ex. 6006) we used in the above command from your local computer:
```bash
ssh username@v.vectorinstitute.ai -L 6006:localhost:6006
```
