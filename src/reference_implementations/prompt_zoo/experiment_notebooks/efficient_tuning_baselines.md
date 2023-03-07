# Experiment Description

<p> As a finetuning experiment, we are going to experiment with the following baselines including soft-prompt tuning. </p>
<ul>

  <li><b>All-finetuning</b>: We update every parameter in the encoder-decoder T5 on the downstream task. For this baseline a very small learning rate such as 0.0005 is effective.
  </li></br>

  <li><b>output-finetuning</b>: We update only the language modelling head on top of the T5 decoder on the downstream task. For this baseline a learning rate such as 0.001 is effective.
  </li></br>

  <li><b>input-finetuning</b>: We update only input embedding table for the T5 encoder on the downstream task. For this baseline a learning rate such as 0.001 is effective.
  </li></br>

  <li><b>classifier-finetuning</b>: We update only feedforward classifier which is built on the top of the T5 encoder on the downstream task. For this baseline a learning rate such as 0.001 is effective.
  </li></br>

  <li><b>soft-prompt tuning</b>: We update only prompt table included in the encoder-decoder T5 on the downstream task.
  For soft-prompt tuning, a large learning rate around 0.3 is effective. In the experiments, we use 100 prompt tokens. Therefore our prompt length is 100.
  </li></br>
</ul>


<b>We are training these baselines on the semeval-2018 sentiment dataset for up to 30 epochs. We will use the vector's GPU cluster and the slurm scheduler to submit four GPU jobs to train these models. For this experiment, we don't need to login to a specific GPU node and we can submit the jobs from the login nodes on the vector vaughan cluster.</b>

# Installing the virtual env
To run the experiments, we need to install the `prompt_torch` virtual environment on the vector cluster:
```bash
bash setup.sh OS=vcluster ENV_NAME=prompt_torch DEV=true
```

Then activate the `prompt_torch` environment to launch the training jobs.

*Note*: This assumes that your are at the top directory. If you are not, you should manipulate the directory to point to the environment. All jobs should also be run from the top directory.
```bash
source ./prompt_torch-env/bin/activate
```
# Submitting the Training Jobs on SemEval Dataset

We need to create the following directories to save the model checkpoints on the vector's cluster.
Note that the following directories are created under the username `snajafi`.
You should use your dedicated username to create similar directories. These directories will hold the training and evaluation results for each of the experiments. They can be viewed with tensorboard following the comment below. These experiments can be run for SST2 by replacing `semeval` with `sst2` everywhere in the sbatch calls and changing the appropriate directory names.

```bash
mkdir -p /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semeval
mkdir -p /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semeval/all_finetune
mkdir -p /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semeval/input_finetune
mkdir -p /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semeval/output_finetune
mkdir -p /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semeval/classifier_finetuning
mkdir -p /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semeval/soft_prompt_finetune
```
*NOTE*: In the following also be sure to change the scratch directory in the bash commands

## Fine Tuning all weights of T5
submitting the job for `all_finetuning` baseline with the learning rate 0.0005:
```bash
sbatch src/reference_implementations/run_singlenode_prompt.slrm \
    src/reference_implementations/prompt_zoo/training_scripts/finetuning_sentiment.sh \
    ./torch-prompt-tuning-exps-logs \
    all_finetune \
    semeval \
    /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semeval/all_finetune \
    0.0005
```

## Fine tuning only the input layer of T5
submitting the job for `input_finetuning` baseline with the learning rate 0.001:
```bash
sbatch src/reference_implementations/run_singlenode_prompt.slrm \
    src/reference_implementations/prompt_zoo/training_scripts/finetuning_sentiment.sh \
    ./torch-prompt-tuning-exps-logs \
    input_finetune \
    semeval \
    /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semeval/input_finetune \
    0.001
```

## Fine tuning only the output layer of T5
submitting the job for `output_finetuning` baseline with the learning rate 0.001:
```bash
sbatch src/reference_implementations/run_singlenode_prompt.slrm \
    src/reference_implementations/prompt_zoo/training_scripts/finetuning_sentiment.sh \
    ./torch-prompt-tuning-exps-logs \
    output_finetune \
    semeval \
    /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semeval/output_finetune \
    0.001
```

## Fine tuning only a classifier on the output activations of T5
submitting the job for classifier `finetuning` baseline with the learning rate 0.001:
```bash
sbatch src/reference_implementations/run_singlenode_prompt.slrm \
    src/reference_implementations/prompt_zoo/training_scripts/finetuning_sentiment.sh \
    ./torch-prompt-tuning-exps-logs \
    classifier_finetuning \
    semeval \
    /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semeval/classifier_finetuning \
    0.001
```

## Soft Prompt Tuning 100 continuous prompt tokens for T5
submitting the job for `soft_prompt_tuning` with the learning rate 0.3:
```bash
sbatch src/reference_implementations/run_singlenode_prompt.slrm \
    src/reference_implementations/prompt_zoo/training_scripts/soft_prompt_sentiment.sh \
    ./torch-prompt-tuning-exps-logs \
    soft_prompt_finetune \
    semeval \
    /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semeval/soft_prompt_finetune \
    0.3 \
    100
```

To view the tensorboard with the training status for all of the submitted jobs:
```
tensorboard --logdir=/scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semeval/ --port=6008
```

*NOTE*: You will need to create a tunnel directory to the v instance that you are starting the tensorboard on. This will be one of `v1`, `v2`, or `v3`. It is written in your prompt as `username@v#`... so replace `v` in the command below with the `v#` that you have on your command line

The tensorboard command will finish and stall in the terminal you're working with. Now, in aother terminal window, create an ssh tunnel to the port 6008 we used in the above command from your local computer:
```bash
ssh username@v.vectorinstitute.ai -L 6008:localhost:6008
```

Then visit `https://localhost:6008`.

__NOTE__: If you get an issue where the port is already in use, change all instances of `6008` above to another port number

# Running hyper-parameter search and training of fine-tuning baselines on `SST2`:

This will kick off a learning rate hyper-parameter sweep for both the various fine-tuning strategies and the soft prompt tuning algorithm.

```bash
source prompt_torch-env/bin/activate
bash ./train_scripts/run_sst2_sentiment_experiments.sh
```
