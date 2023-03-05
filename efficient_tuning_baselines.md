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
```bash
source ./prompt_torch-env/bin/activate
```



# Submitting the Training Jobs on SemEval Dataset

We need to create the following directories to save the model checkpoints on the vector's cluster.
Note that the following directories are created under the username `snajafi`.
You should use your dedicated username to create similar directories.

```bash
mkdir -p /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semval
mkdir -p /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semval/all_finetune
mkdir -p /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semval/input_finetune
mkdir -p /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semval/output_finetune
mkdir -p /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semval/classifier_finetuning
mkdir -p /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semval/soft_prompt_finetune
```

submitting the job for `all_finetuning` baseline with the learning rate 0.0005:
```bash
sbatch src/reference_implementations/run_singlenode_prompt.slrm \
    src/reference_implementations/prompt_zoo/finetuning_sentiment.sh \
    ./torch-prompt-tuning-exps-logs \
    all_finetune \
    semeval \
    /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semval/all_finetune \
    0.0005
```

submitting the job for `input_finetuning` baseline with the learning rate 0.001:
```bash
sbatch src/reference_implementations/run_singlenode_prompt.slrm \
    src/reference_implementations/prompt_zoo/finetuning_sentiment.sh \
    ./torch-prompt-tuning-exps-logs \
    input_finetune \
    semeval \
    /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semval/input_finetune \
    0.001
```

submitting the job for `output_finetuning` baseline with the learning rate 0.001:
```bash
sbatch src/reference_implementations/run_singlenode_prompt.slrm \
    src/reference_implementations/prompt_zoo/finetuning_sentiment.sh \
    ./torch-prompt-tuning-exps-logs \
    output_finetune \
    semeval \
    /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semval/output_finetune \
    0.001
```

submitting the job for classifier `finetuning` baseline with the learning rate 0.001:
```bash
sbatch src/reference_implementations/run_singlenode_prompt.slrm \
    src/reference_implementations/prompt_zoo/finetuning_sentiment.sh \
    ./torch-prompt-tuning-exps-logs \
    classifier_finetuning \
    semeval \
    /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semval/classifier_finetuning \
    0.001
```

submitting the job for `soft_prompt_tuning` with the learning rate 0.3:
```bash
sbatch src/reference_implementations/run_singlenode_prompt.slrm \
    src/reference_implementations/prompt_zoo/soft_prompt_sentiment.sh \
    ./torch-prompt-tuning-exps-logs \
    soft_prompt_finetune \
    semeval \
    /scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semval/soft_prompt_finetune \
    0.3 \
    100
```

To view the tensorboard with the training status for all of the submitted jobs:
```
tensorboard --logdir=/scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/semval/ --port=6008
```

Then create an ssh tunnel to the port 6008 we used in the above command from your local computer:
```bash
ssh username@v.vectorinstitute.ai -L 6008:localhost:6008
```

Then visit `https://localhost:6008`.

# Running hyper-parameter search and training of fine-tuning baselines on `SST2`:
```bash
source prompt_torch-env/bin/activate
bash ./train_scripts/run_sst2_sentiment_experiments.sh
```
