# Using google-prompt-tuning (Not Fully Supported During Prompt Engineering Lab)

The following lines outline the steps to train the google's soft-prompt code based on t5x for the binary sentiment analysis task on the vector's cluster.

Make sure you install the cluster dependencies via the following command:
```bash
bash setup.sh OS=vcluster ENV_NAME=google_prompt_tuning DEV=true
```
*Note*: The command above will take a few moments to run

Source the environment with

```bash
source google_prompt_tuning-env/bin/activate
```

Then submit the following slurm job for training prompts for binary sentiment analysis.
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
       src/reference_implementations/google_prompt_tuning/train_sst2.sh \
       ./google-prompt-tuning-exps-logs \
       /scratch/ssd004/scratch/snajafi/data_temp/google-prompt-tuning-exps/model \
       /scratch/ssd004/scratch/snajafi/data_temp/google-prompt-tuning-exps/data
```

Then, you can monitor the training status using `tensorboard` by specifying the directory used for saving models:
```
tensorboard --logdir=/scratch/ssd004/scratch/snajafi/data_temp/google-prompt-tuning-exps/model --bind_all
```

*Note* that `snajafi` is a cluster username. Make sure that you replace that part of any paths with your own cluster username.

Inference:

The following commands read the input file `src/reference_implementations/google_prompt_tuning/resources/example_input_sentences.csv` and store the predictions at `/scratch/ssd004/scratch/snajafi/data_temp/google-prompt-tuning-exps/model/inference_eval/`:
```
source google_prompt_tuning-env/bin/activate
sbatch src/reference_implementations/run_multinode_2_2.slrm \
       src/reference_implementations/google_prompt_tuning/infer_sst2.sh \
       ./google-prompt-tuning-infer-exps-logs \
       /scratch/ssd004/scratch/snajafi/data_temp/google-prompt-tuning-exps/model \
       /scratch/ssd004/scratch/snajafi/data_temp/google-prompt-tuning-exps/data/infer
```
