# Training Scripts

This folder houses training scripts to run various experiemnts in `src/reference_implementations/prompt_zoo`. More details for running each script is given below.

## Large scale experiments

The two larger scale scripts for running experiments are `run_sst2_sentiment_experiments.sh` and `run_sentiment_experiments_gradient_search.sh`

*NOTE*: Before either is run you must activate your environment. If you are using our pre-built environment run 

```bash
source /ssd003/projects/aieng/public/prompt_engineering/bin/activate
```
if you are using one you created locally run 
```bash
source prompt_torch-env/bin/activate
```

### run_sst2_sentiment_experiments (Hyper parameter search)

This script orchestrates running a hyper parameter search for the learning rate on various kinds of prompt tuning and fine tuning setups. For more information on how to run each experiment individually see `experiments_notebooks/efficient_tuning_baselines.md`. REMEMBER to activate your environment (as instructed above). This script can be run from the top level directory as

```bash
bash ./src/reference_implementations/prompt_zoo/training_scripts/run_sst2_sentiment_experiments.sh
```

After all of the experiments are complete you can view the results on tensorboard by running. Note that `snajafi` should be replaced with your own username both in the script and below in the tensorboard command.

To view the tensorboard with the training status for all of the submitted jobs:
```
tensorboard --logdir=/scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/sst2/ --bind_all
```

*NOTE*: You will need to create a tunnel directory to the v instance that you are starting the tensorboard on. This will be one of `v1`, `v2`, or `v3`. It is written in your prompt as `username@v#`... so replace `v` in the command below with the `v#` that you have on your command line

The tensorboard command will finish and stall in the terminal you're working with. Now, in aother terminal window, create an ssh tunnel to the port 6008 we used in the above command from your local computer:
```bash
ssh username@v.vectorinstitute.ai -L 6006:localhost:6006
```

Then visit `https://localhost:6006`.

### run_sentiment_experiments_gradient_search

This script orchestrates running gradient-based discrete prompt search for semeval and sst-2. REMEMBER to activate your environment (as instructed above). This script can be run from the top level directory as

```bash
bash ./src/reference_implementations/prompt_zoo/training_scripts/run_sentiment_experiments_gradient_search.sh
```

After all of the experiments are complete you can view the results on tensorboard by running. Note that `snajafi` should be replaced with your own username both in the script and below in the tensorboard command.

To view the tensorboard with the training status for all of the submitted jobs and take note of the port:
```
tensorboard --logdir=/scratch/ssd004/scratch/snajafi/data_temp/torch-prompt/gradient_search/ --bind_all
```

*NOTE*: You will need to create a tunnel directory to the v instance that you are starting the tensorboard on. This will be one of `v1`, `v2`, or `v3`. It is written in your prompt as `username@v#`... so replace `v` in the command below with the `v#` that you have on your command line

The tensorboard command will finish and stall in the terminal you're working with. Now, in aother terminal window, create an ssh tunnel to the port 6008 we used in the above command from your local computer:
```bash
ssh username@v.vectorinstitute.ai -L 6006:localhost:6006
```

Then visit `https://localhost:6006`.

## Other training scripts

The various other training scripts in this directory are used to run different types of fine-tuning and prompt tuning experiments. There use is detailed in `experiments_notebooks/efficient_tuning_baselines.md`