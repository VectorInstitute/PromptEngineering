# Measuring Fairness in Language Models

## Fairness Through Fine-Tuning

In the folder `opt_czarnoska_analysis/`, we measure the fairness of some models through the lens of sentiment analysis. Specifically, we will measure if swapping out the name of the groups (e.g., young vs old) will affect the model's prediction of the sentence's sentiment. We will use additional metrics and visualization techniques devised at the Vector Institute to highlight groups of people that the models are in favor of or biased against.

The test cases presented in this folder are from the following paper.

> Quantifying Social Biases in NLP: A Generalization and Empirical Comparison of Extrinsic Fairness Metrics
> Paula Czarnowska, Yogarshi Vyas, Kashif Shah
> Transaction of the Association for Computational Linguistics (TACL), 2021

(See `resources/czarnowska_templates/sentiment_fairness_tests.tsv`)

### Overview

It takes three simple steps to measure the fairness of any given large language model:
- Construct a prompt that enables the LLM to do three-way (positive/neutral/negative) sentiment analysis.
- Using this prompt, predict the sentiment of the test cases.
- Compute and visualize the biases in these predictions.

During the lab, we've covered a wide range of ways to prompt LLMs for text classification, including sentiment analysis. We know that any of these prompting methods could surface bias. Hence, we designed our pipeline to be modular, so that you can easily measure the fairness of a wide range of prompting techniques by changing only a few lines of code.

#### Step 1: Generating Sentiment labels

The first step is to predict the sentiment of each test case using. The script `fairness_eval_template.py` provides the boilerplate code for loading the test cases and formatting the output.

To provide an overview of the pipeline, we use the following fine-tuned RoBERTa-base model from the HuggingFace hub ([link](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)) for sentiment analysis. While we aren't using prompting in this example, this RoBERTa-base model is simple and efficient enough to demonstrate how the pipeline will interact with your LLM. You can easily adjust `fairness_eval_template.py` to use any prompting approach.

Before you run this script, be sure to adjust the constant values in the script as needed.
```python
# (fairness_eval_template.py)

# Path to sentiment fairness test cases
TEST_FILE_PATH = (
    "src/reference_implementations/fairness_measurement/resources/czarnowska_templates/sentiment_fairness_tests.tsv"
)

# Append formatted predictions to this file.
PREDICTION_FILE_PATH = "src/reference_implementations/fairness_measurement/resources/predictions/predictions.tsv"

# Batch size.
BATCH_SIZE = 8

# HuggingFace Model to load for this demo.
# Feel free to delete this line if you replaced the fine-tuned RoBERTa model with
# LLM + prompt-tuning.
DEMO_HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"

# Data entries below are used only for plotting the fairness diagrams.
MODEL = "RoBERTa-base fine-tuned"
DATASET = "TweetEval"  # Name of the labelled task-specific dataset
NUM_PARAMS = 0.125  # billions
RUN_ID = "example_run_a"  # E.g., distinguishes between different random seeds.
```

On a GPU machine, go to the root folder of this project (`PromptEngineering`). Create a virtual environment and run the fairness evaluation script:
```bash
python3 -m venv env
source env/bin/activate

python3 -m pip install -U transformers tqdm
python3 src/reference_implementations/fairness_measurement/fairness_eval_template.py
```

By default, this script will append predictions and info about the model to `PREDICTION_FILE_PATH`. If this file doesn't exist, the script will create a new one with the TSV header row.

#### Step 2: Calculate and Visualize biases

Open the Jupyter notebook `fairness_measurement/group-fairness-plots.ipynb` to load the predictions and interactively evaluate the bias of each model.

You do not need a GPU for this notebook. However, you might need to install additional packages for plotting. Refer to the notebook for more details.

Note that the notebook is capable of visualizing confidence intervals. However, to see the intervals, you will need to run Step 1 at least twice with the same `MODEL`, `DATASET`, `NUM_PARAMS`, but different `RUN_ID` values. Otherwise, the CI widths in the notebook will be NaN and no confidence interval will be plotted.


## BBQ: A Question-Answering Framework for Bias Probing

In the folder `bbq_fairness_example/` there is a small notebook that works through a few demonstrative examples of the BBQ task that was recently introduced to probe the bias inherent in large language models. The notebook describes the task and contains a link to the paper.

Before spinning up the notebooks on a GPU through the cluster, following the instructions in the top level [README](/README.md), make sure you source the `prompt_engineering` environment with the command

```bash
source /ssd003/projects/aieng/public/prompt_engineering/bin/activate
```

If you're running the notebooks launched through the Jupyter Hub, simply select `prompt_engineering` from the available kernels and you should be good to go.

If you want to create your own environment then you can do so by creating your own virtual environment with the command
```bash
python -m venv <name_of_venv>
```
then
```bash
source <name_of_venv>/bin/activate
```
finally run
```bash
pip install torch kscope
```

## Crow-S-Pairs Notebook for Bias Quantification

In the folder `crow_s_pairs/` there is notebook that facilitates running the Crow-S-Pairs task that aims at quantifying bias present in LLMs through estimation of the likelihoods of various pairs of sentences. The details and links to the paper are described in depth in the notebook itself.

Before spinning up the notebooks on a GPU through the cluster, following the instructions in the top level [README](/README.md), make sure you source the `prompt_engineering` environment with the command

```bash
source /ssd003/projects/aieng/public/prompt_engineering/bin/activate
```

If you're running the notebooks launched through the Jupyter Hub, simply select `prompt_engineering` from the available kernels and you should be good to go.

If you want to create your own environment then you can do so by creating your own virtual environment with the command
```bash
python -m venv <name_of_venv>
```
then
```bash
source <name_of_venv>/bin/activate
```
finally run
```bash
pip install transformers kscope pandas
```
