# Introduction to some NLP metrics and Hugging Face Fine-tuning Setups

## HuggingFace Fine-tuning Examples

The folder `hf_fine_tuning_examples/` contains a few basic examples of using HuggingFace (HF) for inference and fine-tuning for several downstream tasks. There are examples, among others, of:

1. Fine-tuning a pre-trained RoBERTa-base model for the AG news text classification task.
2. Fine-tuning a pre-trained GPT2 model for the AG news text classification task.
3. Performing inference on a summarization task with a pre-trained BART model from the model hub and measuring performance on a benchmark dataset.

There are notebooks for running the training in (1) and (2) and for performing the inference and measurements in (3).

### VENV Installation

The notebooks in this folder require certain dependencies. Before spinning up the notebooks on a GPU through the cluster, following the instructions in the top level [README](/README.md), make sure you source the `prompt_engineering` environment with the command

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
pip install transformers evaluate datasets torch absl-py rouge_score
```

### Running scripts instead of notebooks

There are also python scripts for launching training on the cluster through a slurm script, should you want to do so. You need only source our pre-built venv and then launch a slurm request.

1) Source the venv:
```bash
source /ssd003/projects/aieng/public/prompt_engineering/bin/activate
```
2) Run the python script on a GPU __with your venv active__. Make sure you run the sbatch command from the top directory.
```bash
sbatch src/reference_implementations/run_singlenode_fine_tune.slrm \
    src/reference_implementations/hugging_face_basics/training_script/finetuning_roberta.sh \
    ./hf_fine_tuning_logs
```

## NLP Metrics Examples

The notebook in this folder entitled `nlp_metrics_examples.ipynb` focuses on introducing some standard NLP metrics. It specifically emphasizes metrics commonly used for evaluating models performing natural language generation (NLG) tasks.

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
pip install evaluate torch transformers nltk absl-py rouge_score bert_score
```
