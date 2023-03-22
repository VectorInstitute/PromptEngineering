### Introduction to some NLP metrics

### HuggingFace Fine-tuning Examples

The folder `hf_fine_tuning_examples` contains a few basic examples of using HuggingFace for inference and fine-tuning for a few downstream tasks. The examples are

1) Fine-tuning a pre-trained RoBERTa-base model for the AG news text classification task.
2) Fine-tuning a pre-trained GPT2 model for the AG news text classification task.
3) Performing inference on a summarization task with a pre-trained BART model from the model hub and measuring performance on a benchmark dataset.

There are notebooks for running the training in (1) and (2) and for performing the inference and measurements in (3). These notebooks have self-contained pip installs at the beginning. However, there are also python scripts for launching training on the cluster through a slurm script, should you want to do so. However, there isn't any venv setup here. Simply create your own venv using the pip installs from the notebooks if you would like to run them as scripts.

If you're running the scripts rather than working in the notebooks, you can do the following

1) Create a venv with
```bash
module load python/3.9.10
python -m venv <name_of_venv>
source <name_of_venv>/bin/activate
pip install transformers datasets torch
```
2) Run the python script on a GPU after logging into the cluster. Make sure you run the sbatch command from the top directory.
```bash
sbatch src/reference_implementations/run_singlenode_fine_tune.slrm \
    src/reference_implementations/hugging_face_basics/training_script/finetuning_roberta.sh \
    ./hf_fine_tuning_logs
```

### NLP Metrics Examples

The notebook in this folder entitled `nlp_metrics_examples.ipynb` focuses on introducing some standard NLP metrics, specifically focusing on metrics common for natural language generation (NLG) tasks.

The notebook has a pip install as one of the cells. If you are developing on the JupyterHub or a notebook launched from the cluster, you will have an isolated python environment to install into.

__However__: If you are working on this notebook locally, be sure to create a virtual environment to install the dependencies into. This can be done, for example, with the command
```
python -m venv <name_of_venv>
```
then
```
source <name_of_venv>/bin/activate
```
