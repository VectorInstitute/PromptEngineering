### Introduction to some NLP metrics

### HuggingFace Fine-tuning Examples

The folder `hf_fine_tuning_examples` contains a few basic examples of using HuggingFace for inference and fine-tuning for a few downstream tasks. The examples are

1) Fine-tuning a pre-trained RoBERTa-base model for the AG news text classification task.
2) Fine-tuning a pre-trained GPT2 model for the AG news text classification task.
3) Performing inference on a summarization task with a pre-trained BART model from the model hub and measuring performance on a benchmark dataset.

There are notebooks for running the training in (1) and (2) and for performing the inference and measurements in (3). These notebooks have self-contained pip installs at the beginning. However, there are also python scripts for launching training on the cluster through a slurm script, should you want to do so. However, there isn't any venv setup here. Simply created your own venv using the pip installs from the notebooks if you would like to run them as scripts.


### NLP Metrics Examples

The notebook in this folder entitled `nlp_metrics_examples.ipynb` focuses on introducing some standard NLP metrics, specifically focusing on metrics common for natural language generation (NLG) tasks.

Tnotebook has a pip install as one of the cells. If you are developing on the JupyterHub or a notebook launched from the cluster, you will have an isolated python environment to install into.

__However__: If you are working on this notebook locally, be sure to create a virtual environment to install the dependencies into. This can be done, for example, with the command
```
python -m venv <name_of_venv>
```
then
```
source <name_of_venv>/bin/activate
```
