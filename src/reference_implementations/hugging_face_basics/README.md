## Introduction to some NLP metrics

### HuggingFace Fine-tuning Examples

The folder `hf_fine_tuning_examples` contains a few basic examples of using HuggingFace for inference and fine-tuning for a few downstream tasks. There are examples, among others, of:

1) Fine-tuning a pre-trained RoBERTa-base model for the AG news text classification task.
2) Fine-tuning a pre-trained GPT2 model for the AG news text classification task.
3) Performing inference on a summarization task with a pre-trained BART model from the model hub and measuring performance on a benchmark dataset.

There are notebooks for running the training in (1) and (2) and for performing the inference and measurements in (3).

If you're running the notebooks on the cluster, simply select `prompt_engineering` from the available kernels and you should be good to go.

If you want to create your own environment then you can do so by creating your own virtual environment with the command
```
python -m venv <name_of_venv>
```
then
```
source <name_of_venv>/bin/activate
```
finally run
```bash
pip install transformers evaluate datasets torch absl-py rouge_score
```

__Note__: You can also source `prompt_engineering` for running scripts by using
```bash
source /ssd003/projects/aieng/public/prompt_engineering/bin/activate
```

#### Running scripts instead of notebooks

There are also python scripts for launching training on the cluster through a slurm script, should you want to do so. However, there isn't any venv setup here. Simply create your own venv using the pip installs from the notebooks if you would like to run them as scripts.

1) Create a venv with
```bash
module load python/3.9.10
python -m venv <name_of_venv>
source <name_of_venv>/bin/activate
pip install transformers datasets torch absl-py rouge_score
```
2) Run the python script on a GPU after logging into the cluster. Make sure you run the sbatch command from the top directory.
```bash
sbatch src/reference_implementations/run_singlenode_fine_tune.slrm \
    src/reference_implementations/hugging_face_basics/training_script/finetuning_roberta.sh \
    ./hf_fine_tuning_logs
```

### NLP Metrics Examples

The notebook in this folder entitled `nlp_metrics_examples.ipynb` focuses on introducing some standard NLP metrics, specifically focusing on metrics common for natural language generation (NLG) tasks.

If you're running the notebooks on the cluster, simply select `prompt_engineering` from the available kernels and you should be good to go.

If you want to create your own environment then you can do so by creating your own virtual environment with the command
```
python -m venv <name_of_venv>
```
then
```
source <name_of_venv>/bin/activate
```
finally run
```bash
pip install evaluate torch transformers nltk absl-py rouge_score bert_score
```

__Note__: You can also source `prompt_engineering` for running scripts by using
```bash
source /ssd003/projects/aieng/public/prompt_engineering/bin/activate
```
