# LLaMA-2 and the Czarnowska Templates

The code in this folder is used to produce predictions on the Czarnowska templates housed in `src/reference_implementations/fairness_measurement/resources/czarnowska_templates/`. These templates are described in detail in the readme in the directory above this folder and in the paper [here](https://aclanthology.org/2021.tacl-1.74/).

## Environment Setup

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
pip install transformers torch kscope pandas
```

The code is then run with
```bash
python -m src.reference_implementations.fairness_measurement.czarnowska_analysis.prompting_czarnowska_templates
```

The script uses the kaleidoscope tool to prompt the large language models to perform sentiment inference on the Czarnoswka templates file. Producing sentiment predictions for each sentence. We use 5-shot prompts drawn from the SST5 dataset.

__NOTE__: Making predictions on the entire Czarnowska templates takes a __VERY__ long time. We have already run OPT-6.7B, OPT-175B, LLaMA-2-7B, LLaMA-2-70B on these templates and stored the predictions in `src/reference_implementations/fairness_measurement/resources/predictions/` to visualize using the notebook `src/reference_implementations/fairness_measurement/fairness_eval_template.py`. If you would like to generate your own, please strongly consider pairing the templates down to only the groups you would like to investigate.
