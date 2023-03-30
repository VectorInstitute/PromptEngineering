### OPT and Czarnowska Templates

The code in this folder is used to produce predictions on the Czarnowska templates housed in `src/reference_implementations/fairness_measurement/resources/czarnowska_templates`. These templates are described in detail in the readme in the directory above this folder.

### Environment Setup

If you're running the code on the cluster, simply source `prompt_engineering` with
```bash
source /ssd003/projects/aieng/public/prompt_engineering/bin/activate
```
and you should be good to go.

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
pip install transformers torch kscope pandas
```

The code is then run with
```bash
python -m src.reference_implementations.fairness_measurement.opt_czarnoska_analysis.opt_fairness_eval
```
The script uses the kaleidoscope tool to prompt the large language models to perform sentiment inference on the Czarnoswka templates file. Producing sentiment predictions for each sentence. We use 5-shot prompts drawn from the SST5 dataset.
__NOTE__: Making predictions on the entire Czarnowska templates takes a __very__ long time. We have already run OPT-6.7B and OPT-175B on these templates and stored the predictions in `src/reference_implementations/fairness_measurement/resources/predictions/` to consider. If you would like to generate your own, please strongly consider pairing the templates down to only the groups you would like to investigate.