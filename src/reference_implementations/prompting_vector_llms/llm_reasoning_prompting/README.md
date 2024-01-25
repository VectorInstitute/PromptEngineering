## Using Reasoning in the Prompting of LLMs

For an in-depth description of this task see the notebooks or the upper readme at `src/reference_implementations/prompting_vector_llms/README.MD`

## VENV Installation

The notebooks in this folder require certain dependencies. Before spinning up the notebooks on a GPU through the cluster, following the instructions in the top level [README](/README.md), make sure you source the `prompt_engineering` environment with the command

```bash
source /ssd003/projects/aieng/public/prompt_engineering/bin/activate
```

If you're running the notebooks launched through the Jupyter Hub, simply select `prompt_engineering` from the available kernels and you should be good to go.

If you want to create your own environment that you can modify, then you can do so by creating your own virtual environment with the command
```
python -m venv <name_of_venv>
```
then
```
source <name_of_venv>/bin/activate
```
and finally
```
pip install -r requirements.txt
```
