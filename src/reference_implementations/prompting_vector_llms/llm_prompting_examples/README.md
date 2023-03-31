## Prompting of LLMs on Vector's Cluster

For an in-depth desription of each of these tasks see the notebooks or the upper readme at `src/reference_implementations/prompting_vector_llms/README.MD`

### VENV Installation

Each of the notebooks in this folder require certain dependencies. If you're running the notebooks on the cluster, simply select `prompt_engineering` from the available kernels and you should be good to go.

If you want to create your own environment then you can do so by creating your own virtual environment with the command
``` bash
python -m venv <name_of_venv>
```
then
``` bash
source <name_of_venv>/bin/activate
```
and finally
``` bash
pip install -r requirements.txt
```

__Note__: You can also source `prompt_engineering` for running scripts by using
```bash
source /ssd003/projects/aieng/public/prompt_engineering/bin/activate
```
