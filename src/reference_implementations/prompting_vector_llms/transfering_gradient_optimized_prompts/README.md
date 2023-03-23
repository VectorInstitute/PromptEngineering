### Dependency Installation

Each of the notebooks in this folder require certain dependencies. In order to install the proper dependencies, each notebook has a pip install from the provided requirements file. If you are developing on the JupyterHub or a notebook launched from the cluster, you will have an isolated python environment to install into.

__However__: If you are working on these notebooks locally, be sure to create a virtual environment to install the dependencies into. This can be done, for example, with the command
```
python -m venv <name_of_venv>
```
then
```
source <name_of_venv>/bin/activate
```
