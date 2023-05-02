# Some instructions and tips for using the code in Prompt Zoo

Methods covered in this folder:

* [Prompt Tuning](https://aclanthology.org/2021.emnlp-main.243.pdf)
* [Gradient-based Discrete search (AutoPrompt)](https://arxiv.org/pdf/2010.15980.pdf)
* [GrIPS](https://arxiv.org/abs/2203.07281)

Below are some tips on using the cluster to run the code in this folder. More specific instructions about how to run experiments and examine the results are discussed in the readmes under the `training_scripts/` and `experiment_notebooks/` folders. The `src/reference_implementations/prompt_zoo/experiment_notebooks/efficient_tuning_baselines.md` markdown is also an important resource. Once you've read through this readme, see those files to learn how to run experiments.

## Logging into the Vaughn Cluster

Login to Vector's Vaughan cluster. The username `username` is an example, you should replace it with your cluster username
```bash
ssh username@v.vectorinstitute.ai
```

You'll need to follow the forking and clone instructions on the cluster to get a copy of the repository onto the cluster to work from. An overview of these instructions may be found [here](/Forking_Instructions.md)

__NOTE__: The implementations in this folder generally require an A40 GPU. Therefore, it is advised that you follow the instructions below to spin up any notebooks directly on an A40 GPU rather than spinning one up through the JupyterHub tool, as those notebooks are limited to T4v2s.

## Direct Access to a GPU Node.

From the login node of the Vector Vaughan cluster (v1,v2,v3) start a tmux session.
```bash
tmux
```

Let's first request an interactive node with GPU access. We will use A40 NVIDIA gpus from vector's cluster. The implementations in this folder require A40s to run well.
Note the resulting hostname of this gpu node (`gpu001` as an example).
```bash
srun --gres=gpu:1 -c 8 --mem 16G -p a40 --pty bash
```

__Virtual Environment Setup__: You have two options.

1. If you want to use our pre-built environment to run experiments, simply run
    ```bash
    source /ssd003/projects/aieng/public/prompt_zoo/bin/activate
    ```

    If you are using the pre-built environments you do not have permissions to modify it. That is, you cannot pip install any new dependencies.

2. The first option is recommended. However, you can build your own venv that you are free to modify by running the commands below. We install our prompt module in the development mode so if we change code, the env gets updated with our changes.

    ```bash
    bash setup.sh OS=vcluster ENV_NAME=prompt_torch DEV=true
    ```
    then activate the environment
    ```bash
    source ./prompt_torch-env/bin/activate
    ```

    __Note__: If the env already exists in your repository you need not run the setup again. Just source it as instructed. The above will take a few moments to complete

## Starting a notebook from a GPU Node.

Now we can run a jupyter notebook on our requested gpu node. We start the notebook on the example port `8888`: If the port `8888` is taken, try another random port between 1024 and 65000. Also note the URL output by the command below to be used later. (ex. http://127.0.0.1:8888/?token=7ba0ba5c3e9f5668f92518e4c5e723fea8b69aca065b4d57)

```bash
jupyter notebook --ip 0.0.0.0 --port 8888
```

__Note__: The above command will fail if you have not activated your python env.

Using a new terminal window from our laptop, we need to create an ssh tunnel to that specific port of the gpu node:

Note that `gpu001` is the name of the gpu we reserved at the beginnging. Remember that the port needs to be the same as your jupyter notebook port above (8888 in the example).

```bash
ssh username@v.vectorinstitute.ai -L 8888:gpu001:8888
```

Keep the new connection alive by starting a tmux session in the new local terminal:
```bash
tmux
```

Now we can access the notebooks using our local browser. Copy the URL given by the jupyter notebook server into your local webbrowser (Example Token):
```bash
http://127.0.0.1:8888/?token=7ba0ba5c3e9f5668f92518e4c5e723fea8b69aca065b4d57
```

You should now be able to navigate to the notebooks and run them.

**Don't close the local terminal windows in your personal laptop!**
