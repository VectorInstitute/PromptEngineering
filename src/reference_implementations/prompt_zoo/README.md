
# Some instructions and tips for using the code in Prompt Zoo

Below are some tips on using the cluster to run the code in this folder. More specific instructions about how to run experiments and examine the results are discussed in the readmes under the `training_scripts` and `experiment_notebooks` folders and the `src/reference_implementations/prompt_zoo/experiment_notebooks/efficient_tuning_baselines.md` markdown. Once you've read through this readme, see those files to learn how to run experiments.

## Direct Access to a GPU Node.

We assume you are following this instruction online from the vector's github repo for PromptEngineering:
https://github.com/VectorInstitute/PromptEngineering


Login to Vector's Vaughan cluster. I use the username `username` as an example.
```bash
ssh username@v.vectorinstitute.ai
```

Then start a tmux session.
```bash
tmux
```

Let's first request an interactive node with GPU access. We will use A40 NVIDIA gpus from vector's cluster.
Note the resulting hostname of this gpu node (`gpu001` as an example).
```bash
srun --gres=gpu:1 -c 8 --mem 16G -p a40 --pty bash
```

Create a directory and clone this repo.
```bash
mkdir my_codes
cd my_codes
git clone https://github.com/VectorInstitute/PromptEngineering
cd PromptEngineering
```

*Virtual Environment Setup*: You have two options.

1) If you want to use our pre-built environment to run experiments, simply run
```bash
source /ssd003/projects/aieng/public/prompt_zoo/bin/activate
```
If you are using the pre-built environments *do not* modify it, as it will affect all users of the venv.

2) You can build your own venv that you are free to modify by running the commands below. We install our prompt module in the development mode so if we change code, the env gets updated with our changes.
```bash
bash setup.sh OS=vcluster ENV_NAME=prompt_torch DEV=true
```
then activate the environment
```bash
source ./prompt_torch-env/bin/activate
```
*Note*: If the env already exists in your repository you need not run the setup again. Just source it as instructed below. The above will take a few moments to complete

Now we can run a jupyter notebook on this gpu node. We start the notebook on the example port `8888`:
If the port `8888` is taken, try another random port between 1024 and 65000.
Also note the URL output by the command to be used later. (ex. http://127.0.0.1:8888/?token=7ba0ba5c3e9f5668f92518e4c5e723fea8b69aca065b4d57)

```bash
jupyter notebook --ip 0.0.0.0 --port 8888
```

Using a new terminal window from our personal laptop, we need to create an ssh tunnel to that specific port of the gpu node:
Note that `gpu001` is the name of the gpu we reserved at the beginnging. Remember that the port needs to be the same as your jupyter notebook port above.
```bash
ssh username@v.vectorinstitute.ai -L 8888:gpu001:8888
```

Keep the new connection alive by starting a tmux session in the new local terminal:
```bash
tmux
```

Now we can access the notebooks using our local browser. Copy the URL given by the jupyter notebook server into your local webbrowser:
```bash
(Example Token)
http://127.0.0.1:8888/?token=7ba0ba5c3e9f5668f92518e4c5e723fea8b69aca065b4d57
```

You should now be able to navigate to the notebooks and run them.

**Don't close the local terminal windows in your personal laptop!**
