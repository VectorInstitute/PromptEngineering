# Experiment Description

<p> In this set of experiments, we are going to test four different kinds of <b>manually designed templates</b> for
prompting a <b>T5-large</b> language model for the binary sentiment classification task on <b>SST-2</b> dataset and 3-way sentiment classification task on the <b>SemEval 2018</b> dataset. </p>

<p> With manual templates, we no longer need any training examples and we will only make predictions on the corresponding development sets for these two datasets. </p>


# Installation On Vector's Cluster.

We assume you are following this instruction online from the vector's github repo for PromptEngineering:
https://github.com/VectorInstitute/PromptEngineering


Login to Vector's Vaughan cluster. I use the username `snajafi` as an example.
```bash
ssh snajafi@v.vectorinstitute.ai
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

Install the virtual env with the required libraries. We install our prompt module in the development mode so if we change code, the env gets updated with our changes.
```bash
bash setup.sh OS=vcluster ENV_NAME=prompt_torch DEV=true
```

Now we can run a jupyter notebook on this gpu node. We start the notebook on the example port `8888`:
```bash
source ./prompt_torch-env/bin/activate
jupyter notebook --ip 0.0.0.0 --port 8888
```


Using a new terminal window from our personal laptop, we need to create an ssh tunnel to that specific port of the gpu node:
```bash
ssh snajafi@v.vectorinstitute.ai -L 8888:gpu001:8888
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


**Follow the instructions in the jupyter notebook `template_experiments.ipynb`.**

**Don't close the local terminal windows in your personal laptop!**