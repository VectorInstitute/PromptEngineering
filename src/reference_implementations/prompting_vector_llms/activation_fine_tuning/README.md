### Fine-tuning a classifier on the embeddings from an LLM

We are aiming to train a small classifier on top of the activations of an intermediate layer for the OPT language model hosted on Vector's cluster. As an example we'll work on the IMDB sentiment classification task. It's a boolean task that asks the model to classify the sentiment associated with a short movie review.

The dataset consists of 25,000 highly polar movie reviews for training, and another 25,000 for testing. The task is binary sentiment classification, where labels indicate either a positive or negative review.

Sample Review:

"This movie sucked. It really was a waste of my life. The acting was atrocious, the plot completely implausible. Long, long story short, these people get "terrorized" by this pathetic "crazed killer", but completely fail to fight back in any manner. And this is after they take a raft on a camping trip, with no gear, and show up at a campsite that is already assembled and completely stocked with food and clothes and the daughters headphones. Additionally, after their boat goes missing, they panic that they're stuck in the woods, but then the daughters boyfriend just shows up and they apparently never consider that they could just hike out of the woods like he did to get to them. Like I said, this movie sucks. A complete joke. Don't let your girlfriend talk you into watching it."

Label: Negative

### VENV Installation

Each of the notebooks in this folder require certain dependencies. In order to install the proper dependencies, each notebook has a pip install from the provided requirements file. If you are developing on the JupyterHub or a notebook launched from the cluster, you will have an isolated python environment to install into.

__However__: If you are working on these notebooks locally, be sure to create a virtual environment to install the dependencies into. This can be done, for example, with the command
```
python -m venv <name_of_venv>
```
then
```
source <name_of_venv>/bin/activate
```

#### Activation Computation

The first step is to compute activations for a set of training inputs and the same set of activations for testing inputs. Because the dataset is so large and the model is quite heavy, we'll just consider a few-shot set of training inputs (100 samples) and a randomly sampled set of points from the test dataset (300 samples). These will be stored in the resources folder and we'll train/test a classifier using these. Because the model is so expressive, this few-shot set of training inputs should still perform very well on the test set. The notebook to compute these is `compute_activations.ipynb`.

We also want to consider whether training with a prompt as part of the input helps the downstream model perform the task more accurately. So we'll save activations with and without a few-shot prompt as part of the input.

#### Classifier Training

After pre-computing the activations with and without prompting, we train a simple DNN to perform sentiment classification on the two different training sets. We'll compare performance to determine whether the prompting helped produce better downstream performance. The notebook to do the training is `train_on_activations.ipynb`
