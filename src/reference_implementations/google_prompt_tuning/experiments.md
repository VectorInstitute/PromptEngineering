# Binary Sentiment Analysis on the SST2 dataset.
Following the [soft-prompt paper](https://aclanthology.org/2021.emnlp-main.243.pdf), I have been training the soft-prompts for the binary sentiment analysis task on the vector's cluster using the released T5x base model. This is a test that we can indeed train on the vector's t4v2 GPUs. I have been only using 4 gpus each with 16 GB of GPU RAM.

As suggested by the paper, we use the gin-config that initializes the soft-prompts using embeddings for the class labels.

The accuracy of the every 1000 training step is reported in the following tensorboard diagram:

<img width="746" alt="experiment-running" src="./resources/sst2-acc.png">
