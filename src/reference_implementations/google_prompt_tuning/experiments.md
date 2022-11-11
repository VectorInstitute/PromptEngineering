# Binary Sentiment Analysis on the SST2 dataset.
Following the soft-prompt paper for the binary sentiment analysis, I have been training the soft-prompts for the binary sentiment analysis task on the vector's cluster. This is a test that we can indeed train on the vector's t4v2 GPUs. I have been only using 4 gpus each with 16 GB of GPU Ram. This is for the released T5 base model by the soft-prompt repo.

As suggested by the paper, we use the gin-config that initializes the soft-prompts with class labels.

The accuracy of the every 1000 training step is reported in the following tensorboard diagram:

<img width="746" alt="experiment-running" src="https://user-images.githubusercontent.com/12455298/201135283-518662d3-68d5-43c3-a0e6-0cb050a0cd6c.png">
