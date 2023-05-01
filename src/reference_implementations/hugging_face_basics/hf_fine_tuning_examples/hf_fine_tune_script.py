# %%
import torch
import torch.nn as nn
from custom_dataloaders import construct_dataloaders
from hf_trainer import infer, train
from roberta_classification_model import RobertaClsModel
from torch import cuda
import random
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# %% [markdown]
# Choose your dataset. Make sure that the number of classes in your model matches the number of different labels in that dataset.

# %%
# AG News Dataset for classifying news headlines.
# dataset_name = "ag_news"
# dataset_num_labels = 4

dataset_name = "jacobthebanana/sst5-mapped-extreme"
dataset_num_labels = 3
dataset_config = None

# dataset_name = "tweet_eval"
# dataset_config = "sentiment"
# dataset_num_labels = 3

# Uncomment the code below to use the SST2 dataset for sentiment analysis.
# NOTE: If you're going to use the SST2 dataset, you need to make sure that use_hf_sequence_classification = True
# The custom RoBERTa model is only defined for ag_news
# NOTE: For SST2 to train well, you'll need to adjust the learning rate and weight decay in the hf_trainer file
# A good place to start is lr=0.00001, weight_decay=0.001
# dataset_name = "SetFit/sst2"
# dataset_num_labels = 2

# %% [markdown]
# Choose your pre-trained model and setup the dataloaders.
# 
# By default, the HuggingFace Transformer models will provide the dense hidden states of the last layer, one vector for each token in the input. These vectors are not directly usable for our task of classification at the sequence level. 
# 
# One popular way to address this limitation is to add a "classification head"- a linear projection layer (`nn.Dense`)- on top of one of these token vectors in the output. For bi-directional encoder-only transformers such as BERT and RoBERTa, this layer will be added to the virtual token \[CLS\] at the beginning of the input. For decoder-only transformers such as GPT and OPT, this projection layer might be added to the last non-pad token in the sentence.
# 
# HuggingFace provides a convenient way to add this layer to your pre-trained model. For a wide range of base models including RoBERTa and OPT, you can load the pre-trained model with the projection layer added and initialized for you using the `AutoModelForSequenceClassification` class:
# 
# ```python
# model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
# ```
# 
# To demonstrate how this useful abstraction works, we've manually added a classification head on top of a HuggingFace **RoBERTa** model in a custom torch.nn module. We encourage you to take a look at our implementation in *roberta_classification_model.py* and see whether the behavior differs from that of AutoModelForSequenceClassification. Note that there is also an implementation of the "decoder-only" style head in *gpt2_classification_model.py*.
# 
# Please note that if you need to experiment with a base model other than RoBERTa- for example, OPT- you will need to set `use_hf_sequence_classification = False` and use the HuggingFace AutoModelForSequenceClassification instead. 

# %%
# NOTE: If you're going to use the SST2 dataset, you need to make sure that use_hf_sequence_classification = True
# The custom RoBERTa model is only defined for ag_news
use_hf_sequence_classification = True  # set to True to use the HuggingFace abstraction
hf_model_name = "roberta-base"
# hf_model_name = "roberta-large"
# hf_model_name = "facebook/opt-125m"
# hf_model_name = "facebook/opt-350m"
# hf_model_name = "facebook/opt-1.3b"


# Uncomment the code below to use facebook/opt-125m as the base model.
# Note that using OPT-125m requires the use_hf_sequence_classification = True
# use_hf_sequence_classification = True
# hf_model_name = "facebook/opt-125m"  # Also try "facebook/opt-125m" for OPT.

# %%
tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

# Set the maximum number of tokens in each input.
tokenizer.model_max_length = 512
train_dataloader, val_dataloader, test_dataloader = construct_dataloaders(
    batch_size=8, train_split_ratio=0.8, tokenizer=tokenizer, dataset_name=dataset_name, dataset_config=dataset_config
)

# %% [markdown]
# Setup the different variables we'd like for training

# %%
device = "cuda" if cuda.is_available() else "cpu"
print(f"Detected Device {device}")
# We'll provide two options. First we create our own model on top of the vanilla RoBERTa model. The second is to use
# HuggingFace's AutoModel class, which essentially does the same thing for RoBERTa, but with support additional base
# models such as OPT and GPT-J.
classifier_model = (
    AutoModelForSequenceClassification.from_pretrained(hf_model_name, num_labels=dataset_num_labels)
    if use_hf_sequence_classification
    else RobertaClsModel()
)
loss_function = nn.CrossEntropyLoss()
n_training_epochs = 1
n_training_steps = 300

# %% [markdown]
# Train the model on the training dataset

# %%
print("Begin Model Training...")
train(
    classifier_model,
    train_dataloader,
    val_dataloader,
    loss_function,
    device,
    n_training_epochs,
    n_training_steps,
)
print("Training Complete")

# %% [markdown]
# Save the final model to disk

# %%
print("Saving model...")
models_path = "/h/fkohankh/fk-models/"
hf_model_name_formatted = hf_model_name.split("/")[-1]
dataset_name_formatted = dataset_name.split("/")[-1]

seed = random.randint(0, 10000)
output_model_file = f"{models_path}{hf_model_name_formatted}_{dataset_name_formatted}_{str(seed)}/"
classifier_model.save_pretrained(output_model_file)
tokenizer.save_pretrained(output_model_file)

# output_model_file = f"{models_path}{hf_model_name_formatted}_{dataset_name_formatted}_{str(seed)}torch_model.bin"
# torch.save(classifier_model, output_model_file)

print("Model saved to", output_model_file)

# %% [markdown]
# Load model back up and perform inference on the test set

# %%
# print("Loading model...")
# classifier_model = torch.load(output_model_file)
# print("Model loaded.")

print("Evaluating model on test set...")
test_accuracy, test_loss = infer(classifier_model, loss_function, test_dataloader, device)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}%")
print("Model evaluated.")

with open(output_model_file + "fine_tune_log.txt", 'w') as f:
    f.write(f"Test Accuracy: {test_accuracy}%")
    f.write('\n')
    f.write(f"Test Loss: {test_loss}")