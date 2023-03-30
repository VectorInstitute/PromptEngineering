import torch
import torch.nn as nn
from torch import cuda
from transformers import GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer

from src.reference_implementations.hugging_face_basics.hf_fine_tuning_examples.custom_dataloaders import (
    construct_dataloaders,
)
from src.reference_implementations.hugging_face_basics.hf_fine_tuning_examples.gpt2_classification_model import (
    Gpt2ClsModel,
)
from src.reference_implementations.hugging_face_basics.hf_fine_tuning_examples.hf_trainer import infer, train

gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
# Define PAD Token = EOS Token = 50256
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
pad_token_id = gpt2_tokenizer.encode(gpt2_tokenizer.eos_token)[0]

train_dataloader, val_dataloader, test_dataloader = construct_dataloaders(
    batch_size=8, train_split_ratio=0.8, tokenizer=gpt2_tokenizer, dataset_name="ag_news"
)

device = "cuda" if cuda.is_available() else "cpu"
print(f"Detected Device {device}")
# We'll provide two options. First we create our own model on top of the vanilla RoBERTa model. The second is to use
# HuggingFace's GPT2ForSequenceClassification class, which essentially does the same thing.
use_hf_sequence_classification = True
gpt2_model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path="gpt2", num_labels=4)
# The pad_token_id is used to determine when a sequence of inputs ends.
gpt2_model_config.pad_token_id = pad_token_id
gpt2_classifier_model = (
    GPT2ForSequenceClassification.from_pretrained("gpt2", config=gpt2_model_config)
    if use_hf_sequence_classification
    else Gpt2ClsModel(pad_token_id=pad_token_id)
)
loss_function = nn.CrossEntropyLoss()
n_training_epochs = 1
n_training_steps = 300

print("Begin Model Training...")
train(
    gpt2_classifier_model, train_dataloader, val_dataloader, loss_function, device, n_training_epochs, n_training_steps
)
print("Training Complete")

print("Saving model...")
output_model_file = "./gpt2_ag_news.bin"
torch.save(gpt2_classifier_model, output_model_file)
print("Model saved.")

print("Loading model...")
gpt2_classifier_model = torch.load(output_model_file)
print("Model loaded.")

print("Evaluating model on test set...")
test_accuracy, test_loss = infer(gpt2_classifier_model, loss_function, test_dataloader, device)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}%")
print("Model evaluated.")
