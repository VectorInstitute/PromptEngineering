import torch
import torch.nn as nn
from torch import cuda
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from src.reference_implementations.hugging_face_basics.hf_fine_tuning_examples.custom_dataloaders import (
    construct_dataloaders,
)
from src.reference_implementations.hugging_face_basics.hf_fine_tuning_examples.hf_trainer import infer, train
from src.reference_implementations.hugging_face_basics.hf_fine_tuning_examples.roberta_classification_model import (
    RobertaClsModel,
)

roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
train_dataloader, val_dataloader, test_dataloader = construct_dataloaders(
    batch_size=8, train_split_ratio=0.8, tokenizer=roberta_tokenizer, dataset_name="ag_news"
)

device = "cuda" if cuda.is_available() else "cpu"
print(f"Detected Device {device}")
# We'll provide two options. First we create our own model on top of the vanilla RoBERTa model. The second is to use
# HuggingFace's RobertaForSequenceClassification class, which essentially does the same thing.
use_hf_sequence_classification = False
roberta_classifier_model = (
    RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=4)
    if use_hf_sequence_classification
    else RobertaClsModel()
)
loss_function = nn.CrossEntropyLoss()
n_training_epochs = 1
n_training_steps = 300

print("Begin Model Training...")
train(
    roberta_classifier_model,
    train_dataloader,
    val_dataloader,
    loss_function,
    device,
    n_training_epochs,
    n_training_steps,
)
print("Training Complete")

print("Saving model...")
output_model_file = "./roberta_ag_news.bin"
torch.save(roberta_classifier_model, output_model_file)
print("Model saved.")

print("Loading model...")
roberta_classifier_model = torch.load(output_model_file)
print("Model loaded.")

print("Evaluating model on test set...")
test_accuracy, test_loss = infer(roberta_classifier_model, loss_function, test_dataloader, device)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}%")
print("Model evaluated.")
