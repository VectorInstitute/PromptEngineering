import math
from typing import Tuple, Union

import datasets
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def make_train_val_datasets(dataset: datasets.Dataset, split_ratio: float) -> Tuple[Dataset, Dataset]:
    assert 0.0 < split_ratio < 1.0
    # cut the train set into train/val using split ratio as the percentage to put into train.
    original_length = len(dataset)
    train_length = math.floor(original_length * split_ratio)
    lengths = [train_length, original_length - train_length]
    return torch.utils.data.random_split(dataset, lengths)


def construct_dataloaders(
    batch_size: int,
    train_split_ratio: float,  # This value is used only the dataset is missing a "validation" split.
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    dataset_name: str,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    dataset_dict = load_dataset(dataset_name)
    assert isinstance(dataset_dict, datasets.DatasetDict)

    # Tokenize the text data using the model tokenizer
    tokenized_dataset_dict = dataset_dict.map(
        lambda row: tokenizer(row["text"], truncation=True, padding="max_length"), batched=True
    )

    train_dataset = tokenized_dataset_dict["train"]
    # Ensure that the dataloader yields PyTorch tensors, not lists of lists.
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    if "validation" in tokenized_dataset_dict.keys():
        val_dataset = tokenized_dataset_dict["validation"]
        val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    else:
        # Some datasets (e.g., AG news) just has train and test sets (no validation set)
        # split the original training dataset into a training and validation set.
        train_dataset, val_dataset = make_train_val_datasets(train_dataset, train_split_ratio)

    # Create the test set.
    test_dataset = tokenized_dataset_dict["test"]
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Create pytorch dataloaders from the dataset objects.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    example_train_batch = next(iter(train_dataloader))
    example_encode_batch = example_train_batch["input_ids"]
    example_decode = tokenizer.batch_decode(example_encode_batch, skip_special_tokens=True)[0]
    print(f"Training data example encoding: {example_encode_batch[0]}")
    print(f"Training data example decoding: {example_decode}")
    return train_dataloader, val_dataloader, test_dataloader
