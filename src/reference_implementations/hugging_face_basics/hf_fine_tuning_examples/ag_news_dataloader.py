import math
from typing import Tuple, Union

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def make_train_val_datasets(dataset: Dataset, split_ratio: float) -> Tuple[Dataset, Dataset]:
    assert 0.0 < split_ratio < 1.0
    # cut the train set into train/val using split ratio as the percentage to put into train.
    original_length = len(dataset)
    train_length = math.floor(original_length * split_ratio)
    lengths = [train_length, original_length - train_length]
    return torch.utils.data.random_split(dataset, lengths)


def construct_ag_news_dataloaders(
    batch_size: int, train_split_ratio: float, tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast]
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # AG news just has train and test sets (no validation set)
    train_dataset = load_dataset("ag_news")["train"]
    # Tokenize the text data using the model tokenizer
    train_dataset = train_dataset.map(
        lambda row: tokenizer(row["text"], truncation=True, padding="max_length"), batched=True
    )
    train_label_list = train_dataset.features["label"]._int2str
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    # split the original AG news training dataset into a training and validation set.
    train_dataset, val_dataset = make_train_val_datasets(train_dataset, train_split_ratio)

    # Create the AG news test set.
    test_dataset = load_dataset("ag_news")["test"]
    # Tokenize the text data using the model tokenizer
    test_dataset = test_dataset.map(
        lambda row: tokenizer(row["text"], truncation=True, padding="max_length"), batched=True
    )
    test_label_list = test_dataset.features["label"]._int2str
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Ensure the order of the labels matches.
    assert all([train_label == test_label for train_label, test_label in zip(train_label_list, test_label_list)])

    # Create pytorch dataloaders from the dataset objects.
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    example_train_batch = next(iter(train_dataloader))
    example_encode_batch = example_train_batch["input_ids"]
    example_decode = tokenizer.batch_decode(example_encode_batch, skip_special_tokens=True)[0]
    print(f"Training data example encoding: {example_encode_batch[0]}")
    print(f"Training data example decoding: {example_decode}")
    return train_dataloader, val_dataloader, test_dataloader
