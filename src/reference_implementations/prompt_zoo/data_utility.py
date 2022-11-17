"""This module implements the functions for preprocessing the data files into
pytorch datasets."""

from typing import Dict, List

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer


def white_space_fix(text: str) -> str:
    """Remove extra spaces in text."""
    return " ".join(text.split())


def preprocess_semeval_sentiment(text: str) -> str:
    """convert '-3: very negative emotional state can be inferred' to
    'negative'."""

    # convert a 7-class sentiment analysis task to a 3 class sentiment analysis.
    sentiment_mapper = {
        "moderately negative": "negative",
        "very negative": "negative",
        "slightly negative": "negative",
        "slightly positive": "positive",
        "very positive": "positive",
        "moderately positive": "positive",
        "neutral or mixed": "neutral",
    }
    sentiment = text.replace("emotional state can be inferred", "").strip().split(":")[1].strip()
    return sentiment_mapper[sentiment]


def read_semeval_sentiment_file(file_path: str) -> tuple[List[str], List[str]]:
    """This function reads the semeval 2018 data files for sentiment analysis.

    Example header: 'ID  Tweet Affect Dimension  Intensity Class'
    """
    df = pd.read_csv(file_path, delimiter="\t")
    tweets = [white_space_fix(tweet) for tweet in df["Tweet"].tolist()]
    sentiments = [preprocess_semeval_sentiment(sent) for sent in df["Intensity Class"].tolist()]

    # add end of sequence token:
    inputs = [tweet + " </s>" for tweet in tweets]
    outputs = [sent + " </s>" for sent in sentiments]
    return inputs, outputs


class SentimentDataset(Dataset):
    """Subclass the pytorch's Dataset to build my own dataset for the sentiment
    analysis task."""

    def __init__(self, encodings: Dict[str, List[int]]) -> None:
        """store the reference to the tokenized encodings."""
        self.encodings = encodings
        return

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return the elements for example index 'idx' as a dictionary with
        tensor values."""
        row = {}
        for key, val in self.encodings.items():
            row[key] = torch.tensor(val[idx])
        return row

    def __len__(self) -> int:
        """Return the length of the dataset."""
        return len(self.encodings["input_ids"])


def create_semeval_sentiment_dataset(
    tokenizer: T5Tokenizer,
    batch_size: int,
    source_max_length: int,
    decoder_max_length: int,
    file_name: str,
    shuffle: bool,
) -> DataLoader:
    """Function to create the required huggingface dataset to train the T5
    models on the semeval sentiment analysis task."""
    inputs, outputs = read_semeval_sentiment_file(file_name)

    encodings = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=source_max_length,
        add_special_tokens=False,
    )
    output_encodings = tokenizer(
        outputs,
        truncation=True,
        padding="max_length",
        max_length=decoder_max_length,
        add_special_tokens=False,
    )

    encodings["labels"] = output_encodings.input_ids
    encodings["target_attention_mask"] = output_encodings.attention_mask

    # because HuggingFace automatically shifts the labels, the labels correspond exactly to `target_ids`.
    # we have to make sure that the PAD token is ignored.
    # it ignores if label is -100!

    labels = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in encodings["labels"]
    ]
    encodings["labels"] = labels
    dataloader = DataLoader(SentimentDataset(encodings), batch_size=batch_size, shuffle=shuffle)
    return dataloader
