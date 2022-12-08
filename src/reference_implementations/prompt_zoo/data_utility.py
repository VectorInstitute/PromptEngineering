"""This module implements the functions for preprocessing the data files into
pytorch datasets."""

from typing import Dict, List, Tuple, Union

import pandas as pd
import torch
from absl import flags
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer

FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 16, "The batch size used for training or inference.")
flags.DEFINE_integer("source_max_length", 128, "The maximum number of tokens consider in the input sequence.")
flags.DEFINE_integer("decoder_max_length", 128, "The maximum number of tokens consider in the output sequence.")


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


def read_semeval_sentiment_file(
    file_path: str, repeat_input: bool = False, with_instructions: bool = False
) -> Tuple[List[str], List[str], List[int]]:
    """This function reads the semeval 2018 data files for sentiment analysis.

    Example header: 'ID  Tweet Affect Dimension  Intensity Class'
    """
    df = pd.read_csv(file_path, delimiter="\t")
    tweets = [white_space_fix(tweet) for tweet in df["Tweet"].tolist()]
    sentiments = [preprocess_semeval_sentiment(sent) for sent in df["Intensity Class"].tolist()]

    # store class information for classification modules.
    # for converting the class label to its index, we string sort the set to block
    # different permutations of the class labels. required if we call function multiple times.
    all_classes = sorted(list(set(sentiments)))
    assert all_classes == sorted(["positive", "negative", "neutral"])

    class_to_id = {label: index for index, label in enumerate(all_classes)}
    if with_instructions:
        instruction = "Generate the sentiment of the next sentence from the labels {}.".format(" ".join(all_classes))
        tweets = ["instruction: {} sentence: {} sentiment:".format(instruction, tweet) for tweet in tweets]

    if repeat_input:
        # repeat every input for every possible output class.
        # the inference will compute the score for every possible
        # label and then select the label with the max score given by the LM.
        inputs = []
        outputs = []
        class_indices = []
        for tweet in tweets:
            for label in all_classes:
                inputs.append(f"{tweet} </s>")
                outputs.append(f"{label} </s>")
                class_indices.append(class_to_id[label])
        return inputs, outputs, class_indices

    # add end of sequence token:
    inputs = [f"{tweet} </s>" for tweet in tweets]
    outputs = [f"{sent} </s>" for sent in sentiments]
    class_indices = [class_to_id[sent] for sent in sentiments]
    return inputs, outputs, class_indices


class SentimentDataset(Dataset):
    """Subclass the pytorch's Dataset to build my own dataset for the sentiment
    analysis task."""

    def __init__(self, data: Dict[str, Union[List[int], List[List[int]]]]) -> None:
        """store the reference to the tokenized data."""
        self.data = data

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Return the elements for example index 'idx' as a dictionary with
        tensor values."""
        return {key: torch.tensor(val[idx]) for key, val in self.data.items()}

    def __len__(self) -> int:
        """Return the length of the data."""
        return len(self.data["input_ids"])


def create_semeval_sentiment_dataset(
    tokenizer: T5Tokenizer,
    file_name: str,
    shuffle: bool,
    repeat_input: bool = False,
    with_instructions: bool = False,
) -> DataLoader:
    """Function to create the required huggingface dataset to train the T5
    models on the semeval sentiment analysis task."""
    inputs, outputs, class_indices = read_semeval_sentiment_file(file_name, repeat_input, with_instructions)

    input_encodings = tokenizer(
        inputs,
        truncation=True,
        padding="max_length",
        max_length=FLAGS.source_max_length,
        add_special_tokens=False,
    )
    output_encodings = tokenizer(
        outputs,
        truncation=True,
        padding="max_length",
        max_length=FLAGS.decoder_max_length,
        add_special_tokens=False,
    )

    data = {
        "input_ids": input_encodings.input_ids,
        "attention_mask": input_encodings.attention_mask,
        "labels": output_encodings.input_ids,
        "target_attention_mask": output_encodings.attention_mask,
        "class_indices": class_indices,
    }

    dataloader = DataLoader(SentimentDataset(data), batch_size=FLAGS.batch_size, shuffle=shuffle)
    return dataloader
