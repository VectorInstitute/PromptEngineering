"""This module implements the functions for preprocessing the data files into
pytorch datasets."""

from dataclasses import dataclass
from typing import Dict, List, Union

import pandas as pd
import torch
from absl import flags
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer

FLAGS = flags.FLAGS
flags.DEFINE_integer("batch_size", 16, "The batch size used for training or inference.")
flags.DEFINE_integer("source_max_length", 128, "The maximum number of tokens consider in the input sequence.")
flags.DEFINE_integer("decoder_max_length", 128, "The maximum number of tokens consider in the output sequence.")


@dataclass
class SentimentRawData:
    """Input/Output/Classes for sentiment classification raw data."""

    inputs: List[str]
    outputs: List[str]
    class_indices: List[int]


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


def template_data(
    all_classes: List[str], sentences: List[str], labels: List[str], with_instructions: bool, repeat_input: bool
) -> SentimentRawData:
    """Helper function to format the data for the models.

    if with_instructions is True, we will add an instruction to the input sentence
    and make the input a template with special keywords "instructions:", "sentence:", and "sentiment:".

    if the repeat_input is True, we will repeat the input multiple times for every possible output class.

    Finally, the end of sentence token </s> used with T5 models are added to both input and output.
    """
    class_to_id = {label: index for index, label in enumerate(all_classes)}
    if with_instructions:
        instruction = "what would be the sentiment of the sentence?"
        sentences = [f"question: {instruction} context: {sent}" for sent in sentences]
    else:
        sentences = [f"context: {sent}" for sent in sentences]

    if repeat_input:
        # repeat every input for every possible output class.
        # the inference will compute the score for every possible
        # label and then select the label with the max score given by the LM.
        inputs = []
        outputs = []
        class_indices = []
        for sent in sentences:
            for label in all_classes:
                inputs.append(f"{sent} </s>")
                outputs.append(f"{label} </s>")
                class_indices.append(class_to_id[label])
        return SentimentRawData(inputs=inputs, outputs=outputs, class_indices=class_indices)

    # add end of sequence token:
    inputs = [f"{sent} </s>" for sent in sentences]
    outputs = [f"{label} </s>" for label in labels]
    class_indices = [class_to_id[label] for label in labels]
    return SentimentRawData(inputs=inputs, outputs=outputs, class_indices=class_indices)


def read_semeval_sentiment_file(
    file_path: str, repeat_input: bool = False, with_instructions: bool = False
) -> SentimentRawData:
    """This function reads the semeval 2018 data files for sentiment analysis.

    Example header: 'ID  Tweet Affect Dimension  Intensity Class'
    """
    df = pd.read_csv(file_path, delimiter="\t")
    tweets = [white_space_fix(tweet) for tweet in df["Tweet"].tolist()]
    sentiments = [preprocess_semeval_sentiment(sent) for sent in df["Intensity Class"].tolist()]

    all_classes = set(sentiments)
    assert all_classes.issubset({"positive", "negative", "neutral"})
    return template_data(list(all_classes), tweets, sentiments, with_instructions, repeat_input)


def read_sst2_sentiment_file(
    split_name: str, repeat_input: bool = False, with_instructions: bool = False
) -> SentimentRawData:
    """Load the sst2 sentiment analysis split for train, validation or test."""
    assert split_name in {"train", "validation", "test"}
    dataset = load_dataset("sst2", split=split_name)

    def process_row(row: Dict[str, str]) -> Dict[str, str]:
        """Helper function to process each row of the dataset."""
        label = "negative" if str(row["label"]) == "0" else "positive"
        return {"sentence": white_space_fix(row["sentence"]), "sentiment": label}

    new_dataset = dataset.map(
        process_row,
        remove_columns=["idx", "label"],
    )

    sentences = []
    labels = []
    for row in new_dataset:
        sentences.append(row["sentence"])
        labels.append(row["sentiment"])

    all_classes = sorted(list(set(labels)))
    return template_data(all_classes, sentences, labels, with_instructions, repeat_input)


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


def create_sentiment_dataset(
    tokenizer: T5Tokenizer,
    file_name: str,
    task_name: str,
    shuffle: bool,
    repeat_input: bool = False,
    with_instructions: str = "False",
) -> DataLoader:
    """Function to create the required huggingface dataset to train the T5
    models on the sentiment analysis task."""

    # to fix bug with boolean arguments.
    with_instruct = with_instructions == "True"
    if task_name == "semeval":
        rawdata = read_semeval_sentiment_file(file_name, repeat_input, with_instruct)
    elif task_name == "sst2":
        rawdata = read_sst2_sentiment_file(file_name, repeat_input, with_instruct)
    else:
        raise Exception(f"this {task_name} is not supported!")

    input_encodings = tokenizer(
        rawdata.inputs,
        truncation=True,
        padding="max_length",
        max_length=FLAGS.source_max_length,
        add_special_tokens=False,
    )
    output_encodings = tokenizer(
        rawdata.outputs,
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
        "class_indices": rawdata.class_indices,
    }

    dataloader = DataLoader(SentimentDataset(data), batch_size=FLAGS.batch_size, shuffle=shuffle)
    return dataloader
