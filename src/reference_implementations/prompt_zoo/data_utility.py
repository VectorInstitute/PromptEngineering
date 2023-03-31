"""This module implements the functions for preprocessing the data files into
pytorch datasets."""

import random
from dataclasses import dataclass
from typing import Dict, List, Union

import pandas as pd
import torch
from absl import flags
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import T5Tokenizer

FLAGS = flags.FLAGS
flags.DEFINE_integer("train_batch_size", 16, "The batch size used for training.")
flags.DEFINE_integer("eval_batch_size", 2048, "The batch size used for inference on the test or validation data.")
flags.DEFINE_integer("source_max_length", 128, "The maximum number of tokens consider in the input sequence.")
flags.DEFINE_integer("decoder_max_length", 128, "The maximum number of tokens consider in the output sequence.")


@dataclass
class SentimentRawData:
    """Input/Output/Classes for sentiment classification raw data."""

    inputs: List[str]
    outputs: List[str]
    class_indices: List[int]
    gold_outputs: List[str]


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
    class_to_id: Dict[str, int],
    sentences: List[str],
    labels: List[str],
    instruction_type: str,
    repeat_input: bool,
) -> SentimentRawData:
    """Helper function to format the data for the models. The instruction_type
    could have 4 options:

        - "no_instruction": we just feed the input as is to the model.
        - "instruction_at_start": we include an instruction as the prefix to the input sentence.
        - "instruction_at_end": we inlcude an instruction as the suffix to the input sentence.
        - "qa": we convert the input to the question-answering format used by the T5 model.

    If the repeat_input is True, we will repeat the input multiple times for every possible output class.

    Finally, the end of sentence token </s> used with T5 models are added to both input and output.
    """
    if instruction_type == "qa":
        instruction = "what would be the sentiment of the sentence?"
        print(f"Prompt has the form: question: {instruction} context: input_text")
        sentences = [f"question: {instruction} context: {sent}" for sent in sentences]
    elif instruction_type == "instruction_at_start":
        instruction = "Generate the sentiment of the next sentence."
        print(f"Prompt has the form: {instruction} input_text")
        sentences = [f"{instruction} {sent}" for sent in sentences]
    elif instruction_type == "no_instruction":
        print("No instruction used, just input_text")
        sentences = sentences
    elif instruction_type == "instruction_at_end":
        instruction = "Generate the sentiment of the previous sentence."
        print(f"Prompt has the form: input_text {instruction}")
        sentences = [f"{sent} {instruction}" for sent in sentences]

    if repeat_input:
        # repeat every input for every possible output class.
        # the inference will compute the score for every possible
        # label and then select the label with the max score given by the LM.
        inputs = []
        outputs = []
        class_indices = []
        gold_outputs = []
        for idx, sent in enumerate(sentences):
            for label, index in class_to_id.items():
                inputs.append(f"{sent} </s>")
                outputs.append(f"{label} </s>")
                gold_outputs.append(labels[idx])
                class_indices.append(index)
        return SentimentRawData(inputs=inputs, outputs=outputs, class_indices=class_indices, gold_outputs=gold_outputs)

    # add end of sequence token:
    inputs = [f"{sent} </s>" for sent in sentences]
    outputs = [f"{label} </s>" for label in labels]
    class_indices = [class_to_id[label] for label in labels]

    return SentimentRawData(inputs=inputs, outputs=outputs, class_indices=class_indices, gold_outputs=labels)


def read_semeval_sentiment_file(
    file_path: str, instruction_type: str, repeat_input: bool = False, is_grips: bool = False
) -> SentimentRawData:
    """This function reads the semeval 2018 data files for sentiment analysis.

    Example header: 'ID  Tweet Affect Dimension  Intensity Class'
    """
    df = pd.read_csv(file_path, delimiter="\t")
    class_to_id = {"negative": 0, "neutral": 1, "positive": 2}
    tweets = [white_space_fix(tweet) for tweet in df["Tweet"].tolist()]
    sentiments = [preprocess_semeval_sentiment(sent) for sent in df["Intensity Class"].tolist()]

    if "train" in file_path.lower() and is_grips:
        # also shuffle dataset here.
        # required for grips experiments. for grips, we like to keep the repeated inputs together, therefore
        # we shuffle before creating the repeated rows. the dataloader shuffle for the grips experiment is set to
        # False.
        rand_indices = list(range(len(tweets)))
        random.shuffle(rand_indices)
        tweets = [tweets[i] for i in rand_indices]
        sentiments = [sentiments[i] for i in rand_indices]

    # the test data may have examples for some of the labels.
    assert set(sentiments).issubset({"positive", "negative", "neutral"})
    return template_data(class_to_id, tweets, sentiments, instruction_type, repeat_input)


def read_sst2_sentiment_file(
    split_name: str, instruction_type: str, repeat_input: bool = False, is_grips: bool = False
) -> SentimentRawData:
    """Load the sst2 sentiment analysis split for train, validation or test."""
    assert split_name in {"train", "validation", "test"}
    dataset = load_dataset("sst2", split=split_name)
    class_to_id = {"negative": 0, "positive": 1}

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

    if split_name == "train" and is_grips:
        # also shuffle dataset here.
        # required for grips experiments. for grips, we like to keep the repeated inputs together, therefore
        # we shuffle before creating the repeated rows. the dataloader shuffle for the grips experiment is set to
        # False.
        rand_indices = list(range(len(sentences)))
        random.shuffle(rand_indices)
        sentences = [sentences[i] for i in rand_indices]
        labels = [labels[i] for i in rand_indices]

    # the test data may only have examples with one label.
    assert set(labels).issubset({"positive", "negative"})
    return template_data(class_to_id, sentences, labels, instruction_type, repeat_input)


class SentimentDataset(Dataset):
    """Subclass the pytorch's Dataset to build my own dataset for the sentiment
    analysis task."""

    def __init__(self, data: Dict[str, Union[List[int], List[List[int]]]]) -> None:
        """store the reference to the tokenized data."""
        self.data = data

    def __getitem__(self, idx: int) -> Dict[str, Union[str, torch.Tensor]]:
        """Return the elements for example index 'idx' as a dictionary with
        tensor values if they are not strings."""
        ret: Dict[str, Union[str, torch.Tensor]] = {}
        for key, val in self.data.items():
            if isinstance(val[idx], str):
                ret[key] = val[idx]
            else:
                ret[key] = torch.tensor(val[idx])
        return ret

    def __len__(self) -> int:
        """Return the length of the data."""
        return len(self.data["input_ids"])


def create_sentiment_dataset(
    tokenizer: T5Tokenizer,
    file_name: str,
    task_name: str,
    shuffle: bool,
    instruction_type: str,
    repeat_input: bool = False,
) -> DataLoader:
    """Function to create the required huggingface dataset to train the T5
    models on the sentiment analysis task."""

    if task_name == "semeval":
        rawdata = read_semeval_sentiment_file(file_name, instruction_type, repeat_input, FLAGS.t5_exp_type == "grips")
    elif task_name == "sst2":
        rawdata = read_sst2_sentiment_file(file_name, instruction_type, repeat_input, FLAGS.t5_exp_type == "grips")
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
        "gold_classes": rawdata.gold_outputs,
    }

    if FLAGS.t5_exp_type == "grips":
        # repeat batch size since we are going to read the repeated inputs.
        FLAGS.eval_batch_size = len(set(rawdata.gold_outputs)) * FLAGS.eval_batch_size

    if shuffle:
        # this is training phase.
        dataloader = DataLoader(SentimentDataset(data), batch_size=FLAGS.train_batch_size, shuffle=True)
    else:
        # this is inference phase.
        dataloader = DataLoader(SentimentDataset(data), batch_size=FLAGS.eval_batch_size, shuffle=False)
    return dataloader
