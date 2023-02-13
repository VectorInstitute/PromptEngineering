"""This module implements different metrics used to evaluate the T5 predictions
for the downstream tasks."""

import numpy as np
import pandas as pd

from src.reference_implementations.prompt_zoo.data_utility import read_semeval_sentiment_file, read_sst2_sentiment_file


def sentiment_metric(gold_file: str, prediction_file: str, task_name: str, intruction_type: str) -> float:
    """Compute the classification accuracy for sentiment classification."""

    if task_name == "semeval":
        rawdata = read_semeval_sentiment_file(gold_file, intruction_type, repeat_input=False)
    elif task_name == "sst2":
        rawdata = read_sst2_sentiment_file(gold_file, intruction_type, repeat_input=False)
    else:
        raise Exception(f"this {task_name} is not supported!")

    gold_labels = rawdata.outputs
    gold_labels = [label.strip(" </s>") for label in gold_labels]

    # pick the class with the highest score among the possible class labels!
    num_labels = len(set(gold_labels))
    df = pd.read_csv(prediction_file, delimiter=",")

    # This relies on the assumption that there is a prediction score for every label. (i.e. n label scores per input)
    predictions = [label.strip(" </s>") for label in df["potential_class"].tolist()]
    scores = df["prediction_score"].tolist()

    assert len(predictions) % num_labels == 0
    prediction_labels = np.array(predictions).reshape((len(predictions) // num_labels, num_labels))
    prediction_scores = np.array(scores).reshape((len(predictions) // num_labels, num_labels))
    max_predictions = np.argmax(prediction_scores, axis=1)
    max_labels = prediction_labels[:, max_predictions][0]

    corrects = 0.0
    total = 0.0
    for index, gold in enumerate(gold_labels):
        total += 1.0
        if gold == max_labels[index]:
            corrects += 1.0
    return corrects / total


def classifier_sentiment_metric(gold_file: str, prediction_file: str, task_name: str, intruction_type: str) -> float:
    """Compute the classification accuracy for sentiment classification where
    we have classifier on top of the T5 encoder compared to generation of the
    classes in the decoder."""

    if task_name == "semeval":
        rawdata = read_semeval_sentiment_file(gold_file, intruction_type, repeat_input=False)
    elif task_name == "sst2":
        rawdata = read_sst2_sentiment_file(gold_file, intruction_type, repeat_input=False)
    else:
        raise Exception(f"this {task_name} is not supported!")

    df = pd.read_csv(prediction_file, delimiter=",")
    prediction_indices = df["predicted_class"].tolist()

    corrects = 0.0
    total = 0.0
    for index, gold in enumerate(rawdata.class_indices):
        total += 1.0
        if gold == prediction_indices[index]:
            corrects += 1.0
    return corrects / total
