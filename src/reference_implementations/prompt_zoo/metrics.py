"""This module implements different metrics used to evaluate the T5 predictions
for the downstream tasks."""

import numpy as np
import pandas as pd

from src.reference_implementations.prompt_zoo.data_utility import read_semeval_sentiment_file


def semeval_sentiment_metric(gold_file: str, prediction_file: str) -> float:
    """Compute the classification accuracy for semeval sentiment
    classification."""

    _, gold_labels, _ = read_semeval_sentiment_file(gold_file, repeat_input=False, with_instructions=False)
    gold_labels = [label.strip(" </s>") for label in gold_labels]

    # pick the class with the highest score among the possible class labels!
    num_labels = len(set(gold_labels))
    df = pd.read_csv(prediction_file, delimiter=",")
    predictions = [label.strip(" </s>") for label in df["potential_class"].tolist()]
    scores = df["prediction_score"].tolist()
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


def semeval_classifier_sentiment_metric(gold_file: str, prediction_file: str) -> float:
    """Compute the classification accuracy for semeval sentiment classification
    where we have classifier on top of the T5 encoder compared to generation of
    the classes in the decoder."""

    _, _, class_indices = read_semeval_sentiment_file(gold_file, repeat_input=False, with_instructions=False)
    df = pd.read_csv(prediction_file, delimiter=",")
    prediction_indices = df["predicted_class"].tolist()

    corrects = 0.0
    total = 0.0
    for index, gold in enumerate(class_indices):
        total += 1.0
        if gold == prediction_indices[index]:
            corrects += 1.0
    return corrects / total
