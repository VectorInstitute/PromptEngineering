"""This module implements different metrics used to evaluate the T5 predictions
for the downstream tasks."""

import pandas as pd

from src.reference_implementations.prompt_zoo.data_utility import read_semeval_sentiment_file


def semeval_sentiment_metric(gold_file: str, prediction_file: str) -> float:
    """Compute the classification accuracy for semeval sentiment
    classification."""

    _, gold_labels = read_semeval_sentiment_file(gold_file)
    gold_labels = [label.strip(" </s>") for label in gold_labels]
    df = pd.read_csv(prediction_file, delimiter=",")
    prediction_labels = df["prediction"].tolist()
    corrects = 0.0
    total = 0.0
    for index, pred in enumerate(prediction_labels):
        total += 1.0
        if gold_labels[index] == pred:
            corrects += 1.0
    return corrects / total
