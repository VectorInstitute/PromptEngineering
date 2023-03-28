"""This module implements different metrics used to evaluate the T5 predictions
for the downstream tasks."""

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn.metrics import balanced_accuracy_score

from src.reference_implementations.prompt_zoo.data_utility import read_semeval_sentiment_file, read_sst2_sentiment_file


def sentiment_metric(gold_file: str, prediction_file: str, task_name: str) -> float:
    """Compute the classification accuracy for sentiment classification."""

    if task_name == "semeval":
        rawdata = read_semeval_sentiment_file(gold_file, instruction_type="None", repeat_input=False)
    elif task_name == "sst2":
        rawdata = read_sst2_sentiment_file(gold_file, instruction_type="None", repeat_input=False)
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


def classifier_sentiment_metric(gold_file: str, prediction_file: str, task_name: str) -> float:
    """Compute the classification accuracy for sentiment classification where
    we have classifier on top of the T5 encoder compared to generation of the
    classes in the decoder."""

    if task_name == "semeval":
        rawdata = read_semeval_sentiment_file(gold_file, instruction_type="None", repeat_input=False)
    elif task_name == "sst2":
        rawdata = read_sst2_sentiment_file(gold_file, instruction_type="None", repeat_input=False)
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


def grips_sentiment_metric(prediction_file: str) -> float:
    """Compute the balanced accuracy + entropy for sentiment classification
    used in grips training."""
    # pick the class with the highest score among the possible class labels!
    df = pd.read_csv(prediction_file, delimiter=",")
    gold_labels = [label.strip() for label in df["gold_class"].tolist()]
    num_labels = len(set(gold_labels))
    golds = np.array(gold_labels).reshape((len(gold_labels) // num_labels, num_labels))[:, 0]

    # This relies on the assumption that there is a prediction score for every label. (i.e. n label scores per input)
    predictions = [label.strip(" </s>") for label in df["potential_class"].tolist()]
    scores = df["prediction_score"].tolist()

    assert len(predictions) % num_labels == 0
    prediction_labels = np.array(predictions).reshape((len(predictions) // num_labels, num_labels))
    prediction_scores = np.array(scores).reshape((len(predictions) // num_labels, num_labels))
    max_predictions = np.argmax(prediction_scores, axis=1)
    max_labels = prediction_labels[:, max_predictions][0]

    per_label_correct = {g_label: 0 for g_label in list(set(gold_labels))}
    total = 0.0
    for index, gold in enumerate(golds):
        total += 1.0
        if gold == max_labels[index]:
            per_label_correct[gold] += 1

    per_label_frequencies = [count / total for count in per_label_correct.values()]
    balanced_acc = balanced_accuracy_score(y_true=golds, y_pred=max_labels)

    # 10 is a factor used in the grips implementation.
    return np.round(100 * balanced_acc, 2) + 10 * entropy(np.array(per_label_frequencies))
