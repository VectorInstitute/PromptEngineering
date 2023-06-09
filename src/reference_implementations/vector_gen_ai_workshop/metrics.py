from random import sample
from typing import List

import numpy as np
from sklearn.metrics import confusion_matrix


def map_predictions_to_labels(preds: List[str], label_set: List[str]) -> List[str]:
    # If prediction matches a label, we use it. Otherwise we select a random label
    mapped_preds = []
    for pred in preds:
        stripped_pred = pred.strip().lower()
        if stripped_pred in label_set:
            mapped_preds.append(stripped_pred)
        else:
            print(f"{stripped_pred} does not match any labels in {label_set}")
            mapped_preds.append(sample(label_set, 1)[0])

    return mapped_preds


def report_metrics(preds: List[str], labels: List[str], labels_order: List[str]) -> None:
    # Preprocess the predictions and labels to be lowercase to help with matching
    labels = [label.lower() for label in labels]
    labels_order = [label.lower() for label in labels_order]
    preds = map_predictions_to_labels(preds, labels_order)

    # The label ordering just fixes the order of the labels in the confusion matrix.
    matrix = confusion_matrix(labels, preds, labels=labels_order)
    FP = matrix.sum(axis=0) - np.diag(matrix)
    FN = matrix.sum(axis=1) - np.diag(matrix)
    TP = np.diag(matrix)

    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1 = 2 * (precision * recall) / (precision + recall)

    print(f"Prediction Accuracy: {TP.sum()/(matrix.sum())}")
    print(f"Confusion Matrix with ordering {labels_order}")
    print(matrix)
    print("========================================================")
    for label_index, label_name in enumerate(labels_order):
        print(
            f"Label: {label_name}, F1: {f1[label_index]}, Precision: {recall[label_index]}, "
            f"Recall: {precision[label_index]}"
        )
