from typing import Dict, List

import numpy as np
from sklearn.metrics import confusion_matrix


def map_ag_news_int_labels(raw_labels: List[str], int_to_string_map: Dict[int, str]) -> List[str]:
    return [int_to_string_map[int(raw_label)] for raw_label in raw_labels]


def report_metrics(preds: List[str], labels: List[str], labels_order: List[str]) -> None:
    # The label ordering just fixes the order of the labels in the confusion matrix. The default is the labels
    # associated with the AG News dataset.
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
