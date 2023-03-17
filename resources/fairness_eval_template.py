from typing import List, Iterable, Tuple, Dict, Literal
import os
from tqdm.auto import tqdm

TEST_FILE_PATH = "datasets/sentiment_fairness_tests.tsv"
PREDICTION_FILE_PATH = "predictions.tsv"  # Append results to this file.
MODEL = "roberta-base"
DATASET = "TweetEval"  # Labeled task-specific dataset
NUM_PARAMS = 0.125  # billions
RUN_ID = "example_run_a"
BATCH_SIZE = 8

"""
Initialize model here.

Example: fine-tuned RoBERTa model via HuggingFace pipeline
"""
from transformers import pipeline

# HuggingFace pipeline combining model and tokenizer.
pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")
label_lookup = {
    "LABEL_0": 0,  # Negative
    "LABEL_1": 1,  # Neutral
    "LABEL_2": 2,  # Positive
}  # Maps string labels to integers.


def get_predictions_batched(input_texts: List[str]) -> Iterable[int]:  # type: ignore
    """
    Your Code Here.

    This function takes in a list ("batch") of sentences and return
    an iterable of integers (a list, an array, etc.)
    """
    model_output: List[Dict[Literal["label"], str]] = pipe(input_texts)  # type: ignore
    # [{"label": "LABEL_0", "score": 0.7}, {"label": "LABEL_0", "score": 0.8}]

    predictions = [label_lookup[prediction["label"]] for prediction in model_output]
    assert len(predictions) == len(input_texts)
    return predictions


TrueLabel = int
PredictedLabel = int
Category = str
Group = str
TestText = str
Model = str
RunID = str
Dataset = str
NumParams = float
TestEntry = Tuple[TrueLabel, Category, Group, TestText]
OutputEntry = Tuple[
    PredictedLabel,
    TrueLabel,
    Category,
    Group,
    TestText,
    Model,
    RunID,
    Dataset,
    NumParams,
]

tests: List[TestEntry] = []

with open(TEST_FILE_PATH, "r") as template_file:
    for line in template_file.readlines():
        test_entry = tuple(line.rstrip().split("\t"))
        assert len(test_entry) > 1, "Test input must be tab-delimited."
        tests.append(test_entry)  # type: ignore


batch: List[TestEntry] = []
text_batch: List[str] = []
output: List[OutputEntry] = []

for test_entry in tqdm(tests):
    batch.append(test_entry)
    text_batch.append(test_entry[-1])

    if len(batch) == BATCH_SIZE:
        predictions = get_predictions_batched(text_batch)

        for prediction, test_entry in zip(predictions, batch):
            output_entry = (prediction, *test_entry, MODEL, RUN_ID, DATASET, NUM_PARAMS)
            output.append(output_entry)

        batch = []
        text_batch = []

# Handle tailing entries.
if len(batch) > 0:
    predictions = get_predictions_batched(text_batch)

    for prediction, test_entry in zip(predictions, batch):
        output_entry = (prediction, *test_entry, MODEL, RUN_ID, DATASET, NUM_PARAMS)
        output.append(output_entry)

if not os.path.exists(PREDICTION_FILE_PATH):
    header_row = (
        "\t".join(
            [
                "y_true",
                "y_pred",
                "category",
                "group",
                "text",
                "model",
                "run_id",
                "dataset",
                "num_params",
            ]
        )
        + "\n"
    )
    with open(PREDICTION_FILE_PATH, "w") as prediction_file:
        prediction_file.write(header_row)

# Append to the output file instead of overwriting.
with open(PREDICTION_FILE_PATH, "a") as prediction_file:
    output_lines = []
    for output_entry in output:
        output_lines.append(
            "\t".join(map(str, output_entry)) + "\n"
        )  # Convert integers to string before concatenating.

    prediction_file.writelines(output_lines)
