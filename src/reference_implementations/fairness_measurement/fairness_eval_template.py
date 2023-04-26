import os
import sys
from typing import Dict, Iterable, List, Literal, Tuple
from torch import cuda
from tqdm.auto import tqdm
from transformers import pipeline


# Path to sentiment fairness test cases
TEMPLATE_NAME = sys.argv[3]

TEST_FILE_PATH = (
    f"src/reference_implementations/fairness_measurement/resources/race_templates/{TEMPLATE_NAME}_samples.tsv"
)

# Batch size.
BATCH_SIZE = 512

# HuggingFace Model to load for this demo.
# Feel free to delete this line if you replaced the fine-tuned RoBERTa model with
# LLM + prompt-tuning.
# DEMO_HF_MODEL = "src/reference_implementations/fairness_measurement/resources/fk-models/roberta-base_sst5-mapped-extreme_47"
# DEMO_HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
MODEL_NAME = sys.argv[1]
IDX = sys.argv[2]
DEMO_HF_MODEL = f"/h/fkohankh/fk-models/{MODEL_NAME}_sst5-mapped-extreme_{IDX}"

# Data entries below are used only for plotting the fairness diagrams.
MODEL = f"{MODEL_NAME} fine-tuned"
# DATASET = "TweetEval"  # Name of the labelled task-specific dataset
DATASET = "SST5"  # Name of the labelled task-specific dataset
NUM_PARAMS = 0.350  # billions
if MODEL_NAME == "opt-125m":
    NUM_PARAMS = 0.125
RUN_ID = f"r{IDX}"  # E.g., distinguishes between different random seeds.


# Append formatted predictions to this file.

PREDICTION_FILE_PATH = f"src/reference_implementations/fairness_measurement/resources/predictions/{TEMPLATE_NAME}_{MODEL_NAME}_{DATASET}.tsv"


# Initialize model here.
# Example: fine-tuned RoBERTa model via HuggingFace pipeline

# HuggingFace pipeline combining model and tokenizer.
device = 0 if cuda.is_available() else -1
device_str = "cuda" if device == 0 else "cpu"
print(f"Detected Device {device_str}")

sentiment_analysis_pipeline = pipeline("text-classification", model=DEMO_HF_MODEL, device=device)
label_lookup = {
    "LABEL_0": 0,  # Negative
    "LABEL_1": 1,  # Neutral
    "LABEL_2": 2,  # Positive
}  # Maps string labels to integers.


# Determine whether cuda and a GPU are available to speed up processing


def get_predictions_batched(input_texts: List[str]) -> Iterable[int]:
    """
    This function applies the ML model to predict the sentiment of the input texts.
    The input is a list of sentences, and this function is supposed to return
    a list/array of integer labels.

    Note that 0 is for negative emotions, 1 is for neutral, and 2 is for positive.

    As an example, the implementation below returns the predictions from a fine-tuned
    model loaded from the HuggingFace hub.

    You may want to modify this function to evaluate the fairness other
    models, APIs, and prompt-based sentiment analysis approaches.
    """
    model_output: List[Dict[Literal["label"], str]] = sentiment_analysis_pipeline(input_texts)  # type: ignore
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
        label_str, attribute, group, text = tuple(line.rstrip().split("\t"))
        # convert the label string to an int
        label = int(label_str)
        tests.append((label, attribute, group, text))  # type: ignore


batch: List[TestEntry] = []
text_batch: List[str] = []
output: List[OutputEntry] = []

test_batches = [tests[x : x + BATCH_SIZE] for x in range(0, len(tests), BATCH_SIZE)]
for batch in tqdm(test_batches):
    text_batch = [test_case[-1] for test_case in batch]  # Extract texts from the batch.
    predictions = get_predictions_batched(text_batch)

    for prediction, test_entry in zip(predictions, batch):
        label, attribute, group, text = test_entry
        output_entry = (label, prediction, attribute, group, text, MODEL, RUN_ID, DATASET, NUM_PARAMS)
        output.append(output_entry)  # type: ignore

# If the prediction file doesn't exist, we create a new one and append the tsv header row.
if not os.path.exists(PREDICTION_FILE_PATH):
    header_row = "\t".join(
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
    header_row = header_row + "\n"
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
