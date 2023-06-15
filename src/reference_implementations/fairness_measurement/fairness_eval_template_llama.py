import os
import sys
from typing import Dict, Iterable, List, Literal, Tuple
from torch import cuda, nn
import torch
from tqdm.auto import tqdm
from transformers import pipeline
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM


MODEL_NAME = sys.argv[1]
IDX = sys.argv[2]
TEMPLATE_NAME = sys.argv[3]

# Path to sentiment fairness test cases
TEST_FILE_PATH = (
    f"src/reference_implementations/fairness_measurement/resources/race_templates/{TEMPLATE_NAME}_samples.tsv"
)

# Batch size.
BATCH_SIZE = 16

# HuggingFace Model to load for this demo.
# Feel free to delete this line if you replaced the fine-tuned RoBERTa model with
# LLM + prompt-tuning.
# DEMO_HF_MODEL = "src/reference_implementations/fairness_measurement/resources/fk-models/roberta-base_sst5-mapped-extreme_47"
# DEMO_HF_MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
DEMO_HF_MODEL = f"/h/fkohankh/fk-models/{MODEL_NAME}_sst5-mapped-extreme_{IDX}"

# Data entries below are used only for plotting the fairness diagrams.
MODEL = f"{MODEL_NAME} fine-tuned"
# DATASET = "TweetEval"  # Name of the labelled task-specific dataset
DATASET = "SST5"  # Name of the labelled task-specific dataset
NUM_PARAMS = 0.350  # billions
if MODEL_NAME == "opt-125m":
    NUM_PARAMS = 0.125
if MODEL_NAME == "llama-7b":
    NUM_PARAMS = 7
RUN_ID = f"r{IDX}"  # E.g., distinguishes between different random seeds.

# Append formatted predictions to this file.

PREDICTION_FILE_PATH = f"src/reference_implementations/fairness_measurement/resources/predictions/{TEMPLATE_NAME}_{MODEL_NAME}_{DATASET}.tsv"


# Initialize model here.
# Example: fine-tuned RoBERTa model via HuggingFace pipeline

device = 0 if cuda.is_available() else -1
device_str = "cuda" if device == 0 else "cpu"
print(f"Detected Device {device_str}")

causal_model = AutoModelForCausalLM.from_pretrained(DEMO_HF_MODEL)
causal_model.to(device)
tokenizer = AutoTokenizer.from_pretrained(DEMO_HF_MODEL)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.bos_token

print("Loaded model.")


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
    # model_output: List[Dict[Literal["label"], str]] = sentiment_analysis_pipeline(input_texts, max_length=1)  # type: ignore


    tokenizer_output = tokenizer(input_texts, return_tensors='pt', padding=True)
    tokenizer_output.pop("token_type_ids")
    tokenizer_output = {k: v.to(device=device, non_blocking=True) for k, v in tokenizer_output.items()}

    raw_logits = causal_model(**tokenizer_output).logits
    classification_logits = raw_logits[:, -1, [8178, 21104, 5374]]

    labels = classification_logits.argmax(-1)

    predictions = labels.tolist()
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

print("Creating tests.")
tests: List[TestEntry] = []

with open(TEST_FILE_PATH, "r") as template_file:
    for line in template_file.readlines():
        label_str, attribute, group, text = tuple(line.rstrip().split("\t"))
        # convert the label string to an int
        label = int(label_str)
        tests.append((label, attribute, group, text))  # type: ignore

print("Tests are created.")

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
