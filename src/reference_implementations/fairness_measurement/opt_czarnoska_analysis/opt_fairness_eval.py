import os
import time
from typing import List, Tuple, Union

import kscope
import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

PATH_STUB = "src/reference_implementations/fairness_measurement/resources"
TEST_FILE_PATH = f"{PATH_STUB}/czarnowska_templates/sentiment_fairness_tests_subset.tsv"
# Append results to this file.
PREDICTION_FILE_PATH = f"{PATH_STUB}/predictions/opt_175_predictions.tsv"

MODEL = "opt-175b"
DATASET = "SST5"  # Labeled task-specific dataset
NUM_PARAMS = 175.0  # billions
RUN_ID = "run_1"
BATCH_SIZE = 10

# Initialize model here.
# Example: fine-tuned RoBERTa model via HuggingFace pipeline

# HuggingFace pipeline combining model and tokenizer.
client = kscope.Client(gateway_host="llm.cluster.local", gateway_port=3001)
print(f"Models Status: {client.model_instances}")
model = client.load_model("OPT-175B")
# If this model is not actively running, it will get launched in the background.
# In this case, wait until it moves into an "ACTIVE" state before proceeding.
while model.state != "ACTIVE":
    time.sleep(1)

# We're interested in the activations from the last layer of the model, because this will allow us to caculation the
# likelihoods
last_layer_name = model.module_names[-1]
last_layer_name

short_generation_config = {"max_tokens": 2, "top_k": 4, "top_p": 3, "rep_penalty": 1.2, "temperature": 1.0}

label_lookup = {
    "negative": 0,  # Negative
    "neutral": 1,  # Neutral
    "positive": 2,  # Positive
}  # Maps string labels to integers.

reverse_label_lookup = {label_int: label_str for label_str, label_int in label_lookup.items()}

number_of_demonstrations = 10

opt_tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")


def create_demonstrations() -> str:
    path = "src/reference_implementations/fairness_measurement/opt_czarnoska_analysis/resources/processed_sst5.tsv"
    sampled_df = pd.read_csv(path, sep="\t", header=0).sample(number_of_demonstrations)
    texts = sampled_df["Text"].tolist()
    valences = sampled_df["Valence"].apply(lambda x: int(x)).tolist()
    demonstrations = "Classify the sentiment of the text.\n\n"
    for text, valence in zip(texts, valences):
        demonstrations = f"{demonstrations}Text: {text} The sentiment is {reverse_label_lookup[valence]}.\n\n"
    print("Example of demonstrations")
    print("---------------------------------------------------------------------")
    print(demonstrations)
    print("---------------------------------------------------------------------")
    return demonstrations


def create_prompt_for_text(text: str, demonstrations: str) -> str:
    return f"{demonstrations}Text: {text} The sentiment is"


def create_prompts_for_batch(input_texts: List[str], demonstrations: str) -> List[str]:
    prompts = []
    for input_text in input_texts:
        prompts.append(create_prompt_for_text(input_text, demonstrations))
    return prompts


def get_label_with_highest_likelihood(
    layer_matrix: torch.Tensor,
    label_token_ids: torch.Tensor,
) -> int:
    # The activations we can about are the last token (corresponding to our label token) and the values for our label
    #  vocabulary
    label_activations = layer_matrix[-1][label_token_ids].float()
    softmax = nn.Softmax(dim=0)
    # Softmax is not strictly necessary, but it helps to contextualize the "probability" the model associates with each
    # label relative to the others
    label_distributions = softmax(label_activations)
    # We select the label index with the largest value
    max_label_index = torch.argmax(label_distributions)
    # We then map that index back tot the label string that we care about via the map provided.
    return max_label_index.item()


def get_predictions_batched(input_texts: List[str], demonstrations: str, label_token_ids: List[int]) -> List[int]:
    predicted_labels = []
    prompts = create_prompts_for_batch(input_texts, demonstrations)
    activations = model.get_activations(prompts, [last_layer_name], short_generation_config)
    for activations_single_prompt in activations.activations:
        # For each prompt we extract the activations and calculate which label had the high likelihood.
        last_layer_matrix = activations_single_prompt[last_layer_name]
        predicted_label = get_label_with_highest_likelihood(last_layer_matrix, label_token_ids)
        predicted_labels.append(predicted_label)
    assert len(predicted_labels) == len(input_texts)
    return predicted_labels


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
        tests.append((label, attribute, group, text))


batch: List[TestEntry] = []
text_batch: List[str] = []


def get_label_token_ids(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], prompt_template: str, label_words: List[str]
) -> List[int]:
    # Need to consider the token ids of our labels in the context of the prompt, as they may be different in context.
    tokenized_inputs = tokenizer(
        [f"{prompt_template} {label_word}" for label_word in label_words], return_tensors="pt"
    )["input_ids"]
    label_token_ids = tokenized_inputs[:, -1]
    return label_token_ids


test_batches = [tests[x : x + BATCH_SIZE] for x in range(0, len(tests), BATCH_SIZE)]
demonstrations = create_demonstrations()
label_token_ids = get_label_token_ids(
    opt_tokenizer, create_prompt_for_text("Dummy sentence.", demonstrations), ["negative", "neutral", "positive"]
)

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
    for batch in tqdm(test_batches):
        output: List[OutputEntry] = []
        text_batch = [test_case[-1] for test_case in batch]  # Extract texts from the batch.
        predictions = get_predictions_batched(text_batch, demonstrations, label_token_ids)

        for prediction, test_entry in zip(predictions, batch):
            label, attribute, group, text = test_entry
            output_entry = (prediction, label, attribute, group, text, MODEL, RUN_ID, DATASET, NUM_PARAMS)
            output.append(output_entry)

        output_lines = []
        for output_entry in output:
            output_lines.append(
                "\t".join(map(str, output_entry)) + "\n"
            )  # Convert integers to string before concatenating.

        prediction_file.writelines(output_lines)
        prediction_file.flush()
