import os
import random
import re
import time
from typing import List, Tuple

import kscope
import pandas as pd
from tqdm.auto import tqdm

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

PATH_STUB = "src/reference_implementations/fairness_measurement/resources"
TEST_FILE_PATH = f"{PATH_STUB}/czarnowska_templates/sentiment_fairness_tests_cut.tsv"
# Append results to this file.
PREDICTION_FILE_PATH = f"{PATH_STUB}/predictions/llama_2_7b_predictions_r2.tsv"

MODEL = "llama2-7b"
DATASET = "SST5"  # Labeled task-specific dataset
NUM_PARAMS: float = 7  # billions
RUN_ID = "run_1"
BATCH_SIZE = 10

client = kscope.Client(gateway_host="llm.cluster.local", gateway_port=3001)
print(f"Models Status: {client.model_instances}")
model = client.load_model(MODEL)
# # If this model is not actively running, it will get launched in the background.
# # In this case, wait until it moves into an "ACTIVE" state before proceeding.
while model.state != "ACTIVE":
    time.sleep(1)

# We're interested in the activations from the last layer of the model, because this will allow us to calculate the
# likelihoods
last_layer_name = model.module_names[-1]
last_layer_name

# For a discussion of the configuration parameters see:
# src/reference_implementations/prompting_vector_llms/CONFIG_README.md
short_generation_config = {"max_tokens": 5, "top_k": 1, "top_p": 1.0, "temperature": 1.0}

label_lookup = {
    "negative": 0,  # Negative
    "neutral": 1,  # Neutral
    "positive": 2,  # Positive
}  # Maps string labels to integers.

reverse_label_lookup = {label_int: label_str for label_str, label_int in label_lookup.items()}

number_of_demonstrations = 8
number_of_demonstrations_per_label = number_of_demonstrations // 3
number_of_random_demonstrations = number_of_demonstrations - number_of_demonstrations_per_label * 3


def create_demonstrations() -> str:
    path = "src/reference_implementations/fairness_measurement/czarnowska_analysis/resources/processed_sst5.tsv"
    df = pd.read_csv(path, sep="\t", header=0)
    # Trying to balance the number of labels represented in the demonstrations
    sample_df_0 = df.Valence[df.Valence.eq(0)].sample(number_of_demonstrations_per_label).index
    sample_df_1 = df.Valence[df.Valence.eq(1)].sample(number_of_demonstrations_per_label).index
    sample_df_2 = df.Valence[df.Valence.eq(2)].sample(number_of_demonstrations_per_label).index
    random_sampled_df = df.sample(number_of_random_demonstrations).index
    sampled_df = df.loc[sample_df_0.union(sample_df_1).union(sample_df_2).union(random_sampled_df)]
    texts = sampled_df["Text"].tolist()
    valences = sampled_df["Valence"].apply(lambda x: int(x)).tolist()
    demonstrations = ""
    for text, valence in zip(texts, valences):
        demonstrations = (
            f"{demonstrations}Text: {text}\nWhat is the sentiment of the text? {reverse_label_lookup[valence]}.\n\n"
        )
    print("Example of demonstrations")
    print("---------------------------------------------------------------------")
    print(demonstrations)
    print("---------------------------------------------------------------------")
    return demonstrations


def create_prompt_for_text(text: str, demonstrations: str) -> str:
    return f"{demonstrations}Text: {text}\nWhat is the sentiment of the text?"


def create_prompts_for_batch(input_texts: List[str], demonstrations: str) -> List[str]:
    prompts = []
    for input_text in input_texts:
        prompts.append(create_prompt_for_text(input_text, demonstrations))
    return prompts


def extract_predicted_label(sequence: str) -> str:
    match = re.search(r"positive|negative|neutral", sequence, flags=re.IGNORECASE)
    if match:
        return match.group().lower()
    else:
        # If no part of the generated response matches our label space, randomly choose one.
        print(f"Unable to match to a valid label in {sequence}")
        return random.choice(["positive", "negative", "neutral"])


def get_predictions_batched(input_texts: List[str], demonstrations: str) -> List[str]:
    predicted_labels = []
    prompts = create_prompts_for_batch(input_texts, demonstrations)
    batched_sequences = model.generate(prompts, short_generation_config).generation["sequences"]
    for prompt_sequence in batched_sequences:
        predicted_label = extract_predicted_label(prompt_sequence)
        predicted_labels.append(predicted_label)
    assert len(predicted_labels) == len(input_texts)
    return predicted_labels


tests: List[TestEntry] = []

with open(TEST_FILE_PATH, "r") as template_file:
    for line in template_file.readlines():
        label_str, attribute, group, text = tuple(line.rstrip().split("\t"))
        # convert the label string to an int
        label = int(label_str)
        tests.append((label, attribute, group, text))


batch: List[TestEntry] = []
text_batch: List[str] = []

test_batches = [tests[x : x + BATCH_SIZE] for x in range(0, len(tests), BATCH_SIZE)]
demonstrations = create_demonstrations()
example_prompt = create_prompt_for_text("I did not like that movie at all.", demonstrations)
print(f"Example Prompt\n{example_prompt}")

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
        predictions = get_predictions_batched(text_batch, demonstrations)

        for prediction, test_entry in zip(predictions, batch):
            label, attribute, group, text = test_entry
            output_entry = (
                label,
                label_lookup[prediction],
                attribute,
                group,
                text,
                MODEL,
                RUN_ID,
                DATASET,
                NUM_PARAMS,
            )
            output.append(output_entry)

        output_lines = []
        for output_entry in output:
            output_lines.append(
                "\t".join(map(str, output_entry)) + "\n"
            )  # Convert integers to string before concatenating.

        prediction_file.writelines(output_lines)
        prediction_file.flush()
