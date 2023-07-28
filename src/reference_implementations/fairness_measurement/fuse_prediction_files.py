import os
from typing import List, Tuple

semeval_tsv_location = (
    "/Users/david/Documents/SoftPromptTuningFairness/TACLRevisionsData/July_27_2023_SemEval_Experiments/"
    "predictions-eval-sweep-20230718a2.tsv"
)

old_predictions_txt_dir = (
    "/Users/david/Documents/SoftPromptTuningFairness/TACLRevisionsData/ACLSubmissionPredictions/preds/"
)


def read_tsv_filter_opt_predictions() -> List[str]:
    with open(semeval_tsv_location, "r") as f:
        lines = f.readlines()
        lines = [line for line in lines if "facebook/opt-125m" not in line]
        return lines


def parse_prediction_file_name(name: str) -> Tuple[str, str, str, str]:
    split_name = name.split("_")
    run_id = split_name[0]
    model_name = split_name[3]
    n_tokens = split_name[4]
    accuracy = split_name[5]
    return run_id, model_name, n_tokens, accuracy


def parse_old_pred_file_line(line: str) -> Tuple[str, str, str, str, str]:
    split_line = line.split("\t")
    y_pred = split_line[0]
    y_true = split_line[4]
    category = split_line[5]
    group = split_line[6]
    text = split_line[7].strip()
    return y_pred, y_true, category, group, text


def process_old_prediction_file(prediction_file: str) -> List[str]:
    run_id, model_name, n_tokens, accuracy = parse_prediction_file_name(prediction_file)
    accuracy = accuracy.replace(".txt", "")
    new_lines = []
    with open(os.path.join(old_predictions_txt_dir, prediction_file), "r") as f:
        lines = f.readlines()
        for line in lines:
            y_pred, y_true, category, group, text = parse_old_pred_file_line(line)
            new_line = (
                f"{y_true}\t{y_pred}\t{category}\t{group}\t{text}\tfacebook/{model_name}\t"
                f"{run_id}\tdata/processed/semeval_3\t{n_tokens}\t0.{accuracy}\t0.{accuracy}\n"
            )
            new_lines.append(new_line)
    return new_lines


old_prediction_files_to_process = [
    file_name for file_name in os.listdir(old_predictions_txt_dir) if file_name.endswith(".txt")
]
old_semeval_pred_files_to_process = [
    file_name for file_name in old_prediction_files_to_process if "_semeval_3_" in file_name
]

tsv_lines = read_tsv_filter_opt_predictions()
for old_pred_file in old_semeval_pred_files_to_process:
    tsv_lines.extend(process_old_prediction_file(old_pred_file))

out_path = "src/reference_implementations/fairness_measurement/fused_semeval_predictions.tsv"
with open(out_path, "w") as f:
    f.writelines(tsv_lines)
