"""This is the main module to launch the training of the different experiments
with the T5 model.

The module also has some utility functions useful during model
training/inference.
"""

import csv
import io
import os
from typing import Any, Callable, Iterator, Tuple

import numpy as np
import torch
from absl import app, flags
from torch.utils.tensorboard import SummaryWriter

from src.reference_implementations.prompt_zoo.classifier_over_t5 import ClassifierT5
from src.reference_implementations.prompt_zoo.data_utility import create_sentiment_dataset
from src.reference_implementations.prompt_zoo.gradient_search import SearchT5
from src.reference_implementations.prompt_zoo.grips import GRIPSSearch
from src.reference_implementations.prompt_zoo.metrics import classifier_sentiment_metric, sentiment_metric
from src.reference_implementations.prompt_zoo.prompted_t5 import FineTuneT5, MyBaseT5

FLAGS = flags.FLAGS
flags.DEFINE_integer("max_epochs", 20, "The maximum number of epochs for training.")
flags.DEFINE_integer("training_steps", 100, "The number of training steps for each epoch.")
flags.DEFINE_integer("steps_per_checkpoint", 50, "keep checkpoint of the model every this number of steps")
flags.DEFINE_string("prediction_file", "/tmp/predictions.csv", "the path/name for saving the predictions.")
flags.DEFINE_string("dev_file", "/tmp/dev.csv", "the path/name of the dev file.")
flags.DEFINE_string("test_file", "/tmp/test.csv", "the path/name of the test file.")
flags.DEFINE_string("task_name", "semeval_3_class_sentiment", "the name of the downstream nlp task.")
flags.DEFINE_string("train_file", "/tmp/train.csv", "the path/name of the train file.")
flags.DEFINE_string("instruction_type", "qa", "The instruction type to format the input sentences.")


def start_predicting(model: MyBaseT5, dataloader: torch.utils.data.DataLoader, prediction_file: str) -> None:
    """Read batches from the dataloader and predict the outputs from the model
    for the correct experiment and save the results in the prediction_file as
    csv format row by row."""
    with io.open(prediction_file, mode="w", encoding="utf-8") as out_fp:
        writer = csv.writer(out_fp, quotechar='"', quoting=csv.QUOTE_ALL)
        header_written = False
        for batch in dataloader:
            for ret_row in model.predict(batch):
                if not header_written:
                    headers = ret_row.keys()
                    writer.writerow(headers)
                    header_written = True
                writer.writerow(list(ret_row.values()))
    return


def start_training(model: MyBaseT5, dataloader: torch.utils.data.DataLoader) -> Iterator[Tuple[int, float]]:
    """Pick a batch from the dataloader, and train the model for one step."""
    step = 0
    for batch in dataloader:
        loss_values = model.train(batch)
        step += 1
        yield step, loss_values["loss_value"]


def train_model(
    model: MyBaseT5,
    metric: Callable[[str, str, str], float],
    train_dataloader: torch.utils.data.DataLoader,
    eval_dataloader: torch.utils.data.DataLoader,
) -> None:
    """Run the model on input data; for training or inference."""
    if FLAGS.mode == "train":
        writer = SummaryWriter(FLAGS.model_path)
        epoch = 0
        global_step = 0
        total_loss = []
        best_score = float("-inf")
        eval_file = os.path.join(FLAGS.model_path, "temp_eval.csv")
        while epoch < FLAGS.max_epochs and global_step < FLAGS.training_steps:
            print("\nEpoch:{0}\n".format(epoch))
            epoch_loss = []

            # run prediction on the dev data to adjust the best_score.
            start_predicting(model, eval_dataloader, eval_file)
            score = metric(FLAGS.dev_file, eval_file, FLAGS.task_name)  # type: ignore
            writer.add_scalar("Score/dev", score, global_step)
            if score > best_score:
                best_score = score
                # default checkpoint name is "best_step".
                model.save()

            for step, loss in start_training(model, train_dataloader):
                global_step += 1
                total_loss.append(loss)
                epoch_loss.append(loss)
                mean_total_loss = np.mean(total_loss)
                mean_epoch_loss = np.mean(epoch_loss)
                print(
                    f"\rEpoch: {epoch} | Batch: {step} | Mean Loss: {mean_total_loss} | "
                    f"Epoch Loss: {mean_epoch_loss} | Loss: {loss}\n"
                )
                if global_step % FLAGS.steps_per_checkpoint == 0:
                    start_predicting(model, eval_dataloader, eval_file)
                    score = metric(FLAGS.dev_file, eval_file, FLAGS.task_name)  # type: ignore
                    writer.add_scalar("Score/dev", score, global_step)
                    if score > best_score:
                        best_score = score
                        # default checkpoint name is "best_step".
                        model.save()
                    elif score < best_score and FLAGS.t5_exp_type in ["gradient_search", "grips"]:
                        # re-load the best previous template searched so far!
                        # the previous templates was not good!
                        model.load_from_checkpoint()

                writer.add_scalar("Mean_Total_Loss/train", mean_total_loss, global_step)
                writer.add_scalar("Mean_Epoch_Loss/train", mean_epoch_loss, global_step)
                writer.flush()
                if global_step == FLAGS.training_steps:
                    # stop training in this epoch.
                    break

            # do final evaluation on the dev data at the end of epoch.
            start_predicting(model, eval_dataloader, eval_file)
            score = metric(FLAGS.dev_file, eval_file, FLAGS.task_name)  # type: ignore
            writer.add_scalar("Score/dev", score, global_step)
            if score > best_score:
                best_score = score
                model.save()
            elif score < best_score and FLAGS.t5_exp_type in ["gradient_search", "grips"]:
                # re-load the best previous template searched so far!
                # the previous templates was not good!
                model.load_from_checkpoint()

            epoch += 1

        writer.close()

        # delete the eval_file
        os.remove(eval_file)
    else:
        raise Exception(f"the mode {FLAGS.mode} is not for training.")


def test_model(
    model: MyBaseT5,
    metric: Callable[[str, str, str], float],
    test_dataloader: torch.utils.data.DataLoader,
) -> None:
    writer = SummaryWriter(FLAGS.model_path)
    if FLAGS.mode in ["test", "inference", "eval", "no_finetune_test"]:
        print("Predicting...")
        start_predicting(model, test_dataloader, FLAGS.prediction_file)
        score = metric(FLAGS.test_file, FLAGS.prediction_file, FLAGS.task_name)
        writer.add_scalar("Score", score, 0)
        print(f"The performance on the {FLAGS.test_file} is {score}")
    else:
        raise Exception(f"the mode {FLAGS.mode} is not for testing.")


def launch_test_or_train() -> None:
    """launch the testing or training phase for the prompting experiments
    without having the classifier on top."""

    if FLAGS.mode == "train":
        if FLAGS.t5_exp_type == "gradient_search":
            model = SearchT5()
        elif FLAGS.t5_exp_type == "grips":
            model = GRIPSSearch()
        else:
            model = FineTuneT5()
        if FLAGS.t5_exp_type == "grips":
            # For grips, we use train dataset as the search set and compute the balanced accuracy on it.
            # we should repeat the input for prediction and with set shuffle false to
            # keep the repeated inputs next to each other.
            train_dataloader = create_sentiment_dataset(
                tokenizer=model.tokenizer,
                file_name=FLAGS.train_file,
                task_name=FLAGS.task_name,
                shuffle=False,
                instruction_type=FLAGS.instruction_type,
                repeat_input=True,
            )
        else:
            # for other experiments, we don't repeat the inputs as the loss will be
            # calculated based on the gold output.
            train_dataloader = create_sentiment_dataset(
                tokenizer=model.tokenizer,
                file_name=FLAGS.train_file,
                task_name=FLAGS.task_name,
                shuffle=True,
                instruction_type=FLAGS.instruction_type,
                repeat_input=False,
            )
        eval_dataloader = create_sentiment_dataset(
            tokenizer=model.tokenizer,
            file_name=FLAGS.dev_file,
            task_name=FLAGS.task_name,
            shuffle=False,
            instruction_type=FLAGS.instruction_type,
            repeat_input=True,
        )
        train_model(
            model=model, metric=sentiment_metric, train_dataloader=train_dataloader, eval_dataloader=eval_dataloader
        )
    elif FLAGS.mode in ["test", "inference", "no_finetune_test"]:
        if FLAGS.t5_exp_type == "gradient_search":
            model = SearchT5()
        elif FLAGS.t5_exp_type == "grips":
            model = GRIPSSearch()
        else:
            model = FineTuneT5()
        test_dataloader = create_sentiment_dataset(
            tokenizer=model.tokenizer,
            file_name=FLAGS.test_file,
            task_name=FLAGS.task_name,
            shuffle=False,
            instruction_type=FLAGS.instruction_type,
            repeat_input=True,
        )
        test_model(model=model, metric=sentiment_metric, test_dataloader=test_dataloader)


def launch_classifier_train() -> None:
    """launch the training phase for the classifier over the T5 encoder."""

    FLAGS.mode = "train"
    model = ClassifierT5()
    train_dataloader = create_sentiment_dataset(
        tokenizer=model.tokenizer,
        file_name=FLAGS.train_file,
        task_name=FLAGS.task_name,
        shuffle=True,
        instruction_type=FLAGS.instruction_type,
        repeat_input=False,
    )
    eval_dataloader = create_sentiment_dataset(
        tokenizer=model.tokenizer,
        file_name=FLAGS.dev_file,
        task_name=FLAGS.task_name,
        shuffle=False,
        instruction_type=FLAGS.instruction_type,
        repeat_input=False,
    )
    train_model(
        model=model,
        metric=classifier_sentiment_metric,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )


def launch_no_finetune_predict() -> None:
    """launch the predict phase for the no prompting experiments without
    finetuning any parameters and only relying on the T5 model."""

    FLAGS.mode = "no_finetune_test"
    model = FineTuneT5()
    eval_dataloader = create_sentiment_dataset(
        tokenizer=model.tokenizer,
        file_name=FLAGS.test_file,
        task_name=FLAGS.task_name,
        shuffle=False,
        instruction_type=FLAGS.instruction_type,
        repeat_input=True,
    )
    test_model(
        model=model,
        metric=sentiment_metric,
        test_dataloader=eval_dataloader,
    )


def main(argv: Any) -> None:
    """Main function to switch over the t5 experiment type and launch the
    correct train script."""
    if FLAGS.t5_exp_type in [
        "soft_prompt_finetune",
        "all_finetune",
        "input_finetune",
        "output_finetune",
        "gradient_search",
        "grips",
    ]:
        launch_test_or_train()
    elif FLAGS.t5_exp_type == "no_finetune":
        launch_no_finetune_predict()
    elif FLAGS.t5_exp_type == "classifier_finetune":
        launch_classifier_train()


if __name__ == "__main__":
    app.run(main)
