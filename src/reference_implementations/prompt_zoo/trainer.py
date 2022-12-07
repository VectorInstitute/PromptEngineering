"""This is the main module to launch the training of the different experiments
with the T5 model.

The module also has some utility functions useful during model
training/inference.
"""

import csv
import io
import os
from typing import Any, Callable, Iterator, Optional, Tuple

import numpy as np
import torch
from absl import app, flags
from torch.utils.tensorboard import SummaryWriter

from src.reference_implementations.prompt_zoo.data_utility import create_semeval_sentiment_dataset
from src.reference_implementations.prompt_zoo.metrics import (
    semeval_classifier_sentiment_metric,
    semeval_sentiment_metric,
)
from src.reference_implementations.prompt_zoo.prompted_t5 import ClassifierT5, FineTuneT5, MyBaseT5

FLAGS = flags.FLAGS
flags.DEFINE_integer("max_epochs", 10, "The maximum number of epochs for training.")
flags.DEFINE_integer("training_steps", 100, "The number of training steps for each epoch.")
flags.DEFINE_integer("steps_per_checkpoint", 100, "keep checkpoint of the model every this number of steps")
flags.DEFINE_string("prediction_file", "/tmp/predictions.csv", "the path/name for saving the predictions.")
flags.DEFINE_string("dev_file", "/tmp/dev.csv", "the path/name of the dev file.")
flags.DEFINE_string("task_name", "semeval_3_class_sentiment", "the name of the downstream nlp task.")
flags.DEFINE_string("train_file", "/tmp/train.csv", "the path/name of the train file.")
flags.DEFINE_bool("with_instructions", False, "Whether to augment the input to have instructions or not.")


def start_training(model: MyBaseT5, dataloader: torch.utils.data.DataLoader) -> Iterator[Tuple[int, float]]:
    """Pick a batch from the dataloader, and train the model for one step."""
    step = 0
    for batch in dataloader:
        loss_values = model.train(batch)
        step += 1
        yield step, loss_values["loss_value"]


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


def run_model(
    model: MyBaseT5,
    train_dataloader: Optional[torch.utils.data.DataLoader] = None,
    eval_dataloader: Optional[torch.utils.data.DataLoader] = None,
    metric: Optional[Callable[[str, str], float]] = None,
) -> None:
    """Run the model on input data; for training or inference."""
    if FLAGS.mode == "train":
        writer = SummaryWriter(FLAGS.model_path)
        epoch = 0
        global_step = 0
        total_loss = []
        best_score = float("-inf")
        eval_file = os.path.join(FLAGS.model_path, "temp_eval.csv")
        while epoch < FLAGS.max_epochs:
            print("\nEpoch:{0}\n".format(epoch))
            epoch_loss = []
            for step, loss in start_training(model, train_dataloader):
                global_step += 1
                total_loss.append(loss)
                epoch_loss.append(loss)
                mean_total_loss = np.mean(total_loss)
                mean_epoch_loss = np.mean(epoch_loss)
                print(
                    "\rEpoch:{0} | Batch:{1} | Mean Loss:{2} | Epoch Loss:{3} | Loss:{4}\n".format(
                        epoch, step, mean_total_loss, mean_epoch_loss, loss
                    )
                )
                if global_step % FLAGS.steps_per_checkpoint == 0:
                    start_predicting(model, eval_dataloader, eval_file)
                    score = metric(FLAGS.dev_file, eval_file)  # type: ignore
                    writer.add_scalar("Score/dev", score, global_step)
                    if score > best_score:
                        best_score = score
                        model.save("best_step")

                writer.add_scalar("Mean_Total_Loss/train", mean_total_loss, global_step)
                writer.add_scalar("Mean_Epoch_Loss/train", mean_epoch_loss, global_step)
                writer.flush()
                if global_step == FLAGS.training_steps:
                    # stop training in this epoch.
                    break

            # do final evaluation on the dev data at the end of epoch.
            start_predicting(model, eval_dataloader, eval_file)
            score = metric(FLAGS.dev_file, eval_file)  # type: ignore
            writer.add_scalar("Score/dev", score, global_step)
            if score > best_score:
                best_score = score
                model.save("best_step")
            epoch += 1

        writer.close()

        # delete the eval_file
        os.remove(eval_file)

    if FLAGS.mode in ["test", "inference", "eval"]:
        print("Predicting...")
        start_predicting(model, eval_dataloader, FLAGS.prediction_file)


def launch_train() -> None:
    """launch the training phase for the prompting experiments without having
    the classifier on top."""

    FLAGS.mode = "train"

    model = FineTuneT5()
    if FLAGS.task_name == "semeval_3_class_sentiment":
        train_dataloader = create_semeval_sentiment_dataset(
            tokenizer=model.tokenizer,
            file_name=FLAGS.train_file,
            shuffle=True,
            repeat_input=False,
            with_instructions=FLAGS.with_instructions,
        )
        eval_dataloader = create_semeval_sentiment_dataset(
            tokenizer=model.tokenizer,
            file_name=FLAGS.dev_file,
            shuffle=False,
            repeat_input=True,
            with_instructions=FLAGS.with_instructions,
        )
        run_model(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            metric=semeval_sentiment_metric,
        )


def launch_classifier_train() -> None:
    """launch the training phase for the classifier over the T5 encoder."""

    FLAGS.mode = "train"
    model = ClassifierT5()
    if FLAGS.task_name == "semeval_3_class_sentiment":
        train_dataloader = create_semeval_sentiment_dataset(
            tokenizer=model.tokenizer,
            file_name=FLAGS.train_file,
            shuffle=True,
            repeat_input=False,
            with_instructions=FLAGS.with_instructions,
        )
        eval_dataloader = create_semeval_sentiment_dataset(
            tokenizer=model.tokenizer,
            file_name=FLAGS.dev_file,
            shuffle=False,
            repeat_input=False,
            with_instructions=FLAGS.with_instructions,
        )
        run_model(
            model=model,
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            metric=semeval_classifier_sentiment_metric,
        )
    return


def launch_no_finetune_predict() -> None:
    """launch the predict phase for the no prompting experiments without
    finetuning any parameters and only relying on the T5-base."""

    FLAGS.mode = "no_finetune_test"
    model = FineTuneT5()
    if FLAGS.task_name == "semeval_3_class_sentiment":
        eval_dataloader = create_semeval_sentiment_dataset(
            tokenizer=model.tokenizer,
            file_name=FLAGS.dev_file,
            shuffle=False,
            repeat_input=True,
            with_instructions=FLAGS.with_instructions,
        )
        run_model(
            model=model,
            train_dataloader=None,
            eval_dataloader=eval_dataloader,
            metric=None,
        )


def main(argv: Any) -> None:
    """Main function to switch over the t5 experiment type and launch the
    correct train script."""
    if FLAGS.t5_exp_type in [
        "soft_prompt_finetune",
        "all_finetune",
        "input_finetune",
        "output_finetune",
        "input_output_finetune",
    ]:
        launch_train()
    elif FLAGS.t5_exp_type == "no_finetune":
        launch_no_finetune_predict()
    elif FLAGS.t5_exp_type in ["classifier_finetune", "soft_prompt_classifier_finetune"]:
        launch_classifier_train()


if __name__ == "__main__":
    app.run(main)
