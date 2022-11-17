"""Some utility functions useful during model training/inference."""

import csv
import gc
import io
import random
from typing import Iterator, Optional

import numpy
import numpy as np
import torch
from absl import flags
from torch.utils.tensorboard import SummaryWriter

from src.reference_implementations.prompt_zoo.prompted_t5 import PromptedT5

FLAGS = flags.FLAGS
flags.DEFINE_integer("max_epochs", 10, "The maximum number of epochs for training.")
flags.DEFINE_integer("training_steps", 100, "The number of training steps for each epoch.")
flags.DEFINE_integer("steps_per_checkpoint", 100, "keep checkpoint of the model every this number of steps")
flags.DEFINE_string("prediction_file", "/tmp/predictions.csv", "the path/name for saving the predictions.")


def set_random_seed(seed: int) -> None:
    """Set the random seed, which initializes the random number generator.

    Ensures that runs are reproducible and eliminates differences due to
    randomness.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return


def clear_cache() -> None:
    """Clean unused GPU Cache!"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return


def start_training(model: PromptedT5, dataloader: torch.utils.data.DataLoader) -> Iterator[tuple[int, float]]:
    """Pick a batch from the dataloader, and train the model for one step."""
    step = 0
    for batch in dataloader:
        loss_values = model.train(batch)
        step += 1
        yield step, loss_values["loss_value"]


def start_predicting(model: PromptedT5, dataloader: torch.utils.data.DataLoader, prediction_file: str) -> None:
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
    model: PromptedT5,
    train_dataloader: Optional[torch.utils.data.DataLoader] = None,
    eval_dataloader: Optional[torch.utils.data.DataLoader] = None,
) -> None:
    """Run the model on input data; for training or inference."""
    if FLAGS.mode == "train":
        writer = SummaryWriter(FLAGS.model_path)
        epoch = 0
        while epoch < FLAGS.max_epochs:
            print("\nEpoch:{0}\n".format(epoch))
            total_loss = [0.0]  # total_loss for each epoch
            for step, loss in start_training(model, train_dataloader):
                total_loss.append(loss)
                mean_loss = np.mean(total_loss)

                print("\rEpoch:{0} | Batch:{1} | Mean Loss:{2} | Loss:{3}\n".format(epoch, step, mean_loss, loss))
                if step > 0 and (step % FLAGS.steps_per_checkpoint == 0):
                    model.save(str(epoch) + "_step_" + str(step))

                writer.add_scalar("Mean_Loss/train", mean_loss, step)
                writer.flush()
                if step + 1 == FLAGS.training_steps:
                    # stop training in this epoch.
                    break
            model.save(str(epoch))
            epoch += 1

        writer.close()
    if FLAGS.mode in ["test", "inference", "eval"]:
        print("Predicting...")
        start_predicting(model, eval_dataloader, FLAGS.prediction_file)
