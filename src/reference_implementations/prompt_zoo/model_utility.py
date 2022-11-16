"""Some utility functions useful during model training/inference."""

import csv
import gc
import io
import os
import random
from configparser import ConfigParser
from typing import Iterator, Optional

import numpy
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from src.reference_implementations.prompt_zoo.prompted_t5 import ConfigParameters, PromptedT5


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


def remove_prefix(text: str, prefix: str) -> str:
    """This function is used to remove prefix key from the text."""
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text  # or whatever


def torch_save(model: torch.nn.Module, path: str) -> None:
    """Save the model at the specified path."""
    torch.save(model.state_dict(), path)
    return


def save(model: torch.nn.Module, model_path: str, checkpoint_name: str) -> None:
    """Save the model to the specified path name along with a checkpoint
    name."""
    torch_save(model, model_path + "_" + checkpoint_name)
    return


def load_module(model: torch.nn.Module, model_path: str, checkpoint_name: str) -> None:
    """Load the model from the checkpoint."""
    loaded_weights = torch.load(
        model_path + checkpoint_name,
        map_location=lambda storage, loc: storage,
    )

    # sometimes the main model is wrapped inside a dataparallel or distdataparallel object.
    new_weights = {}
    for key, val in loaded_weights.items():
        new_weights[remove_prefix(key, "module.")] = val
    model.load_state_dict(new_weights)
    return


def clear_cache() -> None:
    """Clean unused GPU Cache!"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return


def start_training(
    model: PromptedT5, dataloader: torch.utils.data.DataLoader, exp_type: str = "all_fine_tune"
) -> Iterator[tuple[int, float]]:
    """Pick a batch from the dataloader, and train the model for one step."""
    step = 0
    for batch in dataloader:
        loss_values = model.train(batch, exp_type)
        step += 1
        yield step, loss_values["loss_value"]


def start_predicting(
    model: PromptedT5, dataloader: torch.utils.data.DataLoader, prediction_file: str, exp_type: str = "all_finetune"
) -> None:
    """Read batches from the dataloader and predict the outputs from the model
    for the correct experiment and save the results in the prediction_file as
    csv format row by row."""
    with io.open(prediction_file, mode="w", encoding="utf-8") as out_fp:
        writer = csv.writer(out_fp, quotechar='"', quoting=csv.QUOTE_ALL)
        header_written = False
        for batch in dataloader:
            for ret_row in model.predict(batch, exp_type):
                if not header_written:
                    headers = ret_row.keys()
                    writer.writerow(headers)
                    header_written = True
                writer.writerow(list(ret_row.values()))
    return


def save_config(config: ConfigParameters, path: str) -> None:
    """Saving ConfigParameters dataclass."""

    config_dict = vars(config)
    parser = ConfigParser()
    parser.add_section("config-parameters")
    for key, value in config_dict.items():
        parser.set("config-parameters", str(key), str(value))
    # save to a file
    with io.open(os.path.join(path, "config.ini"), mode="w", encoding="utf-8") as configfile:
        parser.write(configfile)


def run_model(
    model: PromptedT5,
    config: ConfigParameters,
    train_dataloader: Optional[torch.utils.data.DataLoader] = None,
    eval_dataloader: Optional[torch.utils.data.DataLoader] = None,
) -> None:
    """Run the model on input data; for training or inference."""

    model_path = config.model_path
    max_epochs = config.max_epochs
    max_train_steps = config.training_steps
    mode = config.mode
    writer = SummaryWriter(model_path)
    if mode == "train":
        epoch = 0
        while epoch < max_epochs:
            print("\nEpoch:{0}\n".format(epoch))
            total_loss = [0.0]  # total_loss for each epoch
            for step, loss in start_training(model, train_dataloader, config.t5_exp_type):
                total_loss.append(loss)
                mean_loss = np.mean(total_loss)

                print("\rEpoch:{0} | Batch:{1} | Mean Loss:{2} | Loss:{3}\n".format(epoch, step, mean_loss, loss))
                if step > 0 and (step % config.steps_per_checkpoint == 0):
                    model.save(str(epoch) + "_step_" + str(step))

                writer.add_scalar("Mean_Loss/train", mean_loss, step)
                writer.flush()
                if step + 1 == max_train_steps:
                    # stop training in this epoch.
                    break
            model.save(str(epoch))
            epoch += 1

        writer.close()
        save_config(config, model_path)
