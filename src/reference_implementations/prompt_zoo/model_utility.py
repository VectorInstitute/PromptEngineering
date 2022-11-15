"""Some utility functions useful during model training/inference."""

import gc
import random

import numpy
import torch


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
