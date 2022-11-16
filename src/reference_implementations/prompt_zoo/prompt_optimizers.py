"""This module defines the required functions to setup the optimizers for each
experiment type with the T5 models."""

from typing import Dict, Optional, Union

import torch
from torch.optim.optimizer import Optimizer
from transformers import Adafactor

TRAIN_ARGS_TYPE = Dict[str, Union[torch.nn.Module, float, torch.nn.Module]]


def construct_optimizer(
    model: torch.nn.Module, learning_rate: float, second_model: Optional[torch.nn.Module] = None
) -> Optimizer:
    """Define the adafactor optimizer over the parameters."""

    # Configurations suggested by the T5 paper.
    # https://discuss.huggingface.co/t/t5-finetuning-tips/684/35
    # to know more about Adafactor: https://arxiv.org/abs/1804.04235
    # Adafactor has small memory footprint compared to adam in transformers.

    params = list(model.parameters())
    if second_model is not None:
        # concatinate the second module parameters and register in the optimizer.
        params += list(second_model.parameters())

    optimizer = Adafactor(
        params,
        lr=learning_rate,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )
    return optimizer


def all_weights_opt(train_args: TRAIN_ARGS_TYPE) -> Optimizer:
    """Define the optimizer that fine-tunes all the weights in the T5 model."""
    return construct_optimizer(model=train_args["t5_model"], learning_rate=train_args["learning_rate"])


def input_embeddings_opt(train_args: TRAIN_ARGS_TYPE) -> Optimizer:
    """Define the optimizer that only fine-tunes the shared input embedding
    layer of the T5 encoder/decoder."""

    t5_model: torch.nn.Module = train_args["t5_model"]
    for name, param in t5_model.named_parameters():
        if name == "shared.weight":
            param.requires_grad = True
        else:
            param.requires_grad = False

    return construct_optimizer(model=t5_model, learning_rate=train_args["learning_rate"])


def output_embeddings_opt(train_args: TRAIN_ARGS_TYPE) -> Optimizer:
    """Define the optimizer that only fine-tunes the output LM head of the T5
    decoder."""

    t5_model: torch.nn.Module = train_args["t5_model"]
    for name, param in t5_model.named_parameters():
        if name == "lm_head.weight":
            param.requires_grad = True
        else:
            param.requires_grad = False

    return construct_optimizer(model=t5_model, learning_rate=train_args["learning_rate"])


def input_output_embeddings_opt(train_args: TRAIN_ARGS_TYPE) -> Optimizer:
    """Define the optimizer that fine-tunes both the shared input embedding
    layer + the output embedding layer of the T5 decoder."""

    t5_model: torch.nn.Module = train_args["t5_model"]
    for name, param in t5_model.named_parameters():
        if name in ["lm_head.weight", "shared.weight"]:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return construct_optimizer(model=t5_model, learning_rate=train_args["learning_rate"])


def no_weights_opt(train_args: TRAIN_ARGS_TYPE) -> Optimizer:
    """Define the optimizer that does not fine-tune any weights, however, the
    input will be augmented with some prompt instructions + in-context
    examples."""

    t5_model: torch.nn.Module = train_args["t5_model"]
    # don't waste time storing grad data.
    for _, param in t5_model.named_parameters():
        param.requires_grad = False

    return construct_optimizer(model=t5_model, learning_rate=train_args["learning_rate"])


def soft_prompt_opt(train_args: TRAIN_ARGS_TYPE) -> Optimizer:
    """Define the optimizer that only fine-tunes the prompt vectors on the
    downstream task."""

    t5_model: torch.nn.Module = train_args["t5_model"]
    # don't waste time storing grad data.
    for _, param in t5_model.named_parameters():
        param.requires_grad = False

    return construct_optimizer(
        model=t5_model, learning_rate=train_args["learning_rate"], second_model=train_args["prompt_model"]
    )


# store the functions that setup the optimizer for each experiment type.
optimizer_definer = {
    "all_finetune": all_weights_opt,
    "input_finetune": input_embeddings_opt,
    "output_finetune": output_embeddings_opt,
    "input_output_finetune": input_output_embeddings_opt,
    "no_finetune": no_weights_opt,
    "soft_prompt_tune": soft_prompt_opt,
}
