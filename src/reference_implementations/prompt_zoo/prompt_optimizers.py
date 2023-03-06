"""This module defines the required functions to setup the optimizers for each
experiment type with the T5 models."""

from typing import Dict, Optional

import torch
from absl import flags
from torch.optim.optimizer import Optimizer
from transformers import Adafactor

FLAGS = flags.FLAGS
flags.DEFINE_float("learning_rate", 0.005, "The learning rate used in the optimizer", lower_bound=0.0)
flags.DEFINE_float("weight_decay_rate", 0.0, "The weight decay rate used in the adafactor optimizer.", lower_bound=0.0)

OPTIMIZER_ARGS_TYPE = Dict[str, torch.nn.Module]


def construct_optimizer(model: torch.nn.Module, second_model: Optional[torch.nn.Module] = None) -> Optimizer:
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
        lr=FLAGS.learning_rate,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=FLAGS.weight_decay_rate,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False,
    )

    return optimizer


def all_weights_opt(opt_args: OPTIMIZER_ARGS_TYPE) -> Optimizer:
    """Define the optimizer that fine-tunes all the weights in the T5 model."""
    return construct_optimizer(model=opt_args["t5_model"])


def input_embeddings_opt(opt_args: OPTIMIZER_ARGS_TYPE) -> Optimizer:
    """Define the optimizer that only fine-tunes the shared input embedding
    layer of the T5 encoder/decoder."""

    t5_model: torch.nn.Module = opt_args["t5_model"]
    for name, param in t5_model.named_parameters():
        if name == "shared.weight":
            param.requires_grad = True
        else:
            param.requires_grad = False

    return construct_optimizer(model=t5_model)


def output_embeddings_opt(opt_args: OPTIMIZER_ARGS_TYPE) -> Optimizer:
    """Define the optimizer that only fine-tunes the output LM head of the T5
    decoder."""

    t5_model: torch.nn.Module = opt_args["t5_model"]
    for name, param in t5_model.named_parameters():
        if name == "lm_head.weight":
            param.requires_grad = True
        else:
            param.requires_grad = False

    return construct_optimizer(model=t5_model)


def no_weights_opt(opt_args: OPTIMIZER_ARGS_TYPE) -> Optimizer:
    """Define the optimizer that does not fine-tune any weights, however, the
    input will be augmented with some prompt instructions + in-context
    examples."""

    t5_model: torch.nn.Module = opt_args["t5_model"]
    # don't waste time storing grad data.
    for _, param in t5_model.named_parameters():
        param.requires_grad = False

    return construct_optimizer(model=t5_model)


def prompt_model_opt(opt_args: OPTIMIZER_ARGS_TYPE) -> Optimizer:
    """Define the optimizer that only fine-tunes the prompt vectors on the
    downstream task."""

    t5_model: torch.nn.Module = opt_args["t5_model"]
    for name, param in t5_model.named_parameters():
        if name == "shared.prompt_embedder.weight":
            param.requires_grad = True
        else:
            param.requires_grad = False

    return construct_optimizer(model=t5_model)


def classifier_model_opt(opt_args: OPTIMIZER_ARGS_TYPE) -> Optimizer:
    """Define the optimizer that only fine-tunes the classifier on top of the
    T5 encoder for the downstream task."""

    t5_encoder: torch.nn.Module = opt_args["t5_encoder"]
    # don't waste time storing grad data.
    for _, param in t5_encoder.named_parameters():
        param.requires_grad = False

    return construct_optimizer(model=t5_encoder, second_model=opt_args["classifier_model"])


# store the functions that setup the optimizer for each experiment type.
optimizer_definer = {
    "all_finetune": all_weights_opt,
    "input_finetune": input_embeddings_opt,
    "output_finetune": output_embeddings_opt,
    "no_finetune": no_weights_opt,
    "soft_prompt_finetune": prompt_model_opt,
    "classifier_finetune": classifier_model_opt,
}
