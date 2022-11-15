"""This module implements different ideas for fine-tuning a T5 model, which is
adapted with the prefix language modelling, on some downstream NLP datasets.

The module implements the following baselines:
1 - Fully fine-tuning all the parameters of the T5 model.
2 - Only fine-tuning the shared input embedding layer of the T5 encoder/decoder.
3 - Only fine-tuning the output embedding layer of the T5 decoder.
4 - Fine-tuning both the shared input embedding layer +
    the output embedding layer of the T5 decoder.
5 - No fine-tuning, however, augment input with prompt instructions + in-context examples.
6 - Search for the discrete prompts to augment the input using the gradient of the T5.
7 - Initialize some soft-prompt vectors and augment to the input embedding matrix and
    only fine-tune those prompt vectors on the downstream task.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch.optim.optimizer import Optimizer
from transformers import Adafactor, T5ForConditionalGeneration, T5Tokenizer

from src.reference_implementations.prompt_zoo.model_utility import set_random_seed

# the t5-base model with the extra LM adaptation steps.
# https://huggingface.co/google/t5-base-lm-adapt
MODEL_NAME = "google/t5-base-lm-adapt"


@dataclass
class ConfigParameters:
    """To store, edit and share general Model configurations."""

    model_path: Optional[str] = None
    batch_size: int = 16
    source_max_length: int = 256
    decoder_max_length: int = 32
    config_file: str = "config.ini"
    gpu: bool = False
    learning_rate: float = 0.0005
    max_epochs: int = 16
    mode: str = "train"
    train_file: Optional[str] = None
    test_file: Optional[str] = None
    dev_file: Optional[str] = None
    prediction_output_file: Optional[str] = None
    seed: int = 8
    checkpoint: Optional[str] = None
    training_steps: Optional[int] = 1

    # Related to decoding.
    no_repeat_ngram_size: Optional[int] = 2

    # Which T5 checkpoint to download from huggingface?
    model_name: str = MODEL_NAME


class PromptedT5(torch.nn.Module):
    """Wrapper class around the T5 Model to experiment with different prompt
    ideas."""

    def __init__(self, cfg: ConfigParameters) -> None:
        super(PromptedT5, self).__init__()
        self.config = cfg

        set_random_seed(cfg.seed)

        # construct tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)

        # construct the underlying t5 model
        self.model = T5ForConditionalGeneration.from_pretrained(self.config.model_name)
        return


def construct_optimizer(t5_model: torch.nn.Module, learning_rate: float) -> Optimizer:
    """Define the adafactor optimizer over the parameters."""

    # Configurations suggested by the T5 paper.
    # https://discuss.huggingface.co/t/t5-finetuning-tips/684/35
    # to know more about Adafactor: https://arxiv.org/abs/1804.04235
    # Adafactor has small memory footprint compared to adam in transformers.
    optimizer = Adafactor(
        t5_model.parameters(),
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


def all_weights_opt(t5_model: torch.nn.Module, learning_rate: float) -> Optimizer:
    """Define the optimizer that fine-tunes all the weights in the T5 model."""
    return construct_optimizer(t5_model, learning_rate)


def input_embeddings_opt(t5_model: torch.nn.Module, learning_rate: float) -> Optimizer:
    """Define the optimizer that only fine-tunes the shared input embedding
    layer of the T5 encoder/decoder."""

    for name, param in t5_model.named_parameters():
        if name == "shared.weight":
            param.requires_grad = True
        else:
            param.requires_grad = False

    return construct_optimizer(t5_model, learning_rate)


def output_embeddings_opt(t5_model: torch.nn.Module, learning_rate: float) -> Optimizer:
    """Define the optimizer that only fine-tunes the output LM head of the T5
    decoder."""

    for name, param in t5_model.named_parameters():
        if name == "lm_head.weight":
            param.requires_grad = True
        else:
            param.requires_grad = False

    return construct_optimizer(t5_model, learning_rate)


def input_output_embeddings_opt(t5_model: torch.nn.Module, learning_rate: float) -> Optimizer:
    """Define the optimizer that fine-tunes both the shared input embedding
    layer + the output embedding layer of the T5 decoder."""

    for name, param in t5_model.named_parameters():
        if name in ["lm_head.weight", "shared.weight"]:
            param.requires_grad = True
        else:
            param.requires_grad = False

    return construct_optimizer(t5_model, learning_rate)


def no_weights_opt(t5_model: torch.nn.Module, learning_rate: float) -> Optimizer:
    """Define the optimizer that does not fine-tune any weights, however, the
    input will be augmented with some prompt instructions + in-context
    examples."""

    for _, param in t5_model.named_parameters():
        param.requires_grad = False

    return construct_optimizer(t5_model, learning_rate)
