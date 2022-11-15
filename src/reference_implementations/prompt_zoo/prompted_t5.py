"""This module implements different ideas for fine-tuning a T5 model, which is
adapted with the prefix language modelling, on some downstream NLP datasets.

The module implements the following baselines:
1 - Fully fine-tuning all the parameters of the T5 model.
2 - Only fine-tuning the input embedding layer of the T5 encoder.
3 - Only fine-tuning the output embedding layer of the T5 decoder.
4 - Fine-tuning both the input embedding layer of the T5 encoder +
    the output embedding layer of the T5 decoder.
5 - No fine-tuning, however, augment input with prompt instructions + in-context examples.
6 - Search for the discrete prompts to augment the input using the gradient of the T5.
7 - Initialize some soft-prompt vectors and augment to the input embedding matrix and
    only fine-tune those prompt vectors on the downstream task.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

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
