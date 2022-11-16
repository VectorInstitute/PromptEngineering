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
from typing import Dict, Iterator, List, Optional

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.reference_implementations.prompt_zoo.model_utility import clear_cache, set_random_seed
from src.reference_implementations.prompt_zoo.prompt_optimizers import optimizer_definer

# the t5-base model with the extra LM adaptation steps.
# https://huggingface.co/google/t5-base-lm-adapt
MODEL_NAME = "google/t5-base-lm-adapt"


@dataclass
class ConfigParameters:
    """To store, edit and share general Model configurations."""

    model_path: str = "/tmp/"
    batch_size: int = 16
    source_max_length: int = 256
    decoder_max_length: int = 32
    config_file: str = "config.ini"
    gpu: bool = False
    device: Optional[str] = None
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
    steps_per_checkpoint: int = 100

    # Which T5 checkpoint to download from huggingface?
    model_name: str = MODEL_NAME

    # Decide what type of experiment we want with respect to prompts and T5s.
    t5_exp_type: str = "all_finetune"


class PromptedT5(torch.nn.Module):
    """Wrapper class around the T5 Model to experiment with different prompt
    ideas."""

    def __init__(self, cfg: ConfigParameters, exp_type: str, prompt_model: Optional[torch.nn.Module] = None) -> None:
        super(PromptedT5, self).__init__()
        self.config = cfg

        # exp_type is one of the options in optimizer_definer.
        self.config.t5_exp_type = exp_type

        set_random_seed(cfg.seed)

        # check the gpu actually exists and setup device.
        self.config.gpu = self.config.gpu and torch.cuda.is_available()
        self.config.device = torch.device("cuda" if self.config.gpu else "cpu")

        # construct tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(self.config.model_name)

        # construct the underlying t5 model
        self.model = T5ForConditionalGeneration.from_pretrained(self.config.model_name)

        # put model on gpu or cpu.
        self.model.to(self.config.device)
        self.prompt_model = prompt_model
        if self.prompt_model is not None:
            self.prompt_model.to(self.config.device)

        self.setup_optimizer()
        return

    def setup_optimizer(self) -> None:
        """Based on the experiment type, setup the optimizer."""
        train_args = {
            "t5_model": self.model,
            "learning_rate": self.config.learning_rate,
            "prompt_model": self.prompt_model,
        }
        exp_type = self.config.t5_exp_type
        self.optimizer = optimizer_definer[exp_type](**train_args)
        return

    def train_mode_on(self) -> None:
        """Before every forward-backward iteration over batch, clear gpu cache,
        turn on tain mode, and zero the optimizer gradient state."""

        clear_cache()

        # turn on training mode which enables dropout.
        self.model.train()

        if self.prompt_model is not None:
            self.prompt_model.train()

        self.optimizer.zero_grad()
        return

    def predict_mode_on(self) -> None:
        """For each iteration of prediction over batch, clear gpu cache, turn
        on eval mode."""

        clear_cache()

        # turn on eval mode which disables dropout.
        self.model.eval()

        if self.prompt_model is not None:
            self.prompt_model.eval()

        return

    def move_to_gpu(self, batch: torch.utils.data.Dataset, keys: List[str]) -> Dict[str, torch.Tensor]:
        """If gpu flag is set, move the batch tensors specified by keys into
        the gpu and return a dictionary to access the gpu tensors."""
        ret = {}
        for key in keys:
            val = batch[key]
            if self.config.gpu:
                val = val.to(self.config.device)
            ret[key] = val
        return ret

    def no_prompt_train(self, batch: torch.utils.data.Dataset) -> Dict[str, float]:
        """The main train loop for the following cases of the T5 experiments:

        1 - Fully fine-tuning all the parameters of the T5 model.
        2 - Only fine-tuning the shared input embedding layer of the T5 encoder/decoder.
        3 - Only fine-tuning the output embedding layer of the T5 decoder.
        4 - Fine-tuning both the shared input embedding layer +
            the output embedding layer of the T5 decoder.
        """

        self.train_mode_on()
        loaded_batch = self.move_to_gpu(batch, keys=["input_ids", "attention_mask", "target_attention_mask", "labels"])
        output = self.model(
            input_ids=loaded_batch["input_ids"],
            attention_mask=loaded_batch["input_mask"],
            decoder_attention_mask=loaded_batch["target_mask"],
            labels=loaded_batch["labels"],
        )

        loss = output.loss
        loss_value = loss.item()

        # backProp
        loss.backward()

        # optimize
        self.optimizer.step()

        return {"loss_value": loss_value}

    def no_prompt_predict(self, batch: torch.utils.data.Dataset) -> Iterator[Dict[str, str]]:
        """The main prediction loop for the following cases of the T5
        experiments:

        1 - Fully fine-tuning all the parameters of the T5 model.
        2 - Only fine-tuning the shared input embedding layer of the T5 encoder/decoder.
        3 - Only fine-tuning the output embedding layer of the T5 decoder.
        4 - Fine-tuning both the shared input embedding layer +
            the output embedding layer of the T5 decoder.
        """

        self.predict_mode_on()
        loaded_batch = self.move_to_gpu(batch, keys=["input_ids", "attention_mask"])

        # use greedy decoding.
        predictions = self.model.generate(
            input_ids=loaded_batch["input_ids"],
            attention_mask=loaded_batch["input_mask"],
        )

        # all transformer special tokens will be removed
        predictions_str = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # not efficient, but let's pair input along the predictions.
        inputs_str = self.tokenizer.batch_decode(loaded_batch["input_ids"], skip_special_tokens=True)
        for index, pred_str in enumerate(predictions_str):
            output_row = {
                "prediction": pred_str,
                "input": inputs_str[index],
            }
            yield output_row

    def train(self, batch: torch.utils.data.Dataset, exp_type: str = "all_finetune") -> Dict[str, float]:
        """Switch over exp_type and call the correct train function over batch
        for each experiment type."""
        if exp_type in ["all_finetune", "input_finetune" "output_finetune", "input_output_finetune"]:
            return self.no_prompt_train(batch)
        else:
            # TODO: implement the prompt train.
            return self.no_prompt_train(batch)

    def predict(self, batch: torch.utils.data.Dataset, exp_type: str = "all_finetune") -> Iterator[Dict[str, str]]:
        """Switch over exp_type and call the correct predict function over
        batch for each experiment type."""
        if exp_type in ["all_finetune", "input_finetune" "output_finetune", "input_output_finetune"]:
            return self.no_prompt_predict(batch)
        else:
            # TODO: implement the prompt predict.
            return self.no_prompt_predict(batch)
