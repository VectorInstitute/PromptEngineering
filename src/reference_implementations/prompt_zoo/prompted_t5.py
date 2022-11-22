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

import gc
import os
import random
from typing import Dict, Iterator, List, Optional

import numpy
import torch
from absl import flags
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.reference_implementations.prompt_zoo.prompt_optimizers import optimizer_definer

FLAGS = flags.FLAGS
flags.DEFINE_string("t5_exp_type", "all_finetune", "The type of experiment with the T5 model.")
flags.DEFINE_integer("seed", 42, "the seed number")
flags.DEFINE_bool("gpu", False, "Whether to put the model on gpu or not?")

# the t5-base model with the extra LM adaptation steps.
# https://huggingface.co/google/t5-base-lm-adapt
flags.DEFINE_string("t5_pretrained_model", "google/t5-base-lm-adapt", "initial pre-trained model to use as T5.")

flags.DEFINE_string("mode", "train", "the mode of run? train or test")
flags.DEFINE_string("model_path", "/tmp/", "main directory to save or load the model from")
flags.DEFINE_string("checkpoint", None, "checkpoint name to load from.")


def clear_cache() -> None:
    """Clean unused GPU Cache!"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return


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


class PromptedT5(torch.nn.Module):
    """Wrapper class around the T5 Model to experiment with different prompt
    ideas."""

    def __init__(self, prompt_model: Optional[torch.nn.Module] = None) -> None:
        super(PromptedT5, self).__init__()

        set_random_seed(FLAGS.seed)

        # check the gpu actually exists and setup device.
        self.gpu_check = FLAGS.gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.gpu_check else "cpu")

        # construct tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(FLAGS.t5_pretrained_model)

        # construct the underlying t5 model
        self.model = T5ForConditionalGeneration.from_pretrained(FLAGS.t5_pretrained_model)

        # put model on gpu or cpu.
        self.model.to(self.device)
        self.prompt_model = prompt_model
        if self.prompt_model is not None:
            self.prompt_model.to(self.device)

        if FLAGS.mode == "train":
            # create optimizer only for training.
            self.setup_optimizer()
        elif FLAGS.mode in ["test", "inference", "eval"]:
            # load from the given checkpoint.
            self.load()
        return

    def load(self) -> None:
        """Loads the weights from the given checkpoint."""
        m_path = FLAGS.model_path
        ckp_name = FLAGS.checkpoint
        try:
            model_ckp = os.path.join(m_path, "model_") + ckp_name
            prompt_ckp = os.path.join(m_path, "prompt_model_") + ckp_name
            self.model.load_state_dict(
                torch.load(
                    model_ckp,
                    map_location=lambda storage, loc: storage,
                )
            )
            if self.prompt_model is not None:
                self.prompt_model.load_state_dict(
                    torch.load(
                        prompt_ckp,
                        map_location=lambda storage, loc: storage,
                    )
                )
        except Exception as e:
            print("Could not load the checkpoint due to error: {}".format(e))
            print("Using the initial checkpoint for T5.")
        return

    def save(self, checkpoint_name: str) -> None:
        """Save the modules to the model_path for the specified checkpoint
        name."""
        m_path = FLAGS.model_path
        if not os.path.exists(m_path):
            os.makedirs(m_path)
        torch.save(self.model.state_dict(), os.path.join(m_path, "model_") + checkpoint_name)
        if self.prompt_model is not None:
            torch.save(self.prompt_model.state_dict(), os.path.join(m_path, "prompt_model_") + checkpoint_name)
        return

    def setup_optimizer(self) -> None:
        """Based on the experiment type, setup the optimizer."""
        print(FLAGS.t5_exp_type)
        optimizer_args = {
            "t5_model": self.model,
            "prompt_model": self.prompt_model,
        }
        self.optimizer = optimizer_definer[FLAGS.t5_exp_type](optimizer_args)
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
            if self.gpu_check:
                val = val.to(self.device)
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
            attention_mask=loaded_batch["attention_mask"],
            decoder_attention_mask=loaded_batch["target_attention_mask"],
            labels=loaded_batch["labels"],
        )

        loss = output.loss
        loss_value = loss.item()

        # backProp
        loss.backward()

        # optimize
        self.optimizer.step()

        return {"loss_value": loss_value}

    def prompt_train(self, batch: torch.utils.data.Dataset) -> Dict[str, float]:
        """The train function for prompt tuning using T5 as the underlying
        LM."""
        pass

    def predict(self, batch: torch.utils.data.Dataset) -> Iterator[Dict[str, str]]:
        """The main prediction loop for a given potential class label."""

        self.predict_mode_on()

        # define loss function to compute token probabilities.
        # pad tokens have index -100 in huggingface.
        # don't reduce loss, compute loss per token.
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        if self.gpu_check:
            loss_fct = loss_fct.to(self.device)

        loaded_batch = self.move_to_gpu(batch, keys=["input_ids", "attention_mask", "target_attention_mask", "labels"])

        output = self.model(
            input_ids=loaded_batch["input_ids"],
            attention_mask=loaded_batch["attention_mask"],
            decoder_attention_mask=loaded_batch["target_attention_mask"],
            decoder_input_ids=self.model._shift_right(loaded_batch["labels"]),
            labels=None,
        )

        log_p = -loss_fct(
            output.logits.view(-1, output.logits.size(-1)),
            loaded_batch["labels"].view(-1),
        )

        # b: batch size
        # sz: sequence size
        # v: vocab size
        b, sz, v = output.logits.size()
        log_p = log_p.view(b, sz)
        good_log_p = log_p.masked_fill_(loaded_batch["labels"] == -100, 0.0)
        class_log_p = torch.sum(good_log_p, dim=1).squeeze().cpu().detach().numpy()

        # not efficient, but let's pair input and potential class along the prediction scores.
        # all transformer special tokens will be removed

        labels = loaded_batch["labels"]
        # change -100 tokens to pad tokens.
        labels = labels.masked_fill_(labels == -100, self.tokenizer.pad_token_id)
        potentials_str = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        inputs_str = self.tokenizer.batch_decode(loaded_batch["input_ids"], skip_special_tokens=True)

        for index, input_str in enumerate(inputs_str):
            output_row = {
                "potential_class": potentials_str[index],
                "prediction_score": class_log_p[index],
                "input": input_str,
            }
            yield output_row

    def train(self, batch: torch.utils.data.Dataset) -> Dict[str, float]:
        """Switch over t5_exp_type and call the correct train function over
        batch for each experiment type."""
        if FLAGS.t5_exp_type in ["all_finetune", "input_finetune", "output_finetune", "input_output_finetune"]:
            return self.no_prompt_train(batch)
        else:
            return self.prompt_train(batch)
