"""This module implements different ideas for fine-tuning a T5 model, which is
adapted with the prefix language modelling, on some downstream NLP datasets."""

import os
from abc import abstractmethod
from typing import Dict, Iterator, List, Optional, Union

import torch
from absl import flags
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.reference_implementations.prompt_zoo.model_utility import (
    clear_cache,
    log_of_labels,
    modify_inputs_outputs,
    set_random_seed,
)
from src.reference_implementations.prompt_zoo.prompt_optimizers import optimizer_definer
from src.reference_implementations.prompt_zoo.soft_prompt_modules import (
    create_softprompt_T5_for_conditional_generation,
)

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 42, "the seed number")

# the t5-base model with the extra LM adaptation steps.
# https://huggingface.co/google/t5-large-lm-adapt
flags.DEFINE_string("t5_pretrained_model", "google/t5-large-lm-adapt", "initial pre-trained model to use as T5.")
flags.DEFINE_string("mode", "train", "the mode of run? train or test")
flags.DEFINE_string("model_path", "/tmp/", "main directory to save or load the model from")
flags.DEFINE_string("checkpoint", "best_step", "checkpoint name to load from.")
flags.DEFINE_float("dropout_rate", 0.1, "dropout_rate used in T5 base.")


NO_SOFT_PROMPT_EXPS = {"all_finetune", "input_finetune", "output_finetune", "no_finetune"}


class MyBaseT5(torch.nn.Module):
    """Base class for different finetuning + prompt-tuning experiments."""

    def __init__(self) -> None:
        super().__init__()

        set_random_seed(FLAGS.seed)

        # check the gpu actually exists and setup device.
        self.gpu_check = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.gpu_check else "cpu")

        # will contain a dictionary with model name as the key
        # and the actual model as the value.
        self.model_pool: Dict[str, torch.nn.Module] = {}

        # for some subclasses, we will compute per token log probabilities.
        # pad tokens have index -100 in huggingface.
        # don't reduce loss (log likelihood), compute loss per token.
        self.loss_func = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")

    def setup_models(self) -> None:
        """Setup optimizer in training or load from the checkpoint for
        testing."""
        # put model on gpu or cpu.
        for model in self.model_pool.values():
            model.to(self.device)

        self.loss_func = self.loss_func.to(self.device)

        if FLAGS.mode == "train":
            if FLAGS.t5_exp_type not in ["gradient_search", "grips"]:
                # create optimizer only for training.
                # based on the experiment type, setup the optimizer.
                self.optimizer = optimizer_definer[FLAGS.t5_exp_type](self.model_pool)
            else:
                # gradient_search and GRIPS does not require optimizer.
                pass
        elif FLAGS.mode in ["test", "inference", "eval"]:
            # load from the given checkpoint, whether load model or the searched prompt.
            self.load_from_checkpoint()
        elif FLAGS.mode in ["no_finetune_test"]:
            # just rely on the pre-trained T5 or default prompt template
            # for prediction and no loading from the checkpoint.
            pass
        else:
            raise Exception("Wrong mode {}!".format(FLAGS.mode))

    def load_from_checkpoint(self) -> None:
        """Loads the weights from the given checkpoint."""
        m_path = FLAGS.model_path
        ckp_name = FLAGS.checkpoint
        try:
            for m_name, model in self.model_pool.items():
                model_ckp = os.path.join(m_path, f"{m_name}_{ckp_name}")
                model.load_state_dict(
                    torch.load(
                        model_ckp,
                        map_location=lambda storage, loc: storage,
                    )
                )
        except Exception as e:
            raise Exception("Could not load the checkpoint due to error:{}".format(e))

    def save(self) -> None:
        """Save the modules to the model_path for the specified checkpoint
        name."""
        m_path = FLAGS.model_path
        checkpoint_name = FLAGS.checkpoint
        if not os.path.exists(m_path):
            os.makedirs(m_path)
        for m_name, model in self.model_pool.items():
            torch.save(model.state_dict(), os.path.join(m_path, f"{m_name}_{checkpoint_name}"))

    def train_mode_on(self) -> None:
        """Before every forward-backward iteration over batch, clear gpu cache,
        turn on train mode, and zero the optimizer gradient state if
        defined!"""

        clear_cache()

        # turn on training mode which enables dropout.
        for model in self.model_pool.values():
            model.train()

        # the gradient search method doesn't require minimizing a loss function and is a discrete search technique.
        if FLAGS.t5_exp_type != "gradient_search":
            self.optimizer.zero_grad()

    def predict_mode_on(self) -> None:
        """For each iteration of prediction over batch, clear gpu cache, turn
        on eval mode."""

        clear_cache()

        # turn on eval mode which disables dropout.
        for model in self.model_pool.values():
            model.eval()

    def move_to_gpu(self, batch: torch.utils.data.Dataset, keys: List[str]) -> Dict[str, torch.Tensor]:
        """If gpu flag is set, move the batch tensors specified by keys into
        the gpu and return a dictionary to access the gpu tensors."""
        return {key: batch[key].to(self.device) for key in keys}

    @abstractmethod
    def train(self, batch: torch.utils.data.Dataset) -> Dict[str, float]:
        """The abstract train function."""
        pass

    @abstractmethod
    def predict(self, batch: torch.utils.data.Dataset) -> Iterator[Dict[str, Union[str, float]]]:
        """The abstract predict function."""
        pass

    def forward_pass(
        self, batch: torch.utils.data.Dataset, train: bool = False, prompt_lists: Optional[List[List[int]]] = None
    ) -> torch.Tensor:
        """Run a forward computation over the batch for each prompt lists and
        compute the log probability over the batch for that given prompt
        template.

        This function can be called or training or inference or If there
        are no prompts. If there is no prompt, it won't repeat the data
        per prompt template.
        """

        if train:
            self.train_mode_on()
        else:
            self.predict_mode_on()

        loaded_batch = self.move_to_gpu(batch, keys=["input_ids", "attention_mask", "target_attention_mask", "labels"])
        # keep an internal link to the loaded batch on gpu or cpu.
        self.loaded_batch = loaded_batch

        modify_inputs_outputs(loaded_batch, prompt_lists)

        # we have to make sure that the PAD token is ignored.
        # huggingface ignores a pad token if the token is -100!
        orig_labels = loaded_batch["modified_labels"]
        labels = orig_labels.masked_fill(orig_labels == self.tokenizer.pad_token_id, -100)

        t5_model = self.model_pool["t5_model"]

        with torch.set_grad_enabled(train):
            class_log_p = log_of_labels(
                model=t5_model,
                input_ids=loaded_batch["modified_input_ids"],
                input_mask=loaded_batch["modified_attention_mask"],
                decoder_mask=loaded_batch["modified_target_attention_mask"],
                labels=labels,
                loss_func=self.loss_func,
            )

        return class_log_p


class FineTuneT5(MyBaseT5):
    """Wrapper class around the MyBaseT5 Model to experiment with different
    finetuning ideas without having classifier. Used for the following
    experiments:

    1 - Fully fine-tuning all the parameters of the T5 model.
    2 - Only fine-tuning the shared input embedding layer of the T5 encoder/decoder.
    3 - Only fine-tuning the output embedding layer of the T5 decoder.
    4 - Initialize some soft-prompt vectors and augment to the input embedding matrix and
        only fine-tune those prompt vectors on the downstream task.
    """

    def __init__(self) -> None:
        super().__init__()

        # construct tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(FLAGS.t5_pretrained_model)

        # construct the underlying t5 model
        if FLAGS.t5_exp_type in NO_SOFT_PROMPT_EXPS:
            self.model_pool["t5_model"] = T5ForConditionalGeneration.from_pretrained(FLAGS.t5_pretrained_model)
        elif FLAGS.t5_exp_type == "soft_prompt_finetune":
            self.model_pool["t5_model"] = create_softprompt_T5_for_conditional_generation()
        self.setup_models()

    def train(self, batch: torch.utils.data.Dataset) -> Dict[str, float]:
        """The main train loop for generating the class sequence in the decoder
        T5."""
        class_log_ps = self.forward_pass(batch, train=True)

        # average log probs over the batch dimension.
        loss = -class_log_ps.mean(dim=0)
        loss_value = loss.item()

        # backProp
        loss.backward()

        # optimize
        self.optimizer.step()

        return {"loss_value": loss_value}

    def predict(self, batch: torch.utils.data.Dataset) -> Iterator[Dict[str, Union[str, float]]]:
        """The main prediction loop for a given potential class label."""

        class_log_ps = self.forward_pass(batch)
        class_log_ps = class_log_ps.cpu().detach().numpy()

        # not efficient, but let's pair input and potential class along the prediction scores.
        # all transformer special tokens will be removed
        potentials_str = self.tokenizer.batch_decode(self.loaded_batch["labels"], skip_special_tokens=True)
        inputs_str = self.tokenizer.batch_decode(self.loaded_batch["input_ids"], skip_special_tokens=True)

        for index, input_str in enumerate(inputs_str):
            output_row = {
                "potential_class": potentials_str[index],
                "prediction_score": class_log_ps[index],
                "input": input_str,
            }
            yield output_row
