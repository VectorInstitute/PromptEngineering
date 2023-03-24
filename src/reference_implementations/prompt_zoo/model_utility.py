import gc
import random
from typing import List, Optional, Tuple

import numpy
import torch
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_string("t5_exp_type", "all_finetune", "The type of experiment with the T5 model.")

flags.DEFINE_integer("prompt_length", 20, "length of the prompts in the input sequence.")


def prepend_prompt(
    input_ids: torch.LongTensor, mask: torch.LongTensor, prompt_tokens: List[int]
) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """Prepend prompt token ids to the input_ids.

    - input_ids and mask are the raw inputs to t5 to be modified.
    """
    batch_size, _ = input_ids.size()

    prompt_tensor = torch.tensor(prompt_tokens, device=input_ids.device)

    prompt_tensor = prompt_tensor.view(1, FLAGS.prompt_length).expand(batch_size, FLAGS.prompt_length)

    # prompt tokens are always valid so mask is always 1.
    prompt_mask = torch.ones((batch_size, FLAGS.prompt_length), device=mask.device)

    # put prompt tokens before the input tokens.
    prompted_input_ids = torch.cat((prompt_tensor, input_ids), dim=1)
    prompted_mask = torch.cat((prompt_mask, mask), dim=1)
    return prompted_input_ids, prompted_mask


def modify_inputs_outputs(batch: torch.utils.data.Dataset, prompt_lists: Optional[List[List[int]]] = None) -> None:
    """This function will modify the input_ids and mask in the batch to include
    prompt tokens if needed.

    If we have multiple prompt tokens to augment the input with, it will
    repeat the outputs per prompt template.
    """
    batch_size, sequence_length = batch["input_ids"].size()
    if FLAGS.t5_exp_type == "soft_prompt_finetune":
        # This is used for soft prompt tuning! Then for a prompt with length |P|,
        # we add dummy prompt token ids from [0, |P|-1] to map
        # those into |P| vectors from the prompt embedder.
        batch["modified_input_ids"], batch["modified_attention_mask"] = prepend_prompt(
            batch["input_ids"], batch["attention_mask"], prompt_tokens=list(range(FLAGS.prompt_length))
        )

        # no change in the output for soft-prompting.
        batch["modified_target_attention_mask"] = batch["target_attention_mask"]
        batch["modified_labels"] = batch["labels"]

    elif FLAGS.t5_exp_type in ["gradient_search", "grips"] and prompt_lists:
        input_ids_stack = []
        input_mask_stack = []
        num_prompts = len(prompt_lists)
        for prompt_tokens in prompt_lists:
            input_ids, mask = prepend_prompt(batch["input_ids"], batch["attention_mask"], prompt_tokens)
            input_ids_stack.append(input_ids)
            input_mask_stack.append(mask)

        batch["modified_input_ids"] = torch.stack(input_ids_stack, dim=1).view(num_prompts * batch_size, -1)
        batch["modified_attention_mask"] = torch.stack(input_mask_stack, dim=1).view(num_prompts * batch_size, -1)

        # repeat the output mask and output ids for every prompt template in the batch dimension.
        batch["modified_target_attention_mask"] = (
            batch["target_attention_mask"].repeat(1, num_prompts).view(num_prompts * batch_size, -1)
        )
        batch["modified_labels"] = batch["labels"].repeat(1, num_prompts).view(num_prompts * batch_size, -1)

    else:
        batch["modified_target_attention_mask"] = batch["target_attention_mask"]
        batch["modified_labels"] = batch["labels"]
        batch["modified_input_ids"] = batch["input_ids"]
        batch["modified_attention_mask"] = batch["attention_mask"]


def log_of_labels(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    input_mask: torch.Tensor,
    decoder_mask: torch.Tensor,
    labels: torch.Tensor,
    loss_func: torch.nn.CrossEntropyLoss,
) -> torch.Tensor:
    """Do a forward computation and compute the log probability for the given
    labels."""

    # shift the gold labels one step to the right and do teacher-forcing by giving the gold previous token
    # and then compute the probablity for the next token at each step.
    # labels = [pos, it, ive]
    # decoder_input = [<BOS>, pos, it]
    # we want to see what is log probability for the target sequence "positive".
    output = model(
        input_ids=input_ids,
        attention_mask=input_mask,
        decoder_attention_mask=decoder_mask,
        decoder_input_ids=model._shift_right(labels),
        labels=None,
    )

    log_p = -loss_func(
        output.logits.view(-1, output.logits.size(-1)),
        labels.view(-1),
    )

    batch_size, sequence_length, vocab_size = output.logits.size()
    # compute per-token log probability in a sequence.
    # log_p has log probabilities for the following target output: [pos, it, ive]
    log_p = log_p.view(batch_size, sequence_length)

    # pad tokens have index -100 in huggingface.
    good_log_p = log_p.masked_fill_(labels == -100, 0.0)

    # good_log_p now has the log probability of the output sequence tokens.
    # sum over the sequence length to compute the log probability for a full sequence.
    return torch.sum(good_log_p, dim=1).squeeze()


def clear_cache() -> None:
    """Clean unused GPU Cache!"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


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
