from typing import Dict, List, Union

import torch
import torch.nn as nn
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast


def get_label_token_ids(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast], prompt_template: str, label_words: List[str]
) -> List[int]:
    # Need to consider the token ids of our labels in the context of the prompt, as they may be different in context.
    tokenized_inputs = tokenizer(
        [f"{prompt_template} {label_word}" for label_word in label_words], return_tensors="pt"
    )["input_ids"]

    label_token_ids = tokenized_inputs[:, -1]
    # # If you ever need to move back from token ids, you can use tokenizer.decode or tokenizer.batch_decode
    # tokenizer.decode(label_token_ids)
    return label_token_ids


def get_label_with_highest_likelihood(
    layer_matrix: torch.Tensor,
    label_token_ids: torch.Tensor,
    int_to_label_map: Dict[int, str],
    right_shift: bool = False,
) -> str:
    # The activations we care about are the last token (corresponding to our label token) and the values for our label
    #  vocabulary
    label_activations = layer_matrix[-1][label_token_ids].float()
    softmax = nn.Softmax(dim=0)
    # Softmax is not strictly necessary, but it helps to contextualize the "probability" the model associates with each
    # label relative to the others
    label_distributions = softmax(label_activations)
    # We select the label index with the largest value
    max_label_index = torch.argmax(label_distributions)
    if right_shift:
        # Some labels start at 1, so we add one to argmax
        max_label_index += 1
    # We then map that index back tot the label string that we care about via the map provided.
    return int_to_label_map[max_label_index.item()]


def split_prompts_into_batches(prompts: List[str], batch_size: int = 1) -> List[List[str]]:
    return [prompts[x : x + batch_size] for x in range(0, len(prompts), batch_size)]
