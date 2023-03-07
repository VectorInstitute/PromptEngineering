"""A module that defines the modifications to the T5 encoder-decoder and
defines a new embedding for soft-prompt tuning."""
import random

import torch
from absl import flags
from transformers import T5ForConditionalGeneration

FLAGS = flags.FLAGS


class PromptEmbedding(torch.nn.Module):
    """We implement a new Embedding module for the prompt parameters. We only
    update the prompt vectors during training.

    This PromptEmbedding will have a reference to the normal embedding matrix of the T5
    which will be populated when we load the T5 encoders from the huggingface.

    prompt tokens are always at the first prompt_length steps of the
    input.
    """

    def __init__(
        self, prompt_length: int, embedding_dim: int, normal_embedder: torch.nn.Embedding, normal_vocab_size: int
    ) -> None:
        """
        Args:
            prompt_length (int): length of the prompt tokens which are prepended to the input.
            embedding_dim (int): the size of each embedding vector
            normal_embedder (torch.nn.Embedding): this is the shared embedding table for the normal tokens
            of the input/output sequence used by T5 encoder/decoder.
        """
        super().__init__()
        self.prompt_length = prompt_length

        self.normal_embedder = normal_embedder

        self.prompt_embedder = torch.nn.Embedding(prompt_length, embedding_dim)

        # sample prompt_length vectors from the normal embedding table to initialize the prompt vectors.
        sampled_indices = random.choices(list(range(normal_vocab_size)), k=prompt_length)
        self.prompt_embedder.weight.data = self.normal_embedder.weight.data[sampled_indices, :]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """prompt tokens are always at the first prompt_length steps of the
        input.

        split the input sequences into two parts:
            1 - the first prompt_length tokens should be mapped to prompt vectors.
            2 - the rest should be embedded by the normal embedding table of T5 defined for english tokens.

        concatinate the embedded splits into a single split along the sequence dimension.
        """
        batch_size, sequence_length = input.size()

        prompt_input, normal_input = torch.split(
            input, [self.prompt_length, sequence_length - self.prompt_length], dim=1
        )

        # prompt_embedded has shape: (batch_size,  self.prompt_length, embedding_dim)
        prompt_embedded = self.prompt_embedder(prompt_input)

        # normal_input_embedded has shape: (batch_size,  sequence_length - self.prompt_length, embedding_dim)
        normal_input_embedded = self.normal_embedder(normal_input)
        # concat along the dimension 1
        return torch.cat((prompt_embedded, normal_input_embedded), dim=1)


def create_softprompt_T5_for_conditional_generation() -> torch.nn.Module:
    """This function implements the modifications to the T5 module of the
    huggingface to include the soft prompt vectors in the input.

    see the original huggingface implementation:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L1480

    Wrapping the T5ForConditionalGeneration to introduce our new PromptEmbedding
    module.
    """
    # prompt length
    p_len = FLAGS.prompt_length

    # let the T5ForConditionalGeneration load the initial checkpoint of the T5
    # with the normal embedding table.
    t5_model = T5ForConditionalGeneration.from_pretrained(FLAGS.t5_pretrained_model)

    # t5_model.config.d_model is from the class T5ForConditionalGeneration
    # which is the embedding dimension of the embedding table of the T5.
    d_model = t5_model.config.d_model
    vocab_size = t5_model.config.vocab_size

    prompt_embedding = PromptEmbedding(p_len, d_model, t5_model.shared, vocab_size)

    # update the general shared embedding module of huggingface T5.
    # now every call by t5_model.shared(input_ids) will use our forward method of the PromptEmbedding
    # we don't want to update the decoder embedding to add the prompt tokens for the output tokens.
    # see https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L1344
    t5_model.shared = prompt_embedding
    t5_model.encoder.embed_tokens = prompt_embedding
    return t5_model
