"""This module implements the modifications to the T5 module of the huggingface
to include the soft prompt vectors in the input."""

import torch


class PromptEmbedding(torch.nn.Module):
    """We implement a new Embedding module where we also have prompt
    parameters. We only update the prompt vectors during training.

    prompt tokens are always at the first prompt_length steps of the
    input.
    """

    def __init__(self, num_embeddings: int, prompt_length: int, embedding_dim: int) -> None:
        """
        Args:
            num_embeddings (int): size of the dictionary of embeddings for normal tokens.
            embedding_dim (int): the size of each embedding vector
            prompt_length (int): length of the prompt tokens which are prepended to the input.
        """
        super(PromptEmbedding, self).__init__()
        self.prompt_length = prompt_length

        # this is the shared embedding table for the normal tokens of the input/output sequence
        # used by T5 encoder/decoder.
        self.shared = torch.nn.Embedding(num_embeddings, embedding_dim)

        self.prompt_embedder = torch.nn.Embedding(prompt_length, embedding_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """prompt tokens are always at the first prompt_length steps of the
        input.

        split the input sequences into two parts:
            1 - the first prompt_length steps should be mapped to prompt vectors.
            2 - the second part should be embedded by the normal embedding table of T5 defined for english tokens.

        concatinate the embedded splits into a single split along the sequence dimension.
        """

        # b_sz: batch_size
        # seq_len: sequence length
        b_sz, seq_len = input.size()

        prompt_input, normal_input = torch.split(input, [self.prompt_length, seq_len - self.prompt_length], dim=1)

        # prompt_embedded has shape: (b_sz,  self.prompt_length, embedding_dim)
        prompt_embedded = self.prompt_embedder(prompt_input)

        # normal_input_embedded has shape: (b_sz,  seq_len - self.prompt_length, embedding_dim)
        normal_input_embedded = self.shared(normal_input)

        # concat along the dimension 1
        return torch.cat((prompt_embedded, normal_input_embedded), dim=1)
