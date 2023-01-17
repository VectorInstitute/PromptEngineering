import operator
from dataclasses import dataclass, field
from typing import List

import torch
from absl import flags
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.reference_implementations.prompt_zoo.prompted_t5 import MyBaseT5

FLAGS = flags.FLAGS


@dataclass(order=True)
class PromptTemplate:
    """A dataclass to define the prompt template with two attributes:

    1 - tokens: a list of token indices from the vocabulary table.
        tokens have size prompt_length.
    2 - score: the log likelihood of the label given this prompt template
                at a training step over a training batch.
    This dataclass is used for implementing a PromptSearchMemory for the gradient-search
        discrete prompt optimization augmented with beam search for moving from one search step to another step
        while scoring the prompt tokens.
    """

    tokens: List[int] = field(compare=False)
    score: float


class PromptSearchMemory:
    """A search memory that maps the training step index i to a sorted beam
    (list) that stores the top beam_size templates according to the their label
    likelihood computed over a batch of training data.

    This memory is used to easily store, read and update the
    PromptTemplates at different stages of the gradient-search prompting
    augmented with beam search.
    """

    def __init__(self, train_steps: int, prompt_length: int, top_k: int, init_token_id: int, beam_size: int) -> None:
        """This initializes the search memory for the training and will be
        dumped to disk for prediction."""
        self.train_steps = train_steps
        self.prompt_length = prompt_length
        self.top_k = top_k
        self.beam_size = beam_size
        self.memory = {}

        # allocate memory for the first training step.
        beam = [PromptTemplate(tokens=[init_token_id] * prompt_length, score=-float("inf"))] * beam_size
        self.memory[0] = beam

        # flag for the current training iteration.
        self.current_step = 0

    def update_beam(self, beam_candidates: List[PromptTemplate]) -> None:
        """For the next training step, select the top beam_size prompt templates out of beam_size * top_k template candidates
        inside the beam_candidates. The sorting is based on the score attribute of the PromptTemplate.

        Increment the self.current_step by 1.
        """
        # sort the prompt templates by their score in descending order.
        sorted_candidates = sorted(beam_candidates, key=operator.attrgetter("score"), reverse=True)

        # keep the top beam_size prompt templates.
        self.memory[self.current_step + 1] = sorted_candidates[: self.beam_size]
        self.current_step += 1

    '''
    def generate_beam_candidates(
        self, embedding_weight: torch.Tensor, losses: torch.Tensor, prompt_step: int
    ) -> List[PromptTemplate]:
        """For each prompt template inside the beam at the current step,
        compute the gradient with respect to the embedding vector of the prompt
        token at step prompt_step of the template. Perform dot product with the
        embedding_weight table to compute a score for every possible word
        replacement.

        Then consider the top_k replacement tokens and generate the new
        beam_candidates.
        """
        beam_candidates = []
        embedding_grads = []
        for beam_idx in self.beam_size:
            prompt_template = self.memory[self.current_step][beam_idx]
            prompt_token_idx = prompt_template.tokens[prompt_step]
            loss = losses[beam_idx]
            embedding_weight.grad.data.zero_()
            loss.backward()
            embedding_grad = embedding_weight.grad[prompt_token_idx]

        vocab_scores = torch.matmul(embedding_weight * embedding_grad).squeeze().cpu().detach().numpy()
        top_k_token_indices = numpy.argsort(vocab_scores)[: self.top_k]
        for top_k_token_idx in top_k_token_indices:
            candidate_template = copy.copy(prompt_template)
            candidate_template.tokens[prompt_step] = top_k_token_idx
            candidate_template.score = -float("inf")
            beam_candidates.append(candidate_template)

        return beam_candidates
    '''


class SearchT5(MyBaseT5):
    """Subclassing the mybase T5 class to introduce the T5 for gradient-
    search."""

    def __init__(self) -> None:
        super().__init__()

        # construct tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(FLAGS.t5_pretrained_model)

        # construct the underlying t5 model
        self.model_pool["t5_model"] = T5ForConditionalGeneration.from_pretrained(FLAGS.t5_pretrained_model)
        self.setup_models()

    def score_templates(
        self, batch: torch.utils.data.Dataset, prompt_templates: List[PromptTemplate], train: bool = False
    ) -> torch.Tensor:
        """Run a forward computation over the batch for each prompt templates
        and compute the log probability over the batch for that given prompt
        template.

        This function can be called multiple times for training or
        inference.
        """
        batch_size, _ = batch["input_ids"].size()
        prompt_lists = [template.tokens for template in prompt_templates]
        class_log_ps = self.forward_pass(batch, train, prompt_lists)

        # average log probs over the batch dimension.
        template_scores = class_log_ps.view(len(prompt_templates), batch_size).mean(dim=1)
        return template_scores
