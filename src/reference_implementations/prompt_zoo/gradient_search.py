import copy
import operator
import os
import pickle
from dataclasses import dataclass, field
from typing import Dict, Iterator, List

import torch
from absl import flags
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.reference_implementations.prompt_zoo.prompted_t5 import MyBaseT5

FLAGS = flags.FLAGS

flags.DEFINE_integer("top_k", 20, "Number of candidate tokens to replace the prompt token.")
flags.DEFINE_integer("beam_size", 20, "Number of prompt templates to consider for beam search.")


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
    """A search memory that keeps a sorted beam
    (list) which stores the top beam_size templates according to the their label
    likelihood computed over a batch of training data.

    This memory is used to easily store and read
    PromptTemplates at different stages of the gradient-search prompting
    augmented with beam search.
    """

    def __init__(self, init_token_id: int) -> None:
        """This initializes the search memory for the training and its beam will be
        dumped to disk while saving the model."""
        # allocate memory for the current beam of templates.
        self.beam = [
            PromptTemplate(tokens=[init_token_id] * FLAGS.prompt_length, score=-float("inf"))
        ] * FLAGS.beam_size

    def update_beam(self, beam_candidates: List[PromptTemplate]) -> None:
        """For the next training step, select the top beam_size prompt templates out of beam_size * top_k template candidates
        inside the beam_candidates. The sorting is based on the score attribute of the PromptTemplate.
        """
        # In the comparison, also include the current beam templates.
        template_pool = beam_candidates + self.beam
        # sort the prompt templates by their score in descending order.
        sorted_pool = sorted(template_pool, key=operator.attrgetter("score"), reverse=True)

        # keep the top beam_size prompt templates.
        self.beam = sorted_pool[: FLAGS.beam_size]

    def get_beam_loss(self) -> float:
        """Return the list of template scores inside the beam.
        then consider the averaged negative label log-likelihood over a beam of templates as the loss.
        """
        searched_scores = [template.score for template in self.beam]
        return -sum(searched_scores) / len(searched_scores)

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
        for beam_idx, prompt_template in enumerate(self.beam):
            prompt_token_idx = prompt_template.tokens[prompt_step]
            loss = losses[beam_idx]
            embedding_weight.grad.data.zero_()
            loss.backward()
            embedding_grad = embedding_weight.grad[prompt_token_idx]
            embedding_grads.append(embedding_grad)

        embedding_grads_tensor = torch.stack(embedding_grads, dim=1)
        vocab_scores = torch.matmul(embedding_weight, embedding_grads_tensor)
        top_scores, top_indices = torch.topk(vocab_scores, FLAGS.top_k, dim=0, largest=True, sorted=True)
        for prompt_template in self.beam:
            # memory is on RAM and not on GPU.
            for top_index in top_indices.tolist():
                candidate_template = copy.copy(prompt_template)
                candidate_template.tokens[prompt_step] = top_index
                candidate_template.score = -float("inf")
                beam_candidates.append(candidate_template)

        return beam_candidates


class SearchT5(MyBaseT5):
    """Subclassing the mybase T5 class to introduce the T5 for gradient-
    search. We also define a search memory to keep templates as we are scoring them during training."""

    def __init__(self) -> None:
        super().__init__()

        # construct tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(FLAGS.t5_pretrained_model)

        # construct the underlying t5 model
        t5_model = T5ForConditionalGeneration.from_pretrained(FLAGS.t5_pretrained_model)

        self.model_pool["t5_model"] = t5_model

        # use one of the sentinel tokens
        # t5 uses last vocab indices as sentinel tokens
        # https://github.com/google-research/text-to-text-transfer-transformer/blob/main/t5/data/preprocessors.py#L3039
        self.search_memory = PromptSearchMemory(init_token_id=t5_model.config.vocab_size - 1)
        self.setup_models()

    def load_from_checkpoint(self) -> None:
        """Load the optimized prompt templates from the specified checkpoint
        name and update the internal beam."""
        m_path = FLAGS.model_path
        ckp_name = FLAGS.checkpoint
        try:
            with open(os.path.join(m_path, f"{ckp_name}.pkl"), "rb") as inp:
                self.search_memory.beam = pickle.load(inp)
        except Exception as e:
            raise Exception("Could not load the checkpoint due to error:{}".format(e))

    def save(self, checkpoint_name: str) -> None:
        """Save the optimized prompt templates to the model_path for the specified checkpoint
        name."""
        m_path = FLAGS.model_path
        if not os.path.exists(m_path):
            os.makedirs(m_path)

        with open(os.path.join(m_path, f"{checkpoint_name}.pkl"), "wb") as outp:
            pickle.dump(self.search_memory.beam, outp, pickle.HIGHEST_PROTOCOL)

    def score_templates(
        self, batch: torch.utils.data.Dataset, prompt_templates: List[PromptTemplate], train: bool = False
    ) -> torch.Tensor:
        """Run a forward computation over the batch for each prompt templates
        and compute the log probability over the batch for that given prompt
        template.

        This function can be called for training or
        inference.
        """
        batch_size, _ = batch["input_ids"].size()
        prompt_lists = [template.tokens for template in prompt_templates]
        class_log_ps = self.forward_pass(batch, train, prompt_lists)

        template_scores = class_log_ps.view(len(prompt_templates), batch_size)
        return template_scores

    def train(self, batch: torch.utils.data.Dataset) -> Dict[str, float]:
        """The train loop for gradient-search method."""
        train_batch_one, train_batch_two = torch.utils.data.random_split(
            batch, lengths=[len(batch) // 2, len(batch) - len(batch) // 2]
        )
        for prompt_index in range(FLAGS.prompt_length):
            template_losses = self.score_templates(train_batch_one, self.search_memory.beam, train=True)
            template_losses = template_losses.mean(dim=1)  # mean across batch_size
            beam_candidates = self.search_memory.generate_beam_candidates(
                embedding_weight=self.model_pool["t5_model"].shared.weight,
                losses=template_losses,
                prompt_step=prompt_index,
            )
            beam_candidate_scores = self.score_templates(train_batch_two, beam_candidates, train=False)
            beam_candidate_scores = beam_candidate_scores.mean(dim=1)  # mean across batch_size
            for index, score in enumerate(beam_candidate_scores.tolist()):
                beam_candidates[index].score = score
            self.search_memory.update_beam(beam_candidates)

        return {"loss_value": self.search_memory.get_beam_loss()}

    def predict(self, batch: torch.utils.data.Dataset) -> Iterator[Dict[str, str]]:
        """The main prediction loop for a given potential class label using a beam of templates."""
        class_log_ps = self.score_templates(batch, self.search_memory.beam, train=False)
        class_log_ps = class_log_ps.mean(dim=0)  # mean across the beam size.
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
