"""This module implements the GRIPS: Gradient-free, Edit-based Instruction
Search for Prompting Large Language Models over T5 Large Language Model.

This code is based on the following paper and official codebase.
paper: https://arxiv.org/pdf/2203.07281.pdf
github link: https://github.com/archiki/GrIPS
"""

import csv
import io
import os
import pickle
import string
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Tuple, Union

import nltk
import numpy as np
import torch
from absl import flags
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from supar import Parser
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, T5ForConditionalGeneration, T5Tokenizer

from src.reference_implementations.prompt_zoo.metrics import grips_sentiment_metric
from src.reference_implementations.prompt_zoo.prompted_t5 import MyBaseT5

FLAGS = flags.FLAGS

flags.DEFINE_integer("num_compose", default=1, help="Number of edits composed to get one candidate")
flags.DEFINE_string("level", default="phrase", help="level at which edit operations occur")
flags.DEFINE_string("meta_dir", default="grips_logs/", help="folder location to store metadata of search")
flags.DEFINE_string("meta_name", default="search.txt", help="file name to store metadata of search")
flags.DEFINE_integer("num_candidates", default=5, help="Number of candidates in each iteration (m)")
flags.DEFINE_string(
    "grips_initial_prompt",
    "In this task, your job is to generate the sentiment of the next sentence in the output.",
    "An initial instruction to append to the start of the sentence.",
)

nltk.download("punkt")


@dataclass
class GripsPromptTemplate:
    """A dataclass to define the prompt template with two attributes:

    1 - tokens: a list of token indices from the vocabulary table.
        tokens have size prompt_length.
    2 - score: the final balanced accuracy given this prompt template
                across a search dataset.
    """

    tokens: List[int]
    score: float


class GRIPSSearch(MyBaseT5):
    """GRIPS: Gradient-free, Edit-based Instruction Search for
    Prompting Large Language Models over T5 large language model."""

    def __init__(self, edit_operations: List[str] = ["del", "swap", "sub", "add"]) -> None:
        """This initializes parser used to extract phrases."""
        super().__init__()

        # space of edit ops to be considered
        self.edit_operations = edit_operations

        meta_path = os.path.join(FLAGS.meta_dir, FLAGS.meta_name)
        self.meta_file = open(meta_path, "w+")

        self.parser = Parser.load("crf-con-en")
        self.use_add = "add" in self.edit_operations

        # construct main T5 tokenizer.
        self.tokenizer = T5Tokenizer.from_pretrained(FLAGS.t5_pretrained_model)

        # construct the underlying main T5 model.
        t5_model = T5ForConditionalGeneration.from_pretrained(FLAGS.t5_pretrained_model)

        self.model_pool["t5_model"] = t5_model

        if "sub" in self.edit_operations:
            # the paraphrase model used by the GRIPS paper.
            para_model_name = "tuner007/pegasus_paraphrase"
            # this is the tokenizer and the model for the paraphrase Pegasus.
            self.para_tokenizer = PegasusTokenizer.from_pretrained(para_model_name)
            self.para_model = PegasusForConditionalGeneration.from_pretrained(para_model_name).to(self.device)

        # initialize the base candidate into a prompt template.
        self.run_pre_train_loop(FLAGS.grips_initial_prompt)

        self.setup_models()

    def run_pre_train_loop(self, base_instruction: str) -> None:
        """Define the prompt template based on the given base candidate."""
        base_candidate = self.detokenize(word_tokenize(base_instruction))

        self.current_candidate = base_candidate
        self.original_candidate = base_candidate

        instruction_ids = self.tokenizer(self.current_candidate, add_special_tokens=False)["input_ids"]

        # dynamically adjust the prompt length given the current instruction.
        FLAGS.prompt_length = len(instruction_ids)

        self.current_candidate_template = GripsPromptTemplate(tokens=instruction_ids, score=-float("inf"))
        self.operations_tracker: List[Any] = []
        self.meta_file.write(f"Base Candidate:\t {self.current_candidate} \n")
        self.meta_file.write(f"Base Score:\t {str(self.current_candidate_template.score)} \n")
        self.meta_file.write("\n")
        self.delete_tracker: List[Any] = []

    def update_candidate(self, candidate: str) -> None:
        """Update the prompt template based on the given candidate."""
        # https://www.nltk.org/api/nltk.tokenize.html
        # nltk tokenize can detect puctuation and separate on those.
        # $2.5 -> [$, 2.5]
        base_candidate = self.detokenize(word_tokenize(candidate))
        assert word_tokenize(base_candidate) == word_tokenize(candidate)

        instruction_ids = self.tokenizer(base_candidate, add_special_tokens=False)["input_ids"]
        FLAGS.prompt_length = len(instruction_ids)
        self.current_candidate = base_candidate
        self.current_candidate_template.tokens = instruction_ids
        self.current_candidate_template.score = -float("inf")

    def load_from_checkpoint(self) -> None:
        """Load the optimized prompt template from the specified checkpoint
        name and update the internal candidate."""
        m_path = FLAGS.model_path
        ckp_name = FLAGS.checkpoint
        try:
            with open(os.path.join(m_path, f"{ckp_name}.pkl"), "rb") as inp:
                self.current_candidate_template = pickle.load(inp)
                FLAGS.prompt_length = len(self.current_candidate_template.tokens)
                self.current_candidate = self.tokenizer.decode(
                    self.current_candidate_template.tokens, skip_special_tokens=True
                )
                self.current_candidate = self.detokenize(word_tokenize(self.current_candidate))

        except Exception as e:
            raise Exception("Could not load the checkpoint due to error:{}".format(e))

    def save(self) -> None:
        """Save the optimized prompt template to the model_path for the
        specified checkpoint name."""
        m_path = FLAGS.model_path
        checkpoint_name = FLAGS.checkpoint
        if not os.path.exists(m_path):
            os.makedirs(m_path)

        with open(os.path.join(m_path, f"{checkpoint_name}.pkl"), "wb") as outp:
            pickle.dump(self.current_candidate_template, outp, pickle.HIGHEST_PROTOCOL)

    def score_templates(
        self, batch: torch.utils.data.Dataset, prompt_templates: List[GripsPromptTemplate]
    ) -> torch.Tensor:
        """Run a forward computation over the batch for each prompt templates
        and compute the log probability over the batch for each prompt
        template."""
        batch_size, _ = batch["input_ids"].size()
        prompt_lists = [template.tokens for template in prompt_templates]

        # for grips, we always use the backbone LM for inference.
        # no need to compute gradients: train=False.
        class_log_ps = self.forward_pass(batch, train=False, prompt_lists=prompt_lists)

        template_scores = class_log_ps.view(batch_size, len(prompt_templates))
        return template_scores

    def predict(self, batch: torch.utils.data.Dataset) -> Iterator[Dict[str, Union[str, float]]]:
        """The main prediction loop for a given candidate over a batch from the
        search set."""
        class_log_ps = self.score_templates(batch, [self.current_candidate_template])
        # mean across the prompt templates.
        # for grips, we only evaluate one candidate prompt template at a time, so the mean doesn't have an effect.
        class_log_ps = class_log_ps.mean(dim=1)
        class_log_ps = class_log_ps.cpu().detach().numpy()

        # not efficient, but let's pair potential class along the prediction scores.
        # all transformer special tokens will be removed.
        # same labels have been repeated once per template in beam.
        potentials_str = self.tokenizer.batch_decode(batch["labels"], skip_special_tokens=True)
        prompt_str = self.tokenizer.batch_decode(self.current_candidate_template.tokens, skip_special_tokens=True)
        print("evaluating batch with prompt template:", prompt_str)
        for index, potential_class in enumerate(potentials_str):
            output_row = {
                "potential_class": potential_class,
                "prediction_score": class_log_ps[index],
                "prompt_str": prompt_str,
                "gold_class": batch["gold_classes"][index],
            }
            yield output_row

    def grips_score(self, batch: torch.utils.data.Dataset, prediction_file: str) -> float:
        """Predict over the search batch using the current prompt template and
        then return the balanced accuracy + entropy used in the GRIPS paper."""

        # save prediction in a file.
        with io.open(prediction_file, mode="w", encoding="utf-8") as out_fp:
            writer = csv.writer(out_fp, quotechar='"', quoting=csv.QUOTE_ALL)
            header_written = False
            for ret_row in self.predict(batch):
                if not header_written:
                    headers = ret_row.keys()
                    writer.writerow(headers)
                    header_written = True
                writer.writerow(list(ret_row.values()))

        return grips_sentiment_metric(prediction_file)

    def train(self, batch: torch.utils.data.Dataset) -> Dict[str, float]:
        """The train loop for grips method over search set given by batch."""

        # we need to compute the score of the current candidate on the current search set.
        self.current_candidate_template.score = self.grips_score(batch, prediction_file="grips_temp_scores.csv")
        base_score = self.current_candidate_template.score

        # keeping track of what value phrases got deleted to create the key candidate.
        deleted: Dict[str, List[str]] = {}

        # keeping track of what value indices from the delete_tracker got added to create the key candidate.
        added: Dict[str, List[int]] = {}

        phrase_lookup = self.get_phrase_lookup(self.current_candidate)
        if self.current_candidate == self.original_candidate:
            for p in phrase_lookup.values():
                print(p)
        if self.use_add:
            if len(self.delete_tracker):
                # if initially we had add, but then add got deleted from potential edits,
                # consider the add operation again only if we have previously deleted a token.
                if "add" not in self.edit_operations:
                    self.edit_operations.append("add")
            else:
                # if previously we have not deleted anything, we cannot have the add operation in this iteration.
                if "add" in self.edit_operations:
                    self.edit_operations.remove("add")
        if FLAGS.num_compose == 1:
            edits = np.random.choice(self.edit_operations, FLAGS.num_candidates)
        else:
            edits = []
            for n in range(FLAGS.num_candidates):
                edits.append(np.random.choice(self.edit_operations, FLAGS.num_compose))

        print("edits:", edits)

        # generate candidates
        candidates = []
        for edit in edits:
            if isinstance(edit, str):
                self.meta_file.write(f"Performing edit:\t {edit} \n")
                candidate, indices = self.perform_edit(
                    edit, self.current_candidate, phrase_lookup, self.delete_tracker
                )
                if len(candidate.split()) < 3:
                    # ignore too short candidates
                    continue
                self.meta_file.write(f"Generated candidate:\t {candidate} \n")
                candidates.append(candidate)
                if edit == "del":
                    # keep track of the deleted token resulting in "candidate".
                    deleted[candidate] = [phrase_lookup[indices[0]]]
                if edit == "add":
                    if len(indices):
                        # keep track of the index of the added token from the delete tracker
                        # that created the new candidate.
                        added[candidate] = indices
            else:
                self.meta_file.write(f"Performing edit:\t {' '.join(edit)} \n")
                old_candidate = self.current_candidate
                composed_deletes = []
                composed_adds = []
                for op in edit:
                    phrase_lookup = self.get_phrase_lookup(old_candidate)
                    new_candidate, indices = self.perform_edit(op, old_candidate, phrase_lookup, self.delete_tracker)
                    if len(new_candidate.split()) < 3:
                        # ignore too short candidates resulting from multiple delete edits.
                        continue
                    if op == "del":
                        composed_deletes.append(phrase_lookup[indices[0]])
                    if op == "add":
                        if len(indices):
                            composed_adds.append(indices[0])
                    old_candidate = new_candidate
                self.meta_file.write(f"Generated candidate:\t {new_candidate} \n")
                candidates.append(new_candidate)
                if "del" in edit:
                    deleted[new_candidate] = composed_deletes
                if "add" in edit and len(composed_adds) > 0:
                    added[new_candidate] = composed_adds

        scores = []
        current_candidate_temp = self.current_candidate
        for c_idx, candidate in enumerate(candidates):
            # update the current candidate and compute its score.
            self.update_candidate(candidate)
            candidate_score = self.grips_score(batch, prediction_file="grips_temp_scores.csv")
            self.current_candidate_template.score = candidate_score
            scores.append(candidate_score)
            self.meta_file.write(f"Score for Candidate {str(c_idx)} : \t {str(candidate_score)} \n")

        # find the best new candidate if possible.
        if scores:
            best_idx = np.argmax(scores)
            best_score = scores[best_idx]
            if best_score > base_score:
                new_candidate = candidates[best_idx]
                self.operations_tracker.append(edits[best_idx])
                self.meta_file.write("\n New Candidate Found \n")
                self.meta_file.write(f"New Candidate Index:\t {str(best_idx)} \n")
                self.meta_file.write(f"New Candidate:\t {new_candidate} \n")
                self.meta_file.write(f"New Candidate Score:\t {str(best_score)} \n")
                try:
                    self.meta_file.write(f"New Candidate Edit:\t {edits[best_idx]} \n")
                except Exception:
                    self.meta_file.write(f"New Candidate Edit:\t {' '.join(edits[best_idx])} \n")

                print("New Candidate: ", new_candidate)
                if new_candidate in added.keys():
                    print("Notice! Prev tracker: ", self.delete_tracker)
                    # update the delete_tracker if the new candidate is resulting from adding a token
                    # from the delete_tracker.
                    for token_index in added[new_candidate]:
                        try:
                            self.delete_tracker.pop(token_index)
                        except Exception:
                            pass
                    print("Notice! New tracker: ", self.delete_tracker)
                if new_candidate in deleted.keys():
                    # if the new candidate is from deleting a token, then update the delete_tracker
                    # with new delete tokens for future add operations.
                    self.delete_tracker.extend(deleted[new_candidate])

                # update the current candidate with best new_candidate.
                self.update_candidate(new_candidate)
                self.current_candidate_template.score = best_score
                return {"loss_value": 100.00 - best_score}

        # no new best candidate found.
        self.update_candidate(current_candidate_temp)
        self.current_candidate_template.score = base_score
        return {"loss_value": 100.00 - base_score}

    def detokenize(self, tokens: List[str]) -> str:
        """constructs the string back from the nltk tokenizer."""
        return TreebankWordDetokenizer().detokenize(tokens)

    def traverse_tree(self, parsed_tree: nltk.tree.tree.Tree) -> List[str]:
        phrases = []
        for tree in parsed_tree:
            if tree.label() == "_":
                continue
            phrases.append(self.detokenize(tree.leaves()))
            for subtree in tree:
                if type(subtree) == nltk.tree.Tree:
                    if subtree.label() == "_":
                        continue
                    phrases.append(self.detokenize(subtree.leaves()))
                    phrases.extend(self.traverse_tree(subtree))
        return phrases

    def check_child(self, tree: nltk.tree.tree.Tree) -> bool:
        check = False
        count = 0
        total_count = 0
        for subtree in tree:
            total_count += 1
            if type(subtree) == nltk.tree.Tree:
                if subtree.label() == "_":
                    count += 1
        if count >= total_count - count:
            check = True

        return check

    def collect_leaves(self, parsed_tree: nltk.tree.tree.Tree) -> List[str]:
        """A recursive function to traverse a parsed tree and collect
        leaves."""
        leaves = []
        for tree in parsed_tree:
            if type(parsed_tree) != nltk.tree.Tree:
                continue
            if tree.label() == "_":
                leaves.append(self.detokenize(tree.leaves()))
                continue
            if self.check_child(tree):
                leaves.append(self.detokenize(tree.leaves()))
            else:
                leaves.extend(self.collect_leaves(tree))
        return leaves

    def get_phrases(self, instruction: str) -> List[str]:
        """Main function to split instruction into phrases using a CRF based
        Constituency Parser.

        Link of the  parser used: https://github.com/yzhangcs/parser
        This is the way of obtaining disjoint phrases used by the GRIPS
        paper.
        """

        phrases = []
        for sentence in sent_tokenize(instruction):
            parsed_tree = self.parser.predict(word_tokenize(sentence), verbose=False).sentences[0].trees[0]
            leaves = self.collect_leaves(parsed_tree)
            phrases.extend(leaves)
        phrases = [
            self.detokenize(word_tokenize(phrase))
            for phrase in phrases
            if phrase not in string.punctuation or phrase == ""
        ]
        return phrases

    def get_response(self, input_text: str, num_return_sequences: int, num_beams: int) -> List[str]:
        """This function generates a paraphrase version of the input_text using
        the paraphrase model.

        This is useful to support the substitution operation.
        """
        paraphrase_batch = self.para_tokenizer(
            [input_text], truncation=True, padding="longest", max_length=60, return_tensors="pt"
        ).to(self.device)
        translated = self.para_model.generate(
            **paraphrase_batch,
            max_length=60,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            temperature=1.5,
        )
        tgt_text = self.para_tokenizer.batch_decode(translated, skip_special_tokens=True)
        return tgt_text

    def delete_phrase(self, candidate: str, phrase: str) -> str:
        """To support the delete operation of a phrase from a candidate."""
        if candidate.find(" " + phrase) > 0:
            answer = candidate.replace(" " + phrase, " ")
        elif candidate.find(phrase + " ") > 0:
            answer = candidate.replace(phrase + " ", " ")
        else:
            answer = candidate.replace(phrase, "")
        return answer

    def add_phrase(self, candidate: str, phrase: str, after: str) -> str:
        """To support the addition of a phrase after a keyword in the
        candidate."""
        if after == "":
            answer = phrase + " " + candidate
        else:
            if candidate.find(" " + after) > 0:
                answer = candidate.replace(" " + after, " " + after + " " + phrase)
            elif candidate.find(after + " ") > 0:
                answer = candidate.replace(after + " ", after + " " + phrase + " ")
            else:
                answer = candidate.replace(after, after + phrase)
        return answer

    def swap_phrases(self, candidate: str, phrase_1: str, phrase_2: str) -> str:
        """Swap two phrases in the candidate."""
        if candidate.find(" " + phrase_1 + " ") >= 0:
            answer = candidate.replace(" " + phrase_1 + " ", " <1> ")
        else:
            answer = candidate.replace(phrase_1, "<1>")
        if candidate.find(" " + phrase_2 + " ") >= 0:
            answer = candidate.replace(" " + phrase_2 + " ", " <2> ")
        else:
            answer = candidate.replace(phrase_2, "<2>")
        answer = answer.replace("<1>", phrase_2)
        answer = answer.replace("<2>", phrase_1)
        return answer

    def substitute_phrase(self, candidate: str, phrase: str) -> str:
        """Find a paraphrase of the 'phrase' and then replace the 'phrase' with
        its paraphrase in the candidate."""
        # The following beam values are suggested by the original GRIPS paper.
        paraphrases = self.get_response(phrase, num_return_sequences=10, num_beams=10)
        paraphrase = np.random.choice(paraphrases, 1)[0]
        paraphrase = paraphrase.strip(".")
        if candidate.find(" " + phrase) > 0:
            answer = candidate.replace(" " + phrase, " " + paraphrase)
        elif candidate.find(phrase + " ") > 0:
            answer = candidate.replace(phrase + " ", paraphrase + " ")
        else:
            answer = candidate.replace(phrase, paraphrase)
        return answer

    def perform_edit(
        self, edit: str, base: str, phrase_lookup: Dict[int, str], delete_tracker: List[str]
    ) -> Tuple[str, List[int]]:
        """Perform all possible edit operations: del, swap, sub, add."""
        assert edit in {"del", "swap", "sub", "add"}
        if edit == "del":
            [i] = np.random.choice(list(phrase_lookup.keys()), 1)
            return self.delete_phrase(base, phrase_lookup[i]), [i]
        elif edit == "swap":
            try:
                [i, j] = np.random.choice(list(phrase_lookup.keys()), 2, replace=False)
            except Exception:
                # not enough 2 distinct elements.
                [i, j] = np.random.choice(list(phrase_lookup.keys()), 2, replace=True)
            return self.swap_phrases(base, phrase_lookup[i], phrase_lookup[j]), [i, j]
        elif edit == "sub":
            [i] = np.random.choice(list(phrase_lookup.keys()), 1)
            return self.substitute_phrase(base, phrase_lookup[i]), [i]
        else:
            # for add operation.
            # we first need to pick a phrase and then add one of the deleted tokens
            # after this phrase we picked as "after".
            keys = list(phrase_lookup.keys())
            keys.append(-1)
            [i] = np.random.choice(keys, 1)
            if i >= 0:
                after = phrase_lookup[i]
            else:
                # if the sampled i is -1, then we will add to the start of the base.
                after = ""

            if len(delete_tracker) == 0:
                # if we have not deleted any phrase previously, then we skip the add operation.
                return base, []

            [i] = np.random.choice(range(len(delete_tracker)), 1)

            # i is the index in the delete_tracker.
            return self.add_phrase(base, delete_tracker[i], after), [i]

    def get_phrase_lookup(self, base_candidate: str) -> Dict[int, str]:
        """Create a dictionary of potential phrases from the base_candidate and
        store information as a dictionary based on the experiment type."""
        assert FLAGS.level in {"phrase", "word", "sentence", "span"}
        if FLAGS.level == "phrase":
            phrase_lookup = {p: phrase for p, phrase in enumerate(self.get_phrases(base_candidate))}
        elif FLAGS.level == "word":
            words = word_tokenize(base_candidate)
            words = [w for w in words if w not in string.punctuation or w != ""]
            phrase_lookup = {p: phrase for p, phrase in enumerate(words)}
        elif FLAGS.level == "sentence":
            sentences = sent_tokenize(base_candidate)
            phrase_lookup = {p: phrase for p, phrase in enumerate(sentences)}
        elif FLAGS.level == "span":
            phrases = []
            for sentence in sent_tokenize(base_candidate):
                spans_per_sentence = np.random.choice(range(2, 5))  # split sentence into 2, 3, 4, 5 chunks
                spans = np.array_split(word_tokenize(sentence), spans_per_sentence)
                spans = [self.detokenize(s) for s in spans]
                phrases.extend(spans)
            phrase_lookup = {p: phrase for p, phrase in enumerate(phrases)}
        else:
            raise ValueError()
        return phrase_lookup
