"""This module implements the GRIPS: Gradient-free, Edit-based Instruction
Search for Prompting Large Language Models over T5 large language model.

paper: https://arxiv.org/pdf/2203.07281.pdf
github link: https://github.com/archiki/GrIPS
"""

import os
import string
from typing import Dict, List, Tuple, Union

import nltk
import numpy as np
from absl import flags
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from supar import Parser
from transformers import PegasusForConditionalGeneration, PegasusTokenizer, T5ForConditionalGeneration, T5Tokenizer

from src.reference_implementations.prompt_zoo.prompted_t5 import MyBaseT5

FLAGS = flags.FLAGS

flags.DEFINE_string("instruction_mode", default="Instruction Only", help="Type mode of instructions/prompts")
flags.DEFINE_integer("num-compose", default=1, help="Number of edits composed to get one candidate")
flags.DEFINE_string("level", default="phrase", help="level at which edit operations occur")
flags.DEFINE_string("meta-dir", default="grips_logs/", help="folder location to store metadata of search")
flags.DEFINE_string("meta-name", default="search.txt", help="file name to store metadata of search")
flags.DEFINE_integer("patience", default=2, help="Type in the max patience P (counter)")
flags.DEFINE_integer("num-candidates", default=5, help="Number of candidates in each iteration (m)")
flags.DEFINE_integer("num-iter", default=10, help="Max number of search iterations")


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

        self.setup_models()

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
            temperature=1.5
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
    ) -> Tuple[str, List[Union[str, int]]]:
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
            keys = list(phrase_lookup.keys())
            keys.append(-1)
            [i] = np.random.choice(keys, 1)
            if i >= 0:
                after = phrase_lookup[i]
            else:
                after = ""
            if len(delete_tracker) == 0:
                return base, []
            phrase = np.random.choice(delete_tracker, 1)[0]
            return self.add_phrase(base, phrase, after), [phrase]

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
