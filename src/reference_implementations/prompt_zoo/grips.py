"""This module implements the GRIPS: Gradient-free, Edit-based Instruction
Search for Prompting Large Language Models over T5 large language model.

paper: https://arxiv.org/pdf/2203.07281.pdf
github link: https://github.com/archiki/GrIPS
"""

import os
import string
from typing import List

import nltk
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
            self.model_pool["para_model"] = PegasusForConditionalGeneration.from_pretrained(para_model_name)

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
