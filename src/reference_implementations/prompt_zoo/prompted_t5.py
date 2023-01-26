"""This module implements different ideas for fine-tuning a T5 model, which is
adapted with the prefix language modelling, on some downstream NLP datasets.

The module implements the following baselines:
1 - Fully fine-tuning all the parameters of the T5 model.
2 - Only fine-tuning the shared input embedding layer of the T5 encoder/decoder.
3 - Only fine-tuning the output embedding layer of the T5 decoder.
4 - Fine-tuning both the shared input embedding layer +
    the output embedding layer of the T5 decoder.
5 - No fine-tuning, however, augment input with prompt instructions + in-context examples.
6 - Search for the discrete prompts to augment the input using the gradient of the T5.
7 - Initialize some soft-prompt vectors and augment to the input embedding matrix and
    only fine-tune those prompt vectors on the downstream task.
8 - Just train a classifier on top of the encoder of T5.
9 - Consider (7) and (8) together; augment the input with prompt vectors and
    train a classifier on top.
"""

import os
import random
from abc import abstractmethod
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from absl import flags
from transformers import T5EncoderModel, T5ForConditionalGeneration, T5Tokenizer

from src.reference_implementations.prompt_zoo.model_utility import (
    clear_cache,
    log_of_labels,
    modify_inputs_outputs,
    set_random_seed,
)
from src.reference_implementations.prompt_zoo.prompt_optimizers import optimizer_definer

FLAGS = flags.FLAGS

flags.DEFINE_integer("seed", 42, "the seed number")
flags.DEFINE_bool("gpu", False, "Whether to put the model on gpu or not?")

# the t5-base model with the extra LM adaptation steps.
# https://huggingface.co/google/t5-large-lm-adapt
flags.DEFINE_string("t5_pretrained_model", "google/t5-large-lm-adapt", "initial pre-trained model to use as T5.")

flags.DEFINE_string("mode", "train", "the mode of run? train or test")
flags.DEFINE_string("model_path", "/tmp/", "main directory to save or load the model from")
flags.DEFINE_string("checkpoint", None, "checkpoint name to load from.")
flags.DEFINE_integer("num_classes", 3, "Number of classes for sentiment analysis. Only used in linear classifier.")
flags.DEFINE_float("dropout_rate", 0.1, "dropout_rate used in T5 base.")
flags.DEFINE_integer("classifier_hidden_d", 128, "The number of hidden units used in the classifier.")

NO_SOFT_PROMPT_EXPS = {"all_finetune", "input_finetune", "output_finetune", "input_output_finetune", "no_finetune"}


class FFClassifier(torch.nn.Module):
    """A feedforward multinomial logistic regression over the T5 encoder hidden
    states."""

    def __init__(self, model_d: int) -> None:
        """Arguments:
        model_d (int): The hidden dimension of T5; 768 in T5 base.
        """
        super().__init__()

        # we wish to compare to a case where we have a prompt matrix with FLAGS.prompt_length * model_d parameters.
        # we therefore define a classifier such that we have approximately the same number of extra parameters.
        self.layer = torch.nn.Linear(model_d, FLAGS.classifier_hidden_d, bias=True)

        # using gelu activation over relu
        # https://arxiv.org/abs/1606.08415v4
        self.act = torch.nn.GELU()
        self.classifier = torch.nn.Linear(FLAGS.classifier_hidden_d, FLAGS.num_classes, bias=True)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.loss_fun = torch.nn.NLLLoss()

    def forward(self, hidden_states: torch.Tensor, input_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pass the hidden_vector into the classifier."""
        # mask the correct hidden_states from non-masked tokens.
        # masked tokens are zero!
        b_sz, seq_len, h_dim = hidden_states.size()
        extended_mask = input_mask.view(b_sz, seq_len, 1).expand_as(hidden_states)
        good_hidden_states = hidden_states * extended_mask

        # average pooling as the input feature vector.
        hidden_vector = torch.mean(good_hidden_states, dim=1)

        feature_vector = self.act(self.layer(hidden_vector))
        scores = self.classifier(feature_vector)
        logits = self.log_softmax(scores)
        return scores, logits

    def compute_loss(
        self, hidden_states: torch.Tensor, input_mask: torch.Tensor, class_indices: torch.Tensor
    ) -> torch.Tensor:
        """Compute the cross-entropy loss for the above classifier."""
        _, logits = self.forward(hidden_states, input_mask)
        loss = self.loss_fun(logits, class_indices)
        return loss


class MyBaseT5(torch.nn.Module):
    """Base class for different finetuning + prompt-tuning experiments."""

    def __init__(self) -> None:
        super().__init__()

        set_random_seed(FLAGS.seed)

        # check the gpu actually exists and setup device.
        self.gpu_check = FLAGS.gpu and torch.cuda.is_available()
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
            if FLAGS.t5_exp_type != "gradient_search":
                # create optimizer only for training.
                # based on the experiment type, setup the optimizer.
                self.optimizer = optimizer_definer[FLAGS.t5_exp_type](self.model_pool)
            else:
                # gradient_search does not require optimizer.
                pass
        elif FLAGS.mode in ["test", "inference", "eval"]:
            # load from the given checkpoint.
            self.load_from_checkpoint()
        elif FLAGS.mode in ["no_finetune_test"]:
            # just rely on the pre-trained T5 for prediction and no loading from the checkpoint.
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

    def save(self, checkpoint_name: str) -> None:
        """Save the modules to the model_path for the specified checkpoint
        name."""
        m_path = FLAGS.model_path
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
    def two_batch_train(
        self, batch: torch.utils.data.Dataset, next_batch: torch.utils.data.Dataset
    ) -> Dict[str, float]:
        """The abstract train function."""
        pass

    @abstractmethod
    def predict(self, batch: torch.utils.data.Dataset) -> Iterator[Dict[str, str]]:
        """The abstract predict function."""
        pass

    def forward_pass(
        self, batch: torch.utils.data.Dataset, train: bool = False, prompt_lists: Optional[List[List[int]]] = None
    ) -> torch.Tensor:
        """Run a forward computation over the batch for each prompt lists and
        compute the log probability over the batch for that given prompt
        template.

        This function can be called multiple times for training or
        inference. If there is not prompt, it won't repeat the data per
        prompt template.
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
        orig_labels = loaded_batch["labels"]
        labels = orig_labels.masked_fill(orig_labels == self.tokenizer.pad_token_id, -100)

        t5_model = self.model_pool["t5_model"]

        with torch.set_grad_enabled(train):
            class_log_p = log_of_labels(
                model=t5_model,
                input_ids=loaded_batch["input_ids"],
                input_mask=loaded_batch["attention_mask"],
                decoder_mask=loaded_batch["target_attention_mask"],
                labels=labels,
                loss_func=self.loss_func,
            )

        return class_log_p


class PromptEmbedding(torch.nn.Module):
    """We implement a new Embedding module for the prompt parameters. We only
    update the prompt vectors during training.

    This PromptEmbedding will have a reference to the normal embedding matrix of the T5
    which will be populated when we load the T5 encoders from the huggingface.

    prompt tokens are always at the first prompt_length steps of the
    input after the BOS token (first token).
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
        input after the BOS token.

        split the input sequences into three parts:
            1 - the first BOS token to be embedded by the normal embedding.
            2 - the next prompt_length tokens should be mapped to prompt vectors.
            3 - the rest should be embedded by the normal embedding table of T5 defined for english tokens.

        concatinate the embedded splits into a single split along the sequence dimension.
        """
        batch_size, sequence_length = input.size()

        bos_input, prompt_input, normal_input = torch.split(
            input, [1, self.prompt_length, sequence_length - self.prompt_length - 1], dim=1
        )

        # prompt_embedded has shape: (batch_size,  self.prompt_length, embedding_dim)
        prompt_embedded = self.prompt_embedder(prompt_input)

        # normal_input_embedded has shape: (batch_size,  sequence_length - self.prompt_length, embedding_dim)
        normal_input_embedded = self.normal_embedder(normal_input)
        bos_input_embedded = self.normal_embedder(bos_input.view(batch_size, 1))

        # concat along the dimension 1
        return torch.cat((bos_input_embedded, prompt_embedded, normal_input_embedded), dim=1)


def create_softprompt_T5_encoder() -> torch.nn.Module:
    """This function implements the modifications to the T5 Encoder module of
    the huggingface to include the soft prompt vectors in the input.

    see the original huggingface implementation:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L1765

    Wrapping the T5EncoderModel to introduce our new PromptEmbedding
    module.
    """
    # prompt length
    p_len = FLAGS.prompt_length

    # let the T5EncoderModel load the initial checkpoint of the T5 encoder
    # with the normal embedding table.
    t5_encoder = T5EncoderModel.from_pretrained(FLAGS.t5_pretrained_model)

    # t5_encoder.config.d_model is from the class T5EncoderModel
    # which is the embedding dimension of the embedding table of the T5.
    d_model = t5_encoder.config.d_model
    vocab_size = t5_encoder.config.vocab_size

    prompt_embedding = PromptEmbedding(p_len, d_model, t5_encoder.shared, vocab_size)

    # update the general shared embedding module of huggingface T5.
    # now every call by t5_encoder.shared(input_ids) will use our forward method of the PromptEmbedding
    # hugging face also keeps an internal reference to the shared encoder in the encoder stack module, so we
    # have to modify that as well.
    # we don't want to update the decoder embedding to add the prompt tokens for the output tokens.
    # see https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py#L1344
    t5_encoder.shared = prompt_embedding
    t5_encoder.encoder.embed_tokens = prompt_embedding
    return t5_encoder


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


class FineTuneT5(MyBaseT5):
    """Wrapper class around the MyBaseT5 Model to experiment with different
    finetuning ideas without having classifier."""

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

    def predict(self, batch: torch.utils.data.Dataset) -> Iterator[Dict[str, str]]:
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


class ClassifierT5(MyBaseT5):
    """Wrapper class around the T5 Model with a classifier on top of the T5
    encoder."""

    def __init__(self) -> None:
        super().__init__()

        # construct tokenizer
        self.tokenizer = T5Tokenizer.from_pretrained(FLAGS.t5_pretrained_model)

        # construct the underlying t5 model
        if FLAGS.t5_exp_type == "classifier_finetune":
            self.model_pool["t5_encoder"] = T5EncoderModel.from_pretrained(FLAGS.t5_pretrained_model)
        elif FLAGS.t5_exp_type == "soft_prompt_classifier_finetune":
            self.model_pool["t5_encoder"] = create_softprompt_T5_encoder()

        # use the d_model from the t5 config defined internally from huggingface.
        self.model_pool["classifier_model"] = FFClassifier(self.model_pool["t5_encoder"].config.d_model)

        self.setup_models()

    def predict(self, batch: torch.utils.data.Dataset) -> Iterator[Dict[str, str]]:
        """The main prediction loop using a separate classifier."""

        self.predict_mode_on()
        loaded_batch = self.move_to_gpu(batch, keys=["input_ids", "attention_mask"])

        modify_inputs_outputs(loaded_batch)

        t5_encoder = self.model_pool["t5_encoder"]
        classifier_model = self.model_pool["classifier_model"]

        output = t5_encoder(
            input_ids=loaded_batch["input_ids"],
            attention_mask=loaded_batch["attention_mask"],
        )

        encoder_hidden_states = output.last_hidden_state

        _, logits = classifier_model(encoder_hidden_states, loaded_batch["attention_mask"])

        predictions = torch.argmax(logits, dim=1).cpu().detach().numpy()

        # not efficient, but let's pair input along the prediction class.
        inputs_str = self.tokenizer.batch_decode(loaded_batch["input_ids"], skip_special_tokens=True)

        for index, input_str in enumerate(inputs_str):
            output_row = {
                "predicted_class": predictions[index],
                "input": input_str,
            }
            yield output_row

    def train(self, batch: torch.utils.data.Dataset) -> Dict[str, float]:
        """The classifier training step."""
        self.train_mode_on()
        loaded_batch = self.move_to_gpu(batch, keys=["input_ids", "attention_mask", "class_indices"])

        modify_inputs_outputs(loaded_batch)

        t5_encoder = self.model_pool["t5_encoder"]
        classifier_model = self.model_pool["classifier_model"]

        output = t5_encoder(
            input_ids=loaded_batch["input_ids"],
            attention_mask=loaded_batch["attention_mask"],
        )
        encoder_hidden_states = output.last_hidden_state
        loss = classifier_model.compute_loss(
            encoder_hidden_states, loaded_batch["attention_mask"], loaded_batch["class_indices"]
        )
        loss_value = loss.item()

        # backProp
        loss.backward()

        # optimize
        self.optimizer.step()

        return {"loss_value": loss_value}
