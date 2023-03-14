"""This is a module to train a classifier on top of the encoder of T5."""
from typing import Dict, Iterator, Tuple, Union

import torch
from absl import flags
from transformers import T5EncoderModel, T5Tokenizer

from src.reference_implementations.prompt_zoo.prompted_t5 import MyBaseT5

FLAGS = flags.FLAGS

flags.DEFINE_integer("classifier_hidden_d", 128, "The number of hidden units used in the classifier.")
flags.DEFINE_integer("num_classes", 3, "Number of classes for sentiment analysis. Only used in linear classifier.")


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
        hidden_vector = torch.sum(good_hidden_states, dim=1) / torch.sum(extended_mask, dim=1)

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

        # use the d_model from the t5 config defined internally from huggingface.
        self.model_pool["classifier_model"] = FFClassifier(self.model_pool["t5_encoder"].config.d_model)

        self.setup_models()

    def predict(self, batch: torch.utils.data.Dataset) -> Iterator[Dict[str, Union[str, float]]]:
        """The main prediction loop using a separate classifier."""

        self.predict_mode_on()
        loaded_batch = self.move_to_gpu(batch, keys=["input_ids", "attention_mask"])

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
