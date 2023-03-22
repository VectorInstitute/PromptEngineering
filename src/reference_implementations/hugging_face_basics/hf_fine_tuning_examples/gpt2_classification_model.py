import torch
import torch.nn as nn
from transformers import GPT2Model


class Gpt2ClsModel(nn.Module):
    def __init__(self, pad_token_id: int) -> None:
        super().__init__()
        self.vanilla_gpt2 = GPT2Model.from_pretrained("gpt2")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 4)
        self.pad_token_id = pad_token_id

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Take in a batch of tokenized input and attention masks
        # Their sizes are (batch size, input length, hidden dimension)
        gpt2_activations = self.vanilla_gpt2(input_ids=input_ids, attention_mask=attention_mask)
        # the last hidden state is the last layer activations of the decoder model.
        # It has shape (batch_size, input length = 1024, hidden dimension = 768)
        hidden_states = gpt2_activations.last_hidden_state

        # Need to find the last non-pad token for each component of the batch.
        sequence_lengths = torch.ne(input_ids, self.pad_token_id).sum(-1) - 1

        batch_size = input_ids.shape[0]
        # Get the hidden states of the last non-pad token in each batch. Since we're padding on the right, this is just
        # the value at sequence length.
        last_token_hidden_states = hidden_states[range(batch_size), sequence_lengths]
        # take the token embeddings and send through a DNN layer
        out = self.pre_classifier(last_token_hidden_states)
        out = torch.nn.ReLU()(out)
        # apply dropout to the output
        out = self.dropout(out)
        # send through final classification layer to get activations for the 4 classes
        output = self.classifier(out)
        return output
