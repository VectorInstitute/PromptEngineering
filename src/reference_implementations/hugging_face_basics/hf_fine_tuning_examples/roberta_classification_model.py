import torch
import torch.nn as nn
from transformers import RobertaModel


class RobertaClsModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vanilla_roberta = RobertaModel.from_pretrained("roberta-base")
        self.pre_classifier = torch.nn.Linear(768, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 4)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Take in a batch of tokenized input and attention masks
        # Their sizes are (batch size, input length, hidden dimension)
        roberta_activations = self.vanilla_roberta(input_ids=input_ids, attention_mask=attention_mask)
        # The first output is the cls token (which is the same as the <s> token for RoBERTa), which is typically
        # used for sequence classification
        last_layer_activation = roberta_activations.last_hidden_state
        # The first output of the sequence is the cls token (which is the same as the <s> token for RoBERTa),
        # which is typically used for sequence classification
        cls_activation = last_layer_activation[:, 0, :]
        # take the token embeddings and send through a DNN layer
        out = self.pre_classifier(cls_activation)
        out = torch.nn.ReLU()(out)
        # apply dropout to the output
        out = self.dropout(out)
        # send through finall classification layer to get activations for the 4 classes
        output = self.classifier(out)
        return output
