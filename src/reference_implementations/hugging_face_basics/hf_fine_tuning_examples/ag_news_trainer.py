from typing import Tuple

import torch
import torch.nn as nn
from transformers.modeling_outputs import SequenceClassifierOutput, SequenceClassifierOutputWithPast


def calcuate_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> int:
    n_correct = (preds == targets).sum().item()
    return n_correct


def infer(
    model: nn.Module, loss_function: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str
) -> Tuple[float, float]:
    model.to(device)
    # set model to eval mode (disable dropout etc.)
    model.eval()
    n_correct = 0
    n_total = 0
    # disable gradient calculations
    with torch.no_grad():
        for _, batch in enumerate(dataloader):
            # send the batch components to proper deviceX
            ids = batch["input_ids"].to(device, dtype=torch.long)
            mask = batch["attention_mask"].to(device, dtype=torch.long)
            targets = batch["label"].to(device, dtype=torch.long)

            # forward pass
            outputs = model(ids, mask)
            loss = loss_function(outputs, targets)
            pred_label = torch.argmax(outputs.data, dim=1)
            n_correct += calcuate_accuracy(pred_label, targets)
            n_total += targets.size(0)
    return n_correct * 100 / n_total, loss.item()


def train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    loss_func: nn.Module,
    device: str,
    n_epochs: int = 1,
) -> None:
    # move model to the GPU (if available)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.0001)
    n_steps_per_report = 100
    model.to(device)
    total_epoch_loss = 0.0

    n_correct = 0
    n_total = 0

    model.train()
    for epoch_number in range(n_epochs):
        print(f"Starting Epoch {epoch_number}")
        total_epoch_loss = 0.0
        total_steps_loss = 0.0
        train_batches = len(train_dataloader)
        for batch_number, batch in enumerate(train_dataloader):
            # send the batch components to proper device
            # ids has shape (batch size, input length = 512)
            ids = batch["input_ids"].to(device, dtype=torch.long)
            # mask has shape (batch size, input length = 512), zeros indicate padding tokens
            mask = batch["attention_mask"].to(device, dtype=torch.long)
            # targets has shape (batch size)
            targets = batch["label"].to(device, dtype=torch.long)

            # forward pass
            outputs = model(input_ids=ids, attention_mask=mask)
            if type(outputs) in {SequenceClassifierOutput, SequenceClassifierOutputWithPast}:
                # For a SequenceClassifierOutput object, we want logits which are of shape (batch size, 4)
                loss = loss_func(outputs.logits, targets)
                pred_label = torch.argmax(outputs.logits, dim=1)
            else:
                # calculate loss for batch
                loss = loss_func(outputs, targets)
                pred_label = torch.argmax(outputs, dim=1)

            batch_loss = loss.item()
            total_steps_loss += batch_loss
            total_epoch_loss += batch_loss

            n_correct += calcuate_accuracy(pred_label, targets)
            n_total += targets.size(0)

            if batch_number % n_steps_per_report == 0 and batch_number > 1:
                print(f"Completed batch number: {batch} of {train_batches}")
                print(f"Training Loss over last {n_steps_per_report} steps: {total_steps_loss/n_steps_per_report}")
                print(f"Training Accuracy over last {n_steps_per_report} steps: {(n_correct*100)/n_total}%")
                val_accuracy, val_loss = infer(model, loss_func, val_dataloader, device)
                print(f"Validation Loss over last {n_steps_per_report} steps: {val_loss}")
                print(f"Validation Accuracy over last {n_steps_per_report} steps: {val_accuracy}%")
                n_correct = 0
                n_total = 0
                total_steps_loss = 0.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss = total_epoch_loss / len(train_dataloader)
        val_accuracy, val_loss = infer(model, loss_func, val_dataloader, device)
        print("------------------------------------------------")
        print(f"Training Loss Epoch: {epoch_loss}")
        print(f"Validation Loss: {val_loss}")
        print(f"Final validation accuracy: {val_accuracy}")
        print("------------------------------------------------")
