# # You are not allowed to import any other libraries or modules.

import torch
import torch.nn as nn
import numpy as np

def train(model, criterion, optimizer, train_dataloader, num_epoch, device):
    model.to(device)
    avg_train_loss, avg_train_acc = [], []

    for epoch in range(num_epoch):
        model.train()
        batch_train_loss, batch_train_acc = train_one_epoch(model, criterion, optimizer, train_dataloader, device)
        avg_train_acc.append(np.mean(batch_train_acc))
        avg_train_loss.append(np.mean(batch_train_loss))

        print(f'Epoch [{epoch}] Average training loss: {avg_train_loss[-1]:.4f}, '
              f'Average training accuracy: {avg_train_acc[-1]:.4f}')

    return model


def train_one_epoch(model, criterion, optimizer, train_dataloader, device):
    batch_train_loss, batch_train_acc = [], []

    for inputs, targets in train_dataloader:
        # Move inputs and targets to the designated device
        inputs, targets = inputs.to(device), targets.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Calculate predicted classes from outputs
        predicted = torch.argmax(outputs, dim=1)  # Get class indices from output logits
        
        # Convert targets from one-hot to class indices for accuracy calculation
        target_indices = torch.argmax(targets, dim=1)

        # Calculate accuracy
        correct = (predicted == target_indices).sum().item()
        accuracy = correct / targets.size(0)

        # Store the loss and accuracy
        batch_train_loss.append(loss.item())
        batch_train_acc.append(accuracy)

    return batch_train_loss, batch_train_acc


def test(model, test_dataloader, device):
    model.to(device)
    model.eval()
    batch_test_acc = []

    # Disable gradient computation during testing
    with torch.no_grad():
        for inputs, targets in test_dataloader:
            # Move inputs and targets to the designated device
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            predicted = torch.argmax(outputs, dim=1)  # Get predicted class indices
            
            # Convert targets from one-hot to class indices for accuracy calculation
            target_indices = torch.argmax(targets, dim=1)

            # Calculate accuracy
            correct = (predicted == target_indices).sum().item()
            accuracy = correct / targets.size(0)
            batch_test_acc.append(accuracy)

    print(f"The test accuracy is {torch.mean(torch.tensor(batch_test_acc)):.4f}.\n")