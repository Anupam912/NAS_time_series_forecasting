# nas/train_and_evaluate.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def create_dataloader(features, target, batch_size):
    """
    Create a DataLoader for the given features and target.
    :param features: Scaled features.
    :param target: Scaled target.
    :param batch_size: Size of the batches for training.
    :return: DataLoader object.
    """
    dataset = TensorDataset(torch.tensor(features, dtype=torch.float32), torch.tensor(target, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def train_and_evaluate(model, config, train_data, val_data, device):
    """
    Train the model and evaluate its performance using MSE.
    :param model: The neural network model to be trained.
    :param config: Configuration containing hyperparameters.
    :param train_data: Tuple containing train features and targets.
    :param val_data: Tuple containing validation features and targets.
    :param device: The device (CPU or GPU) to train the model on.
    :return: Validation loss (MSE).
    """
    train_features, train_target = train_data
    val_features, val_target = val_data

    # Move model to device (GPU or CPU)
    model.to(device)

    # Create DataLoaders
    train_loader = create_dataloader(train_features, train_target, config['batch_size'])
    val_loader = create_dataloader(val_features, val_target, config['batch_size'])

    # Set up optimizer and loss function (MSE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(config['epochs']):
        for batch_features, batch_targets in train_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_features, batch_targets in val_loader:
            batch_features, batch_targets = batch_features.to(device), batch_targets.to(device)
            val_outputs = model(batch_features)
            val_loss += criterion(val_outputs, batch_targets).item()

    # Return average validation loss
    return val_loss / len(val_loader)
