# models/tcn_model.py

import torch
import torch.nn as nn

class TCNBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, dropout):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_size, output_size, kernel_size, padding=(kernel_size - 1) // 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        return x

class TCNModel(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout):
        super(TCNModel, self).__init__()
        layers = []
        for i in range(len(num_channels)):
            layers.append(TCNBlock(input_size if i == 0 else num_channels[i-1], num_channels[i], kernel_size, dropout))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = self.network(x.transpose(1, 2))  # Convert to (batch_size, channels, sequence_length)
        x = self.fc(x[:, -1, :])
        return x

def build_tcn(input_size, num_channels, kernel_size, dropout):
    """
    Build and return a TCN model.
    :param input_size: Number of input features.
    :param num_channels: Number of channels in each TCN layer.
    :param kernel_size: Kernel size for TCN layers.
    :param dropout: Dropout rate for TCN layers.
    """
    return TCNModel(input_size, num_channels, kernel_size, dropout)
