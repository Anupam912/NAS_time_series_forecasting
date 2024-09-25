# models/gru_model.py

import torch
import torch.nn as nn

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.gru(x)
        output = self.fc(output[:, -1, :])  # Take output of the last time step
        return output

def build_gru(input_size, hidden_size, num_layers, dropout):
    """
    Build and return a GRU model.
    :param input_size: Number of input features.
    :param hidden_size: Number of units in GRU hidden layer.
    :param num_layers: Number of GRU layers.
    :param dropout: Dropout rate between GRU layers.
    """
    return GRUModel(input_size, hidden_size, num_layers, dropout)
