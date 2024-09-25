# models/lstm_model.py

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = self.fc(output[:, -1, :])  # Take output of the last time step
        return output

def build_lstm(input_size, hidden_size, num_layers, dropout):
    """
    Build and return an LSTM model.
    :param input_size: Number of input features.
    :param hidden_size: Number of units in LSTM hidden layer.
    :param num_layers: Number of LSTM layers.
    :param dropout: Dropout rate between LSTM layers.
    """
    return LSTMModel(input_size, hidden_size, num_layers, dropout)
