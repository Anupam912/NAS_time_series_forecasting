# models/lstm_tcn_model.py

import torch
import torch.nn as nn
from models.tcn_model import TCNBlock

class LSTMTCNModel(nn.Module):
    def __init__(self, input_size, lstm_hidden_size, lstm_num_layers, tcn_channels, kernel_size, dropout):
        super(LSTMTCNModel, self).__init__()
        # LSTM part
        self.lstm = nn.LSTM(input_size, lstm_hidden_size, lstm_num_layers, batch_first=True, dropout=dropout)
        # TCN part
        self.tcn_blocks = nn.ModuleList([TCNBlock(lstm_hidden_size, c, kernel_size, dropout) for c in tcn_channels])
        self.fc = nn.Linear(tcn_channels[-1], 1)

    def forward(self, x):
        output, _ = self.lstm(x)
        output = output.transpose(1, 2)  # Convert to (batch_size, channels, sequence_length) for TCN
        for block in self.tcn_blocks:
            output = block(output)
        output = self.fc(output[:, -1, :])
        return output

def build_lstm_tcn(input_size, lstm_hidden_size, lstm_num_layers, tcn_channels, kernel_size, dropout):
    """
    Build and return a hybrid LSTM-TCN model.
    :param input_size: Number of input features.
    :param lstm_hidden_size: Number of hidden units in LSTM.
    :param lstm_num_layers: Number of LSTM layers.
    :param tcn_channels: List of channel sizes for TCN layers.
    :param kernel_size: Kernel size for TCN layers.
    :param dropout: Dropout rate.
    """
    return LSTMTCNModel(input_size, lstm_hidden_size, lstm_num_layers, tcn_channels, kernel_size, dropout)
