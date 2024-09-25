# models/transformer_tcn_model.py

import torch
import torch.nn as nn
from models.tcn_model import TCNBlock

class TransformerTCNModel(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, hidden_size, tcn_channels, kernel_size, dropout):
        super(TransformerTCNModel, self).__init__()
        # Transformer encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        # TCN part
        self.tcn_blocks = nn.ModuleList([TCNBlock(input_size, c, kernel_size, dropout) for c in tcn_channels])
        self.fc = nn.Linear(tcn_channels[-1], 1)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)  # Convert to (batch_size, channels, sequence_length) for TCN
        for block in self.tcn_blocks:
            x = block(x)
        x = self.fc(x[:, -1, :])
        return x

def build_transformer_tcn(input_size, num_heads, num_layers, hidden_size, tcn_channels, kernel_size, dropout):
    """
    Build and return a hybrid Transformer-TCN model.
    :param input_size: Number of input features.
    :param num_heads: Number of attention heads in Transformer.
    :param num_layers: Number of Transformer layers.
    :param hidden_size: Hidden size of Transformer feedforward layers.
    :param tcn_channels: List of channel sizes for TCN layers.
    :param kernel_size: Kernel size for TCN layers.
    :param dropout: Dropout rate.
    """
    return TransformerTCNModel(input_size, num_heads, num_layers, hidden_size, tcn_channels, kernel_size, dropout)
