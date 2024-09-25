# models/transformer_model.py

import torch
import torch.nn as nn

class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, hidden_size, dropout):
        super(TransformerModel, self).__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_size, 1)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :])  # Output the last time step
        return x

def build_transformer(input_size, num_heads, num_layers, hidden_size, dropout):
    """
    Build and return a Transformer model.
    :param input_size: Number of input features.
    :param num_heads: Number of attention heads in Transformer.
    :param num_layers: Number of Transformer layers.
    :param hidden_size: Hidden size of Transformer feedforward layers.
    :param dropout: Dropout rate.
    """
    return TransformerModel(input_size, num_heads, num_layers, hidden_size, dropout)
