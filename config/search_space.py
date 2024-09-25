# config/search_space.py

search_space = {
    'model': ['LSTM', 'GRU', 'TCN', 'Transformer', 'LSTM-TCN', 'GRU-TCN', 'Transformer-TCN'],  # Includes hybrid models
    'num_layers': [1, 2, 3, 4],                        # Number of layers
    'units_per_layer': [32, 64, 128, 256],             # Number of units in each layer
    'dropout': [0.1, 0.2, 0.3, 0.4],                   # Dropout rate
    'learning_rate': [0.001, 0.0001, 0.00001],         # Learning rates
    'batch_size': [16, 32, 64],                        # Batch sizes

    # Transformer-specific hyperparameters
    'num_heads': [4, 8],                               # Number of attention heads in Transformer
    'hidden_size': [128, 256, 512],                    # Hidden size in Transformer

    # Specific for TCN part (used in hybrid models)
    'tcn_channels': [[64, 128], [128, 256], [64, 128, 256]],  # Channels in TCN
    'kernel_size': [3, 5, 7]                           # Kernel sizes for TCN
}

def get_search_space():
    """
    Return the search space for the NAS.
    """
    return search_space
