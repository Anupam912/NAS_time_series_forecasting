# tests/test_training.py

import pytest
import torch
from nas.train_and_evaluate import train_and_evaluate
from nas.trainer import build_model
from data.preprocess import preprocess_data
import numpy as np
import pandas as pd

@pytest.fixture
def sample_data():
    data = {
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'target': np.random.randn(100)
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_config():
    return {
        'model': 'LSTM',
        'input_size': 2,
        'units_per_layer': 64,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'batch_size': 16,
        'epochs': 1
    }

def test_train_and_evaluate(sample_data, sample_config):
    # Preprocess the data and split into train and validation sets
    train_data, val_data, _, _ = preprocess_data(sample_data, target_column='target')

    # Build the model
    model = build_model(sample_config)

    # Use CPU for testing
    device = torch.device('cpu')

    # Test training and evaluation
    validation_loss = train_and_evaluate(model, sample_config, train_data, val_data, device)
    
    assert validation_loss >= 0  # Loss should be a non-negative value
