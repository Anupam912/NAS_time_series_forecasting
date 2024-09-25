# tests/test_model_building.py

import pytest
from nas.trainer import build_model

@pytest.fixture
def sample_lstm_config():
    return {
        'model': 'LSTM',
        'input_size': 10,
        'units_per_layer': 64,
        'num_layers': 2,
        'dropout': 0.2
    }

@pytest.fixture
def sample_gru_config():
    return {
        'model': 'GRU',
        'input_size': 10,
        'units_per_layer': 64,
        'num_layers': 2,
        'dropout': 0.2
    }

@pytest.fixture
def sample_tcn_config():
    return {
        'model': 'TCN',
        'input_size': 10,
        'tcn_channels': [64, 128],
        'kernel_size': 3,
        'dropout': 0.2
    }

def test_build_lstm_model(sample_lstm_config):
    model = build_model(sample_lstm_config)
    assert model is not None
    assert hasattr(model, 'lstm')  # Check if LSTM layer exists

def test_build_gru_model(sample_gru_config):
    model = build_model(sample_gru_config)
    assert model is not None
    assert hasattr(model, 'gru')  # Check if GRU layer exists

def test_build_tcn_model(sample_tcn_config):
    model = build_model(sample_tcn_config)
    assert model is not None
    assert hasattr(model, 'network')  # Check if TCN layers exist
