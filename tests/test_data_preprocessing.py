# tests/test_data_preprocessing.py

import pytest
from data.preprocess import load_data, preprocess_data
import pandas as pd
import numpy as np

# Sample dataset for testing
@pytest.fixture
def sample_data():
    data = {
        'feature1': np.random.randn(100),
        'feature2': np.random.randn(100),
        'target': np.random.randn(100)
    }
    return pd.DataFrame(data)

def test_load_data(tmpdir):
    # Create a temporary CSV file to test load_data
    filepath = tmpdir.join('sample.csv')
    data = pd.DataFrame({
        'feature1': [1, 2, 3],
        'feature2': [4, 5, 6],
        'target': [7, 8, 9]
    })
    data.to_csv(filepath, index=False)
    
    # Test load_data function
    loaded_data = load_data(str(filepath))
    assert not loaded_data.empty
    assert list(loaded_data.columns) == ['feature1', 'feature2', 'target']

def test_preprocess_data(sample_data):
    # Test preprocess_data function
    train_data, val_data, test_data, scaler = preprocess_data(sample_data, target_column='target')
    
    # Ensure output is correctly scaled and split
    assert len(train_data[0]) > 0
    assert len(val_data[0]) > 0
    assert len(test_data[0]) > 0
    assert len(train_data[0]) + len(val_data[0]) + len(test_data[0]) == len(sample_data)
    assert scaler is not None
