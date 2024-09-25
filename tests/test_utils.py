# tests/test_utils.py

import pytest
import os
from utils.logger import initialize_logger, log_result
from utils.helpers import Timer, save_dict_to_file, load_dict_from_file

@pytest.fixture
def sample_log_file(tmpdir):
    return tmpdir.join("nas_log.csv")

@pytest.fixture
def sample_result():
    return {
        'timestamp': '2024-01-01 12:00:00',
        'architecture': 'LSTM',
        'validation_loss': 0.03,
        'learning_rate': 0.001,
        'num_layers': 2
    }

def test_initialize_logger(sample_log_file):
    # Test that logger initializes and writes the header
    fields = ['timestamp', 'architecture', 'validation_loss', 'learning_rate', 'num_layers']
    initialize_logger(str(sample_log_file), fields)

    assert os.path.exists(sample_log_file)

def test_log_result(sample_log_file, sample_result):
    # Test logging a result to the CSV file
    fields = ['timestamp', 'architecture', 'validation_loss', 'learning_rate', 'num_layers']
    initialize_logger(str(sample_log_file), fields)
    log_result(str(sample_log_file), sample_result)

    with open(sample_log_file, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 2  # One header, one result

def test_timer():
    # Test the Timer utility
    with Timer("Sample process"):
        x = sum([i for i in range(100000)])

def test_save_and_load_dict(tmpdir):
    # Test saving and loading a dictionary
    sample_dict = {'key1': 'value1', 'key2': 'value2'}
    filepath = tmpdir.join("sample_dict.json")

    save_dict_to_file(sample_dict, str(filepath))
    loaded_dict = load_dict_from_file(str(filepath))

    assert sample_dict == loaded_dict
