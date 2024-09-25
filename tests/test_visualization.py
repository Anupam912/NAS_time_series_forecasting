# tests/test_visualization.py

import pytest
import matplotlib.pyplot as plt
from visualization.visualize_results import plot_architecture_performance, plot_hyperparameter_distribution, generate_comparison_report

@pytest.fixture
def sample_results():
    return [
        {'architecture': 'LSTM', 'validation_loss': 0.02, 'learning_rate': 0.001, 'num_layers': 2},
        {'architecture': 'GRU', 'validation_loss': 0.03, 'learning_rate': 0.0001, 'num_layers': 3},
        {'architecture': 'TCN', 'validation_loss': 0.015, 'learning_rate': 0.001, 'num_layers': 4}
    ]

def test_plot_architecture_performance(sample_results, tmpdir):
    # Test if the architecture performance plot is generated
    save_dir = tmpdir.mkdir("plots")
    plot_architecture_performance(sample_results, save_dir=str(save_dir))
    
    plot_file = save_dir.join("architecture_performance.png")
    assert plot_file.exists()

def test_plot_hyperparameter_distribution(sample_results, tmpdir):
    # Test if the hyperparameter distribution plot is generated
    save_dir = tmpdir.mkdir("plots")
    plot_hyperparameter_distribution(sample_results, save_dir=str(save_dir))
    
    plot_file = save_dir.join("hyperparameter_distribution.png")
    assert plot_file.exists()

def test_generate_comparison_report(sample_results, tmpdir):
    # Test if the comparison report is generated
    save_dir = tmpdir.mkdir("reports")
    generate_comparison_report(sample_results, save_dir=str(save_dir))
    
    report_file = save_dir.join("architecture_comparison_report.csv")
    assert report_file.exists()
