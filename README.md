# Automated Neural Architecture Search (NAS) for Time Series Forecasting

## Overview

This project implements an **Automated Neural Architecture Search (NAS)** system to automatically optimize the architecture and hyperparameters of neural networks tailored for **time series forecasting** tasks. The NAS system uses either **Reinforcement Learning (RL)** or **Bayesian Optimization** to search for the best model architecture, such as **LSTM**, **GRU**, **TCN**, or **Transformer-based models**. The models are optimized for predicting future trends based on historical data.

The key components of this project include:

- **Model architectures**: LSTM, GRU, TCN, Transformer, and hybrid models like LSTM-TCN, GRU-TCN, and Transformer-TCN.
- **NAS strategies**: Reinforcement Learning (RL) and Bayesian Optimization.
- **GPU support**: Parallelized NAS with support for training on multiple GPUs.
- **Visualization**: Performance and hyperparameter distribution plots, as well as architecture comparison reports.

## Project Structure

```plaintext
nas_time_series_forecasting/
│
├── config/
│   └── search_space.py            # Defines search space for NAS (architectures and hyperparameters)
│
├── data/
│   └── preprocess.py              # Handles data loading and preprocessing (scaling, splitting)
│
├── models/
│   ├── lstm_model.py              # Defines LSTM architecture
│   ├── gru_model.py               # Defines GRU architecture
│   ├── tcn_model.py               # Defines TCN architecture
│   ├── transformer_model.py       # Defines Transformer architecture
│   ├── lstm_tcn_model.py          # Hybrid LSTM-TCN model
│   ├── gru_tcn_model.py           # Hybrid GRU-TCN model
│   └── transformer_tcn_model.py   # Hybrid Transformer-TCN model
│
├── nas/
│   ├── rl_controller.py           # NAS using Reinforcement Learning (RL)
│   ├── bayesian_optimization.py   # NAS using Bayesian Optimization (Optuna)
│   ├── trainer.py                 # NAS trainer (parallel execution, GPU support, early stopping)
│   └── train_and_evaluate.py      # Training and evaluation of models
│
├── utils/
│   ├── logger.py                  # Logs NAS results (architecture, hyperparameters, validation loss)
│   └── helpers.py                 # Timer and save/load helper functions
│
├── visualization/
│   └── visualize_results.py       # Generates plots and comparison reports
│
├── tests/
│   ├── test_data_preprocessing.py # Unit tests for data loading and preprocessing
│   ├── test_model_building.py     # Unit tests for model creation
│   ├── test_training.py           # Unit tests for training and evaluation
│   └── test_visualization.py      # Unit tests for visualization and report generation
│
├── results/
│   ├── logs/                      # Logs architecture and performance results
│   ├── plots/                     # Visualizations (performance, hyperparameter distribution)
│   └── reports/                   # Architecture comparison reports
│
├── main.py                        # Entry point for running the NAS pipeline
└── requirements.txt               # Python dependencies
```

## **Key Features**

- **Neural Architecture Search (NAS)**: Automates the search for optimal neural network architectures and hyperparameters for time series forecasting tasks.
- **Time Series Models**: LSTM, GRU, TCN, Transformer, and hybrid models such as LSTM-TCN, GRU-TCN, and Transformer-TCN.
- **Search Strategies**:
  - **Reinforcement Learning (RL)**: Uses an RL-based controller to sample architectures.
  - **Bayesian Optimization**: Uses **Optuna** to optimize architectures and hyperparameters.
- **Parallel Execution**: Supports running the NAS process in parallel using multiple GPUs.
- **Early Stopping**: Avoids overfitting during model training by stopping training when validation loss stagnates.
- **Visualization**: Generates performance plots, hyperparameter distributions, and comparison reports.
- **GPU Acceleration**: Trains models on GPUs to speed up the NAS process.

## **Getting Started**

Requirements

- Python 3.7+
- Required packages (can be installed via requirements.txt)

### **Installation**

```bash
git clone https://github.com/Anupam912/NAS_time_series_forecasting.git
cd nas_time_series_forecasting
pip install -r requirements.txt
```

### **Dataset**

Ensure that your time series dataset is in CSV format and stored in the data/ folder. The preprocessing script (preprocess.py) will handle scaling and splitting the dataset.

```python
from data.preprocess import load_data, preprocess_data
data = load_data('data/time_series.csv')
train_data, val_data, test_data, scaler = preprocess_data(data, target_column='target')
```

### **Running the NAS Pipeline**

You can start the NAS process by running `main.py`. You can choose between Reinforcement Learning (RL) and Bayesian Optimization as the search strategy.

```python
from nas.trainer import run_nas

if __name__ == '__main__':
    save_dir = 'results/plots'
    log_file = 'results/nas_results.csv'
    run_nas(strategy='Bayesian', save_dir=save_dir, log_file=log_file, early_stopping_patience=5, num_workers=2)
```

### **Visualizations and Reports**

Once the NAS process completes, performance plots and comparison reports will be saved in the `results/folder`:

Performance Plots: `results/plots/architecture_performance.png`
Hyperparameter Distribution: `results/plots/hyperparameter_distribution.png`
Comparison Report: `results/reports/architecture_comparison_report.csv`

### Running Tests

The project includes unit tests for the key components. You can run all tests using pytest:

```bash
pytest
```
