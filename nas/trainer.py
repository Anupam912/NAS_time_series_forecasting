# nas/trainer.py

import os
import torch
import concurrent.futures
from datetime import datetime  # Add this for logging timestamps
from nas.rl_controller import RLController
from nas.bayesian_optimization import run_bayesian_optimization
from config.search_space import get_search_space
from utils.logger import initialize_logger, log_result
from visualization.visualize_results import plot_architecture_performance, plot_hyperparameter_distribution, generate_comparison_report
from nas.train_and_evaluate import train_and_evaluate  # Import the evaluation function
from data.preprocess import load_data, preprocess_data  # Import data preprocessing functions

# Define build_model function
from models.lstm_model import build_lstm
from models.gru_model import build_gru
from models.tcn_model import build_tcn
from models.transformer_model import build_transformer
from models.lstm_tcn_model import build_lstm_tcn
from models.gru_tcn_model import build_gru_tcn
from models.transformer_tcn_model import build_transformer_tcn

def run_nas(strategy='RL', save_dir=None, log_file=None, early_stopping_patience=5, num_workers=4):
    """
    Run NAS using either RL-based or Bayesian Optimization-based strategy.
    :param strategy: The NAS strategy to use ('RL' or 'Bayesian').
    :param save_dir: Directory to save the results (plots, logs, etc.).
    :param log_file: File to log the results.
    :param early_stopping_patience: Patience for early stopping.
    :param num_workers: Number of parallel workers (for parallel execution).
    """
    search_space = get_search_space()
    results = []
    
    # Set the default save directory if not provided
    if save_dir is None:
        save_dir = 'results'
    
    # Log file path for NAS results
    log_file_path = os.path.join(save_dir, 'logs', 'nas_results.csv')
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)  # Ensure directory exists

    # Early stopping object
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    
    # Initialize the logger if the log file is specified
    if log_file:
        fields = ['timestamp', 'architecture', 'validation_loss', 'learning_rate', 'num_layers']
        initialize_logger(log_file_path, fields)
    
    # NAS search strategy
    if strategy == 'RL':
        rl_controller = RLController(search_space)
        sampled_configs = [rl_controller.sample_architecture() for _ in range(num_workers)]
    elif strategy == 'Bayesian':
        sampled_configs = run_bayesian_optimization(n_trials=num_workers)
    
    # Get the list of devices (GPUs or CPU)
    devices = [torch.device(f'cuda:{i}') if torch.cuda.device_count() > i else torch.device('cpu') for i in range(num_workers)]
    
    # Train architectures in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(train_architecture_in_parallel, config, early_stopping, log_file_path, device) 
            for config, device in zip(sampled_configs, devices)
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

    # Visualize and save the results
    plot_architecture_performance(results, os.path.join(save_dir, 'plots'))
    plot_hyperparameter_distribution(results, os.path.join(save_dir, 'plots'))
    generate_comparison_report(results, os.path.join(save_dir, 'reports'))


def build_model(config):
    """
    Build a model based on the architecture type and hyperparameters specified in the config.
    :param config: A dictionary with architecture type and hyperparameters.
    :return: An instance of the selected model.
    """
    if config['model'] == 'LSTM':
        return build_lstm(config['input_size'], config['units_per_layer'], config['num_layers'], config['dropout'])
    elif config['model'] == 'GRU':
        return build_gru(config['input_size'], config['units_per_layer'], config['num_layers'], config['dropout'])
    elif config['model'] == 'TCN':
        return build_tcn(config['input_size'], config['tcn_channels'], config['kernel_size'], config['dropout'])
    elif config['model'] == 'Transformer':
        return build_transformer(config['input_size'], config['num_heads'], config['num_layers'], config['hidden_size'], config['dropout'])
    elif config['model'] == 'LSTM-TCN':
        return build_lstm_tcn(config['input_size'], config['units_per_layer'], config['num_layers'], config['tcn_channels'], config['kernel_size'], config['dropout'])
    elif config['model'] == 'GRU-TCN':
        return build_gru_tcn(config['input_size'], config['units_per_layer'], config['num_layers'], config['tcn_channels'], config['kernel_size'], config['dropout'])
    elif config['model'] == 'Transformer-TCN':
        return build_transformer_tcn(config['input_size'], config['num_heads'], config['num_layers'], config['hidden_size'], config['tcn_channels'], config['kernel_size'], config['dropout'])
    else:
        raise ValueError(f"Unknown model type: {config['model']}")

# Define load_train_val_data function
def load_train_val_data(config):
    """
    Load and preprocess the training and validation data.
    :param config: A dictionary containing the model configuration and file paths.
    :return: Preprocessed training and validation data.
    """
    # Load the dataset (ensure this path points to your dataset)
    data = load_data(config['data_path'])
    
    # Preprocess the data (scale and split)
    train_data, val_data, test_data, scaler = preprocess_data(data, target_column=config['target_column'])
    
    return train_data, val_data

# Early stopping and trainer functions
class EarlyStopping:
    """
    Early stopping mechanism to stop training if validation loss doesn't improve.
    """
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

def get_device():
    """
    Get the available GPU(s) or fallback to CPU if no GPU is available.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def train_architecture_in_parallel(config, early_stopping, log_file=None, device=None):
    """
    Train a single architecture configuration on the specified device.
    :param config: Configuration for the model (hyperparameters, architecture type, etc.).
    :param early_stopping: EarlyStopping object to monitor progress.
    :param log_file: Path to the log file.
    :param device: The device to run the model on (CPU or GPU).
    :return: A dictionary containing architecture performance metrics (e.g., validation loss).
    """
    model = build_model(config).to(device)  # Move model to device (GPU or CPU)

    # Load training and validation data
    train_data, val_data = load_train_val_data(config)

    # Train the model and return validation loss
    validation_loss = train_and_evaluate(model, config, train_data, val_data, device)

    # Early stopping check
    if early_stopping.step(validation_loss):
        return None

    result = {
        'architecture': config['model'],
        'validation_loss': validation_loss,
        'learning_rate': config['learning_rate'],
        'num_layers': config['num_layers']
    }

    if log_file:
        result['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_result(log_file, result)

    return result

def run_nas(strategy='RL', save_dir=None, log_file=None, early_stopping_patience=5, num_workers=4):
    """
    Run NAS using either RL-based or Bayesian Optimization-based strategy.
    :param strategy: The NAS strategy to use ('RL' or 'Bayesian').
    :param save_dir: Directory to save the results (plots, logs, etc.).
    :param log_file: File to log the results.
    :param early_stopping_patience: Patience for early stopping.
    :param num_workers: Number of parallel workers (for parallel execution).
    """
    search_space = get_search_space()
    results = []
    
    # Early stopping object
    early_stopping = EarlyStopping(patience=early_stopping_patience)
    
    if log_file:
        fields = ['timestamp', 'architecture', 'validation_loss', 'learning_rate', 'num_layers']
        initialize_logger(log_file, fields)
    
    if strategy == 'RL':
        rl_controller = RLController(search_space)
        sampled_configs = [rl_controller.sample_architecture() for _ in range(num_workers)]
    elif strategy == 'Bayesian':
        sampled_configs = run_bayesian_optimization(n_trials=num_workers)
    
    # Get the list of devices (GPUs or CPU)
    devices = [torch.device(f'cuda:{i}') if torch.cuda.device_count() > i else torch.device('cpu') for i in range(num_workers)]
    
    # Train architectures in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(train_architecture_in_parallel, config, early_stopping, log_file, device) 
            for config, device in zip(sampled_configs, devices)
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

    # Visualize performance and hyperparameters
    plot_architecture_performance(results, save_dir)
    plot_hyperparameter_distribution(results, save_dir)
    
    # Generate comparison report
    generate_comparison_report(results, save_dir)
