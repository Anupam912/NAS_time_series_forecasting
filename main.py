# main.py
from nas.trainer import run_nas
from data.preprocess import load_data, preprocess_data

if __name__ == '__main__':
    # Load the dataset from a CSV file (ensure the CSV is in the correct path)
    data = load_data('data/time_series.csv')
    
    # Preprocess the data (assuming the target column is called 'target')
    train_data, val_data, test_data, scaler = preprocess_data(data, target_column='target')

    # Specify directories to save the plots and logs
    save_dir = 'results/plots'
    log_file = 'results/nas_results.csv'
    
    # Early stopping configuration
    early_stopping_patience = 5
    
    # Number of workers for parallel training (set based on available CPU cores or GPUs)
    num_workers = 2  # Increase this if you have multiple GPUs or CPU cores

    # Run the NAS pipeline with Bayesian Optimization and parallel execution
    run_nas(
        strategy='Bayesian', 
        save_dir=save_dir, 
        log_file=log_file, 
        early_stopping_patience=early_stopping_patience, 
        num_workers=num_workers
    )
