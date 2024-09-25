# utils/logger.py

import csv
import os

def initialize_logger(log_file, fields):
    """
    Initialize a CSV logger to log the NAS results.
    :param log_file: The path to the CSV file where results will be logged.
    :param fields: The column headers for the CSV file.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Initialize CSV file and write headers if file doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

def log_result(log_file, result):
    """
    Log a result (architecture, hyperparameters, and performance) to the CSV file.
    :param log_file: The path to the CSV file where results will be logged.
    :param result: A dictionary containing the results to log (e.g., architecture, hyperparameters, validation loss).
    """
    with open(log_file, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=result.keys())
        writer.writerow(result)
