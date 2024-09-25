# utils/helpers.py

import time
import os

class Timer:
    """
    Timer utility for measuring the duration of processes.
    Example usage:
        with Timer("Training time"):
            # Code to time
    """
    def __init__(self, name="Process"):
        self.name = name
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        print(f"[{self.name}] started...")

    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = time.time()
        elapsed_time = end_time - self.start_time
        print(f"[{self.name}] completed in {elapsed_time:.2f} seconds.")

def save_dict_to_file(dictionary, filepath):
    """
    Save a dictionary to a file (can be useful for saving search space configurations or final results).
    :param dictionary: The dictionary to save.
    :param filepath: The path to the file where the dictionary will be saved.
    """
    import json
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(dictionary, f, indent=4)

def load_dict_from_file(filepath):
    """
    Load a dictionary from a file (can be useful for loading saved configurations).
    :param filepath: The path to the file where the dictionary is saved.
    :return: The loaded dictionary.
    """
    import json
    with open(filepath, 'r') as f:
        return json.load(f)
