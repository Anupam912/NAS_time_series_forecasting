# data/preprocess.py

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_data(filepath):
    """
    Load time series data from a CSV file.
    :param filepath: Path to the CSV file containing time series data.
    :return: Data as a pandas DataFrame.
    """
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data, target_column):
    """
    Preprocess the time series data by scaling it and returning train, validation, and test splits.
    :param data: pandas DataFrame containing time series data.
    :param target_column: The name of the target column (output).
    :return: Scaled train, validation, and test sets, along with the scaler.
    """
    # Extract target and features
    features = data.drop(columns=[target_column])
    target = data[target_column]

    # Scale the features and target
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features)
    scaled_target = scaler.fit_transform(target.values.reshape(-1, 1))

    # Split into train, validation, and test sets (e.g., 70% train, 15% validation, 15% test)
    train_features, test_features, train_target, test_target = train_test_split(
        scaled_features, scaled_target, test_size=0.3, shuffle=False)
    
    val_features, test_features, val_target, test_target = train_test_split(
        test_features, test_target, test_size=0.5, shuffle=False)

    return (train_features, train_target), (val_features, val_target), (test_features, test_target), scaler
