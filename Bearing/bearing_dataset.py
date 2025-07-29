"""
generate_dataset.py

Dataset Preprocessing Script for Bearing RUL Prediction
--------------------------------------------------------

Description:
This script reads preprocessed CSV files of bearing data, applies standard normalization,
generates sliding window sequences for RUL prediction, and saves the result as PyTorch tensors.

Generated Output:
- train_set, train_label, test_set, test_label: joblib dump files in ./dataresult/
- scaler: StandardScaler object saved for inference
"""

import os
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from joblib import dump

# ------------------------- Configurations -------------------------
DATA_DIR = './dataresult'
WINDOW_SIZE = 10
TRAIN_FILES = ['samples_data_Bearing1_1.csv', 'samples_data_Bearing1_2.csv']
TEST_FILES = [
    'samples_data_FUll_Bearing1_3.csv',
    'samples_data_FUll_Bearing1_4.csv',
    'samples_data_FUll_Bearing1_5.csv',
    'samples_data_FUll_Bearing1_6.csv',
    'samples_data_FUll_Bearing1_7.csv',
]

# ------------------------- Setup -------------------------
os.makedirs(DATA_DIR, exist_ok=True)

# ------------------------- Load CSV Files -------------------------
def load_csv(file_path):
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load {file_path}: {e}")

def extract_features_and_labels(df):
    features = df.drop(columns=['rul']).values
    labels = df[['rul']].values
    return features, labels

# Load and preprocess training data
train_x_list, train_y_list = [], []

for file in TRAIN_FILES:
    df = load_csv(os.path.join(DATA_DIR, file))
    x, y = extract_features_and_labels(df)
    train_x_list.append(x)
    train_y_list.append(y)

x_train = np.vstack(train_x_list)
y_train = np.vstack(train_y_list)

# Load and concatenate testing data
test_x_list, test_y_list = [], []

for file in TEST_FILES:
    df = load_csv(os.path.join(DATA_DIR, file))
    x, y = extract_features_and_labels(df)
    test_x_list.append(x)
    test_y_list.append(y)

x_test = np.vstack(test_x_list)
y_test = np.vstack(test_y_list)

# ------------------------- Normalize -------------------------
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
y_train = scaler.fit_transform(y_train)

x_test = scaler.transform(x_test)
y_test = scaler.transform(y_test)

# Save the scaler
dump(scaler, os.path.join(DATA_DIR, 'scaler'))

# ------------------------- Sliding Window Function -------------------------
def create_windowed_dataset(features, labels, window_size):
    x_list, y_list = [], []
    for i in range(len(features) - window_size):
        x_list.append(features[i:i+window_size])
        y_list.append(labels[i + window_size])
    x_tensor = torch.tensor(np.array(x_list), dtype=torch.float32)
    y_tensor = torch.tensor(np.array(y_list), dtype=torch.float32)
    return x_tensor, y_tensor

# ------------------------- Create Datasets -------------------------
train_set, train_label = create_windowed_dataset(x_train, y_train, WINDOW_SIZE)
test_set, test_label = create_windowed_dataset(x_test, y_test, WINDOW_SIZE)

# ------------------------- Save Datasets -------------------------
dump(train_set, os.path.join(DATA_DIR, 'train_set'))
dump(train_label, os.path.join(DATA_DIR, 'train_label'))
dump(test_set, os.path.join(DATA_DIR, 'test_set'))
dump(test_label, os.path.join(DATA_DIR, 'test_label'))

# ------------------------- Print Info -------------------------
print("Preprocessing complete.")
print(f"Train Set: {train_set.shape}, Train Label: {train_label.shape}")
print(f"Test Set:  {test_set.shape}, Test Label:  {test_label.shape}")
