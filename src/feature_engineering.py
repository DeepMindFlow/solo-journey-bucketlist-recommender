# src/ml_algorithms/feature_engineering.py

import numpy as np
import pandas as pd
from src.data_loader import load_data_from_postgresql # Loading raw data (X, y) from PostgresSQL table, bucket_list_activities



def prepare_numpy_data(X_raw, y_raw):
    """
    Prepare a cleaned version of X_raw (NumPy only) and return a preview array + headers.
    """

    # Original columns assumed:
    # ['user_id', 'activity_id', 'activity_name', 'category', 'activity_type', 'user_mood', 'user_interest_score']
    user_interest_score = X_raw[:, -1].astype(float).reshape(-1, 1)

    # One-hot encode 3 categorical columns
    encoded_parts = []
    encoded_names = []

    for col_idx, col_name in zip([3, 4, 5], ['category', 'activity_type', 'user_mood']):
        col = X_raw[:, col_idx]
        unique_vals = np.unique(col)
        one_hot = (col[:, None] == unique_vals).astype(int)
        encoded_parts.append(one_hot)
        encoded_names.extend([f"{col_name}_{v}" for v in unique_vals])

    # Scale numeric column
    score_scaled = min_max_scaler(user_interest_score)

    # Combine all features
    X_cleaned = np.hstack(encoded_parts + [score_scaled])


    y = np.array(y_raw).astype(int)

    feature_names = encoded_names + ['user_interest_score_scaled']

    return X_cleaned, y, feature_names

def one_hot_encode_column(df, column_name):
    """ Safely one-hot encode a categorical column using NumPy. """
    unique_values = df[column_name].astype(str).unique()
    value_to_index = {val: i for i, val in enumerate(unique_values)}

    encoded = np.zeros((df.shape[0], len(unique_values)))
    for i, val in enumerate(df[column_name].astype(str).values):
        index = value_to_index.get(val.strip(), None)
        if index is not None:
            encoded[i, index] = 1

    column_names = [f"{column_name}_{val}" for val in unique_values]
    return encoded, column_names

def min_max_scaler(x):
    return (x - x.min()) / (x.max() - x.min())

def standardizer(x):
    return (x - x.mean(axis=0)) / x.std(axis=0)


def run_feature_engineering():
    """
    Loads raw data from PostgresSQL and prepares it for machine learning.
    Step 1: Load, convert, and preview types.
    :return: X_raw, y_raw from PostgreSQL


    1. Load raw data from PostgreSQL
    2. Drop irrelevant columns
    3. Encode categorical features
    4. Scale numerical features
    5. Standardize features
    6. Train/test split

    """
    X_raw, y_raw = load_data_from_postgresql()

    # Load data from PostgresSQL
    print("=== Step 1: Raw Data Shapes ===")
    print("X shape:", X_raw.shape)
    print("y shape:", y_raw.shape)

    # Step 2: Convert to pandas dataframe for easier manipulation
    columns = ['user_id', 'activity_id', 'activity_name', 'category',
               'activity_type', 'user_mood', 'user_interest_score']

    df = pd.DataFrame(X_raw, columns=columns)
    print("\n=== Step 2: DataFrame Head ===")
    print(df.head())

    # Convert types
    df['user_interest_score'] = df['user_interest_score'].astype(float)
    y = y_raw.astype(int)

    print("\n=== Preview After Loading and Type Conversion ===")
    print(df.head())
    print(f"\nLabel distribution: {np.bincount(y)}")

    # Drop ID and name columns
    df = df.drop(columns=['user_id', 'activity_id', 'activity_name'])

    # Encode categorical columns
    encoded_parts = []
    encoded_names = []
    for col in ['category', 'activity_type', 'user_mood']:
        encoded, names = one_hot_encode_column(df, col)
        encoded_parts.append(encoded)
        encoded_names.extend(names)

    # Scale numeric feature
    score_scaled = min_max_scaler(df['user_interest_score'].values.reshape(-1, 1))

    # Combine all into feature matrix
    X = np.hstack(encoded_parts + [score_scaled])

    # Optional: standardize to help gradient descent
    X = standardizer(X)
    print("\n=== Step 3: Feature Engineering ===")

    np.random.seed(42)
    indices = np.random.permutation(X.shape[0]) # 0th index = number of rows
    training_split_point = int(0.8 * len(X))

    train_indices = indices[:training_split_point]
    test_indices = indices[training_split_point:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    print("=== Completed Feature Engineering ===")

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = run_feature_engineering()
    print("\n=== Step 4: Feature Engineering ===")
    print(f"\nX_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")






