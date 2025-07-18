# test_data_postgres_preview.py

import numpy as np
from src.data_loader import load_data_from_postgresql

X, y = load_data_from_postgresql()

print("=== Preview of Raw Features (X) ===")
print(X[:5])  # Print first 5 rows of features
print(f"Shape of X: {X.shape}")
print()

print("=== Preview of Class Labels (y) ===")
print(y[:10])  # Print first 10 labels
print(f"Shape of y: {y.shape}")
print()

print(f"Data type of X: {X.dtype}")
print(f"Data type of y: {y.dtype}")