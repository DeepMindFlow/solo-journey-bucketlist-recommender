import logging
from src.data_loader import load_data_from_postgresql

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

X, y = load_data_from_postgresql()

logging.info("✅ Loaded data from PostgreSQL!")
logging.info(f"Feature shape: {X.shape}")
logging.info(f"Labels shape: {y.shape}")
logging.info(f"First row (X): {X[0]}")
logging.info(f"First label (y): {y[0]}")