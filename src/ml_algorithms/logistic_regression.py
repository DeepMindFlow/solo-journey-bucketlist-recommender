import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000, verbose=False):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.verbose = verbose
        self.cost_history = []

        logging.info("Initializing Logistic Regression Class...")
        logging.debug(f"Learning rate: {self.learning_rate}")
        logging.debug(f"Number of iterations: {self.num_iterations}")
        logging.debug(f"Initial weights: {self.weights}")
        logging.debug(f"Initial bias: {self.bias}")

    def sigmoid_function(self, z):
        logging.debug("Applying Sigmoid Activation Function")
        z = np.clip(z, -500, 500)
        result = 1 / (1 + np.exp(-z))
        logging.debug(f"sigmoid(z) sample = {result[:5] if len(result) > 5 else result}")
        return result

    def cost_function_error(self, y_true, y_pred):
        logging.debug("Computing cost function error")
        cost_function_error = -(y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))
        total_cost_function_error = np.mean(cost_function_error)
        
        logging.debug(f"y_true (first 5): {y_true[:5]}")
        logging.debug(f"y_pred (first 5): {y_pred[:5]}")
        logging.debug(f"Cost per sample (first 5): {cost_function_error[:5]}")
        logging.debug(f"Total cost (J): {total_cost_function_error:.6f}")
        
        return total_cost_function_error

    def fit(self, X, y):
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)

        logging.info("Training Logistic Regression Model...")
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0
        logging.debug(f"Initial weights: {self.weights}")
        logging.debug(f"Initial bias: {self.bias}")

        for i in range(self.num_iterations):
            if self.verbose and (i % 100 == 0 or i == self.num_iterations - 1):
                logging.debug(f"Iteration {i + 1}")

            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid_function(z)

            if i % 100 == 0 or i == self.num_iterations - 1:
                cost = self.cost_function_error(y, y_pred)
                logging.debug(f"Cost at iteration {i}: {cost:.4f}")

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        logging.info("Training complete!")
        logging.debug(f"Final Weights (first 5): {self.weights[:5]}")
        logging.debug(f"Final Bias: {self.bias}")
        
        return self

    def predict_proba(self, X):
        X = np.array(X, dtype=float)
        linear_combination_output = np.dot(X, self.weights) + self.bias
        logging.debug(f"z (dot product + bias) sample: {linear_combination_output[:10]}")
        logging.debug(f"z min: {np.min(linear_combination_output)}")
        logging.debug(f"z max: {np.max(linear_combination_output)}")
        return self.sigmoid_function(linear_combination_output)

    def predict(self, X, threshold=0.8):
        X = np.array(X, dtype=float)
        predictions = self.predict_proba(X)
        return np.array([1 if p >= threshold else 0 for p in predictions])

