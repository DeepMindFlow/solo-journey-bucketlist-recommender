import numpy as np
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def sigmoid(z):
    """
    The sigmoid activation function.
    Takes any real number z and squashes it to a value between 0 and 1.
    """
    return 1 / (1 + np.exp(-z))

# Generate a range of z values (from negative to positive)
z_values = np.linspace(-10, 10, 100)
sigmoid_output = sigmoid(z_values)

plt.figure(figsize=(8, 6))
plt.plot(z_values, sigmoid_output, color='blue', linewidth=2)
plt.title('Sigmoid Function $\sigma(z) = \\frac{1}{1 + e^{-z}}$')
plt.xlabel('z')
plt.ylabel('$\sigma(z)$ (Probability)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.axvline(0, color='gray', linestyle='--', label='z = 0')
plt.axhline(0.5, color='red', linestyle='--', label='$\sigma(z)$ = 0.5')
plt.legend()
plt.show()

logging.info(f"Sigmoid(0) = {sigmoid(0)}")
logging.info(f"Sigmoid(large positive) = {sigmoid(100)}")
logging.info(f"Sigmoid(large negative) = {sigmoid(-100)}")