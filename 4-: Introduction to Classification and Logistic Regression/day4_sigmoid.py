import numpy as np
import matplotlib.pyplot as plt

# Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Generate values
z = np.linspace(-10, 10, 100)
sigmoid_values = sigmoid(z)

# Plot
plt.plot(z, sigmoid_values)
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("σ(z")
plt.grid()
plt.show()
