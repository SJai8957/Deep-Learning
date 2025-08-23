#WAP to Visualize Activation Functions (Sigmoid, ReLU, Tanh).

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

x = np.linspace(-10, 10, 400)
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_tanh = tanh(x)

plt.figure(figsize=(12, 4))

# Sigmoid
plt.subplot(1, 3, 1)
plt.plot(x, y_sigmoid, color='blue')
plt.title('Sigmoid Activation')
plt.grid(True)
plt.xlabel('Input')
plt.ylabel('Output')

# ReLU
plt.subplot(1, 3, 2)
plt.plot(x, y_relu, color='green')
plt.title('ReLU Activation')
plt.grid(True)
plt.xlabel('Input')
plt.ylabel('Output')

# Tanh
plt.subplot(1, 3, 3)
plt.plot(x, y_tanh, color='red')
plt.title('Tanh Activation')
plt.grid(True)
plt.xlabel('Input')
plt.ylabel('Output')

# Display plots
plt.tight_layout()
plt.show()
