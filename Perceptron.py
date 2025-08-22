#WAP to Implement Perceptron for understanding Single-Layer Neural Network.

import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=10):
        self.weights = np.zeros(input_size + 1)  
        self.learning_rate = learning_rate
        self.epochs = epochs

    def activation_function(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        x = np.insert(x, 0, 1) 
        weighted_sum = np.dot(self.weights, x)
        return self.activation_function(weighted_sum)

    def train(self, X, y):
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}")
            for xi, target in zip(X, y):
                xi = np.insert(xi, 0, 1) 
                prediction = self.activation_function(np.dot(self.weights, xi))
                error = target - prediction
                self.weights += self.learning_rate * error * xi
                print(f"Input: {xi[1:]}, Target: {target}, Prediction: {prediction}, Updated Weights: {self.weights}")
            print("-" * 50)

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([0, 1, 1, 1]) 

perceptron = Perceptron(input_size=2, learning_rate=0.1, epochs=10)
perceptron.train(X, y)

print("\nTesting Trained Perceptron:")
for x in X:
    result = perceptron.predict(x)
    print(f"Input: {x}, Predicted Output: {result}")
