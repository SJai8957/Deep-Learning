import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# 1. Prepare Data (example: XOR problem)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# 2. Build the Model
model = keras.Sequential([
    layers.Dense(units=4, activation='relu', input_shape=(2,)), # Hidden layer with 4 units and ReLU activation
    layers.Dense(units=1, activation='sigmoid') # Output layer with 1 unit and Sigmoid activation for binary classification
])

# 3. Compile the Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. Train the Model
model.fit(X, y, epochs=1000, verbose=0) # Train for 1000 epochs without verbose output

# 5. Make Predictions
predictions = model.predict(X)
print("Predictions:")
print(predictions)

# Evaluate the model
loss, accuracy = model.evaluate(X, y, verbose=0)
print(f"\nLoss: {loss:.4f}")
print(f"Accuracy: {accuracy:.4f}")