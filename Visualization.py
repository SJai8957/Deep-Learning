#WAP to visualize feature maps from CNN (intermediate feature visualization).

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

inputs = Input(shape=(28, 28, 1))
x = Conv2D(32, (3, 3), activation='relu', name="conv1")(inputs)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', name="conv2")(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
outputs = Dense(10, activation="softmax")(x)

model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=1, batch_size=128, validation_split=0.1)

img = x_test[0].reshape(1, 28, 28, 1)

layer_outputs = [layer.output for layer in model.layers if "conv" in layer.name]
activation_model = Model(inputs=model.input, outputs=layer_outputs)

feature_maps = activation_model.predict(img)

for layer_name, feature_map in zip([layer.name for layer in model.layers if "conv" in layer.name], feature_maps):
    num_filters = feature_map.shape[-1]
    size = feature_map.shape[1]
    display_grid = np.zeros((size, size * num_filters))

    for i in range(num_filters):
        f_map = feature_map[0, :, :, i]
        f_map -= f_map.mean()
        f_map /= (f_map.std() + 1e-6)
        f_map *= 64
        f_map += 128
        f_map = np.clip(f_map, 0, 255).astype("uint8")
        display_grid[:, i * size : (i + 1) * size] = f_map

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(f"Feature maps from layer: {layer_name}")
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')

plt.show()
