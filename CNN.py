#WAP to create and visualize CNN layers.

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input

def create_cnn_model():
    input_layer = Input(shape=(28, 28, 1)) 

    x = Conv2D(32, (3, 3), activation='relu', name='conv1')(input_layer)
    x = MaxPooling2D((2, 2), name='pool1')(x)
    x = Conv2D(64, (3, 3), activation='relu', name='conv2')(x)
    x = MaxPooling2D((2, 2), name='pool2')(x)
    x = Flatten(name='flatten')(x)
    x = Dense(64, activation='relu', name='fc1')(x)
    output_layer = Dense(10, activation='softmax', name='output')(x)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def visualize_layers(model, image):
    layer_outputs = [layer.output for layer in model.layers if 'conv' in layer.name or 'pool' in layer.name]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(np.expand_dims(image, axis=0))
    for layer_name, activation in zip([layer.name for layer in model.layers if 'conv' in layer.name or 'pool' in layer.name], activations):
        n_features = activation.shape[-1]
        size = activation.shape[1]
        display_grid = np.zeros((size, size * n_features))
        for i in range(n_features):
            x = activation[0, :, :, i]
            x -= x.mean()
            x /= (x.std() + 1e-5)
            x *= 64
            x += 128
            x = np.clip(x, 0, 255).astype('uint8')
            display_grid[:, i * size: (i + 1) * size] = x
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(f"Layer: {layer_name}")
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')
        plt.show()

def load_sample_image():
    (x_train, _), _ = tf.keras.datasets.mnist.load_data()
    image = x_train[0].astype('float32') / 255.0
    image = np.expand_dims(image, axis=-1)
    return image

cnn_model = create_cnn_model()
cnn_model.summary()
sample_image = load_sample_image()
visualize_layers(cnn_model, sample_image)
