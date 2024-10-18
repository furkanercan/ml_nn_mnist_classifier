# import numpy as np
# import struct
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Flatten # type: ignore
import matplotlib.pyplot as plt
from process_mnist_files_for_binary import load_idx3_ubyte, load_idx1_ubyte

# File paths
test_image_path = 'data/processed/test-images-binary.idx3-ubyte'
test_label_path = 'data/processed/test-labels-binary.idx1-ubyte'
train_image_path = 'data/processed/train-images-binary.idx3-ubyte'
train_label_path = 'data/processed/train-labels-binary.idx1-ubyte'

# Load train and test data
train_images = load_idx3_ubyte(train_image_path)
train_labels = load_idx1_ubyte(train_label_path)
test_images = load_idx3_ubyte(test_image_path)
test_labels = load_idx1_ubyte(test_label_path)

# Check data shapes
print(f"Train images shape: {train_images.shape}, Train labels shape: {train_labels.shape}")
print(f"Test images shape: {test_images.shape}, Test labels shape: {test_labels.shape}")

# Normalize the images (pixel values between 0 and 1)
train_images = train_images / 255.0
test_images = test_images / 255.0

# Flatten the images outside the model (28x28 to 784)
train_images_flattened = train_images.reshape(train_images.shape[0], -1)  # (12665, 784)
test_images_flattened = test_images.reshape(test_images.shape[0], -1)  # Flatten test set as well
print(f"Flattened train images shape: {train_images_flattened.shape}")
print(f"Flattened test images shape: {test_images_flattened.shape}")


#Describe the model (from Coursera)
model = Sequential(
    [
        tf.keras.Input(shape=(784,)), #specify input here
        Dense(25, activation = 'relu',    name = 'L1'),
        Dense(15, activation = 'relu',    name = 'L2'),
        Dense(1,  activation = 'sigmoid', name = 'L3'),
    ], name = "coursera_model"
)

model.summary()
# [layer1, layer2, layer3] = model.layers
# W1, b1 = layer1.get_weights()
# W2, b2 = layer2.get_weights()
# W3, b3 = layer3.get_weights()
# print(f"W1 shape: {W1.shape}, b1 shape = {b1.shape}")
# print(f"W2 shape: {W2.shape}, b2 shape = {b2.shape}")
# print(f"W3 shape: {W3.shape}, b3 shape = {b3.shape}")

learning_rate = 0.001
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate),
    metrics=['accuracy'] 
)

model.fit(
    train_images_flattened, train_labels,
    epochs = 20
)

### Testing Phase
test_loss, test_accuracy = model.evaluate(test_images_flattened, test_labels, verbose=2)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
