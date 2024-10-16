import numpy as np
import struct
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Flatten # type: ignore
import matplotlib.pyplot as plt
from process_mnist_files_for_binary import load_idx3_ubyte, load_idx1_ubyte

# File paths
test_image_path = 'data/raw/t10k-images.idx3-ubyte'
test_label_path = 'data/raw/t10k-labels.idx1-ubyte'
train_image_path = 'data/raw/train-images.idx3-ubyte'
train_label_path = 'data/raw/train-labels.idx1-ubyte'

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
        Dense(10, activation = 'linear',  name = 'L3'),
    ], name = "coursera_model"
)

model.summary()

learning_rate = 0.001
model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = tf.keras.optimizers.Adam(learning_rate),
    metrics=['accuracy'] 
)

model.fit(
    train_images_flattened, train_labels,
    epochs = 200
)

### Testing Phase
test_loss, test_accuracy = model.evaluate(test_images_flattened, test_labels, verbose=2)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")


def visualize_mismatched_predictions(model, test_images, test_labels, max_images, search_mismatches):
    # Get model predictions (logits)
    logits = model.predict(test_images)
    
    # Convert logits to probabilities using softmax
    probabilities = tf.nn.softmax(logits)
    
    # Get predicted class labels
    predicted_labels = np.argmax(probabilities, axis=1)
    
    # Find mismatched predictions
    if(search_mismatches == 1):
        mismatches = np.where(predicted_labels != test_labels)[0]
    else:
        mismatches = np.where(predicted_labels == test_labels)[0]
    
    # Limit to the first max_images mismatches
    mismatches = mismatches[:max_images]
    
    # Number of mismatches to display
    num_mismatches = len(mismatches)
    
    # Set up the plot for a 10x10 grid (or smaller if fewer than 100 mismatches)
    grid_size = int(np.ceil(np.sqrt(min(max_images, num_mismatches))))
    plt.figure(figsize=(10, 10))
    
    # Plot each mismatched case
    for i, mismatch_idx in enumerate(mismatches):
        plt.subplot(grid_size, grid_size, i + 1)
        plt.imshow(test_images[mismatch_idx].reshape(28, 28), cmap='gray')  # Reshape to 28x28
        plt.title(f"P: {predicted_labels[mismatch_idx]}, T: {test_labels[mismatch_idx]}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    
visualize_mismatched_predictions(model, test_images_flattened, test_labels, max_images=100, search_mismatches=0)