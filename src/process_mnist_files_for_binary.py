import numpy as np
import struct
import matplotlib.pyplot as plt

# Function to load idx3-ubyte file (images)
def load_idx3_ubyte(file_path):
    with open(file_path, 'rb') as f:
        magic_number = struct.unpack('>I', f.read(4))[0]
        num_images = struct.unpack('>I', f.read(4))[0]
        num_rows = struct.unpack('>I', f.read(4))[0]
        num_cols = struct.unpack('>I', f.read(4))[0]

        image_data = np.frombuffer(f.read(), dtype=np.uint8)
        image_data = image_data.reshape((num_images, num_rows, num_cols))
        
        print(f"Loaded {num_images} images from {file_path}")

    return image_data

# Function to load idx1-ubyte file (labels)
def load_idx1_ubyte(file_path):
    with open(file_path, 'rb') as f:
        magic_number = struct.unpack('>I', f.read(4))[0]
        num_labels = struct.unpack('>I', f.read(4))[0]

        label_data = np.frombuffer(f.read(), dtype=np.uint8)
        
        print(f"Loaded {num_labels} images from {file_path}")

    return label_data

# Function to save idx3-ubyte file (filtered images)
def save_idx3_ubyte(file_path, images):
    num_images, num_rows, num_cols = images.shape
    with open(file_path, 'wb') as f:
        f.write(struct.pack('>I', 2051))  # Magic number for idx3-ubyte files
        f.write(struct.pack('>I', num_images))
        f.write(struct.pack('>I', num_rows))
        f.write(struct.pack('>I', num_cols))
        f.write(images.tobytes())
        
        print(f"Saved {num_images} images to {file_path}")

# Function to save idx1-ubyte file (filtered labels)
def save_idx1_ubyte(file_path, labels):
    num_labels = labels.shape[0]
    with open(file_path, 'wb') as f:
        f.write(struct.pack('>I', 2049))  # Magic number for idx1-ubyte files
        f.write(struct.pack('>I', num_labels))
        f.write(labels.tobytes())
        
        print(f"Saved {num_labels} images to {file_path}")
        
# Display first 100 images and labels in a 10x10 grid
def display_images(images, labels, num_images=100):
    plt.figure(figsize=(10, 10))  # Set figure size for the grid
    grid_size = int(np.sqrt(num_images))  # 10x10 grid
    
    for i in range(num_images):
        plt.subplot(grid_size, grid_size, i + 1)  # Create subplots in a grid
        plt.imshow(images[i], cmap='gray')  # Display the image in grayscale
        plt.title(f"Label: {labels[i]}", fontsize=8)  # Add label as title
        plt.axis('off')  # Hide the axes for clarity
    
    plt.tight_layout()  # Adjust layout so titles don't overlap
    plt.show()
        
def process_files(image_file_path, label_file_path, processed_image_path, processed_label_path):
    # Load images and labels
    images = load_idx3_ubyte(image_file_path)
    labels = load_idx1_ubyte(label_file_path)

    # Filter for labels 0 and 1
    filter_mask = (labels == 0) | (labels == 1)
    filtered_images = images[filter_mask]
    filtered_labels = labels[filter_mask]

    # Save the filtered images and labels
    save_idx3_ubyte(processed_image_path, filtered_images)
    save_idx1_ubyte(processed_label_path, filtered_labels)
    
    print(f"Filtered images and labels saved to {processed_image_path} and {processed_label_path}")
    
    return filtered_images, filtered_labels

# Paths to original files
test_image_file_path = 'data/raw/t10k-images.idx3-ubyte'
test_label_file_path = 'data/raw/t10k-labels.idx1-ubyte'
# Paths to save filtered data
test_processed_image_path = 'data/processed/test-images-binary.idx3-ubyte'
test_processed_label_path = 'data/processed/test-labels-binary.idx1-ubyte'

filtered_images, filtered_labels = process_files(test_image_file_path, test_label_file_path, test_processed_image_path, test_processed_label_path)

# Paths to original files
train_image_file_path = 'data/raw/train-images.idx3-ubyte'
train_label_file_path = 'data/raw/train-labels.idx1-ubyte'
# Paths to save filtered data
train_processed_image_path = 'data/processed/train-images-binary.idx3-ubyte'
train_processed_label_path = 'data/processed/train-labels-binary.idx1-ubyte'

filtered_images, filtered_labels = process_files(test_image_file_path, test_label_file_path, test_processed_image_path, test_processed_label_path)

# Display the first 100 images with labels
# display_images(filtered_images, filtered_labels, num_images=100)