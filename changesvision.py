"""
Vision Utils Module for GreenReArchitect

This module contains OpenCV and CNN functions for image processing
and computer vision tasks.

Unit 5: Computer Vision & Satellite Data
"""

import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_image(image_path: str, target_size=(224, 224)):
    """
    Preprocess an image for CNN input.

    Parameters:
    - image_path: Path to the image file
    - target_size: Target size for resizing

    Returns:
    - Preprocessed image array
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")

    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize
    image = cv2.resize(image, target_size)

    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    return image

def detect_hotspots(image: np.ndarray, threshold=0.5):
    """
    Detect hotspots in an image using simple thresholding.

    Parameters:
    - image: Input image array
    - threshold: Threshold for hotspot detection

    Returns:
    - Binary mask of hotspots
    """
    # Convert to grayscale
    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    # Apply threshold
    _, hotspots = cv2.threshold(gray, int(threshold * 255), 255, cv2.THRESH_BINARY)

    return hotspots

def build_cnn_model(input_shape=(224, 224, 3), num_classes=2):
    """
    Build a simple CNN model for image classification.

    Parameters:
    - input_shape: Shape of input images
    - num_classes: Number of output classes

    Returns:
    - Compiled Keras model
    """
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

def create_data_generator(data_dir: str, target_size=(224, 224), batch_size=32):
    """
    Create an ImageDataGenerator for data augmentation.

    Parameters:
    - data_dir: Directory containing image data
    - target_size: Target size for images
    - batch_size: Batch size for generator

    Returns:
    - ImageDataGenerator
    """
    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    train_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    return train_generator, validation_generator

# Example usage
if __name__ == "__main__":
    # Preprocess an image
    image = preprocess_image('path/to/image.jpg')

    # Detect hotspots
    hotspots = detect_hotspots(image)

    # Build CNN model
    model = build_cnn_model()
