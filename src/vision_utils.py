# """
# vision_utils.py
# Temporary lightweight vision utilities
# CNN/GAN logic will be added later using Google Colab
# """

# import numpy as np

# def preprocess_image(image):
#     # Placeholder preprocessing
#     return image

# def detect_hotspots(image):
#     """
#     Dummy hotspot detector
#     Returns random hotspot mask for now
#     """
#     h, w = 100, 100
#     hotspot_mask = np.random.choice([0, 1], size=(h, w), p=[0.9, 0.1])
#     return hotspot_mask
"""
Vision Utils Module for GreenReArchitect

This module contains CNN and GAN functions for image processing
and computer vision tasks.

Unit 4: Deep Learning (CNN)
Unit 6: Generative AI (GAN)
"""

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Input, Concatenate, Conv2DTranspose, LeakyReLU, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

class HeatDetectorCNN:
    """
    CNN for detecting heat patterns in satellite images.
    Unit 4: Deep Learning
    """

    def __init__(self, input_shape=(224, 224, 3)):
        """
        Initialize the CNN model.

        Parameters:
        - input_shape: Shape of input images
        """
        self.input_shape = input_shape
        self.model = None

    def build_model(self):
        """
        Build the CNN architecture for heat detection.
        """
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')  # Binary classification: hot/not hot
        ])

        self.model.compile(optimizer='adam',
                          loss='binary_crossentropy',
                          metrics=['accuracy'])

    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """
        Train the CNN model.

        Parameters:
        - X_train: Training images
        - y_train: Training labels
        - epochs: Number of training epochs
        - batch_size: Batch size
        """
        if self.model is None:
            self.build_model()

        history = self.model.fit(X_train, y_train,
                               epochs=epochs,
                               batch_size=batch_size,
                               validation_split=0.2)

        return history

    def predict_heat_intensity(self, image):
        """
        Predict heat intensity for an image.

        Parameters:
        - image: Input image array

        Returns:
        - Heat intensity score (0-1)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call build_model and train first.")

        # Preprocess image
        processed = self.preprocess_image(image)
        prediction = self.model.predict(processed, verbose=0)[0][0]
        return prediction

    def generate_pixel_level_heatmap(self, image):
        """
        Generate pixel-level heat intensity map.

        Parameters:
        - image: Input satellite image array

        Returns:
        - Heat intensity map array
        """
        if self.model is None:
            raise ValueError("Model not trained. Call build_model and train first.")

        # For pixel-level prediction, we'll use sliding window approach
        # This is a simplified version - in practice, you'd use a fully convolutional network
        height, width = image.shape[:2]
        heatmap = np.zeros((height, width))

        window_size = 224
        stride = 112

        for y in range(0, height - window_size + 1, stride):
            for x in range(0, width - window_size + 1, stride):
                patch = image[y:y+window_size, x:x+window_size]
                if patch.shape[:2] == (window_size, window_size):
                    intensity = self.predict_heat_intensity(patch)
                    heatmap[y:y+window_size, x:x+window_size] = np.maximum(
                        heatmap[y:y+window_size, x:x+window_size], intensity
                    )

        return heatmap

    def save_model(self, filepath):
        """
        Save the trained CNN model.

        Parameters:
        - filepath: Path to save the model
        """
        if self.model:
            self.model.save(filepath)
            print(f"CNN model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load a trained CNN model.

        Parameters:
        - filepath: Path to the model file
        """
        self.model = load_model(filepath)
        print(f"CNN model loaded from {filepath}")

    def preprocess_image(self, image):
        """
        Preprocess image for CNN input.
        """
        # Resize
        resized = cv2.resize(image, (self.input_shape[0], self.input_shape[1]))

        # Normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Add batch dimension
        return np.expand_dims(normalized, axis=0)

class GreenRedesignGAN:
    """
    GAN for generating green space redesigns.
    Unit 6: Generative AI
    """

    def __init__(self, image_shape=(256, 256, 3)):
        """
        Initialize the GAN.

        Parameters:
        - image_shape: Shape of input/output images
        """
        self.image_shape = image_shape
        self.generator = None
        self.discriminator = None
        self.gan = None

    def build_generator(self):
        """
        Build the generator network (U-Net style).
        """
        inputs = Input(self.image_shape)

        # Encoder
        conv1 = Conv2D(64, (3, 3), padding='same')(inputs)
        conv1 = LeakyReLU(alpha=0.2)(conv1)
        conv1 = Conv2D(64, (3, 3), padding='same')(conv1)
        conv1 = LeakyReLU(alpha=0.2)(conv1)
        pool1 = MaxPooling2D((2, 2))(conv1)

        conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
        conv2 = LeakyReLU(alpha=0.2)(conv2)
        conv2 = Conv2D(128, (3, 3), padding='same')(conv2)
        conv2 = LeakyReLU(alpha=0.2)(conv2)
        pool2 = MaxPooling2D((2, 2))(conv2)

        # Bottleneck
        conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
        conv3 = LeakyReLU(alpha=0.2)(conv3)
        conv3 = Conv2D(256, (3, 3), padding='same')(conv3)
        conv3 = LeakyReLU(alpha=0.2)(conv3)

        # Decoder
        up1 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv3)
        up1 = Concatenate()([up1, conv2])
        conv4 = Conv2D(128, (3, 3), padding='same')(up1)
        conv4 = LeakyReLU(alpha=0.2)(conv4)
        conv4 = Conv2D(128, (3, 3), padding='same')(conv4)
        conv4 = LeakyReLU(alpha=0.2)(conv4)

        up2 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv4)
        up2 = Concatenate()([up2, conv1])
        conv5 = Conv2D(64, (3, 3), padding='same')(up2)
        conv5 = LeakyReLU(alpha=0.2)(conv5)
        conv5 = Conv2D(64, (3, 3), padding='same')(conv5)
        conv5 = LeakyReLU(alpha=0.2)(conv5)

        outputs = Conv2D(3, (1, 1), activation='tanh')(conv5)

        self.generator = Model(inputs, outputs)

    def build_discriminator(self):
        """
        Build the discriminator network (PatchGAN).
        """
        inputs = Input(self.image_shape)

        conv1 = Conv2D(64, (4, 4), strides=(2, 2), padding='same')(inputs)
        conv1 = LeakyReLU(alpha=0.2)(conv1)

        conv2 = Conv2D(128, (4, 4), strides=(2, 2), padding='same')(conv1)
        conv2 = BatchNormalization()(conv2)
        conv2 = LeakyReLU(alpha=0.2)(conv2)

        conv3 = Conv2D(256, (4, 4), strides=(2, 2), padding='same')(conv2)
        conv3 = BatchNormalization()(conv3)
        conv3 = LeakyReLU(alpha=0.2)(conv3)

        conv4 = Conv2D(512, (4, 4), strides=(1, 1), padding='same')(conv3)
        conv4 = BatchNormalization()(conv4)
        conv4 = LeakyReLU(alpha=0.2)(conv4)

        outputs = Conv2D(1, (4, 4), padding='same')(conv4)

        self.discriminator = Model(inputs, outputs)
        self.discriminator.compile(loss='mse', optimizer=Adam(0.0002, 0.5))

    def build_gan(self):
        """
        Build the combined GAN model.
        """
        if self.generator is None or self.discriminator is None:
            raise ValueError("Build generator and discriminator first.")

        self.discriminator.trainable = False

        inputs = Input(self.image_shape)
        generated = self.generator(inputs)
        validity = self.discriminator(generated)

        self.gan = Model(inputs, validity)
        self.gan.compile(loss='mse', optimizer=Adam(0.0002, 0.5))

    def generate_green_redesign(self, input_image):
        """
        Generate a green space redesign from input image.

        Parameters:
        - input_image: Input satellite image

        Returns:
        - Redesigned image with added green spaces
        """
        if self.generator is None:
            raise ValueError("Generator not built. Call build_generator first.")

        # Preprocess input
        processed = self.preprocess_image(input_image)

        # Generate redesign
        generated = self.generator.predict(processed)

        # Post-process output
        redesigned = self.postprocess_image(generated[0])

        return redesigned

    def preprocess_image(self, image):
        """
        Preprocess image for GAN input.
        """
        # Resize
        resized = cv2.resize(image, (self.image_shape[0], self.image_shape[1]))

        # Normalize to [-1, 1] for tanh activation
        normalized = (resized.astype(np.float32) - 127.5) / 127.5

        return np.expand_dims(normalized, axis=0)

    def postprocess_image(self, image):
        """
        Post-process generated image.
        """
        # Denormalize from [-1, 1] to [0, 255]
        denormalized = ((image + 1) * 127.5).astype(np.uint8)

        return denormalized

def create_data_generator(data_dir: str, target_size=(224, 224), batch_size=32):
    """
    Create ImageDataGenerator for CNN training.
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
        class_mode='binary',
        subset='training'
    )

    validation_generator = datagen.flow_from_directory(
        data_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode='binary',
        subset='validation'
    )

    return train_generator, validation_generator

# Example usage
if __name__ == "__main__":
    # CNN for heat detection
    cnn = HeatDetectorCNN()
    cnn.build_model()
    print("CNN model built for heat detection.")

    # GAN for green redesign
    gan = GreenRedesignGAN()
    gan.build_generator()
    gan.build_discriminator()
    gan.build_gan()
    print("GAN model built for green redesign.")
