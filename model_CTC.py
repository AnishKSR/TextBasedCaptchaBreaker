import os
import numpy as np
import matplotlib as plt
import pandas as pd
import cv2

import tensorflow as tf
from tensorflow import keras
from keras import layers
from pathlib import Path
from collections import Counter

# Data directory

input_directory = Path('/Users/anishravuri/Desktop/Junior Year/Semester 1/DATS 4001/captcha_dataset/samples')


# Load data

images = sorted(list(map(str, list(input_directory.glob("*.png")))) + list(map(str, list(input_directory.glob("*.jpg"))))) 
labels = [os.path.splitext(os.path.basename(img))[0] for img in images]
characters = set(char for label in labels for char in label)
characters = sorted(list(set(char for label in labels for char in label)))

print(characters)


# Initialize parameters


batch_size = 16
img_width = 200
img_height = 50
downsample = 4

# Function to map characters in labels to integers

char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)

# Inverse of mapping function, returning mapped integers to characters

num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

# Split data into training, validation, and test data.

# Creating CTC Loss Layer

class CTCLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")

        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

    
        return y_pred


# Build Model

def build_model():
    input_img = layers.Input(
        shape=(img_width, img_height, 1), name="image", dtype="float32"
    )
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    # First Convolution Block

    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second Convolution Block

    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # Reshaping and Dense Layers




    # RNNs

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)


    # Output layer

    x = layers.Dense(
        len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
    )(x)

    # Add CTC layer for calculating CTC loss at each step
    output = CTCLayer(name="ctc_loss")(labels, x)



