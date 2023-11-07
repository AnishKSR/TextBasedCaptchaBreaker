
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow import keras
from keras import layers

# Data directory

input_directory = Path('./captcha_dataset/samples')

# Load data

images = sorted(list(map(str, list(input_directory.glob("*.png")))) + list(map(str, list(input_directory.glob("*.jpg"))))) 
labels = [os.path.splitext(os.path.basename(img))[0] for img in images]
characters = set(char for label in labels for char in label)
characters = sorted(list(set(char for label in labels for char in label)))

print(characters)

# Split data into training set and remaining data

X_train, X_rem, y_train, y_rem = train_test_split(images, labels, train_size=0.8, random_state= 42)

# Split remaining data into validation and test sets

X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size = 0.5, random_state=42)

# Initialize parameters

batch_size = 1
img_width = 200
img_height = 50
downsample = 4

# Character map from characters to integers

char_to_num = layers.StringLookup(vocabulary=list(characters), mask_token=None)

# Character map from integers to characters

num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

# Create data dictionary for neural network inputs

def encode_single_sample(img_path, label):
   
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [img_height, img_width])
    img = tf.transpose(img, perm=[1, 0, 2])
    label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))

    return {"image": img, "label": label}

# Create Data set objects (found this code online, may try to recreate in pandas)

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = (
    train_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
validation_dataset = (
    validation_dataset.map(
        encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE
    )
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)


# Visualize data



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

    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)


    # RNNs

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)


    # Output layer

    x = layers.Dense(
        len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
    )(x)

    # Add CTC layer for calculating CTC loss at each step
    
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define Model

    model = keras.models.Model(
    inputs=[input_img, labels], outputs=output, name="captcha_reader_v1"
    )
     
    # Add optimizer 
    
    opt = keras.optimizers.Adam()

    
    # Compile and return model
    
    model.compile(optimizer=opt)
    return model

# Get and summarize model

model = build_model()
model.summary()

## Training the model

epochs = 100
early_stopping_patience = 10

# Add early stopping to model
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
)

# Train model

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs = epochs,
    callbacks = [early_stopping],
)




