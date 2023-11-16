
import os
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
import tensorflow as tf 
from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array 


# Data directory

originals_directory = Path('./captcha_dataset/samples')
augments_directory = Path('./captcha_dataset/augments')

# # Load data from originals
# original_images = sorted(list(map(str, list(originals_directory.glob("*.png")))) + list(map(str, list(originals_directory.glob("*.jpg")))))
# original_labels = [os.path.splitext(os.path.basename(img))[0] for img in original_images]
# max_len_originals = max([len(label) for label in original_labels])

# # Load data from augments
# augment_images = sorted(list(map(str, list(augments_directory.glob("*.png")))) + list(map(str, list(augments_directory.glob("*.jpg")))))
# augment_labels = [os.path.splitext(os.path.basename(img))[0] for img in augment_images]
# max_len_augments = max([len(label) for label in augment_labels])

# # Add padded labels with 0 as padding character to equalize input length
# labels = original_labels + augment_labels
# padded_labels = [label.ljust(10, '0') for label in labels]
# labels = padded_labels

# # Combine original and augmented data
# images = original_images + augment_images

# # Get the characters set
# characters = sorted(list(set(char for label in labels for char in label)))

# Load data

images = sorted(list(map(str, list(originals_directory.glob("*.png")))) + list(map(str, list(originals_directory.glob("*.jpg"))))) 
labels = [os.path.splitext(os.path.basename(img))[0] for img in images]
max_len = max([len(label) for label in labels])
print(max_len)

# Add padded labels with 0 as padding character to equalize input length
padded_labels = [label.ljust(max_len, '0') for label in labels]
labels = padded_labels

characters = set(char for label in labels for char in label)
characters = sorted(list(set(char for label in labels for char in label)))

print(characters)

# Split data into training set and remaining data

X_train, X_rem, y_train, y_rem = train_test_split(images, labels, train_size=0.8, random_state= 42)

# Split remaining data into validation and test sets

X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size = 0.5, random_state=42)

# Initialize parameters

batch_size = 16
img_width = 200
img_height = 50
downsample = 4
max_length = max([len(label) for label in labels])

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

test_dataset = tf.data.Dataset.from_tensor_slices((X_test,y_test))
test_dataset = (
    test_dataset.map(
        encode_single_sample, num_parallel_calls= tf.data.AUTOTUNE
    ).batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE)
)

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
    
# ## Build Models

#  # Model 1

# def build_model1():
#     input_img = layers.Input(
#         shape=(img_width, img_height, 1), name="image", dtype="float32"
#     )
#     labels = layers.Input(name="label", shape=(None,), dtype="float32")

#     # First Convolution Block

#     x = layers.Conv2D(
#         32,
#         (3, 3),
#         activation="relu",
#         kernel_initializer="he_normal",
#         padding="same",
#         name="Conv1",
#     )(input_img)
#     x = layers.MaxPooling2D((2, 2), name="pool1")(x)

#     # Second Convolution Block

#     x = layers.Conv2D(
#         64,
#         (3, 3),
#         activation="relu",
#         kernel_initializer="he_normal",
#         padding="same",
#         name="Conv2",
#     )(x)
#     x = layers.MaxPooling2D((2, 2), name="pool2")(x)

#     # Reshaping and Dense Layers

#     new_shape = ((img_width // 4), (img_height // 4) * 64)
#     x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
#     x = layers.Dense(64, activation="relu", name="dense1")(x)
#     x = layers.Dropout(0.2)(x)


#     # RNNs

#     x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
#     x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)


#     # Output layer

#     x = layers.Dense(
#         len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2"
#     )(x)

#     # Add CTC layer for calculating CTC loss at each step
    
#     output = CTCLayer(name="ctc_loss")(labels, x)

#     # Define Model

#     model = keras.models.Model(
#     inputs=[input_img, labels], outputs=output, name="captcha_reader_v1"
#     )
     
#     # Add optimizer 
    
#     opt = keras.optimizers.legacy.Adam()

    
#     # Compile and return model
    
#     model.compile(optimizer=opt)
#     return model

# Model 2 With extra dense layer

def build_model2():
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

    # Reshaping and Dense Layers - Added Dense layer and dropout layer

    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, activation ='relu', name='dense2')(x)
    x = layers.Dropout(0.2)(x)


    # RNNs

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)


    # Output layer

    x = layers.Dense(
        len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense3"
    )(x)

    # Add CTC layer for calculating CTC loss at each step
    
    output = CTCLayer(name="ctc_loss")(labels, x)

    # Define Model

    model = keras.models.Model(
    inputs=[input_img, labels], outputs=output, name="captcha_reader_v2"
    )
     
    # Add optimizer 
    
    opt = keras.optimizers.legacy.Adam()

    
    # Compile and return model
    
    model.compile(optimizer=opt)
    return model


# Get and summarize model 1

# model1 = build_model1()
# model1.summary()

# Get and summarize model 2

model2 = build_model2()
model2.summary()


## Training the model

epochs = 100
early_stopping_patience = 10

# Add early stopping to model

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True
)

# Train model

# # Model 1

# history1 = model1.fit(
#     train_dataset,
#     validation_data=validation_dataset,
#     epochs = epochs,
#     callbacks = [early_stopping],
# )

# Model 2

history = model2.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs = epochs,
    callbacks = [early_stopping],
)

train_loss = history.history['train_loss']
val_loss = history.history['val_loss']
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']


epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss, 'bo', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.plot(epochs, train_acc, 'bo', label='Training Accuracy')
plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# ## Plots for training and validation loss

# Get the prediction model by extracting layers till the output layer
prediction_model = keras.models.Model(
    model2.get_layer(name="image").input, model2.get_layer(name="dense3").output
)
prediction_model.summary()


# A utility function to decode the output of the network
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    padding_char = '0'
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8").replace(padding_char,'')
        output_text.append(res)
    return output_text


#  check results on some validation samples
for batch in test_dataset.take(1):
    batch_images = batch["image"]
    batch_labels = batch["label"]

    preds = prediction_model.predict(batch_images)
    pred_texts = decode_batch_predictions(preds)

    orig_texts = []
    for label in batch_labels:
        label = tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8")
        orig_texts.append(label)

    _, ax = plt.subplots(4, 4, figsize=(15, 5))
    for i in range(len(pred_texts)):
        img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8)
        img = img.T
        title = f"Prediction: {pred_texts[i]}"
        ax[i // 4, i % 4].imshow(img, cmap="gray")
        ax[i // 4, i % 4].set_title(title)
        ax[i // 4, i % 4].axis("off")
plt.show()
