"""This file applies transfer learning on the Flowers102 dataset"""

# %% Packages

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

from src._functions import loading_images_array, load_labels

# %% Constants

TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# %% Data Loading

# Loading the image data
image_data = loading_images_array(TARGET_SIZE)

# Loading the image labels
one_hot_labels = load_labels()

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    image_data,
    one_hot_labels,
    train_size=0.8,
    random_state=SEED,
    shuffle=True,
    stratify=image_labels,
)

# Splitting train and validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, train_size=0.8, random_state=SEED, shuffle=True, stratify=y_train
)

# Data Augmentation setup for train
train_data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.1,
)

# Data Augmentation setup for test
test_data_gen = ImageDataGenerator()

# Setting up the generators for train
training_generator = train_data_gen.flow(x=X_train, y=y_train, batch_size=BATCH_SIZE)

# Setting up the generators for validation
valid_generator = test_data_gen.flow(x=X_val, y=y_val, batch_size=BATCH_SIZE)

# Setting up the generators for test
test_generator = test_data_gen.flow(x=X_test, y=y_test, batch_size=BATCH_SIZE)

# %% Plotting distribution of labels

np.sum(y_val, axis=0)


# %% Plotting sample images

number_of_example_images = 9
ncols = int(np.sqrt(number_of_example_images))
nrows = ncols
images, _ = next(iter(training_generator))

fig, axs = plt.subplots(figsize=(10, 10), ncols=ncols, nrows=nrows)
axs = axs.ravel()
for i, image in enumerate(images[:number_of_example_images]):
    axs[i].imshow(image / 255)
    axs[i].axis("off")
plt.show()

# %% Base Model

IMG_SHAPE = TARGET_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE, include_top=False, pooling="avg"
)
base_model.trainable = False
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

number_of_categories = len(set(image_labels))

model = Sequential()
model.add(base_model)
model.add(Dense(number_of_categories, activation="softmax"))
model.compile(loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

from tensorflow.keras.callbacks import EarlyStopping

custom_callbacks = [EarlyStopping(monitor="val_accuracy", mode="max", patience=3)]
training_history = model.fit(
    training_generator,
    epochs=15,
    validation_data=valid_generator,
    callbacks=custom_callbacks,
)

a, b = next(iter(training_generator))

len(b[0])
