"""This file applies transfer learning on the Flowers102 dataset"""

# %% Packages

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.ops.gen_math_ops import TruncateMod
from sklearn.model_selection import train_test_split

from src._functions import image_loader

# %% Constants

TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42
BASE_IMAGE_PATH = "./data/jpg"

# %% Data Loading

# Loading the image data
setid_file_path = "./data/setid.mat"
set_ids = loadmat(setid_file_path)
total_ids = (
    set_ids["trnid"].tolist()[0]
    + set_ids["valid"].tolist()[0]
    + set_ids["tstid"].tolist()[0]
)
total_ids.sort()
image_data = image_loader(total_ids, BASE_IMAGE_PATH, TARGET_SIZE)

# Loading the image labels
imagelabels_file_path = "./data/imagelabels.mat"
image_labels = loadmat(imagelabels_file_path)["labels"][0]

# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    image_data,
    image_labels,
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
    shear_range=0.1
)

# Data Augmentation setup for test
test_data_gen = ImageDataGenerator()

# Setting up the generators for train
training_generator = train_data_gen.flow(x=X_train, y=y_train, batch_size=BATCH_SIZE)

# Setting up the generators for validation
valid_generator = test_data_gen.flow(x=X_val, y=y_val, batch_size=BATCH_SIZE)

# Setting up the generators for test
test_generator = test_data_gen.flow(x=X_test, y=y_test, batch_size=BATCH_SIZE)

# %% Plotting sample images

number_of_example_images = 9
ncols = int(np.sqrt(number_of_example_images))
nrows = ncols
images, _ = next(iter(training_generator))

fig, axs = plt.subplots(figsize=(10, 10), ncols=ncols, nrows=nrows)
axs = axs.ravel()
for i, image in enumerate(images[:number_of_example_images]):
    axs[i].imshow(image/255)
    axs[i].axis("off")
plt.show()

# %% Base Model

IMG_SHAPE = TARGET_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SHAPE, include_top=False, weights="imagenet"
)

image_batch, _ = next(iter(training_generator))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False
base_model.summary()


global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)


preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# Building the layers
inputs = tf.keras.Input(shape=IMG_SHAPE)
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# %% Model compiling

base_learning_rate = 0.0001
model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)
model.summary()

initial_epochs = 10
loss0, accuracy0 = model.evaluate(valid_generator)

history = model.fit(
    training_generator, epochs=initial_epochs, validation_data=valid_generator
)

