"""This file applies transfer learning on the Flowers102 dataset"""

# %% Packages

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping

from src.loading_functions import loading_images_array, load_labels
from src.datagenerator_functions import train_val_test_split, create_generators
from src.plotting_functions import (
    plot_distribution,
    plot_example_images,
    plot_model_performance,
)

# %% Constants

TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_STATE = 42
LEARNING_RATE = 1e-3
NUMBER_OF_BASE_EPOCHS = 15
TRAIN_SIZE = 0.8

# %% Data Loading

# Loading the image data
image_data = loading_images_array(TARGET_SIZE)

# Loading the image labels
one_hot_labels = load_labels()

# Splitting the data into train and test
X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X=image_data,
    y=one_hot_labels,
    train_size=TRAIN_SIZE,
    random_state=RANDOM_STATE,
    shuffle=True,
)

train_generator, valid_generator, test_generator = create_generators(
    X_train=X_train,
    X_val=X_val,
    X_test=X_test,
    y_train=y_train,
    y_val=y_val,
    y_test=y_test,
    batch_size=BATCH_SIZE,
)

# %% Example plots

y_dict = {"Train": y_train, "Validation": y_val, "Test": y_test}
plot_distribution(y_dict)
plot_example_images(train_generator)

# %% Base Model - Only training the top layer of the model

IMG_SHAPE = TARGET_SIZE + (3,)
base_model = MobileNetV2(input_shape=IMG_SHAPE, include_top=False, pooling="avg")
base_model.trainable = False

number_of_categories = one_hot_labels.shape[1]
model = Sequential()
model.add(base_model)
model.add(Dense(number_of_categories, activation="softmax"))
model.compile(
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE),
)

model.summary()

custom_callbacks = [EarlyStopping(monitor="val_accuracy", mode="max", patience=3)]
training_history = model.fit(
    train_generator,
    epochs=NUMBER_OF_BASE_EPOCHS,
    validation_data=valid_generator,
    callbacks=custom_callbacks,
)

path = "./models/oxford_flower102.h5"
model.save(filepath=path)
plot_model_performance(training_history)

# %% Fine-tuning

base_model.trainable = True
number_of_all_layers = len(base_model.layers)
ratio_to_be_trained = 1 / 3
non_trained_layers = int(number_of_all_layers * (1 - ratio_to_be_trained))
for layer in base_model.layers[:non_trained_layers]:
    layer.trainable = False

adjusted_learning_rate = LEARNING_RATE / 10

model.compile(
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    optimizer=tf.keras.optimizers.Adam(learning_rate=adjusted_learning_rate),
)
model.summary()
len(model.trainable_variables)

fine_tune_epoch = 10
total_epochs = 15 + fine_tune_epoch

fine_tune_history = model.fit(
    training_generator,
    initial_epoch=training_history.epoch[-1],
    epochs=total_epochs,
    validation_data=valid_generator,
    callbacks=custom_callbacks,
)
