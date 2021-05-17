# %% Packages

import copy
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications import MobileNetV2

# %% Classes


class OxfordFlower102Model:
    def __init__(self, config):
        self.config = config
        self.base_model = self.build_model()

    def build_model(self):

        pre_trained_model = self.initialize_pre_trained_model()
        top_model = self.create_top_layers()

        model = Sequential()
        model.add(pre_trained_model)
        model.add(top_model)

        model.compile(
            loss=self.config.model.loss,
            metrics=[self.config.model.metrics],
            optimizer=tf.keras.optimizers.RMSprop(
                learning_rate=self.config.model.learning_rate
            ),
        )
        return model

    def unfreeze_top_n_layers(self, model, ratio):
        base_model = model.layers[0]
        trained_top_model = model.layers[1]

        base_model.trainable = True
        number_of_all_layers = len(base_model.layers)
        non_trained_layers = int(number_of_all_layers * ratio)
        for layer in base_model.layers[:non_trained_layers]:
            layer.trainable = False

        fine_tune_model = Sequential()
        fine_tune_model.add(base_model)
        fine_tune_model.add(trained_top_model)

        adjusted_learning_rate = (
            self.config.model.learning_rate / self.config.model.learning_rate_shrinker
        )
        fine_tune_model.compile(
            loss=self.config.model.loss,
            metrics=[self.config.model.metrics],
            optimizer=tf.keras.optimizers.RMSprop(learning_rate=adjusted_learning_rate),
        )
        fine_tune_model.summary()
        return fine_tune_model

    def initialize_pre_trained_model(self):
        image_shape = (
            self.config.data_loader.target_size,
            self.config.data_loader.target_size,
            3,
        )
        base_model = MobileNetV2(
            input_shape=image_shape, include_top=False, pooling="avg"
        )
        base_model.trainable = False
        return base_model

    def create_top_layers(self):
        top_model = Sequential()
        top_model.add(
            Dense(self.config.model.number_of_categories, activation="softmax")
        )
        top_model.add(Dropout(rate=self.config.model.dropout_rate))
        return top_model
