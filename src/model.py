# %% Packages

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications import MobileNetV2

# %% Classes


class OxfordFlower102Model:
    def __init__(self, config):
        self.config = config

    def build_model(self):
        self.build_base_model()
        model = self.add_top_layer()

        model.compile(
            loss="categorical_crossentropy",
            metrics=["accuracy"],
            optimizer=tf.keras.optimizers.RMSprop(
                learning_rate=self.config.model.learning_rate
            ),
        )
        return model

    def build_base_model(self):
        IMG_SHAPE = (
            self.config.data_loader.target_size,
            self.config.data_loader.target_size,
            3,
        )
        self.base_model = MobileNetV2(
            input_shape=IMG_SHAPE, include_top=False, pooling="avg"
        )

    def add_top_layer(self):
        top_model = Sequential()
        top_model.add(
            Dense(self.config.model.number_of_categories, activation="softmax")
        )
        top_model.add(Dropout(rate=self.config.model.dropout_rate))
        combined_model = Model(
            input=self.base_model.input, output=top_model(self.base_model.output)
        )
        return combined_model
