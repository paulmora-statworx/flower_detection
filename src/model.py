# %% Packages

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.applications import MobileNetV2

# %% Classes


class OxfordFlower102Model:
    """
    This class is initializing the model
    """

    def __init__(self, config):
        self.config = config
        self.base_model = self.build_model()
        tf.random.set_seed(self.config.model.random_seed)

    def build_model(self):
        """
        This method build the basic model. The basic model describes the pre-trained model plus a dense layer
        on top which is individualized to the number of categories needed. The model is also compiled
        :return: A compiled tensorflow model
        """
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
        model.summary()
        return model

    def unfreeze_top_n_layers(self, model, ratio):
        """
        This method unfreezes a certain number of layers of the pre-trained model and combines it subsequently with the
        pre-trained top layer which was added within the 'create_top_layers' method and trained within the 'build_model'
        class
        :param model: Tensorflow model which was already fitted
        :param ratio: Float of how many layers should not be trained of the entire model
        :return: Compiled tensorflow model
        """
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
        """
        This method calls the pre-trained model. In this case we are loading the MobileNetV2
        :return: Tensorflow model
        """
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
        """
        Creating the tensorflow top-layer of a model
        :return: Tensorflow Sequential model
        """
        top_model = Sequential()
        top_model.add(
            Dense(self.config.model.number_of_categories, activation="softmax")
        )
        top_model.add(Dropout(rate=self.config.model.dropout_rate))
        return top_model
