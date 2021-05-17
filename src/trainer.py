# %% Packages

import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping

# %% Classes


class OxfordFlower102Trainer:
    def __init__(self, model, data_generator, config):
        self.config = config
        self.model = model
        self.train_data_generator = data_generator.train_generator
        self.val_data_generator = data_generator.val_generator
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []

        self._init_callbacks()
        self.train_model()
        self.train_fine_tune()

    def _init_callbacks(self):
        self.custom_callbacks = [
            EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=self.config.trainer.early_stopping_patience,
            )
        ]

    def append_model_data(self, history):
        self.loss.extend(history.history["loss"])
        self.val_loss.extend(history.history["val_loss"])

        self.acc.extend(history.history["accuracy"])
        self.val_acc.extend(history.history["val_accuracy"])

    def train_model(self):
        history = self.model.base_model.fit(
            self.train_data_generator,
            verbose=self.config.trainer.verbose_training,
            epochs=self.config.trainer.number_of_base_epochs,
            validation_data=self.val_data_generator,
            callbacks=self.custom_callbacks,
        )
        self.append_model_data(history)
        self.visualize_data("base_model")

    def train_fine_tune(self):
        total_epochs = (
            self.config.trainer.number_of_base_epochs
            + self.config.trainer.number_of_fine_tune_epochs
        )
        fine_tune_model = self.model.unfreeze_top_n_layers(
            self.model.base_model, self.config.trainer.percentage_of_frozen_layers
        )

        fine_tune_history = fine_tune_model.fit(
            self.train_data_generator,
            verbose=self.config.trainer.verbose_training,
            initial_epoch=self.config.trainer.number_of_base_epochs,
            epochs=total_epochs,
            validation_data=self.val_data_generator,
            callbacks=self.custom_callbacks,
        )
        self.append_model_data(fine_tune_history)
        self.visualize_data("fine_tune")
