# %% Packages

from tensorflow.keras.callbacks import EarlyStopping

# %% Classes


class OxfordFlower102Trainer:
    def __init__(self, model, data_generator, config):
        self.config = config
        self.model = model
        self.data_generator = data_generator
        self.loss = []
        self.acc = []
        self.val_loss = []
        self.val_acc = []

    def train(self):
        custom_callbacks = [
            EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=self.config.trainer.early_stopping_patience,
            )
        ]

        history = self.model.fit(
            self.data_generator,
            verbose=self.config.trainer.verbose_training,
            epochs=self.config.trainer.number_of_base_epochs,
            validation_split=self.config.trainer.validation_split,
            callbacks=custom_callbacks,
        )

        self.loss.extend(history.history["loss"])
        self.acc.extend(history.history["acc"])
        self.val_loss.extend(history.history["val_loss"])
        self.val_acc.extend(history.history["val_acc"])
