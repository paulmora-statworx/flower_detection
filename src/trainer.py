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
        print("Train the base Model!")
        self.train_model()
        print("Fine tune the Model!")
        self.train_fine_tune()
        self.save_model()

    def _init_callbacks(self):
        self.custom_callbacks = [
            EarlyStopping(
                monitor="val_accuracy",
                mode="max",
                patience=self.config.trainer.early_stopping_patience,
            )
        ]

    def train_model(self):
        history = self.model.base_model.fit(
            self.train_data_generator,
            verbose=self.config.trainer.verbose_training,
            epochs=self.config.trainer.number_of_base_epochs,
            validation_data=self.val_data_generator,
            callbacks=self.custom_callbacks,
        )
        self.append_model_data(history)

    def train_fine_tune(self):
        total_epochs = (
            self.config.trainer.number_of_base_epochs
            + self.config.trainer.number_of_fine_tune_epochs
        )
        self.fine_tune_model = self.model.unfreeze_top_n_layers(
            self.model.base_model, self.config.trainer.percentage_of_frozen_layers
        )

        fine_tune_history = self.fine_tune_model.fit(
            self.train_data_generator,
            verbose=self.config.trainer.verbose_training,
            initial_epoch=self.config.trainer.number_of_base_epochs,
            epochs=total_epochs,
            validation_data=self.val_data_generator,
            callbacks=self.custom_callbacks,
        )
        self.append_model_data(fine_tune_history)
        self.plot_history("fine_tune_model")

    def append_model_data(self, history):
        self.loss.extend(history.history["loss"])
        self.val_loss.extend(history.history["val_loss"])

        self.acc.extend(history.history["accuracy"])
        self.val_acc.extend(history.history["val_accuracy"])

    def plot_history(self, title):

        fig, axs = plt.subplots(figsize=(10, 5), ncols=2)
        axs = axs.ravel()
        axs[0].plot(self.loss, label="Training")
        axs[0].plot(self.val_loss, label="Validation")
        axs[0].set_title("Loss")
        axs[0].axvline(
            x=(self.config.trainer.number_of_base_epochs - 1),
            ymin=0,
            ymax=1,
            label="BaseEpochs",
            color="green",
            linestyle="--",
        )
        axs[0].legend()

        axs[1].plot(self.acc, label="Training")
        axs[1].plot(self.val_acc, label="Validation")
        axs[1].set_title("Accuracy")
        axs[1].axvline(
            x=(self.config.trainer.number_of_base_epochs - 1),
            ymin=0,
            ymax=1,
            label="BaseEpochs",
            color="green",
            linestyle="--",
        )
        axs[1].legend()

        fig.savefig(f"./reports/figures/history_{title}.png")

    def save_model(self):
        path = "./models/oxford_flower102_fine_tuning.h5"
        self.fine_tune_model.save(filepath=path)
