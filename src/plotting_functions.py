# %% Packages

import numpy as np
import matplotlib.pyplot as plt

# %% Path

FIGURES_PATH = "./reports/figures"

# %% Functions


def plot_distribution(y_dict):
    fig, axs = plt.subplots(figsize=(15, 5))
    for label, y in y_dict.items():
        y_agg = np.sum(y, axis=0)
        axs.plot(y_agg, label=label)
    axs.legend()
    plt.show()
    path = f"{FIGURES_PATH}/relative_distribution.png"
    fig.savefig(fname=path, bbox_inches="tight")


def plot_example_images(generator):
    number_of_example_images = 9
    ncols = int(np.sqrt(number_of_example_images))
    nrows = ncols
    images, _ = next(iter(generator))

    fig, axs = plt.subplots(figsize=(10, 10), ncols=ncols, nrows=nrows)
    axs = axs.ravel()
    for i, image in enumerate(images[:number_of_example_images]):
        axs[i].imshow(image)
        axs[i].axis("off")
    plt.show()
    path = f"{FIGURES_PATH}/sample_images.png"
    fig.savefig(fname=path, bbox_inches="tight")


def plot_model_performance(history, img_name, vline_level=None):
    training_loss = history.history["loss"]
    validation_loss = history.history["val_loss"]

    training_acc = history.history["accuracy"]
    validation_acc = history.history["val_accuracy"]

    fig, axs = plt.subplots(figsize=(10, 5), ncols=2)
    axs = axs.ravel()
    axs[0].plot(training_loss, label="Training")
    axs[0].plot(validation_loss, label="Validation")
    axs[0].set_title("Loss")
    axs[0].legend()

    if vline_level is not None:
        axs[0].axvline(x=vline_level, ymin=0, ymax=1)

    axs[1].plot(training_acc, label="Training")
    axs[1].plot(validation_acc, label="Validation")
    axs[1].set_title("Accuracy")
    axs[1].legend()

    if vline_level is not None:
        axs[1].axvline(x=vline_level, ymin=0, ymax=1)

    plt.show()

    path = f"{FIGURES_PATH}/loss_acc_{img_name}.png"
    fig.savefig(fname=path, bbox_inches="tight")
    plt.show()
