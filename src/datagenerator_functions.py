# %% Packages

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from src._config import (
    train_data_augmentation_settings,
    test_data_augmentation_settings,
)

# %% Functions


def train_val_test_split(X, y, train_size, random_state, shuffle):

    # Splitting the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=y,
    )

    # Splitting train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        train_size=train_size,
        random_state=random_state,
        shuffle=shuffle,
        stratify=y_train,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def create_generators(X_train, X_val, X_test, y_train, y_val, y_test, batch_size):

    # Data Augmentation setup initialization
    train_data_gen = ImageDataGenerator(**train_data_augmentation_settings)
    test_data_gen = ImageDataGenerator(**test_data_augmentation_settings)

    # Setting up the generators
    training_generator = train_data_gen.flow(
        x=X_train, y=y_train, batch_size=batch_size
    )
    valid_generator = test_data_gen.flow(x=X_val, y=y_val, batch_size=batch_size)
    test_generator = test_data_gen.flow(x=X_test, y=y_test, batch_size=batch_size)

    return training_generator, valid_generator, test_generator
