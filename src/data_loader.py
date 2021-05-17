# %% Packages

import os
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# %% Classes


class OxfordFlower102DataLoader:
    def __init__(self, config):
        self.config = config
        (
            self.train_generator,
            self.val_generator,
            self.test_generator,
        ) = self.create_generators()

    def create_generators(self):

        X_train, X_val, X_test, y_train, y_val, y_test = self._image_and_labels()
        train_augment_settings, test_augment_settings = self._add_preprocess_function()

        # Data Augmentation setup initialization
        train_data_gen = ImageDataGenerator(**train_augment_settings)
        valid_data_gen = ImageDataGenerator(**test_augment_settings)
        test_data_gen = ImageDataGenerator(**test_augment_settings)

        # Setting up the generators
        training_generator = train_data_gen.flow(
            x=X_train, y=y_train, batch_size=self.config.data_loader.batch_size
        )
        validation_generator = valid_data_gen.flow(
            x=X_val, y=y_val, batch_size=self.config.data_loader.batch_size
        )
        test_generator = test_data_gen.flow(
            x=X_test, y=y_test, batch_size=self.config.data_loader.batch_size
        )
        return training_generator, validation_generator, test_generator

    def _add_preprocess_function(self):
        train_augment_settings = self.config.data_loader.train_augmentation_settings
        test_augment_settings = self.config.data_loader.test_augmentation_settings
        train_augment_settings.update(
            {
                "preprocessing_function": tf.keras.applications.mobilenet_v2.preprocess_input
            }
        )
        test_augment_settings.update(
            {
                "preprocessing_function": tf.keras.applications.mobilenet_v2.preprocess_input
            }
        )
        return train_augment_settings, test_augment_settings

    def _image_and_labels(self):
        y = self._load_labels()
        X = self._loading_images_array()
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            train_size=self.config.data_loader.train_size,
            random_state=self.config.data_loader.random_state,
            shuffle=True,
            stratify=y,
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train,
            y_train,
            train_size=self.config.data_loader.train_size,
            random_state=self.config.data_loader.random_state,
            shuffle=True,
            stratify=y_train,
        )
        return X_train, X_val, X_test, y_train, y_val, y_test

    def _load_labels(self):
        imagelabels_file_path = "./data/imagelabels.mat"
        image_labels = loadmat(imagelabels_file_path)["labels"][0]
        image_labels_2d = image_labels.reshape(-1, 1)

        encoder = OneHotEncoder(sparse=False)
        one_hot_labels = encoder.fit_transform(image_labels_2d)
        return one_hot_labels

    def _loading_images_array(self):
        image_path = "./data/jpg"
        image_file_names = os.listdir(image_path)
        image_file_names.sort()
        image_array_list = []
        for image_file_name in image_file_names:
            tf_image = tf.keras.preprocessing.image.load_img(
                path=f"{image_path}/{image_file_name}",
                grayscale=False,
                target_size=(
                    self.config.data_loader.target_size,
                    self.config.data_loader.target_size,
                ),
            )
            img_array = tf.keras.preprocessing.image.img_to_array(tf_image)
            image_array_list.append(img_array)
        return np.array(image_array_list)
