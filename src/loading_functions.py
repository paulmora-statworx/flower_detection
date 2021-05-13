# %% Packages

import os
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder

# %% Loading functions


def load_labels():
    imagelabels_file_path = "./data/imagelabels.mat"
    image_labels = loadmat(imagelabels_file_path)["labels"][0]
    image_labels_2d = image_labels.reshape(-1, 1)

    encoder = OneHotEncoder(sparse=False)
    one_hot_labels = encoder.fit_transform(image_labels_2d)
    return one_hot_labels


def loading_images_array(target_size):
    image_path = "./data/jpg"
    image_file_names = os.listdir(image_path)
    image_file_names.sort()
    image_array_list = []
    for image_file_name in image_file_names:
        tf_image = tf.keras.preprocessing.image.load_img(
            path=f"{image_path}/{image_file_name}",
            grayscale=False,
            target_size=target_size,
        )
        img_array = tf.keras.preprocessing.image.img_to_array(tf_image)
        image_array_list.append(img_array)
    return np.array(image_array_list)


# %%
