
"""This file applies transfer learning on the Flowers102 dataset"""

# %% Packages

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

# %% Getting the labels right

setid_file_path = "./data/setid.mat"
set_ids = loadmat(setid_file_path)

test_ids = set_ids["trnid"].tolist()[0]
train_ids = set_ids["tstid"].tolist()[0]

def adjust_index_names(index_list):
    """This function takes in a list of image labels and pads them with zeros in the beginning in order 

    Args:
        index_list ([list]): List with labels

    Returns:
        [list]: List with padded labels
    """
    changed_list = []
    for index in index_list:
        str_int = str(index)
        number_of_zeros = 5 - len(str_int)
        new_int = "0"*(number_of_zeros) + str_int
        changed_list.append(new_int)
    return new_int

adjust_index_names(test_ids)
