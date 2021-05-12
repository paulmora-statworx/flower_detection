
# %% Packages

import tensorflow as tf
from tensorflow import keras

# %%

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
    return changed_list

def image_loader(index_list, image_path, target_size):
    """This function loads the images one by one, reshapes them and appends them to an image array

    Args:
        index_list ([list]): List of index names
        image_path ([str]): The base path of where the image is located
        target_size ([int]): How large the output image should be

    Returns:
        [list]: Array with tensors of every image
    """
    image_list = []
    adj_indices = adjust_index_names(index_list)
    for index in adj_indices:
        tf_image = tf.keras.preprocessing.image.load_img(
            path=f"{image_path}/image_{index}.jpg",
            grayscale=False,
            target_size=(target_size, target_size),
        )
        img_array = keras.preprocessing.image.img_to_array(tf_image)
        image_list.append(img_array)
    return image_list
