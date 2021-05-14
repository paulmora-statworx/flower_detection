# %% Packages

import tensorflow as tf

# %% DataGenerator Settings

train_data_augmentation_settings = {
    "rotation_range": 20,
    "width_shift_range": 0.2,
    "height_shift_range": 0.2,
    "zoom_range": 0.2,
    "shear_range": 0.1,
    "preprocessing_function": tf.keras.applications.mobilenet_v2.preprocess_input,
}

test_data_augmentation_settings = {
    "preprocessing_function": tf.keras.applications.mobilenet_v2.preprocess_input
}
