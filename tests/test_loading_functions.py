# %% Packages

from src.loading_functions import load_labels, loading_images_array

# %% Fixtures

# %% Tests

def test_load_labels():
    labels = load_labels()
    assert len(labels) == 8189, "Not the right length!"


def test_loading_images_array():
    target_size = 224
    loaded_images = loading_images_array(target_size)
