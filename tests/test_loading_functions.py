# %% Packages


import os
from src.loading_functions import load_labels

# %%


def test_load_labels():
    labels = load_labels()
    assert len(labels) == 8189, "Not the right length!"
