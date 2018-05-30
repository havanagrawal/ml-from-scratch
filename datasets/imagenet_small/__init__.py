import os
import sys
import numpy as np

def load_dataset():
    dirname = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(dirname, "spam.data")

    image_X_train = np.load(os.path.join(dirname, 'train_features.npy'))
    image_y_train = np.load(os.path.join(dirname, 'train_labels.npy'))

    image_X_val = np.load(os.path.join(dirname, 'val_features.npy'))
    image_y_val = np.load(os.path.join(dirname, 'val_labels.npy'))

    return image_X_train, image_X_val, image_y_train, image_y_val
