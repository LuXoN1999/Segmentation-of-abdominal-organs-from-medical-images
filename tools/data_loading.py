from glob import glob
import os
from pydicom import dcmread
import numpy as np


def load_data(path):
    train_x, valid_x = sorted(glob(os.path.join(path, "train", "img", "*"))), sorted(glob(os.path.join(path, "valid", "img", "*")))
    train_y, valid_y = sorted(glob(os.path.join(path, "train", "msk", "*"))), sorted(glob(os.path.join(path, "valid", "msk", "*")))
    return (train_x, train_y), (valid_x, valid_y)


def read_image(path):
    path = path.decode()
    image = dcmread(fp=path).pixel_array
    image = image / 255.0
    return image


def read_mask(path):
    path = path.decode()
    mask = dcmread(fp=path).pixel_array
    mask = mask / 255.0
    mask = np.expand_dims(a=mask, axis=-1)
    return mask
