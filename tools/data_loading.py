from glob import glob
import os
import numpy as np
import cv2


def load_data(path):
    train_x, valid_x = sorted(glob(os.path.join(path, "train", "img", "*"))), sorted(glob(os.path.join(path, "valid", "img", "*")))
    train_y, valid_y = sorted(glob(os.path.join(path, "train", "msk", "*"))), sorted(glob(os.path.join(path, "valid", "msk", "*")))
    return (train_x, train_y), (valid_x, valid_y)


def read_image(path):
    path = path.decode()
    image = cv2.imread(filename=path, flags=cv2.IMREAD_COLOR)
    image = image / 255.0
    return image


def read_mask(path):
    path = path.decode()
    mask = cv2.imread(filename=path, flags=cv2.IMREAD_GRAYSCALE)
    mask = mask / 255.0
    mask = np.expand_dims(a=mask, axis=-1)
    return mask
