from glob import glob
import os
import numpy as np
import cv2


def load_data(path):
    """
    Loads names of images from data folder.To work properly, data folder must be structured like this:
    data_folder(root folder)
        -train
            -img
            -msk
        -valid
            -img
            -msk
    :param path: path to the data root folder
    :return: two tuple pairs, first pair representing train(0) and valid(1) images and second pair representing train(0) and valid(1) masks
    """
    train_x, valid_x = sorted(glob(os.path.join(path, "train", "img", "*"))), sorted(glob(os.path.join(path, "valid", "img", "*")))
    train_y, valid_y = sorted(glob(os.path.join(path, "train", "msk", "*"))), sorted(glob(os.path.join(path, "valid", "msk", "*")))
    return (train_x, train_y), (valid_x, valid_y)


def read_image(path):
    """
    Reads single image from given path and normalizes it.
    :param path: path to image
    :return: normalized image
    """
    path = path.decode()
    image = cv2.imread(filename=path, flags=cv2.IMREAD_COLOR)
    image = image / 255.0
    return image


def read_mask(path):
    """
    Reads single mask from given path and normalizes it. Also extends its dimension because it's a grayscale mask(1 channel).
    :param path: path to mask
    :return: normalized mask with extended extra dimension.
    """
    path = path.decode()
    mask = cv2.imread(filename=path, flags=cv2.IMREAD_GRAYSCALE)
    mask = mask / 255.0
    mask = np.expand_dims(a=mask, axis=-1)
    return mask
