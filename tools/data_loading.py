from glob import glob
import os
import numpy as np
import cv2
import json


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
    image = cv2.imread(filename=path, flags=cv2.IMREAD_COLOR)
    image = image / 255.0
    image = image.astype(np.float32)
    return image


def read_mask(path, colormap):
    """
    Reads single mask from given path and normalizes it.
    :param path: path to mask
    :param colormap: dictionary representing colormap with labels of a single mask
    :return: list of images where each image represents a mask of a specific label
    """
    mask = cv2.imread(path, flags=cv2.IMREAD_COLOR)
    output = []
    for color in colormap:
        cmap = np.all(np.equal(mask, color), axis=-1)
        output.append(cmap)
    output = np.stack(output, axis=-1)
    output = output.astype(np.uint8)
    return output


def get_colormap(path):
    """
    Reads JSON file which contains labels and their pixel values.
    :param path: path to colormap JSON file
    :return: tuple containing list of all labels[0] and all pixel values of labels[1]
    """
    json_file = open(path)
    colormap = json.load(json_file)
    classes = [organ for organ in colormap["labels"].values()]
    colormap = [label for label in colormap["labels"].keys()]
    colormap = [np.uint8(int(label)) for label in colormap]
    return classes, colormap

