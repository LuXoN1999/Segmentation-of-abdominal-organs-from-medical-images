from glob import glob
import os
import numpy as np
import cv2
import json
from sklearn.model_selection import train_test_split


def load_dataset(path, n_images, split=0.25, log_feedback=False):
    """
    Loads and splits names of images and masks from data folder. To work properly, data folder must be structured like this:
    data_folder(root folder)
        -train
            -img
            -msk
        -valid
            -img
            -msk
    :param path: path to the data root folder
    :param n_images: number of images/masks to take from training and validation dataset
    :param split: validation data ratio, defaults to 0.25
    :param log_feedback: prints size of each set if True, else skips logging
    :return: two tuple pairs, first pair representing train(0) and valid(1) images and second pair representing train(0) and valid(1) masks
    """
    images_full_path, masks_full_path = os.path.join(path, "train", "img"), os.path.join(path, "train", "msk")
    n_images = len(os.listdir(images_full_path)) if n_images > len(os.listdir(images_full_path)) else n_images  # in case n_images is greater than the actual number of images in dataset
    images = sorted(glob(os.path.join(images_full_path, "*")))[:n_images]
    masks = sorted(glob(os.path.join(masks_full_path, "*")))[:n_images]
    split_size = int(split * len(images))
    train_x, valid_x = train_test_split(images, test_size=split_size, random_state=42)
    train_y, valid_y = train_test_split(masks, test_size=split_size, random_state=42)
    if log_feedback:
        print(f"NUMBER OF PAIRS:\nTraining: {len(train_x)}/{len(train_y)}\nValidation: {len(valid_x)}/{len(valid_y)}")
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
