import cv2
import os
import numpy as np
import pydicom as dcm
from tqdm import tqdm


def resize_dicom(dicom_file_path, img_width, img_height, log_feedback=False):
    """
    Function which resizes an DICOM image. Logs message if the resizing was successful on log_feedback = True

    :param dicom_file_path: path to the target DICOM image
    :param img_width:  width of the DICOM image
    :param img_height:  height od the DICOM image
    :param log_feedback: value which represents should a log message be printed after successful resizing or not
    :return: resized DICOM image as pixel array with the dimensions of img_width x img_height
    """
    try:
        dicom_file = dcm.dcmread(dicom_file_path)
        pixel_array = dicom_file.pixel_array
        resized_pixel_array = cv2.resize(pixel_array, (img_width, img_height))
    except Exception as e:
        dicom_file_name = os.path.basename(dicom_file_path)
        print(f"Couldn't resize {dicom_file_name} image: {e}")
    else:
        if log_feedback:
            print(f"DICOM image {os.path.basename(dicom_file_path)} resized from {dicom_file.Rows}x{dicom_file.Columns} to {img_width}x{img_height}.")
        return resized_pixel_array


def is_completely_empty(pixel_array):
    """
    Function which checks if the image is completely empty(where every pixel in the image contains the same value).

    :param pixel_array: pixel array representing an image
    :return: True if the image has all pixel values the same, else returns False
    """
    all_zeroes = np.all(pixel_array == 0)
    if all_zeroes:
        return True
    return False


def delete_completely_empty_images(target_dir_path, log_feedback=False):
    """
    Function which deletes all DICOM images which are completely empty(where every pixel in the image contains the same value).

    :param target_dir_path: path to the root directory which contains DICOM images
    :param log_feedback:  value which represents should a log message be printed after successful deletion or not
    """
    n_images = len(os.listdir(target_dir_path))
    n_deleted = 0
    for msk in os.listdir(target_dir_path):
        msk_path = os.path.join(target_dir_path, msk)
        pixel_array = dcm.dcmread(msk_path).pixel_array
        img_path = msk_path.replace("msk", "img")
        if is_completely_empty(pixel_array):
            if log_feedback:
                print(f"Deleted: {os.path.basename(msk_path)}")
            os.remove(img_path)
            os.remove(msk_path)
            n_deleted += 1
    print(f"Number of deleted images: {n_deleted} ({round(n_deleted / n_images * 100, 2)}%)")


def check_percentage_of_classes(pixel_array, image_name):
    """
    Function which checks the percentage of each class within pixel array.

    :param pixel_array: pixel array representing an image
    :param image_name: name of the image
    :return: tuple which represents number of classes and their percentages within the pixel array
    """
    unique_values, counts = np.unique(pixel_array, return_counts=True)
    total_pixels = pixel_array.size
    print("=" * 50)
    print(image_name)
    for value, count in zip(unique_values, counts):
        percentage = (count / total_pixels) * 100
        print(f"Class {value}, Count: {count}, Percentage: {percentage:.2f}%")
    print("=" * 50)
    return unique_values, counts


def dcm_to_jpg(input_dir_path, output_dir_path, img_dims, log_feedback=False):
    """
    Function which converts all DICOM images inside directory to .jpg.

    :param input_dir_path: path to the root directory of targeted DICOM images
    :param output_dir_path: path to the directory where the converted DICOM images will be saved
    :param img_dims: wanted dimensions of .jpg images
    :param log_feedback: value which represents should a log message be printed after successful converting or not
    """
    images = os.listdir(input_dir_path)
    for image in tqdm(images):
        image_path = os.path.join(input_dir_path, image)
        pixel_array = dcm.dcmread(image_path).pixel_array
        pixel_array = cv2.resize(pixel_array, (img_dims[0], img_dims[1]))
        pixel_array = np.array(pixel_array)
        image_name = image.removesuffix(".dcm") + ".jpg"
        try:
            cv2.imwrite(os.path.join(output_dir_path, image_name), pixel_array)
        except Exception as e:
            print(f"Couldn't save {image}: {e}")
        else:
            if log_feedback:
                print(f"Successfully converted: {image}")
