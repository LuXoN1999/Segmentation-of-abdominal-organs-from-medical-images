import sys
import os
from pathlib import Path
from zipfile import ZipFile

import wget

from data_loading import _get_project_path

DATASET_URL = ""  # to be added


def __get_progress_bar(current, total, width=80):
    progress_message = f"Downloading dataset: {round(current / total * 100, 2)}% [{current} / {total}] bytes"
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def __unzip_dataset():
    print("\nExtracting dataset...")
    zip_file_path = "./CHAOS_dataset.zip"
    if not Path(zip_file_path).exists():
        raise FileNotFoundError("Dataset zip file is not found in current directory.")
    with ZipFile(file=zip_file_path, mode="r") as zip_reference:
        zip_reference.extractall(path=_get_project_path())
        print("Dataset extracted!")
    os.remove(path=zip_file_path)


def download_and_unzip_dataset():
    wget.download(url=DATASET_URL, bar=__get_progress_bar)
    __unzip_dataset()