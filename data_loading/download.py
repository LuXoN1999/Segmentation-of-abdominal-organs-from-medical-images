import sys
import os
from pathlib import Path
from zipfile import ZipFile

import wget

from data_loading import get_project_root

DATASET_URL = ""  # to be added


def _get_progress_bar(current, total, width=80):
    progress_message = f"Downloading dataset: {round(current / total * 100, 2)}% [{current} / {total}] bytes"
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()


def _unzip_dataset():
    print("\nExtracting dataset...")
    zip_file_path = "./CHAOS_dataset.zip"
    if not Path(zip_file_path).exists():
        raise FileNotFoundError("Dataset zip file is not found in current directory.")
    with ZipFile(file=zip_file_path, mode="r") as zip_reference:
        zip_reference.extractall(path=get_project_root())
        print("Dataset extracted!")
    os.remove(path=zip_file_path)


def download_and_unzip_dataset():
    wget.download(url=DATASET_URL, bar=_get_progress_bar)
    _unzip_dataset()
