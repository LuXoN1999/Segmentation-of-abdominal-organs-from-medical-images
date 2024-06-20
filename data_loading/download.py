import os
from zipfile import ZipFile
def __unzip_dataset():
    print("Extracting dataset...")
    zip_file_path = "./CHAOS_dataset.zip"
    if not os.path.exists(zip_file_path):
        raise FileNotFoundError("Dataset zip file is not found in current directory.")
    with ZipFile(file=zip_file_path, mode="r") as zip_reference:
        zip_reference.extractall()
        print("Dataset extracted!")
    os.remove(path=zip_file_path)
