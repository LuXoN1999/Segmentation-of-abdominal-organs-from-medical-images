import os
import random
from glob import glob
from pathlib import Path
from typing import Union

import numpy as np
import pydicom
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from data_preprocessing import preprocess_image_and_mask


def _get_project_path() -> str:
    current_file_path = Path(__file__).resolve()
    project_marker = 'README.md'  # using README.md as marker for base level of project
    for parent in current_file_path.parents:
        if (parent / project_marker).exists():
            return str(parent.resolve())
    raise ValueError("Project path not found.")


def _validate_params(dataset_type: str, validation_split: float):
    if dataset_type not in ["train", "valid"]:
        raise ValueError("Parameter 'dataset_type' can only contain values 'train' or 'valid'.")
    if not 0 < validation_split < 1:
        raise ValueError("Parameter 'validation_split' must be in range (0,1)(excluding).")


def _get_x_paths() -> list[Path]:
    """Fetches list of DICOM image paths inside CHAOS dataset folder."""
    dataset_path = os.path.join(_get_project_path(), "CHAOS dataset")
    dcm_paths = sorted(glob(pathname=f"{dataset_path}/**/*.dcm", recursive=True))
    return [Path(dcm_path) for dcm_path in dcm_paths if _get_y_path(path_to_image=dcm_path).exists()]


def _get_y_path(path_to_image: str) -> Path:
    """Fetches mask path for given DICOM image path from CHAOS dataset folder."""
    mask_path = path_to_image.replace("DICOM_anon", "Ground")
    for phase in ["InPhase", "OutPhase"]:
        if phase in mask_path:
            mask_path = mask_path.replace(phase, "")
    mask_path = mask_path.replace(".dcm", ".png")
    return Path(mask_path)


class CHAOSDataset(Dataset):

    def __init__(self, dataset_type: str = "train", validation_split: float = 0.25, log_feedback: bool = False):
        _validate_params(dataset_type=dataset_type, validation_split=validation_split)
        self.dataset_type = dataset_type
        self.image_paths = _get_x_paths()
        n_images = int(validation_split * len(self.image_paths))  # number of images to split
        random.shuffle(self.image_paths)
        self.image_paths = self.image_paths[n_images:] if self.dataset_type == "train" else self.image_paths[:n_images]
        self.mask_paths = [_get_y_path(str(image_path)) for image_path in self.image_paths]
        if log_feedback:
            ds_type_ext = "Training" if self.dataset_type == "train" else "Validation"
            print(f"{ds_type_ext} dataset instance created. \nNumber of images: {len(self.image_paths)}")

    def __getitem__(self, index: Union[int, slice]) -> Union[tuple, list[tuple]]:
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        elif isinstance(index, int):
            image = pydicom.dcmread(fp=self.image_paths[index]).pixel_array
            mask = Image.open(fp=self.mask_paths[index])
            image, mask = np.array(image).astype(np.float32), np.array(mask)
            return preprocess_image_and_mask(image=image, mask=mask)
        else:
            raise TypeError("Index must be an integer or slice element.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __iter__(self) -> tuple:
        for image_path, mask_path in zip(self.image_paths, self.mask_paths):
            yield image_path, mask_path


def generate_dataloaders(batch_size: int, validation_split: float) -> dict:
    train_dataset = CHAOSDataset(dataset_type="train", validation_split=validation_split)
    valid_dataset = CHAOSDataset(dataset_type="valid", validation_split=validation_split)
    return {
        "train": DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        "valid": DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    }
