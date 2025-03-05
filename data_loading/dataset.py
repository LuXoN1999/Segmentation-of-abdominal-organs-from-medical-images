import random
from glob import glob
from pathlib import Path

import numpy as np
import pydicom
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing_extensions import Union

from data_loading import get_project_root
from data_preprocessing import preprocess_image_and_mask


def _get_image_paths() -> list[Path]:
    """Fetches list of valid DICOM image paths inside CHAOS dataset folder."""
    dataset_path = get_project_root() / "CHAOS dataset"
    dcm_paths = sorted(glob(pathname=f"{dataset_path}/**/*.dcm", recursive=True))
    return [Path(dcm_path) for dcm_path in dcm_paths if _get_mask_path(path_to_image=dcm_path).exists()]


def _get_mask_path(path_to_image: str) -> Path:
    """Fetches mask path for given DICOM image path from CHAOS dataset folder."""
    mask_path = path_to_image.replace("DICOM_anon", "Ground")
    mask_path = mask_path.replace("InPhase", "").replace("OutPhase", "")
    mask_path = mask_path.replace(".dcm", ".png")
    return Path(mask_path)


class CHAOSDataset(Dataset):

    def __init__(self, dataset_type: str, validation_split: float = 0.20, log_feedback: bool = False):
        self.dataset_type = dataset_type
        self.image_paths = _get_image_paths()
        random.shuffle(self.image_paths)

        n_images = int(validation_split * len(self.image_paths))  # number of images to split
        self.image_paths = self.image_paths[n_images:] if self.dataset_type == "train" else self.image_paths[:n_images]
        self.mask_paths = [_get_mask_path(str(image_path)) for image_path in self.image_paths]

        if log_feedback:
            ds_type_ext = "Training" if self.dataset_type == "train" else "Validation"
            print(f"{ds_type_ext} dataset instance created. \nNumber of samples: {len(self.image_paths)}")

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

    def __len__(self):
        return len(self.image_paths)

    def __iter__(self):
        for image_path, mask_path in zip(self.image_paths, self.mask_paths):
            yield image_path, mask_path


def generate_dataloaders(batch_size: int, validation_split: float) -> dict:
    train_dataset = CHAOSDataset(dataset_type="train", validation_split=validation_split)
    valid_dataset = CHAOSDataset(dataset_type="valid", validation_split=validation_split)
    return {
        "train": DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0),
        "valid": DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    }
