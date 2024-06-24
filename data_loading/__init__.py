import os
from glob import glob
from pathlib import Path

from torch.utils.data import Dataset


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

    def __init__(self, dataset_type: str = "train", validation_split: float = 0.25):
        _validate_params(dataset_type=dataset_type, validation_split=validation_split)
        self.dataset_type = dataset_type
        self.image_paths = _get_x_paths()
        n_images = int(validation_split * len(self.image_paths))  # number of images to split
        self.image_paths = self.image_paths[n_images:] if self.dataset_type == "train" else self.image_paths[:n_images]
        self.mask_paths = [_get_y_path(str(image_path)) for image_path in self.image_paths]
