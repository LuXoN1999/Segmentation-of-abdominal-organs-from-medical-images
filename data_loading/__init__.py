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


class ChaosDataset(Dataset):

    def __init__(self, dataset_type: str = "train", validation_split: float = 0.25):
        _validate_params(dataset_type=dataset_type, validation_split=validation_split)
        self.dataset_type = dataset_type
