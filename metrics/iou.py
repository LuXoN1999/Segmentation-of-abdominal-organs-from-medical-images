import numpy as np
import torch
from torch import Tensor


def calculate_iou(prediction: Tensor, ground_truth: Tensor, threshold: float = 0.5) -> np.array:
    prediction = torch.sigmoid(prediction)

    prediction = prediction > threshold
    ground_truth = ground_truth > threshold

    prediction = prediction.view(prediction.shape[0], prediction.shape[1], -1)
    ground_truth = ground_truth.view(ground_truth.shape[0], ground_truth.shape[1], -1)

    intersection = (prediction & ground_truth).sum(dim=2)
    union = (prediction | ground_truth).sum(dim=2)

    iou_per_sample = intersection / union
    iou_per_sample[union == 0] = 1.0  # avoid division by 0

    return iou_per_sample.mean(dim=0).numpy()


class CHAOSIoUTracker:
    def __init__(self):
        self.classes = ["liver", "r_kidney", "l_kidney", "spleen"]
        self.n_classes = len(self.classes)
        self.iou_sums = {cls: 0.0 for cls in self.classes}
        self.n_samples = 0

    def update(self, batch_iou: np.array, batch_size: int) -> None:
        for index, key in enumerate(self.iou_sums):
            self.iou_sums[key] += batch_iou[index] * batch_size
        self.n_samples += batch_size

    def get_results(self) -> dict:
        return {cls: round(iou_sum / self.n_samples, 5) for cls, iou_sum in self.iou_sums.items()}

    def reset(self) -> None:
        self.iou_sums = {cls: 0.0 for cls in self.classes}
        self.n_samples = 0
