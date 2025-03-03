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
