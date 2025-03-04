import torch
from torch import Tensor
from torch.nn import functional


def calculate_loss(prediction: Tensor, ground_truth: Tensor, losses: dict, bce_weight: float = 0.5) -> Tensor:
    bce = functional.binary_cross_entropy_with_logits(prediction, ground_truth)
    dice = dice_loss(prediction, ground_truth)
    loss = bce * bce_weight + dice * (1 - bce_weight)

    losses["bce"] += bce.data.cpu().numpy() * ground_truth.size(0)
    losses["dice"] += dice.data.cpu().numpy() * ground_truth.size(0)
    losses["loss"] += loss.data.cpu().numpy() * ground_truth.size(0)

    return loss


def dice_loss(prediction: Tensor, ground_truth: Tensor, smooth: float = 1e-6) -> Tensor:
    prediction = prediction.sigmoid()

    prediction = prediction.view(prediction.shape[0], prediction.shape[1], -1)
    ground_truth = ground_truth.view(ground_truth.shape[0], ground_truth.shape[1], -1)

    intersection = torch.sum(prediction * ground_truth, dim=2)
    union = torch.sum(prediction, dim=2) + torch.sum(ground_truth, dim=2)
    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice.mean()
