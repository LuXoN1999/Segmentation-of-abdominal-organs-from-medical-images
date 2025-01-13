from torch import Tensor, sigmoid
from torch.nn import functional


def calculate_loss(prediction: Tensor, ground_truth: Tensor, bce_weight: float = 0.5) -> Tensor:
    bce = functional.binary_cross_entropy_with_logits(input=prediction, target=ground_truth)
    prediction = sigmoid(input=prediction)
    dice = dice_loss(prediction=prediction, ground_truth=ground_truth)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    return loss


def dice_loss(prediction: Tensor, ground_truth: Tensor, smooth: float = 1.) -> Tensor:
    intersection = (prediction * ground_truth).sum(dim=2).sum(dim=2)
    dice = (2. * intersection + smooth) / (prediction.sum(dim=2).sum(dim=2) + ground_truth.sum(dim=2).sum(dim=2) + smooth)
    loss = 1 - dice
    return loss.mean()
