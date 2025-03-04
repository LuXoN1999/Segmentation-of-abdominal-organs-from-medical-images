from torch import Tensor
from torch.nn import functional


def calculate_loss(prediction: Tensor, ground_truth: Tensor, bce_weight: float = 0.5) -> dict:
    bce = functional.binary_cross_entropy_with_logits(prediction, ground_truth)
    dice = dice_loss(prediction, ground_truth)
    loss = bce * bce_weight + dice * (1 - bce_weight)
    return {"bce": bce, "dice": dice, "loss": loss}


def dice_loss(prediction: Tensor, ground_truth: Tensor, smooth: float = 1e-6) -> Tensor:
    prediction = prediction.sigmoid()

    prediction = prediction.view(prediction.shape[0], prediction.shape[1], -1)
    ground_truth = ground_truth.view(ground_truth.shape[0], ground_truth.shape[1], -1)

    intersection = (prediction * ground_truth).sum(dim=2)
    union = prediction.sum(dim=2) + ground_truth.sum(dim=2) - intersection
    dice = (2. * intersection + smooth) / (union + smooth)

    return 1 - dice.mean()
