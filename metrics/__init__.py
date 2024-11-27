from torch import Tensor

def dice_loss(prediction: Tensor, ground_truth: Tensor, smooth: float = 1.) -> Tensor:
    intersection = (prediction * ground_truth).sum(dim=2).sum(dim=2)
    dice = (2. * intersection + smooth) / (prediction.sum(dim=2).sum(dim=2) + ground_truth.sum(dim=2).sum(dim=2) + smooth)
    loss = 1 - dice
    return loss.mean()
