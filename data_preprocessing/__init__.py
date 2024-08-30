import numpy as np
from torch import Tensor
from torchvision import transforms
from torchvision.transforms import InterpolationMode

IMAGE_PREPROCESSING_PIPELINE = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(size=(128, 128), interpolation=InterpolationMode.BICUBIC),
    transforms.CenterCrop(size=96),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(x.shape).expand(3, -1, -1)),
    transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min()))
])


def _one_hot_encode_mask(mask: Tensor) -> np.array:
    mask = mask.squeeze()

    organ_values = {
        "liver": 0.24705882,
        "r_kidney": 0.49411765,
        "l_kidney": 0.7411765,
        "spleen": 0.9882353
    }

    ones, zeros = np.ones(mask.shape), np.zeros(mask.shape)
    one_hot_encoded_mask = [np.where(mask == uni_val, ones, zeros) for uni_val in organ_values.values()]
    one_hot_encoded_mask = np.stack(one_hot_encoded_mask, axis=0).astype(np.float32)
    return one_hot_encoded_mask
