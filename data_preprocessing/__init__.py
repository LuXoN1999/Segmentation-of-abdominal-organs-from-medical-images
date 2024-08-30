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
