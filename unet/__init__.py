from torch.nn import Module


class UNet(Module):

    def __init__(self, n_classes: int):
        super().__init__()
