from torch.nn import Module, Sequential, Conv2d, ReLU


def conv_block(in_channels: int, out_channels: int):
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
        ReLU(inplace=True),
        Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
        ReLU(inplace=True)
    )


class UNet(Module):

    def __init__(self, n_classes: int):
        super().__init__()
