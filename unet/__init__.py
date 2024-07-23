from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, Upsample


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

        self.max_pool = MaxPool2d(kernel_size=2)
        self.up_conv = Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        # encoder blocks
        self.enc_conv1 = conv_block(in_channels=3, out_channels=64)
        self.enc_conv2 = conv_block(in_channels=64, out_channels=128)
        self.enc_conv3 = conv_block(in_channels=128, out_channels=256)
        self.enc_conv4 = conv_block(in_channels=256, out_channels=512)

        # bottleneck block
        self.b_conv = conv_block(in_channels=512, out_channels=1024)

        # decoder blocks
        self.dec_conv4 = conv_block(in_channels=1024 + 512, out_channels=512)
        self.dec_conv3 = conv_block(in_channels=512 + 256, out_channels=256)
        self.dec_conv2 = conv_block(in_channels=256 + 128, out_channels=128)
        self.dec_conv1 = conv_block(in_channels=128 + 64, out_channels=64)

        self.final_layer = Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1)
