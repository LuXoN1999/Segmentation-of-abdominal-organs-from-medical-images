from torch import cat
from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d, Upsample, BatchNorm2d


def conv_block(in_channels: int, out_channels: int) -> Sequential:
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True),
        BatchNorm2d(num_features=out_channels),
        ReLU(inplace=True),
        Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True),
        BatchNorm2d(num_features=out_channels),
        ReLU(inplace=True)
    )


def up_conv(in_channels: int, out_channels: int) -> Sequential:
    return Sequential(
        Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, padding=1, bias=True),
        BatchNorm2d(num_features=out_channels),
        ReLU(inplace=True)
    )


class UNet(Module):
    def __init__(self, n_classes: int):
        super().__init__()

        self.max_pool = MaxPool2d(kernel_size=2, stride=2)

        # encoder path
        self.enc_conv1 = conv_block(in_channels=3, out_channels=64)
        self.enc_conv2 = conv_block(in_channels=64, out_channels=128)
        self.enc_conv3 = conv_block(in_channels=128, out_channels=256)
        self.enc_conv4 = conv_block(in_channels=256, out_channels=512)
        # bottleneck encoder
        self.enc_conv5 = conv_block(in_channels=512, out_channels=1024)

        # decoder path
        self.up_conv5 = up_conv(in_channels=1024, out_channels=512)
        self.dec_conv5 = conv_block(in_channels=1024, out_channels=512)

        self.up_conv4 = up_conv(in_channels=512, out_channels=256)
        self.dec_conv4 = conv_block(in_channels=512, out_channels=256)

        self.up_conv3 = up_conv(in_channels=256, out_channels=128)
        self.dec_conv3 = conv_block(in_channels=256, out_channels=128)

        self.up_conv2 = up_conv(in_channels=128, out_channels=64)
        self.dec_conv2 = conv_block(in_channels=128, out_channels=64)

        # final layer
        self.fin_conv = Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoder path
        x1 = self.enc_conv1(x)
        x2 = self.enc_conv2(self.max_pool(x1))
        x3 = self.enc_conv3(self.max_pool(x2))
        x4 = self.enc_conv4(self.max_pool(x3))
        x5 = self.enc_conv5(self.max_pool(x4))

        # decoder and concat path
        d5 = self.up_conv5(x5)
        d5 = self.dec_conv5(cat((x4, d5), dim=1))

        d4 = self.up_conv4(d5)
        d4 = self.dec_conv4(cat((x3, d4), dim=1))

        d3 = self.up_conv3(d4)
        d3 = self.dec_conv3(cat((x2, d3), dim=1))

        d2 = self.up_conv2(d3)
        d2 = self.dec_conv2(cat((x1, d2), dim=1))

        return self.fin_conv(d2)
