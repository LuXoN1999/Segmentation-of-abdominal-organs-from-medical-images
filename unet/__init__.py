from torch import cat
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

    def forward(self, x):
        # encoder path
        x_enc1 = self.enc_conv1(x)
        x1 = self.max_pool(x_enc1)

        x_enc2 = self.enc_conv2(x1)
        x2 = self.max_pool(x_enc2)

        x_enc3 = self.enc_conv3(x2)
        x3 = self.max_pool(x_enc3)

        x_enc4 = self.enc_conv4(x3)
        x4 = self.max_pool(x_enc4)

        # bottleneck
        x_b = self.b_conv(x4)
        x5 = self.up_conv(x_b)

        # decoder path
        x_dec4 = self.dec_conv4(cat(tensors=[x5, x_enc4], dim=1))
        x6 = self.up_conv(x_dec4)

        x_dec3 = self.dec_conv3(cat(tensors=[x6, x_enc3], dim=1))
        x7 = self.up_conv(x_dec3)

        x_dec2 = self.dec_conv2(cat(tensors=[x7, x_enc2], dim=1))
        x8 = self.up_conv(x_dec2)

        x_dec1 = self.dec_conv1(cat(tensors=[x8, x_enc1], dim=1))
        x9 = self.up_conv(x_dec1)

        output = self.final_layer(x9)

        return output
