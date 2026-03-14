import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()

        # spatial size: 64 x 64
        self.conv1 = nn.Conv2d(
            in_channels=nc,
            out_channels=ndf,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)

        # spatial size: 32 x 32
        self.conv2 = nn.Conv2d(
            in_channels=ndf,
            out_channels=ndf * 2,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(ndf * 2)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)

        # spatial size: 16 x 16
        self.conv3 = nn.Conv2d(
            in_channels=ndf * 2,
            out_channels=ndf * 4,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(ndf * 4)
        self.lrelu3 = nn.LeakyReLU(0.2, inplace=True)

        # spatial size: 8 x 8
        self.conv4 = nn.Conv2d(
            in_channels=ndf * 4,
            out_channels=ndf * 8,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn4 = nn.BatchNorm2d(ndf * 8)
        self.lrelu4 = nn.LeakyReLU(0.2, inplace=True)

        # spatial size: 4 x 4
        self.conv5 = nn.Conv2d(
            in_channels=ndf * 8,
            out_channels=1,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.lrelu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lrelu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lrelu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.lrelu4(x)

        x = self.conv5(x)
        x = self.sigmoid(x)

        return x