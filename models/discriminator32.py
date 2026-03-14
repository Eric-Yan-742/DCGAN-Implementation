import torch
import torch.nn as nn


class Discriminator32(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator32, self).__init__()

        # input: (nc) x 32 x 32
        self.conv1 = nn.Conv2d(
            in_channels=nc,
            out_channels=ndf,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)

        # state size: (ndf) x 16 x 16
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

        # state size: (ndf*2) x 8 x 8
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

        # state size: (ndf*4) x 4 x 4
        self.conv4 = nn.Conv2d(
            in_channels=ndf * 4,
            out_channels=1,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

        self.pool = nn.AdaptiveMaxPool2d((4, 4))

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
        x = self.sigmoid(x)

        return x

    def extract_features(self, x):

        x = self.lrelu1(self.conv1(x))
        f1 = self.pool(x).flatten(1)

        x = self.lrelu2(self.bn2(self.conv2(x)))
        f2 = self.pool(x).flatten(1)

        x = self.lrelu3(self.bn3(self.conv3(x)))
        f3 = self.pool(x).flatten(1)

        return torch.cat([f1, f2, f3], dim=1)
