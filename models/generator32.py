import torch
import torch.nn as nn


class Generator32(nn.Module):
    def __init__(self, nz, ngf, nc):
        super(Generator32, self).__init__()

        # input: (nz) x 1 x 1
        self.tconv1 = nn.ConvTranspose2d(
            in_channels=nz,
            out_channels=ngf * 4,
            kernel_size=4,
            stride=1,
            padding=0,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(ngf * 4)
        self.relu1 = nn.ReLU(inplace=True)

        # state size: (ngf*4) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(
            in_channels=ngf * 4,
            out_channels=ngf * 2,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(ngf * 2)
        self.relu2 = nn.ReLU(inplace=True)

        # state size: (ngf*2) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(
            in_channels=ngf * 2,
            out_channels=ngf,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.bn3 = nn.BatchNorm2d(ngf)
        self.relu3 = nn.ReLU(inplace=True)

        # state size: (ngf) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(
            in_channels=ngf,
            out_channels=nc,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.tconv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.tconv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.tconv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.tconv4(x)
        x = self.tanh(x)

        return x
