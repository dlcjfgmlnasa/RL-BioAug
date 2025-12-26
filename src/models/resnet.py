# -*- coding:utf-8 -*-
import torch
import torch.nn as nn


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.elu = nn.ELU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.elu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = self.elu(out)
        return out


class ResNet1D18(nn.Module):
    def __init__(self, resnet_out_dim, feat_dim):
        super().__init__()
        self.in_channels = 64
        self.initial = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ELU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.resnet_out_dim = 512

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(resnet_out_dim, 2, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # Projection Head (for self-supervised learning)
        self.projector = nn.Sequential(
            nn.Linear(resnet_out_dim, resnet_out_dim),
            nn.BatchNorm1d(resnet_out_dim),
            nn.ELU(inplace=True),
            nn.Linear(resnet_out_dim, feat_dim)
        )

    def _make_layer(self, out_channels, blocks, stride):
        layers = list()
        layers.append(ResBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ResBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        h = self.avg_pool(x).flatten(1)

        z = self.projector(h)
        return h, z


if __name__ == '__main__':
    net = ResNet1D18(resnet_out_dim=256, feat_dim=128)
    oh, oz = net(
        torch.randn(64, 1, 3000)
    )
    print(oh.shape)
    print(oz.shape)
