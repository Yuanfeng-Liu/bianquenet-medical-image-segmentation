import torch
import torch.nn as nn


class SimpleSegmentationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x
