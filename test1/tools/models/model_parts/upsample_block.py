import torch
import torch.nn as nn
from torch.nn import functional as F


class UpsampleBlock(nn.Module):
    def __init__(self, ch_in, ch_out, skip_in, mode, use_bn=True, drop_value=0):
        super(UpsampleBlock, self).__init__()

        self.drop_value = drop_value

        if self.drop_value > 0:
            self.dropout1 = nn.Dropout2d(p=0.1)

        ch_in = ch_in + skip_in
        self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=(3, 3),
                               stride=1, padding=1, bias=(not use_bn))
        self.bn1 = nn.BatchNorm2d(ch_out) if use_bn else None
        self.relu = nn.ReLU(inplace=True)

        if self.drop_value > 0:
            self.dropout2 = nn.Dropout2d(p=0.1)

        # second convolution
        self.conv2 = nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=(3, 3),
                               stride=1, padding=1, bias=(not use_bn))
        self.bn2 = nn.BatchNorm2d(ch_out) if use_bn else None

        self.mode = mode

    def forward(self, x, size, skip_connection=None):
        x = F.interpolate(x, size=size, mode=self.mode, align_corners=None)

        if skip_connection is not None:
            x = torch.cat([x, skip_connection], dim=1)

        if self.drop_value > 0:
            x = self.dropout1(x)

        x = self.conv1(x)
        x = self.bn1(x) if self.bn1 is not None else x
        x = self.relu(x)

        if self.drop_value > 0:
            x = self.dropout2(x)

        x = self.conv2(x)
        x = self.bn2(x) if self.bn2 is not None else x
        x = self.relu(x)

        return x
