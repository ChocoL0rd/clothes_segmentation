import torch
from torch import nn
from collections import OrderedDict

__all__ = [
    "Bottleneck",
    "NeckSequence"
]


class Bottleneck(nn.Module):
    def __init__(self, channel, hid_channel, use_bn=True, out_channel=None):
        super(Bottleneck, self).__init__()

        self.use_bn = use_bn

        if out_channel is None:
            out_channel = channel

        if self.use_bn:
            self.conv1 = nn.Conv2d(channel, hid_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)
            self.bn1 = nn.BatchNorm2d(hid_channel)
            self.conv2 = nn.Conv2d(hid_channel, hid_channel,
                                   kernel_size=3, stride=1, padding=1, bias=False)
            self.bn2 = nn.BatchNorm2d(hid_channel)
            self.conv3 = nn.Conv2d(hid_channel, out_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channel)
        else:
            self.conv1 = nn.Conv2d(channel, hid_channel,
                                   kernel_size=1, stride=1, padding=0, bias=True)
            self.conv2 = nn.Conv2d(hid_channel, hid_channel,
                                   kernel_size=3, stride=1, padding=1, bias=True)
            self.conv3 = nn.Conv2d(hid_channel, out_channel,
                                   kernel_size=1, stride=1, padding=0, bias=True)

        self.relu = nn.ReLU()

    def forward(self, x):
        if self.use_bn:
            y = self.conv1(x)
            y = self.bn1(y)
            y = self.conv2(y)
            y = self.bn2(y)
            y = self.conv3(y)
            y = self.bn3(y)
        else:
            y = self.conv1(x)
            y = self.conv2(y)
            y = self.conv3(y)

        y = self.relu(y)
        return x + y


class NeckSequence(nn.Sequential):
    def __init__(self, channels, length, use_bn=True):
        super(NeckSequence, self).__init__()
        if isinstance(use_bn, list):
            for i in range(length):
                self.add_module(f"Bottleneck{i}", Bottleneck(channels, channels // 2, use_bn[i]))
        else:
            for i in range(length):
                self.add_module(f"Bottleneck{i}", Bottleneck(channels, channels // 2, use_bn))


class DilatedBottleneck(nn.Module):
    def __init__(self, in_chnls, dilations):
        super(DilatedBottleneck, self).__init__()

        hid_chnls = in_chnls // 2

        self.in_conv = nn.Conv2d(in_chnls, hid_chnls,
                                 kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(hid_chnls)

        self.hidden_layer = nn.ModuleList()

        for dilation in dilations:
            self.hidden_layer.append(nn.Sequential(OrderedDict([
                ("conv", nn.Conv2d(hid_chnls, hid_chnls // len(dilations),
                                   kernel_size=3, stride=1, bias=False,
                                   padding=dilation, dilation=dilation)),
                ("bn", nn.BatchNorm2d(hid_chnls // len(dilations)))
            ])))

        num_hidden_chnls = (hid_chnls // len(dilations)) * len(dilations)
        self.out_conv = nn.Conv2d(num_hidden_chnls, in_chnls,
                                  kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(in_chnls)

        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.bn1(self.in_conv(x))

        hid_outs = []
        for hid_conv in self.hidden_layer:
            hid_outs.append(hid_conv(y))
        y = torch.cat(hid_outs, dim=1)

        y = self.relu(self.bn3(self.out_conv(y)))

        return x + y


class DilatedNeckSequence(nn.Sequential):
    def __init__(self, channels, dilations, length):
        super(DilatedNeckSequence, self).__init__()
        for i in range(length):
            self.add_module(f"Bottleneck{i}", DilatedBottleneck(channels, dilations))



