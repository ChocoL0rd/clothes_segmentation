import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from collections import OrderedDict

from .model_parts.u2net_parts import *


##### U^2-Net ####
class U2NET(nn.Module):
    def __init__(self, cfg):
        super(U2NET, self).__init__()

        out_ch = cfg["out_ch"]

        self.stage1 = RSU7(3, 32, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(512, 256, 512)

        # decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # -------------------- decoder --------------------
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        """
        del hx1, hx2, hx3, hx4, hx5, hx6
        del hx5d, hx4d, hx3d, hx2d, hx1d
        del hx6up, hx5dup, hx4dup, hx3dup, hx2dup
        """

        return d0, d1, d2, d3, d4, d5, d6

    def inference(self, x):
        x = self.forward(x)
        return [nn.Sigmoid()(i) for i in x]

    def get_params(self):
        return [self.parameters()]


### U^2-Net small ###
class U2NETP(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(U2NETP, self).__init__()

        self.stage1 = RSU7(in_ch, 16, 64)
        self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage3 = RSU5(64, 16, 64)
        self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage4 = RSU4(64, 16, 64)
        self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.stage6 = RSU4F(64, 16, 64)

        # decoder
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)

        self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        hx = x

        # stage 1
        hx1 = self.stage1(hx)
        hx = self.pool12(hx1)

        # stage 2
        hx2 = self.stage2(hx)
        hx = self.pool23(hx2)

        # stage 3
        hx3 = self.stage3(hx)
        hx = self.pool34(hx3)

        # stage 4
        hx4 = self.stage4(hx)
        hx = self.pool45(hx4)

        # stage 5
        hx5 = self.stage5(hx)
        hx = self.pool56(hx5)

        # stage 6
        hx6 = self.stage6(hx)
        hx6up = _upsample_like(hx6, hx5)

        # decoder
        hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side1(hx1d)

        d2 = self.side2(hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side3(hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side4(hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side5(hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side6(hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat((d1, d2, d3, d4, d5, d6), 1))

        """
        del hx1, hx2, hx3, hx4, hx5, hx6
        del hx5d, hx4d, hx3d, hx2d, hx1d
        del hx6up, hx5dup, hx4dup, hx3dup, hx2dup
        """

        return d0, d1, d2, d3, d4, d5, d6


#
# ### U^2-Net small ###
# class MiniU2Net(nn.Module):
#     def __init__(self, cfg):
#         super(MiniU2Net, self).__init__()
#         out_ch = cfg["out_ch"]
#         self.stage1 = RSU7(3, 16, 64)
#         self.pool12 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
#
#         self.stage2 = RSU6(64, 16, 64)
#         self.pool23 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
#
#         self.stage3 = RSU5(64, 16, 64)
#         self.pool34 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
#
#         self.stage4 = RSU4(64, 16, 64)
#         self.pool45 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
#
#         self.stage5 = RSU4F(64, 16, 64)
#         self.pool56 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
#
#         self.stage6 = RSU4F(64, 16, 64)
#
#         # decoder
#         self.stage5d = RSU4F(128, 16, 64)
#         self.stage4d = RSU4(128, 16, 64)
#         self.stage3d = RSU5(128, 16, 64)
#         self.stage2d = RSU6(128, 16, 64)
#         self.stage1d = RSU7(128, 16, 64)
#
#         self.side1 = nn.Conv2d(64, out_ch, 3, padding=1)
#         self.side2 = nn.Conv2d(64, out_ch, 3, padding=1)
#         self.side3 = nn.Conv2d(64, out_ch, 3, padding=1)
#         self.side4 = nn.Conv2d(64, out_ch, 3, padding=1)
#         self.side5 = nn.Conv2d(64, out_ch, 3, padding=1)
#         self.side6 = nn.Conv2d(64, out_ch, 3, padding=1)
#
#         self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)
#
#     def forward(self, x):
#         hx = x
#
#         # stage 1
#         hx1 = self.stage1(hx)
#         hx = self.pool12(hx1)
#
#         # stage 2
#         hx2 = self.stage2(hx)
#         hx = self.pool23(hx2)
#
#         # stage 3
#         hx3 = self.stage3(hx)
#         hx = self.pool34(hx3)
#
#         # stage 4
#         hx4 = self.stage4(hx)
#         hx = self.pool45(hx4)
#
#         # stage 5
#         hx5 = self.stage5(hx)
#         hx = self.pool56(hx5)
#
#         # stage 6
#         hx6 = self.stage6(hx)
#         hx6up = _upsample_like(hx6, hx5)
#
#         # decoder
#         hx5d = self.stage5d(torch.cat((hx6up, hx5), 1))
#         hx5dup = _upsample_like(hx5d, hx4)
#
#         hx4d = self.stage4d(torch.cat((hx5dup, hx4), 1))
#         hx4dup = _upsample_like(hx4d, hx3)
#
#         hx3d = self.stage3d(torch.cat((hx4dup, hx3), 1))
#         hx3dup = _upsample_like(hx3d, hx2)
#
#         hx2d = self.stage2d(torch.cat((hx3dup, hx2), 1))
#         hx2dup = _upsample_like(hx2d, hx1)
#
#         hx1d = self.stage1d(torch.cat((hx2dup, hx1), 1))
#
#         # side output
#         d1 = self.side1(hx1d)
#
#         d2 = self.side2(hx2d)
#         d2 = _upsample_like(d2, d1)
#
#         d3 = self.side3(hx3d)
#         d3 = _upsample_like(d3, d1)
#
#         d4 = self.side4(hx4d)
#         d4 = _upsample_like(d4, d1)
#
#         d5 = self.side5(hx5d)
#         d5 = _upsample_like(d5, d1)
#
#         d6 = self.side6(hx6)
#         d6 = _upsample_like(d6, d1)
#
#         d0 = self.outconv(torch.cat([d1, d2, d3, d4, d5, d6], 1))
#
#         """
#         del hx1, hx2, hx3, hx4, hx5, hx6
#         del hx5d, hx4d, hx3d, hx2d, hx1d
#         del hx6up, hx5dup, hx4dup, hx3dup, hx2dup
#         """
#
#         return [d0, d1, d2, d3, d4, d5, d6]
#
#     def inference(self, x):
#         x = self.forward(x)
#         return [nn.Sigmoid()(i) for i in x]
#
#     def get_params(self):
#         return [self.parameters()]
#


### U^2-Net small ###
class MiniU2Net(nn.Module):
    def __init__(self, cfg):
        super(MiniU2Net, self).__init__()
        out_ch = cfg["out_ch"]

        self.encoder = nn.ModuleDict(OrderedDict([
            [
                "lvl1",
                nn.ModuleDict(OrderedDict([
                    ["RSU7", RSU7(3, 16, 64)],
                    ["maxpool", nn.MaxPool2d(2, stride=2, ceil_mode=True)]
                ]))
            ],
            [
                "lvl2",
                nn.ModuleDict(OrderedDict([
                    ["RSU6", RSU6(64, 16, 64)],
                    ["maxpool", nn.MaxPool2d(2, stride=2, ceil_mode=True)]
                ]))
            ],
            [
                "lvl3",
                nn.ModuleDict(OrderedDict([
                    ["RSU5", RSU5(64, 16, 64)],
                    ["maxpool", nn.MaxPool2d(2, stride=2, ceil_mode=True)]
                ]))
            ],
            [
                "lvl4",
                nn.ModuleDict(OrderedDict([
                    ["RSU4", RSU4(64, 16, 64)],
                    ["maxpool", nn.MaxPool2d(2, stride=2, ceil_mode=True)]
                ]))
            ],
            [
                "lvl5",
                nn.ModuleDict(OrderedDict([
                    ["RSU4F", RSU4F(64, 16, 64)],
                    ["maxpool", nn.MaxPool2d(2, stride=2, ceil_mode=True)]
                ]))
            ],
            [
                "lvl6",
                nn.ModuleDict(OrderedDict([
                    ["RSU4F", RSU4F(64, 16, 64)],
                ]))
            ]
        ]))

        self.encoder_params = []
        for i in range(6):
            if cfg["freeze_encoder"][i]:
                for param in self.encoder[f"lvl{i + 1}"].parameters():
                    param.requires_grad = False
            else:
                self.encoder_params.append(
                    self.encoder[f"lvl{i + 1}"].parameters()
                )

        self.decoder = nn.ModuleDict(OrderedDict([
            ["lvl5", RSU4F(128, 16, 64)],
            ["lvl4", RSU4(128, 16, 64)],
            ["lvl3", RSU5(128, 16, 64)],
            ["lvl2", RSU6(128, 16, 64)],
            ["lvl1", RSU7(128, 16, 64)]
        ]))

        self.decoder_params = []
        for i in range(5, 0, -1):
            if cfg["freeze_decoder"][i - 1]:
                for param in self.decoder[f"lvl{i}"].parameters():
                    param.requires_grad = False
            else:
                self.decoder_params.append(
                    self.decoder[f"lvl{i}"].parameters()
                )

        self.side = nn.ModuleList([
            nn.Conv2d(64, out_ch, 3, padding=1),
            nn.Conv2d(64, out_ch, 3, padding=1),
            nn.Conv2d(64, out_ch, 3, padding=1),
            nn.Conv2d(64, out_ch, 3, padding=1),
            nn.Conv2d(64, out_ch, 3, padding=1),
            nn.Conv2d(64, out_ch, 3, padding=1)
        ])

        self.side_params = []
        for i in range(6):
            if cfg["freeze_side"][i]:
                for param in self.side[i].parameters():
                    param.requires_grad = False
            else:
                self.side_params.append(
                    self.side[i].parameters()
                )

        self.outconv = nn.Conv2d(6 * out_ch, out_ch, 1)

    def forward(self, x):
        hx = x

        # encoder
        # lvl1
        hx1 = self.encoder["lvl1"]["RSU7"].forward(hx)
        hx = self.encoder["lvl1"]["maxpool"](hx1)

        # lvl2
        if hx.requires_grad:
            hx2 = checkpoint(self.encoder["lvl2"]["RSU6"].forward, hx)
        else:
            hx2 = self.encoder["lvl2"]["RSU6"].forward(hx)
        hx = self.encoder["lvl2"]["maxpool"](hx2)

        # lvl3
        if hx.requires_grad:
            hx3 = checkpoint(self.encoder["lvl3"]["RSU5"].forward, hx)
        else:
            hx3 = self.encoder["lvl3"]["RSU5"].forward(hx)
        hx = self.encoder["lvl3"]["maxpool"](hx3)

        # lvl4
        if hx.requires_grad:
            hx4 = checkpoint(self.encoder["lvl4"]["RSU4"].forward, hx)
        else:
            hx4 = self.encoder["lvl4"]["RSU4"].forward(hx)
        hx = self.encoder["lvl4"]["maxpool"](hx4)

        # lvl5
        if hx.requires_grad:
            hx5 = checkpoint(self.encoder["lvl5"]["RSU4F"].forward, hx)
        else:
            hx5 = self.encoder["lvl5"]["RSU4F"].forward(hx)
        hx = self.encoder["lvl5"]["maxpool"](hx5)

        # lvl6
        if hx.requires_grad:
            hx6 = checkpoint(self.encoder["lvl6"]["RSU4F"].forward, hx)
        else:
            hx6 = self.encoder["lvl6"]["RSU4F"].forward(hx)
        hx6up = _upsample_like(hx6, hx5)

        # decoder
        # lvl5
        # if hx6up.requires_grad or hx5.requires_grad:
        #     hx5d = checkpoint(self.decoder["lvl5"].forward, torch.cat((hx6up, hx5), 1))
        # else:
        hx5d = self.decoder["lvl5"].forward(torch.cat((hx6up, hx5), 1))
        hx5dup = _upsample_like(hx5d, hx4)

        # lvl4
        # if hx5dup.requires_grad or hx4.requires_grad:
        #     hx4d = checkpoint(self.decoder["lvl4"].forward, torch.cat((hx5dup, hx4), 1))
        # else:
        hx4d = self.decoder["lvl4"].forward(torch.cat((hx5dup, hx4), 1))
        hx4dup = _upsample_like(hx4d, hx3)

        # lvl3
        # if hx4dup.requires_grad or hx3.requires_grad:
        #     hx3d = checkpoint(self.decoder["lvl3"].forward, torch.cat((hx4dup, hx3), 1))
        # else:
        hx3d = self.decoder["lvl3"].forward(torch.cat((hx4dup, hx3), 1))
        hx3dup = _upsample_like(hx3d, hx2)

        # lvl2
        # if hx3dup.requires_grad or hx2.requires_grad:
        #     hx2d = checkpoint(self.decoder["lvl2"].forward, torch.cat((hx3dup, hx2), 1))
        # else:
        hx2d = self.decoder["lvl2"].forward(torch.cat((hx3dup, hx2), 1))
        hx2dup = _upsample_like(hx2d, hx1)

        # lvl1
        # if hx2dup.requires_grad or hx1.requires_grad:
        #     hx1d = checkpoint(self.decoder["lvl1"].forward, torch.cat((hx2dup, hx1), 1))
        # else:
        hx1d = self.decoder["lvl1"].forward(torch.cat((hx2dup, hx1), 1))

        # side output
        d1 = self.side[0](hx1d)

        d2 = self.side[1](hx2d)
        d2 = _upsample_like(d2, d1)

        d3 = self.side[2](hx3d)
        d3 = _upsample_like(d3, d1)

        d4 = self.side[3](hx4d)
        d4 = _upsample_like(d4, d1)

        d5 = self.side[4](hx5d)
        d5 = _upsample_like(d5, d1)

        d6 = self.side[5](hx6)
        d6 = _upsample_like(d6, d1)

        d0 = self.outconv(torch.cat([d1, d2, d3, d4, d5, d6], 1))

        """
        del hx1, hx2, hx3, hx4, hx5, hx6
        del hx5d, hx4d, hx3d, hx2d, hx1d
        del hx6up, hx5dup, hx4dup, hx3dup, hx2dup
        """

        return [d0, d1, d2, d3, d4, d5, d6]

    def inference(self, x):
        x = self.forward(x)
        return [nn.Sigmoid()(i) for i in x]

    def get_params(self):
        return self.encoder_params + self.decoder_params + self.side_params + [self.outconv.parameters()]
