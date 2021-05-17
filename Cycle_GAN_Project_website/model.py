import os
import numpy as np

import torch
import torch.nn as nn

from layer import *

## 네트워크 구축하기
# CycleGAN
# https://arxiv.org/pdf/1703.10593.pdf

class CycleGAN(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, norm='bnorm', nblk=6):
        super(CycleGAN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.nker = nker
        self.norm = norm
        self.nblk = nblk

        if norm == 'bnorm':
            self.bias = False
        else:
            self.bias = True

        self.enc1 = CBR2d(self.in_channels, 1 * self.nker, kernel_size=7, stride=1, padding=3, norm=self.norm, relu=0.0)
        self.enc2 = CBR2d(1 * self.nker, 2 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)
        self.enc3 = CBR2d(2 * self.nker, 4 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)

        if self.nblk:
            res = []

            for i in range(self.nblk):
                res += [ResBlock(4 * self.nker, 4 * self.nker, kernel_size=3, stride=1, padding=1, norm=self.norm, relu=0.0)]

            self.res = nn.Sequential(*res)

        self.dec3 = DECBR2d(4 * self.nker, 2 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)
        self.dec2 = DECBR2d(2 * self.nker, 1 * self.nker, kernel_size=3, stride=2, padding=1, norm=self.norm, relu=0.0)
        self.dec1 = CBR2d(1 * self.nker, self.out_channels, kernel_size=7, stride=1, padding=3, norm=None, relu=None)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        x = self.res(x)

        x = self.dec3(x)
        x = self.dec2(x)
        x = self.dec1(x)

        x = torch.tanh(x)

        return x
