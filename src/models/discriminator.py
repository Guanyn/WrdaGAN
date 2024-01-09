# encoding: utf8
# @Author: qiaoyongchao
# @Email: xiaoxia0722qyc@163.com
from src.models.modules import *

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from src.models.modules import *
from collections import OrderedDict
import clip


# 定义鉴别器网络D
class NetD(nn.Module):
    def __init__(self, ndf, imsize=128, ch_size=3):
        super(NetD, self).__init__()
        self.conv_img = nn.Conv2d(ch_size, ndf, 3, 1, 1)
        # build DBlocks
        self.DBlocks = nn.ModuleList([])
        in_out_pairs = get_D_in_out_chs(ndf, imsize)
        for idx, (in_ch, out_ch) in enumerate(in_out_pairs):
            self.DBlocks.append(D_Block(in_ch, out_ch))

    def forward(self,x):
        out = self.conv_img(x)
        for DBlock in self.DBlocks:
            out = DBlock(out)
        return out

    def embedding(self, x):
        out = self.conv_img(x)
        for DBlock in self.DBlocks:
            out = DBlock(out)
        return out

    def adaptive_mix(self, x_a, x_b, noise_std=0.05, alpha=1.0):

        lam = np.random.beta(alpha, alpha)
        x_mixed = x_a * lam + x_b * (1 - lam)

        out_mixed = self.embedding(x_mixed)
        out_a = self.embedding(x_a)
        out_b = self.embedding(x_b)

        mix_z = out_a * lam + out_b * (1 - lam)

        mix_z_ = mix_z + torch.normal(mean=0., std=noise_std, size=(mix_z.size())).cuda(x_a.device.index)
        return out_mixed, mix_z_


class D_Block(nn.Module):
    def __init__(self, fin, fout, downsample=True):
        super(D_Block, self).__init__()
        self.downsample = downsample
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fout, fout, 3, 1, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_s = nn.Conv2d(fin,fout, 1, stride=1, padding=0)
        self.gamma = nn.Parameter(torch.zeros(fout, 1, 1))

    def forward(self, x):
        res = self.conv_r(x)
        if self.learned_shortcut:
            x = self.conv_s(x)
        if self.downsample:
            x = F.avg_pool2d(x, 2)
        #return x + res
        return x + res * self.gamma


def get_D_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize))-1
    channel_nums = [nf*min(2**idx, 8) for idx in range(layer_num)]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs