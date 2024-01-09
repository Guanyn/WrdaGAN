# encoding: utf8
# @Author: qiaoyongchao
# @Email: xiaoxia0722qyc@163.com

from src.models.modules import *

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import clip
from transformers import ViTImageProcessor


class NetG(nn.Module):
    def __init__(self, ngf, nz, dim, ch_size, clip_model=None, CLIP_ch=768):
        super(NetG, self).__init__()
        self.ngf = ngf
        # input noise (batch_size, 100)
        self.clip = clip_model
        self.CLIP_ch = CLIP_ch
        if self.clip:
            self.clip_mapper = CLIP_Mapper(self.clip)
            self.fc_prompt = nn.Linear(dim + nz, self.CLIP_ch * 8)
            self.conv_fuse = nn.Conv2d(256, CLIP_ch, 5, 1, 2)
            self.conv = nn.Conv2d(768, 256, 5, 1, 2)

        self.fc = nn.Sequential(
            nn.Linear(nz, ngf * 8 * 4 * 4),
            nn.ReLU()
        )

        # # build GBlocks
        # self.GBlocks = nn.ModuleList([])
        #
        # self.GBlocks.append(G_Block(dim + nz, 256, 256, upsample=True))
        # self.GBlocks.append(G_Block(dim + nz, 256, 256, upsample=True))
        # self.GBlocks.append(G_Block(dim + nz, 256, 256, upsample=True))
        # self.GBlocks.append(G_Block(dim + nz, 256, 128, upsample=True))
        # self.GBlocks.append(G_Block(dim + nz, 128, 64, upsample=True))
        # self.GBlocks.append(G_Block(dim + nz, 64, 32, upsample=True))

        self.feat_8 = G_Block(dim + nz, 256, 256, upsample=True)
        self.feat_16 = G_Block(dim + nz, 256, 256, upsample=True)
        self.feat_32 = G_Block(dim + nz, 256, 256, upsample=True)
        self.feat_64 = G_Block(dim + nz, 256, 128, upsample=True)
        self.feat_128 = G_Block(dim + nz, 128, 64, upsample=True)
        self.feat_256 = G_Block(dim + nz, 64, 32, upsample=True)

        self.feat_256_to_128 = G_Block(dim + nz, 512, 128, upsample=True)
        self.feat_512_to_128 = G_Block(dim + nz, 512, 256, upsample=True)
        self.feat_1024_to_256 = G_Block(dim + nz, 512, 256, upsample=True)
        self.feat_512_to_512 = G_Block(dim + nz, 128, 128, upsample=True)

        # WaveUnpool
        self.recon_block2 = WaveUnpool(512, "sum")
        self.recon_block3 = WaveUnpool(512, "sum")
        self.recon_block4 = WaveUnpool(512, "sum")
        # WavePool
        self.pool128 = WavePool(256)
        self.pool256 = WavePool(256)
        self.pool512 = WavePool(256)

        # self.ks = KSBLK(dim + nz, out_ch)
        # to RGB image
        self.to_rgb = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, ch_size, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, c):  # x=noise, c=ent_emb
        c = c.float()
        # concat noise and sentence
        # (bs, 4096)
        out = self.fc(noise)
        # (bs, 256, 4, 4)
        out = out.view(noise.size(0), 8*self.ngf, 4, 4)

        cond = torch.cat((noise, c), dim=1)

        if self.clip:
            # (bs, 8, 768)
            prompts = self.fc_prompt(cond).view(cond.size(0), -1, self.CLIP_ch)
            fuse_feat = self.conv_fuse(out)
            map_feat = self.clip_mapper(fuse_feat, prompts)
            out = self.conv(fuse_feat + 0.1 * map_feat)

        feat_8 = self.feat_8(out, cond)
        LL_8, LH_8, HL_8, HH_8 = self.pool512(
            feat_8)  # (bs, 1024, 4, 4) (bs, 1024, 4, 4) (bs, 1024, 4, 4) (bs, 1024, 4, 4)
        original_8 = self.recon_block4(LL_8, LH_8, HL_8, HH_8)  # (bs, 1024, 8, 8)
        original_8 = self.feat_1024_to_256(original_8, cond)  # (bs, 256, 16, 16)
        feat_16 = self.feat_16(feat_8, cond)  # (bs, 256, 16, 16)
        LL_16, LH_16, HL_16, HH_16 = self.pool256(
            feat_16)  # (bs, 512, 8, 8) (bs, 512, 8, 8) (bs, 512, 8, 8) (bs, 512, 8, 8)
        original_16 = self.recon_block3(LL_16, LH_16, HL_16, HH_16)  # (bs, 512, 16, 16)
        original_16 = self.feat_512_to_128(original_16, cond)  # (bs, 128, 32, 32)
        feat_32 = self.feat_32(feat_16 + original_8, cond)  # (bs, 128, 32, 32)
        LL_32, LH_32, HL_32, HH_32 = self.pool128(feat_32)  # (bs, 256, 16, 16), (bs, 256, 16, 16), (bs, 256, 16, 16), (bs, 256, 16, 16)
        original_32 = self.recon_block2(LL_32, LH_32, HL_32, HH_32)  # (bs, 256, 32, 32)
        original_32 = self.feat_256_to_128(original_32, cond)  # (bs, 128, 64, 64)

        feat_64 = self.feat_64(feat_32 + original_16, cond)  # (bs, 128, 64, 64)
        feat_128 = self.feat_128(feat_64 + original_32, cond)  # (bs, 64, 128, 128)

        feat_256 = self.feat_256(feat_128, cond)  # (bs, 32, 256, 256)

        # convert to RGB image
        # (bs, 3, 256, 256)
        out = self.to_rgb(feat_256)
        return out


class G_Block(nn.Module):
    def __init__(self, cond_dim, in_ch, out_ch, upsample):
        super(G_Block, self).__init__()
        self.upsample = upsample
        self.learnable_sc = in_ch != out_ch
        self.c1 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        self.c2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1)
        self.fuse1 = DFBLK(cond_dim, in_ch)
        self.fuse2 = DFBLK(cond_dim, out_ch)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(in_ch, out_ch, 1, stride=1, padding=0)

    def shortcut(self, x):
        # x:
        if self.learnable_sc:
            #
            x = self.c_sc(x)
        return x

    def residual(self, h, w):
        # h: (bs, 256, 8)  y:(1, 868)
        # (bs, 256, 8, 8)
        h1 = self.fuse1(h, w)
        # (bs, 256, 8, 8)
        h1 = self.c1(h1)
        # (bs, 256, 8, 8)
        h1 = self.fuse2(h1, w)
        # (bs, 256, 8, 8)
        h1 = self.c2(h1)
        return h1

    def forward(self, x, y):
        # x: (bs, 256, 4, 4) y:(1, 868)
        if self.upsample == True:
            # (bs, 256, 8, 8)
            x = F.interpolate(x, scale_factor=2)
        out = self.shortcut(x) + self.residual(x, y)
        return out


def get_G_in_out_chs(nf, imsize):
    layer_num = int(np.log2(imsize)) - 1
    channel_nums = [nf * min(2 ** idx, 8) for idx in range(layer_num)]
    channel_nums = channel_nums[::-1]
    in_out_pairs = zip(channel_nums[:-1], channel_nums[1:])
    return in_out_pairs, channel_nums


class CLIP_Mapper(nn.Module):
    def __init__(self, CLIP):
        super(CLIP_Mapper, self).__init__()
        model = CLIP.visual
        # print(model)
        self.define_module(model)
        for param in model.parameters():
            param.requires_grad = False

    def define_module(self, model):
        self.conv1 = model.conv1
        self.class_embedding = model.class_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_pre = model.ln_pre
        self.transformer = model.transformer

    @property
    def dtype(self):
        return self.conv1.weight.dtype

    def forward(self, img: torch.Tensor, prompts: torch.Tensor):
        # (bs, 768, 4, 4)
        x = img.type(self.dtype)
        # (bs, 8, 768)
        prompts = prompts.type(self.dtype)
        grid = x.size(-1)
        # (bs, 768, 16)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        # (bs, 16, 768)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        # (bs, 50, 768)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding[:17].to(x.dtype)
        x = self.ln_pre(x)
        # NLD -> LND
        x = x.permute(1, 0, 2)
        # Local features
        selected = [1,2,3,4,5,6,7,8]
        begin, end = 0, 12
        prompt_idx = 0
        for i in range(begin, end):
            if i in selected:
                prompt = prompts[:,prompt_idx,:].unsqueeze(0)
                prompt_idx = prompt_idx+1
                x = torch.cat((x,prompt), dim=0)
                x = self.transformer.resblocks[i](x)
                x = x[:-1,:,:]
            else:
                x = self.transformer.resblocks[i](x)
        return x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(-1, 768, grid, grid).contiguous().type(img.dtype)
