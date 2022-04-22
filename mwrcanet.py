# -*- coding: utf-8 -*-
# Yue Cao (cscaoyue@gmail.com) (cscaoyue@hit.edu.cn)
# supervisor : Wangmeng Zuo (cswmzuo@gmail.com)
# github: https://github.com/happycaoyue
# personal link:   happycaoyue.com
from turtle import forward
import torch
import torch.nn as nn
import numpy as np
import torch.nn.init as init
import torch.nn.functional as F

CHECKPOINT_PATH = '.pretrain/dn_mwrcanet_raw_c1.pth'

class HITVPCTeam:
    r"""
        DWT and IDWT block written by: Yue Cao
        """
    class CALayer(nn.Module):
        def __init__(self, channel=64, reduction=16):
            super(HITVPCTeam.CALayer, self).__init__()

            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel//reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel//reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
            )

        def forward(self, x):
            y = self.avg_pool(x)
            y = self.conv_du(y)
            return x * y

    # conv - prelu - conv - sum
    class RB(nn.Module):
        def __init__(self, filters):
            super(HITVPCTeam.RB, self).__init__()
            self.conv1 = nn.Conv2d(filters, filters, 3, 1, 1)
            self.act = nn.PReLU()
            self.conv2 = nn.Conv2d(filters, filters, 3, 1, 1)
            self.cuca = HITVPCTeam.CALayer(channel=filters)

        def forward(self, x):
            c0 = x
            x = self.conv1(x)
            x = self.act(x)
            x = self.conv2(x)
            out = self.cuca(x)
            return out + c0

    class NRB(nn.Module):
        def __init__(self, n, f):
            super(HITVPCTeam.NRB, self).__init__()
            nets = []
            for i in range(n):
                nets.append(HITVPCTeam.RB(f))
            self.body = nn.Sequential(*nets)
            self.tail = nn.Conv2d(f, f, 3, 1, 1)

        def forward(self, x):
            return x + self.tail(self.body(x))

    class DWTForward(nn.Module):
        def __init__(self):
            super(HITVPCTeam.DWTForward, self).__init__()
            ll = np.array([[0.5, 0.5], [0.5, 0.5]])
            lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
            hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
            hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
            filts = np.stack([ll[None,::-1,::-1], lh[None,::-1,::-1],
                              hl[None,::-1,::-1], hh[None,::-1,::-1]],
                             axis=0)
            self.weight = nn.Parameter(
                torch.tensor(filts).to(torch.get_default_dtype()),
                requires_grad=False)
        def forward(self, x):
            C = x.shape[1]
            filters = torch.cat([self.weight,] * C, dim=0)
            y = F.conv2d(x, filters, groups=C, stride=2)
            return y

    class DWTInverse(nn.Module):
        def __init__(self):
            super(HITVPCTeam.DWTInverse, self).__init__()
            ll = np.array([[0.5, 0.5], [0.5, 0.5]])
            lh = np.array([[-0.5, -0.5], [0.5, 0.5]])
            hl = np.array([[-0.5, 0.5], [-0.5, 0.5]])
            hh = np.array([[0.5, -0.5], [-0.5, 0.5]])
            filts = np.stack([ll[None, ::-1, ::-1], lh[None, ::-1, ::-1],
                              hl[None, ::-1, ::-1], hh[None, ::-1, ::-1]],
                             axis=0)
            self.weight = nn.Parameter(
                torch.tensor(filts).to(torch.get_default_dtype()),
                requires_grad=False)

        def forward(self, x):
            C = int(x.shape[1] / 4)
            filters = torch.cat([self.weight, ] * C, dim=0)
            y = F.conv_transpose2d(x, filters, groups=C, stride=2)
            return y


class Net(nn.Module):
    def __init__(self, channels=1, filters_level1=96, filters_level2=256//2, filters_level3=256//2, n_rb=4*5):
        super(Net, self).__init__()

        self.head = HITVPCTeam.DWTForward()

        self.down1 = nn.Sequential(
            nn.Conv2d(channels * 4, filters_level1, 3, 1, 1),
            nn.PReLU(),
            HITVPCTeam.NRB(n_rb, filters_level1))

        # sum 1
        # self.down1 = HITVPCTeam.NRB(n_rb, filters_level1),

        # sum 2
        self.down2 = nn.Sequential(
            HITVPCTeam.DWTForward(),
            nn.Conv2d(filters_level1 * 4, filters_level2, 3, 1, 1),
            nn.PReLU(),
            HITVPCTeam.NRB(n_rb, filters_level2))

        self.down3 = nn.Sequential(
            HITVPCTeam.DWTForward(),
            nn.Conv2d(filters_level2 * 4, filters_level3, 3, 1, 1),
            nn.PReLU())

        self.middle = HITVPCTeam.NRB(n_rb, filters_level3)

        self.up1 = nn.Sequential(
            nn.Conv2d(filters_level3, filters_level2 * 4, 3, 1, 1),
            nn.PReLU(),
            HITVPCTeam.DWTInverse())

        self.up2 = nn.Sequential(
            HITVPCTeam.NRB(n_rb, filters_level2),
            nn.Conv2d(filters_level2, filters_level1 * 4, 3, 1, 1),
            nn.PReLU(),
            HITVPCTeam.DWTInverse())

        self.up3 = nn.Sequential(
            HITVPCTeam.NRB(n_rb, filters_level1),
            nn.Conv2d(filters_level1, channels * 4, 3, 1, 1))

        self.tail = HITVPCTeam.DWTInverse()

    def forward(self, x):
        return self.slide_inference(x)

    def single_forward(self, inputs):
        c0 = inputs
        c1 = self.head(c0)
        c2 = self.down1(c1)
        c3 = self.down2(c2)
        c4 = self.down3(c3)
        m = self.middle(c4)
        c5 = self.up1(m) + c3
        c6 = self.up2(c5) + c2
        c7 = self.up3(c6) + c1
        return self.tail(c7)

    def slide_inference(self, img):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        h_stride, w_stride = 1024, 1024 # remove block
        h_crop, w_crop = 1024, 1024
        batch_size, c_img, h_img, w_img = img.size()
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        rec_img = img.new_zeros((batch_size, c_img, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                # y1 = max(y2 - h_crop, 0)
                # x1 = max(x2 - w_crop, 0)
                if (y2 - y1) % 64 != 0:
                    y1 = max(y2 - (y2 - y1) // 64 * 64 - 64, 0)
                if (x2 - x1) % 64 != 0:
                    x1 = max(x2 - (x2 - x1) // 64 * 64 - 64, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_rec_img = self.net(crop_img)
                rec_img += torch.nn.functional.pad(crop_rec_img,
                                 (int(x1), int(rec_img.shape[3] - x2), int(y1),
                                  int(rec_img.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        rec_img = rec_img / count_mat
        return rec_img

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
