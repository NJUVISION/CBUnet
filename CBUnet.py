import torch
from torch import nn, cat
from utils import differentiable_histogram

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.ReplicationPad2d(padding=1),
        nn.Conv2d(in_channels, out_channels, 3, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.ReplicationPad2d(padding=1),
        nn.Conv2d(out_channels, out_channels, 3, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        nn.ReplicationPad2d(padding=1),
        nn.Conv2d(out_channels, out_channels, 3, padding=0),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
    )


class CABlock(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, reduction=16):
        super(CABlock, self).__init__()

        self.conv_block = conv_block(in_channels, out_channels)

        self.attetion = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // reduction, out_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.conv_block(x)
        y = self.attetion(x)
        out = x * y
        return out


class CAUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(CAUNet, self).__init__()

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.down1 = conv_block(in_channels, 32)
        self.down2 = conv_block(32, 64)
        self.down3 = conv_block(64, 128)
        self.down4 = conv_block(128, 256)

        self.inter_conv = CABlock(256, 512)

        self.up4 = CABlock(512 + 256, 256)
        self.up3 = CABlock(256 + 128, 128)
        self.up2 = CABlock(128 + 64, 64)
        self.up1 = CABlock(64 + 32, 32)

        self.last_conv = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(32, out_channels, 3, padding=0))

    def forward(self, x):

        h, w = x.shape[2:]
        pad_h, pad_w = 0, 0
        target = 32
        if h % target != 0:
            pad_h = h // target * target + target - h
        if w % target != 0:
            pad_w = w // target * target + target - w
        x = nn.functional.pad(x, (0, pad_w, 0, pad_h), 'reflect', 0)

        conv1 = self.down1(x)
        x = self.maxpool(conv1)

        conv2 = self.down2(x)
        x = self.maxpool(conv2)

        conv3 = self.down3(x)
        x = self.maxpool(conv3)

        conv4 = self.down4(x)
        x = self.maxpool(conv4)

        low_features = self.inter_conv(x)

        x = self.upsample(low_features)
        x = cat([x, conv4], dim=1)
        x = self.up4(x)

        x = self.upsample(x)
        x = cat([x, conv3], dim=1)
        x = self.up3(x)

        x = self.upsample(x)
        x = cat([x, conv2], dim=1)
        x = self.up2(x)

        x = self.upsample(x)
        x = cat([x, conv1], dim=1)
        x = self.up1(x)

        x = x - conv1

        out = self.last_conv(x)

        out = out[:, :, :h, :w]

        return out


class Hist_CAUNet(CAUNet):
    def __init__(self):
        super().__init__(in_channels=4, out_channels=3)

        in_channels = 1
        out_channels = 128
        self.global_extract = nn.Sequential(
            nn.ReplicationPad2d(padding=15),
            nn.Conv2d(in_channels, out_channels, 31, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ReplicationPad2d(padding=3),
            nn.Conv2d(out_channels, out_channels, 7, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.ReplicationPad2d(padding=1),
            nn.Conv2d(out_channels, out_channels, 3, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.AdaptiveMaxPool2d((1, 1)),
        )

        self.hist_extract = nn.Sequential(
            nn.Linear(255, 255),
            nn.LeakyReLU(),
            nn.Linear(255, 128),
        )

        self.inter_conv_new = CABlock(256+128+128, 512)

    def forward(self, x, x_full_gray):
        h, w = x.shape[2:]
        pad_h, pad_w = 0, 0
        target = 32
        if h % target != 0:
            pad_h = h // target * target + target - h
        if w % target != 0:
            pad_w = w // target * target + target - w
        x = nn.functional.pad(x, (0, pad_w, 0, pad_h), 'reflect', 0)

        x_full_gray = nn.functional.interpolate(x_full_gray, (224, 224))
        global_info = self.global_extract(x_full_gray)
        hist = differentiable_histogram(x_full_gray, 255)[:, 0]  # [B, 256]
        hist /= 224 * 224
        hist_info = self.hist_extract(hist)

        conv1 = self.down1(x)
        x = self.maxpool(conv1)

        conv2 = self.down2(x)
        x = self.maxpool(conv2)

        conv3 = self.down3(x)
        x = self.maxpool(conv3)

        conv4 = self.down4(x)
        x = self.maxpool(conv4)

        _h, _w = x.shape[2:]
        global_info = global_info.repeat([1, 1, _h, _w])
        hist_info = hist_info[:, :, None, None].repeat([1, 1, _h, _w])
        low_features = torch.cat([x, global_info, hist_info], 1)
        low_features = self.inter_conv_new(low_features)

        x = self.upsample(low_features)
        x = cat([x, conv4], dim=1)
        x = self.up4(x)

        x = self.upsample(x)
        x = cat([x, conv3], dim=1)
        x = self.up3(x)

        x = self.upsample(x)
        x = cat([x, conv2], dim=1)
        x = self.up2(x)

        x = self.upsample(x)
        x = cat([x, conv1], dim=1)
        x = self.up1(x)

        x = x - conv1

        out = self.last_conv(x)

        out = out[:, :, :h, :w]

        return out

