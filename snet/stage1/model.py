"""Stage 1 model: Modified 3D U-Net for spinal centerline regression."""

import torch
import torch.nn as nn


class ConvBlock3D(nn.Module):
    """Two consecutive Conv3d(3x3x3) + LeakyReLU(0.01) blocks."""

    def __init__(self, in_ch, out_ch=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet3D_Stage1(nn.Module):
    """Modified 3D U-Net for Stage-1 centerline heatmap regression.

    Architecture: 5-level encoder-decoder, AvgPool3d, trilinear upsample,
    all 64 filters, 3x3x3 kernels.

    Input:  [B, 1, 64, 64, 128]  (CT volume)
    Output: [B, 1, 64, 64, 128]  (predicted heatmap, NO sigmoid)
    """

    def __init__(self, in_channels=1, num_filters=64):
        super().__init__()
        f = num_filters

        # Encoder
        self.enc1 = ConvBlock3D(in_channels, f)
        self.enc2 = ConvBlock3D(f, f)
        self.enc3 = ConvBlock3D(f, f)
        self.enc4 = ConvBlock3D(f, f)

        # Bottleneck (level 5)
        self.bottleneck = ConvBlock3D(f, f)

        # Pooling & upsampling
        self.pool = nn.AvgPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)

        # Decoder (skip concat doubles channels -> 2f -> f)
        self.dec4 = ConvBlock3D(f * 2, f)
        self.dec3 = ConvBlock3D(f * 2, f)
        self.dec2 = ConvBlock3D(f * 2, f)
        self.dec1 = ConvBlock3D(f * 2, f)

        # Output head — NO sigmoid (MSE loss on raw values)
        self.head = nn.Conv3d(f, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        bn = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up(bn), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))

        return self.head(d1)
