"""Stage 2 model: SpatialConfiguration-Net (SC-Net) for centroid detection."""

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


class LocalAppearanceUNet(nn.Module):
    """5-level U-Net for the Local Appearance branch.

    Input:  [B, 1, 96, 96, 128]
    Output: [B, 64, 96, 96, 128]
    """

    def __init__(self, in_channels=1, filters=64):
        super().__init__()
        f = filters
        self.enc1 = ConvBlock3D(in_channels, f)
        self.enc2 = ConvBlock3D(f, f)
        self.enc3 = ConvBlock3D(f, f)
        self.enc4 = ConvBlock3D(f, f)
        self.bottleneck = ConvBlock3D(f, f)
        self.pool = nn.AvgPool3d(2)
        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=False)
        self.dec4 = ConvBlock3D(f * 2, f)
        self.dec3 = ConvBlock3D(f * 2, f)
        self.dec2 = ConvBlock3D(f * 2, f)
        self.dec1 = ConvBlock3D(f * 2, f)

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
        return d1


class SpatialConfigBranch(nn.Module):
    """Spatial Configuration branch at 1/4 resolution.

    4 consecutive Conv3d(7x7x7) + LeakyReLU at 1/4 res, then upsample back.
    """

    def __init__(self, in_channels=1, filters=64):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv3d(in_channels, filters, kernel_size=7, padding=3, bias=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(filters, filters, kernel_size=7, padding=3, bias=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(filters, filters, kernel_size=7, padding=3, bias=True),
            nn.LeakyReLU(0.01, inplace=True),
            nn.Conv3d(filters, filters, kernel_size=7, padding=3, bias=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.down = nn.AvgPool3d(4)
        self.up = nn.Upsample(scale_factor=4, mode="trilinear", align_corners=False)

    def forward(self, x):
        return self.up(self.branch(self.down(x)))


class SCNet(nn.Module):
    """SpatialConfiguration-Net for Stage 2.

    Input:  [B, 1, 96, 96, 128]
    Output: heatmaps [B, 25, 96, 96, 128], learned_sigmas [25]
    """

    def __init__(self, in_channels=1, num_classes=25, filters=64, sigma_init=5.0):
        super().__init__()
        self.local_appearance = LocalAppearanceUNet(in_channels, filters)
        self.spatial_config = SpatialConfigBranch(in_channels, filters)

        self.combine_conv = nn.Sequential(
            nn.Conv3d(filters * 2, filters, kernel_size=1, bias=True),
            nn.LeakyReLU(0.01, inplace=True),
        )
        self.head = nn.Conv3d(filters, num_classes, kernel_size=1, bias=True)
        self.learned_sigmas = nn.Parameter(torch.ones(num_classes) * sigma_init)

    def forward(self, x):
        local_feat = self.local_appearance(x)
        spatial_feat = self.spatial_config(x)
        combined = torch.cat([local_feat, spatial_feat], dim=1)
        combined = self.combine_conv(combined)
        heatmaps = self.head(combined)
        return heatmaps, self.learned_sigmas
