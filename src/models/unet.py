"""
U-Net architecture for marine snow removal.

A standard encoder-decoder with skip connections, designed for image-to-image
translation where the input is a marine-snow-corrupted image and the output
is the clean image.

Compute cost: ~20-50ms per frame on GPU for 1080p (depending on depth/width).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Double convolution block: Conv-BN-ReLU x2."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """Standard U-Net for marine snow removal.

    Architecture:
        Encoder: ConvBlock → MaxPool, repeated `depth` times
        Bottleneck: ConvBlock
        Decoder: Upsample → Concat(skip) → ConvBlock, repeated `depth` times
        Output: 1x1 Conv → Sigmoid (or raw for residual learning)

    Typical config: depth=4, base_filters=64 → ~31M params
    Lightweight:    depth=3, base_filters=32 → ~2M params
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_filters: int = 64,
        depth: int = 4,
        residual: bool = True,
    ):
        """
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            base_filters: Number of filters in the first encoder layer.
            depth: Number of encoder/decoder levels.
            residual: If True, learn the residual (snow) and subtract from input.
        """
        super().__init__()
        self.depth = depth
        self.residual = residual

        # Encoder
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        channels = in_channels
        for i in range(depth):
            out_ch = base_filters * (2 ** i)
            self.encoders.append(ConvBlock(channels, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            channels = out_ch

        # Bottleneck
        bottleneck_ch = base_filters * (2 ** depth)
        self.bottleneck = ConvBlock(channels, bottleneck_ch)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        channels = bottleneck_ch
        for i in range(depth - 1, -1, -1):
            out_ch = base_filters * (2 ** i)
            self.upconvs.append(
                nn.ConvTranspose2d(channels, out_ch, kernel_size=2, stride=2)
            )
            self.decoders.append(ConvBlock(out_ch * 2, out_ch))  # *2 for skip concat
            channels = out_ch

        # Output
        self.output_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Store input for residual connection
        identity = x

        # Encoder path
        skips = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path
        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            # Handle size mismatches from odd dimensions
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        x = self.output_conv(x)

        if self.residual:
            # Learn the snow pattern and subtract it
            x = identity - x

        return torch.clamp(x, 0.0, 1.0)


class LightweightUNet(nn.Module):
    """Lightweight U-Net variant with depthwise separable convolutions.

    ~5x fewer parameters than standard U-Net, suitable for faster inference.
    Depth=3, base_filters=32 → ~400K params.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        base_filters: int = 32,
        depth: int = 3,
        residual: bool = True,
    ):
        super().__init__()
        self.depth = depth
        self.residual = residual

        # Encoder with depthwise separable convolutions
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        channels = in_channels
        for i in range(depth):
            out_ch = base_filters * (2 ** i)
            self.encoders.append(DepthwiseSeparableConvBlock(channels, out_ch))
            self.pools.append(nn.MaxPool2d(2))
            channels = out_ch

        # Bottleneck
        bottleneck_ch = base_filters * (2 ** depth)
        self.bottleneck = DepthwiseSeparableConvBlock(channels, bottleneck_ch)

        # Decoder
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        channels = bottleneck_ch
        for i in range(depth - 1, -1, -1):
            out_ch = base_filters * (2 ** i)
            self.upconvs.append(
                nn.ConvTranspose2d(channels, out_ch, kernel_size=2, stride=2)
            )
            self.decoders.append(DepthwiseSeparableConvBlock(out_ch * 2, out_ch))
            channels = out_ch

        self.output_conv = nn.Conv2d(base_filters, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        skips = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for upconv, decoder, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = upconv(x)
            if x.shape != skip.shape:
                x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = decoder(x)

        x = self.output_conv(x)

        if self.residual:
            x = identity - x

        return torch.clamp(x, 0.0, 1.0)


class DepthwiseSeparableConvBlock(nn.Module):
    """Depthwise separable convolution block (MobileNet-style)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            # Depthwise
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            # Pointwise
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            # Second depthwise-separable
            nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)
