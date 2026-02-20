"""HRNet baseline for holographic reconstruction.

This module provides the HRNet architecture as described by Ren et al.
(End-to-end deep learning framework for digital holographic reconstruction, 2019).
It utilizes residual blocks, max pooling for downsampling, and subpixel convolution
(PixelShuffle) for upsampling, followed by 1x1 fine-tuning convolutions.
"""

import torch
from torch import nn

from holopaswin.propagator import AngularSpectrumPropagator


class ResBlock(nn.Module):
    """Standard Residual Block with two 3x3 convolutions, BN, and ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """Initialize the residual block."""
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip: nn.Module = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False), nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the residual block."""
        identity: torch.Tensor = self.skip(x)
        out: torch.Tensor = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class HRNet(nn.Module):
    """HRNet for digital holographic reconstruction (Ren et al., 2019).

    Architecture features:
    - Input block: 3x3 Conv, BN, ReLU, 32 channels
    - Feature extraction: Residual blocks with 3 Max Pooling layers (downsample by 8)
    - Reconstruction: Subpixel convolution (upsample by 8)
    - Final tuning: Three 1x1 convolutions
    """

    def __init__(  # noqa: PLR0913
        self,
        img_size: int,
        wavelength: float,
        pixel_size: float,
        z_distance: float,
        num_res_blocks: int = 2,
        residual_mode: bool = True,
    ) -> None:
        """Initialize HRNet baseline.

        Args:
            img_size: Spatial size of input images (assumed square).
            wavelength: Wavelength of light in meters.
            pixel_size: Pixel pitch in meters.
            z_distance: Propagation distance in meters.
            num_res_blocks: Number of residual blocks per scale.
            residual_mode: If True, predict correction; if False, predict clean directly.
        """
        super().__init__()
        self.img_size = img_size
        self.residual_mode = residual_mode

        # Physics propagator
        self.propagator = AngularSpectrumPropagator(
            (img_size, img_size),
            wavelength,
            pixel_size,
            z_distance,
        )

        channels = [32, 64, 128]

        # First unit
        self.input_block = nn.Sequential(
            nn.Conv2d(2, channels[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )

        # Stage 1
        self.stage1 = self._make_res_layer(channels[0], channels[0], num_res_blocks)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 2
        self.stage2 = self._make_res_layer(channels[0], channels[1], num_res_blocks)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 3
        self.stage3 = self._make_res_layer(channels[1], channels[2], num_res_blocks)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Subpixel Convolution upscaling by factor 8
        # upscale factor = 8 -> outputs channels[0] = 32
        upscale_factor = 8
        conv_out_channels = channels[0] * (upscale_factor**2)
        self.subpixel_conv = nn.Sequential(
            nn.Conv2d(channels[2], conv_out_channels, kernel_size=3, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.ReLU(inplace=True),
        )

        # Three 1x1 convolutions for fine tuning
        self.fine_tune = nn.Sequential(
            nn.Conv2d(channels[0], 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1),
        )

    def _make_res_layer(self, in_channels: int, out_channels: int, blocks: int) -> nn.Sequential:
        """Create a sequence of residual blocks."""
        layers = [ResBlock(in_channels, out_channels)]
        layers.extend([ResBlock(out_channels, out_channels) for _ in range(1, blocks)])
        return nn.Sequential(*layers)

    def forward(self, hologram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through HRNet.

        Args:
            hologram: Input hologram intensity tensor of shape (B, 1, H, W).

        Returns:
            Tuple of:
                - clean_object: Cleaned complex object as (B, 2, H, W) [Real, Imag]
                - dirty_field: ASM back-propagation (B, 2, H, W) for loss computation
        """
        complex_input = torch.sqrt(hologram + 1e-8)
        complex_holo = torch.complex(complex_input, torch.zeros_like(complex_input))
        dirty_complex = self.propagator(complex_holo, backward=True)
        dirty_2ch = torch.cat([dirty_complex.real, dirty_complex.imag], dim=1)

        # Feature extraction
        x = self.input_block(dirty_2ch)

        x = self.stage1(x)
        x = self.pool1(x)

        x = self.stage2(x)
        x = self.pool2(x)

        x = self.stage3(x)
        x = self.pool3(x)

        # Reconstruction block
        x = self.subpixel_conv(x)
        output = self.fine_tune(x)

        # Output
        clean_2ch = dirty_2ch + output if self.residual_mode else output

        return clean_2ch, dirty_2ch
