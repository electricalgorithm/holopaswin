"""ResNet-18 U-Net baseline for architecture ablation.

This provides a CNN baseline to compare against the Swin Transformer architecture.
"""

import torch
from torch import nn
from torchvision import models

from holopaswin.propagator import AngularSpectrumPropagator


class ResNetUNet(nn.Module):
    """U-Net with ResNet-18 encoder for holographic reconstruction."""

    def __init__(
        self,
        img_size: int,
        wavelength: float,
        pixel_size: float,
        z_distance: float,
        residual_mode: bool = True,
    ) -> None:
        """Initialize ResNet U-Net.

        Args:
            img_size: Input image size (assumes square)
            wavelength: Laser wavelength in meters
            pixel_size: Pixel pitch in meters
            z_distance: Propagation distance in meters
            residual_mode: If True, predict correction; if False, predict clean field directly
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

        # ResNet-18 encoder (pretrained on ImageNet)
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Modify first conv to accept 2 channels (real, imag)
        self.encoder_conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # Copy pretrained weights for first 2 channels (average RGB weights)
        with torch.no_grad():
            self.encoder_conv1.weight[:, :2, :, :] = resnet.conv1.weight[:, :2, :, :]

        self.encoder_bn1 = resnet.bn1
        self.encoder_relu = resnet.relu
        self.encoder_maxpool = resnet.maxpool

        # ResNet stages
        self.encoder_layer1 = resnet.layer1  # 64 channels, /4
        self.encoder_layer2 = resnet.layer2  # 128 channels, /8
        self.encoder_layer3 = resnet.layer3  # 256 channels, /16
        self.encoder_layer4 = resnet.layer4  # 512 channels, /32

        # Decoder with transposed convolutions
        self.decoder_up1 = self._make_decoder_block(512, 256)  # /32 -> /16
        self.decoder_up2 = self._make_decoder_block(256 + 256, 128)  # /16 -> /8 (with skip)
        self.decoder_up3 = self._make_decoder_block(128 + 128, 64)  # /8 -> /4 (with skip)
        self.decoder_up4 = self._make_decoder_block(64 + 64, 64)  # /4 -> /2 (with skip)

        # Final upsampling to original resolution
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 2, kernel_size=3, padding=1),
        )

    def _make_decoder_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Create a decoder upsampling block."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, hologram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            hologram: Input hologram intensity (B, 1, H, W)

        Returns:
            Tuple of (clean_complex, dirty_complex) both as (B, 2, H, W)
        """
        # ASM back-propagation
        dirty_complex = self.propagator(hologram, backward=True)
        dirty_2ch = torch.cat([dirty_complex.real, dirty_complex.imag], dim=1)

        # Encoder
        x = self.encoder_conv1(dirty_2ch)
        x = self.encoder_bn1(x)
        x = self.encoder_relu(x)
        x = self.encoder_maxpool(x)

        e1 = self.encoder_layer1(x)  # /4, 64
        e2 = self.encoder_layer2(e1)  # /8, 128
        e3 = self.encoder_layer3(e2)  # /16, 256
        e4 = self.encoder_layer4(e3)  # /32, 512

        # Decoder with skip connections
        d1 = self.decoder_up1(e4)  # /16, 256
        d1 = torch.cat([d1, e3], dim=1)  # /16, 512

        d2 = self.decoder_up2(d1)  # /8, 128
        d2 = torch.cat([d2, e2], dim=1)  # /8, 256

        d3 = self.decoder_up3(d2)  # /4, 64
        d3 = torch.cat([d3, e1], dim=1)  # /4, 128

        d4 = self.decoder_up4(d3)  # /2, 64

        correction = self.final_up(d4)  # Original size, 2 channels

        # Residual or direct
        clean_2ch = dirty_2ch + correction if self.residual_mode else correction

        return clean_2ch, dirty_2ch
