"""Standard U-Net baseline for holographic reconstruction.

This module provides a simple encoder-decoder U-Net architecture following
the style of Rivenson et al. for comparison against the Swin Transformer.
"""

import torch
from torch import nn

from holopaswin.propagator import AngularSpectrumPropagator


class UNetBaseline(nn.Module):
    """Standard U-Net for holographic reconstruction.

    A 4-scale encoder-decoder architecture with skip connections.
    - Encoder: Conv-BN-ReLU blocks with MaxPool downsampling
    - Decoder: ConvTranspose-BN-ReLU blocks with concatenation skips

    This is simpler than ResNet-U-Net (no pretrained encoder) to provide
    a fair baseline that doesn't rely on ImageNet pretraining.
    """

    def __init__(
        self,
        img_size: int,
        wavelength: float,
        pixel_size: float,
        z_distance: float,
        base_channels: int = 64,
        residual_mode: bool = True,
    ) -> None:
        """Initialize U-Net baseline.

        Args:
            img_size: Spatial size of input images (assumed square).
            wavelength: Wavelength of light in meters (e.g., 532e-9).
            pixel_size: Pixel pitch in meters (e.g., 4.65e-6).
            z_distance: Propagation distance in meters (e.g., 0.02).
            base_channels: Number of channels in first encoder layer.
            residual_mode: If True, predict correction; if False, predict clean directly.
        """
        super().__init__()
        self.img_size = img_size
        self.residual_mode = residual_mode

        # Physics propagator (same configuration as HoloPASWIN)
        self.propagator = AngularSpectrumPropagator(
            (img_size, img_size),
            wavelength,
            pixel_size,
            z_distance,
        )

        # Channel progression: 64 -> 128 -> 256 -> 512
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        # Encoder
        self.enc1 = self._encoder_block(2, c1)      # 224 -> 224, 64ch
        self.pool1 = nn.MaxPool2d(2, 2)             # 224 -> 112
        self.enc2 = self._encoder_block(c1, c2)     # 112 -> 112, 128ch
        self.pool2 = nn.MaxPool2d(2, 2)             # 112 -> 56
        self.enc3 = self._encoder_block(c2, c3)     # 56 -> 56, 256ch
        self.pool3 = nn.MaxPool2d(2, 2)             # 56 -> 28
        self.enc4 = self._encoder_block(c3, c4)     # 28 -> 28, 512ch
        self.pool4 = nn.MaxPool2d(2, 2)             # 28 -> 14

        # Bottleneck
        self.bottleneck = self._encoder_block(c4, c4 * 2)  # 14 -> 14, 1024ch

        # Decoder
        self.up4 = self._decoder_block(c4 * 2, c4)   # 14 -> 28, 512ch
        self.dec4 = self._encoder_block(c4 * 2, c4)  # Skip + up = 1024 -> 512

        self.up3 = self._decoder_block(c4, c3)       # 28 -> 56, 256ch
        self.dec3 = self._encoder_block(c3 * 2, c3)  # Skip + up = 512 -> 256

        self.up2 = self._decoder_block(c3, c2)       # 56 -> 112, 128ch
        self.dec2 = self._encoder_block(c2 * 2, c2)  # Skip + up = 256 -> 128

        self.up1 = self._decoder_block(c2, c1)       # 112 -> 224, 64ch
        self.dec1 = self._encoder_block(c1 * 2, c1)  # Skip + up = 128 -> 64

        # Final output layer
        self.final = nn.Conv2d(c1, 2, kernel_size=1)  # 64 -> 2 (Real, Imag)

    def _encoder_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Create encoder block: Conv-BN-ReLU x2."""
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _decoder_block(self, in_ch: int, out_ch: int) -> nn.Sequential:
        """Create decoder upsampling block."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, hologram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through U-Net.

        Args:
            hologram: Input hologram intensity tensor of shape (B, 1, H, W).

        Returns:
            Tuple of:
                - clean_object: Cleaned complex object as (B, 2, H, W) [Real, Imag]
                - dirty_field: ASM back-propagation (B, 2, H, W) for loss computation
        """
        # Physics step: ASM back-propagation
        complex_input = torch.sqrt(hologram + 1e-8)
        complex_holo = torch.complex(complex_input, torch.zeros_like(complex_input))
        dirty_complex = self.propagator(complex_holo, backward=True)
        dirty_2ch = torch.cat([dirty_complex.real, dirty_complex.imag], dim=1)

        # Encoder path
        e1 = self.enc1(dirty_2ch)       # (B, 64, 224, 224)
        e2 = self.enc2(self.pool1(e1))  # (B, 128, 112, 112)
        e3 = self.enc3(self.pool2(e2))  # (B, 256, 56, 56)
        e4 = self.enc4(self.pool3(e3))  # (B, 512, 28, 28)

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))  # (B, 1024, 14, 14)

        # Decoder path with skip connections
        d4 = self.up4(b)                    # (B, 512, 28, 28)
        d4 = torch.cat([d4, e4], dim=1)     # (B, 1024, 28, 28)
        d4 = self.dec4(d4)                  # (B, 512, 28, 28)

        d3 = self.up3(d4)                   # (B, 256, 56, 56)
        d3 = torch.cat([d3, e3], dim=1)     # (B, 512, 56, 56)
        d3 = self.dec3(d3)                  # (B, 256, 56, 56)

        d2 = self.up2(d3)                   # (B, 128, 112, 112)
        d2 = torch.cat([d2, e2], dim=1)     # (B, 256, 112, 112)
        d2 = self.dec2(d2)                  # (B, 128, 112, 112)

        d1 = self.up1(d2)                   # (B, 64, 224, 224)
        d1 = torch.cat([d1, e1], dim=1)     # (B, 128, 224, 224)
        d1 = self.dec1(d1)                  # (B, 64, 224, 224)

        # Final output
        output = self.final(d1)             # (B, 2, 224, 224)

        # Apply residual connection if enabled
        clean_2ch = dirty_2ch + output if self.residual_mode else output

        return clean_2ch, dirty_2ch
