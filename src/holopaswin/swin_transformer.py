"""Swin Transformer U-Net module for image refinement.

This module provides a U-Net style architecture using Swin Transformer blocks
as the encoder. It uses a pretrained Swin Transformer from timm as the encoder
and a simple convolutional decoder with skip connections for upsampling.
"""

import timm
import torch
from torch import nn


class SwinTransformerSys(nn.Module):
    """A U-Net style architecture with Swin Transformer blocks.

    This architecture uses a pretrained Swin Transformer from timm as the encoder
    to extract multi-scale features, and a simple convolutional decoder with
    skip connections to upsample back to the original resolution.

    The encoder outputs features at 1/4, 1/8, 1/16, and 1/32 resolutions,
    which are progressively upsampled in the decoder with skip connections
    to preserve fine details.
    """

    def __init__(self, img_size: int = 224, in_chans: int = 1, out_chans: int = 1) -> None:
        """Initialize the Swin Transformer U-Net.

        Args:
            img_size: Spatial size of input images (assumed square, e.g., 224).
                     Defaults to 224.
            in_chans: Number of input channels. Defaults to 1.
            out_chans: Number of output channels. Defaults to 1.
        """
        super().__init__()

        # --- Encoder (Swin Transformer from timm) ---
        # We use swin_tiny_patch4_window7_224 as a lightweight base.
        # We remove the head to get features.
        # It could be better to use microsoft/swinv2-tiny-patch4-window16-256
        # later on, it's not base on timm, so skipping for now.
        self.encoder = timm.create_model(
            "swin_tiny_patch4_window7_224",
            pretrained=True,
            features_only=True,
            img_size=img_size,
            in_chans=in_chans,
        )

        # Get channel counts from the encoder stages
        # Typically: [96, 192, 384, 768] for Swin-Tiny
        encoder_channels = self.encoder.feature_info.channels()  # type: ignore[union-attr, operator]

        # --- Decoder ---
        # We need to upsample back to original size.
        # Swin Tiny outputs features at 1/4, 1/8, 1/16, 1/32 resolutions.

        self.decoder_up1 = self._up_block(encoder_channels[-1], encoder_channels[-2])
        self.decoder_up2 = self._up_block(encoder_channels[-2], encoder_channels[-3])
        self.decoder_up3 = self._up_block(encoder_channels[-3], encoder_channels[-4])

        # Final upsampling from 1/4 scale to 1/1 scale
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(encoder_channels[-4], encoder_channels[-4] // 2, kernel_size=4, stride=4),
            nn.BatchNorm2d(encoder_channels[-4] // 2),
            nn.GELU(),
            nn.Conv2d(encoder_channels[-4] // 2, out_chans, kernel_size=3, padding=1),
            nn.Sigmoid(),  # Assuming object is amplitude 0-1
        )

    def _up_block(self, in_c: int, out_c: int) -> nn.Sequential:
        """Build a decoder upsampling block.

        Creates a block that upsamples features using transposed convolution,
        followed by batch normalization, activation, and a convolution layer.

        Args:
            in_c: Number of input channels.
            out_c: Number of output channels.

        Returns:
            A Sequential module containing the upsampling block.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2),
            nn.BatchNorm2d(out_c),
            nn.GELU(),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Swin Transformer U-Net.

        Args:
            x: Input tensor of shape (B, in_chans, H, W).

        Returns:
            Output tensor of shape (B, out_chans, H, W) with values in [0, 1]
            (due to final Sigmoid activation).
        """
        # --- Encoder ---
        # timm features_only=True returns a list of features at different scales
        features = self.encoder(x)

        # Permute features from (B, H, W, C) to (B, C, H, W) for PyTorch Conv layers
        features_nchw = [f.permute(0, 3, 1, 2) for f in features]

        # features_nchw[0]: 1/4 scale (96 ch)
        # features_nchw[1]: 1/8 scale (192 ch)
        # features_nchw[2]: 1/16 scale (384 ch)
        # features_nchw[3]: 1/32 scale (768 ch)

        # --- Decoder ---
        # Upsample 1/32 -> 1/16 (add skip from features[2])
        d1 = self.decoder_up1(features_nchw[3])
        d1 = d1 + features_nchw[2]  # Simple additive skip connection

        # Upsample 1/16 -> 1/8 (add skip from features[1])
        d2 = self.decoder_up2(d1)
        d2 = d2 + features_nchw[1]

        # Upsample 1/8 -> 1/4 (add skip from features[0])
        d3 = self.decoder_up3(d2)
        d3 = d3 + features_nchw[0]

        # Final restoration to 1/1 resolution
        out = self.final_up(d3)
        return out  # type: ignore[no-any-return]
