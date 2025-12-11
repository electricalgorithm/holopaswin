"""Main model module for holographic reconstruction using physics-aware Swin Transformer.

This module provides the HoloPASWIN model which combines physics-based propagation
with deep learning-based refinement to remove twin image artifacts from holographic
reconstructions.
"""

import torch

from holopaswin.propagator import AngularSpectrumPropagator
from holopaswin.swin_transformer import SwinTransformerSys


class HoloPASWIN(torch.nn.Module):
    """Physics-Aware Swin Transformer for holographic reconstruction.

    The complete end-to-end model that combines:
    1. Physics-based back-propagation using Angular Spectrum Method (ASM)
    2. Deep learning-based refinement using Swin Transformer U-Net

    The model takes a hologram as input, back-propagates it to the object plane
    (which contains twin image artifacts), and then uses a Swin Transformer to
    clean the reconstruction and remove the twin image.

    Input: Hologram intensity (1 channel)
    Internal: ASM Backprop -> Dirty Image (Twin included, Complex)
    Refinement: SwinUNet cleans the dirty image (Real/Imag -> Real/Imag)
    Output: Clean Object Complex
    """

    def __init__(self, img_size: int, wavelength: float, pixel_size: float, z_dist: float) -> None:
        """Initialize the HoloPASWIN model.

        Args:
            img_size: Spatial size of input images (assumed square, e.g., 224).
            wavelength: Wavelength of the light used for holography in meters (e.g., 532e-9).
            pixel_size: Physical size of each pixel in meters (e.g., 4.65e-6).
            z_dist: Propagation distance in meters (e.g., 0.02).
        """
        super().__init__()
        self.propagator = AngularSpectrumPropagator((img_size, img_size), wavelength, pixel_size, z_dist)

        # Input to Swin is the Real and Imag parts of the back-propagated field (2 channels)
        # Output is Clean Real and Imag parts (2 channels)
        self.swin_unet = SwinTransformerSys(img_size=img_size, in_chans=2, out_chans=2)

    def forward(self, hologram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model.

        Args:
            hologram: Input hologram intensity tensor of shape (B, 1, H, W).

        Returns:
            A tuple containing:
                - clean_object: Cleaned object Complex tensor of shape (B, 2, H, W).
                - dirty_field: Dirty reconstruction Complex tensor of shape (B, 2, H, W).
        """
        # 1. Physics Step: Back-propagate Hologram to Object Plane
        # Input hologram is intensity (real). Treat as amplitude with flat phase 0.
        # Standard Digital Holography: Field = sqrt(Hologram)
        # 1e-8 is an epsilon value to handle hologram = 0 cases.
        complex_hologram = torch.complex(torch.sqrt(hologram + 1e-8), torch.zeros_like(hologram))

        # Propagate backwards (-z)
        reconstructed_field = self.propagator(complex_hologram, backward=True)

        # 2. Deep Learning Step: Remove Twin Image
        # Prepare input for Swin: Stack Real and Imag (B, 2, H, W)
        dirty_real = reconstructed_field.real
        dirty_imag = reconstructed_field.imag
        dirty_input = torch.cat([dirty_real, dirty_imag], dim=1)  # (B, 2, H, W)

        # Swin UNet Refinement
        clean_2ch = self.swin_unet(dirty_input)

        # Return 2ch clean object (Real, Imag) and dirty input
        return clean_2ch, dirty_input
