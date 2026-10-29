"""Main model module for holographic reconstruction using physics-aware Swin Transformer.

This module provides the HoloPASWIN model which combines physics-based propagation
with deep learning-based refinement to remove twin image artifacts from holographic
reconstructions.
"""

from typing import Any

import torch

from holopaswin.propagator import AngularSpectrumPropagator
from holopaswin.swin_transformer import SwinTransformerSys

# How the back-propagated field is added around the Swin U-Net. Version 2 adds it
# once, outside the Swin U-Net, in residual mode only.
WIRING_VERSION: int = 2


def _check_wiring(  # noqa: PLR0913
    module: torch.nn.Module,
    state_dict: dict[str, Any],
    prefix: str,
    local_metadata: dict[str, Any],
    strict: bool,
    missing_keys: list[str],
    unexpected_keys: list[str],
    error_msgs: list[str],
) -> None:
    """Refuse weights of another skip wiring before load_state_dict copies them.

    The skips have no weights, so weights of another wiring would load without
    error and reconstruct the wrong field. Only the stored wiring version tells
    them apart, so it must be present and equal to WIRING_VERSION, whatever the
    strictness of the load.

    Args:
        module (torch.nn.Module): The HoloPASWIN model being loaded.
        state_dict (dict[str, Any]): Weights being loaded.
        prefix (str): Prefix of the model's keys in state_dict.
        local_metadata (dict[str, Any]): Metadata of the model's weights (unused).
        strict (bool): Strictness of the load (unused, the check always applies).
        missing_keys (list[str]): Missing keys collected by torch (unused).
        unexpected_keys (list[str]): Unexpected keys collected by torch (unused).
        error_msgs (list[str]): Errors collected by torch (unused).

    Raises:
        ValueError: If the wiring version is missing or not WIRING_VERSION.
    """
    del module, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    stored: torch.Tensor | None = state_dict.get(prefix + "wiring_version")
    if stored is None or int(stored) != WIRING_VERSION:
        found: str = "no wiring version" if stored is None else f"wiring version {int(stored)}"
        msg: str = (
            f"These HoloPASWIN weights have {found}, but HoloPASWIN expects skip wiring {WIRING_VERSION}: "
            "the back-propagated field is added once, in residual mode only. The weights would "
            "reconstruct the wrong field, so retrain the model."
        )
        raise ValueError(msg)


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

    def __init__(  # noqa: PLR0913
        self,
        img_size: int,
        wavelength: float,
        pixel_size: float,
        z_dist: float,
        use_pretrained: bool = True,
        residual_mode: bool = True,
        numerical_aperture: float | None = None,
    ) -> None:
        """Initialize the HoloPASWIN model.

        Args:
            img_size: Spatial size of input images (assumed square, e.g., 224).
            wavelength: Wavelength of the light used for holography in meters (e.g., 532e-9).
            pixel_size: Physical size of each pixel in meters (e.g., 4.65e-6).
            z_dist: Propagation distance in meters (e.g., 0.02).
            use_pretrained: Whether to use ImageNet pretrained weights for Swin Transformer.
            residual_mode: If True, predict correction; if False, predict clean field directly.
            numerical_aperture: NA of the imaging objective for a magnified setup (e.g.,
                0.45), applied in forward propagation by the physics loss. None for a
                lensless setup.
        """
        super().__init__()
        self.residual_mode = residual_mode
        # Saved with the weights and checked on every load by _check_wiring, because the
        # skips around the Swin U-Net have no weights that would fail to load.
        self.register_buffer("wiring_version", torch.tensor(WIRING_VERSION))
        self.register_load_state_dict_pre_hook(_check_wiring)  # type: ignore[no-untyped-call]
        self.propagator = AngularSpectrumPropagator(
            (img_size, img_size), wavelength, pixel_size, z_dist, numerical_aperture=numerical_aperture
        )

        # Input to Swin is the Real and Imag parts of the back-propagated field (2 channels)
        # Output is Clean Real and Imag parts (2 channels)
        self.swin_unet = SwinTransformerSys(
            img_size=img_size,
            in_chans=2,
            out_chans=2,
            use_pretrained=use_pretrained,
        )

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
        correction = self.swin_unet(dirty_input)

        # Apply residual connection if enabled
        clean_2ch = dirty_input + correction if self.residual_mode else correction

        # Return 2ch clean object (Real, Imag) and dirty input
        return clean_2ch, dirty_input
