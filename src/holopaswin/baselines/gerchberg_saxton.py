"""Gerchberg-Saxton iterative phase retrieval algorithm for inline holography.

This module implements the classical GS algorithm for recovering phase from a single
intensity hologram. It alternates between object and sensor planes, applying constraints
at each plane to iteratively refine the phase estimate.
"""

import torch
from torch import nn

from holopaswin.propagator import AngularSpectrumPropagator


class GerchbergSaxton(nn.Module):
    """Gerchberg-Saxton iterative phase retrieval for inline holography.

    This is a classical baseline that recovers phase from intensity measurements
    by alternating between object and sensor planes with constraint application.

    The algorithm:
    1. Initialize at sensor: sqrt(H) with zero phase
    2. Back-propagate to object plane (ASM inverse)
    3. Apply object constraint (support mask from amplitude threshold)
    4. Forward-propagate to sensor plane (ASM forward)
    5. Replace amplitude with measured sqrt(H), keep estimated phase
    6. Repeat for N iterations

    Note: This is NOT a neural network - it's a deterministic iterative algorithm.
    We wrap it as nn.Module for API consistency with other models.
    """

    def __init__(
        self,
        img_size: int,
        wavelength: float,
        pixel_size: float,
        z_distance: float,
        iterations: int = 100,
        support_threshold: float = 0.5,
    ) -> None:
        """Initialize the GS algorithm.

        Args:
            img_size: Spatial size of input images (assumed square).
            wavelength: Wavelength of light in meters (e.g., 532e-9).
            pixel_size: Pixel pitch in meters (e.g., 4.65e-6).
            z_distance: Propagation distance in meters (e.g., 0.02).
            iterations: Number of GS iterations to run.
            support_threshold: Threshold for support constraint (0-1).
                Pixels with amplitude < threshold * max are constrained.
        """
        super().__init__()
        self.iterations = iterations
        self.support_threshold = support_threshold

        # Physics propagator (same as HoloPASWIN for fairness)
        self.propagator = AngularSpectrumPropagator(
            (img_size, img_size),
            wavelength,
            pixel_size,
            z_distance,
        )

    def forward(self, hologram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run GS algorithm on input hologram.

        Args:
            hologram: Input hologram intensity tensor of shape (B, 1, H, W).

        Returns:
            Tuple of:
                - recovered_object: Recovered complex object as (B, 2, H, W) [Real, Imag]
                - dirty_field: Initial back-propagation (for API compatibility)
        """
        # Ensure we're not computing gradients (iterative, not differentiable)
        with torch.no_grad():
            batch_size = hologram.shape[0]
            device = hologram.device

            # Initialize: sqrt(hologram) with zero phase at sensor plane
            # Add small epsilon to avoid sqrt(0) issues
            amplitude_sensor = torch.sqrt(hologram + 1e-8)  # (B, 1, H, W)
            phase_sensor = torch.zeros_like(amplitude_sensor)

            # Initial complex field at sensor plane
            field_sensor = amplitude_sensor * torch.exp(1j * phase_sensor)

            # Back-propagate to get initial dirty estimate (for return value)
            dirty_field_complex = self.propagator(field_sensor, backward=True)
            dirty_2ch = torch.cat([dirty_field_complex.real, dirty_field_complex.imag], dim=1)

            # Build support mask from initial back-propagation
            # This helps constrain the object region
            initial_object = self.propagator(field_sensor, backward=True)
            initial_amp = torch.abs(initial_object)
            # Normalize per batch
            max_amp = initial_amp.view(batch_size, -1).max(dim=1, keepdim=True)[0]
            max_amp = max_amp.view(batch_size, 1, 1, 1) + 1e-8
            support_mask = (initial_amp / max_amp) > self.support_threshold

            # GS iterations
            current_field = field_sensor.clone()

            for _ in range(self.iterations):
                # 1. Back-propagate to object plane
                object_field = self.propagator(current_field, backward=True)

                # 2. Apply object constraint (support mask)
                # Inside support: keep field
                # Outside support: reduce amplitude (soft constraint)
                object_amp = torch.abs(object_field)
                object_phase = torch.angle(object_field)

                # Soft support: multiply amplitude by mask (0 outside, 1 inside)
                # This is a gentler constraint than hard masking
                constrained_amp = object_amp * support_mask.float() + \
                                  object_amp * (1 - support_mask.float()) * 0.1

                # Reconstruct complex field
                object_field = constrained_amp * torch.exp(1j * object_phase)

                # 3. Forward-propagate to sensor plane
                sensor_field = self.propagator(object_field, backward=False)

                # 4. Apply sensor constraint: replace amplitude with measured sqrt(H)
                sensor_phase = torch.angle(sensor_field)
                current_field = amplitude_sensor * torch.exp(1j * sensor_phase)

            # Final back-propagation to get recovered object
            recovered_object = self.propagator(current_field, backward=True)

            # Convert to 2-channel format (Real, Imag) for API consistency
            recovered_2ch = torch.cat(
                [recovered_object.real, recovered_object.imag], dim=1
            )

            return recovered_2ch, dirty_2ch

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"GerchbergSaxton(iterations={self.iterations}, "
            f"support_threshold={self.support_threshold})"
        )
