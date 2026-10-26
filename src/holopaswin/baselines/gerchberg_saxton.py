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
    3. Apply the object constraint (see _constrain_object)
    4. Forward-propagate to sensor plane (ASM forward)
    5. Replace amplitude with measured sqrt(H), keep estimated phase
    6. Repeat for N iterations

    The object constraint is what makes the iterations do anything. Without a
    pupil, ASM is unitary, so sqrt(H) with zero phase is a fixed point of steps
    2-5 when the constraint leaves the field unchanged, and every iteration then
    returns the plain back-propagation. A support mask from thresholding the
    back-propagated amplitude does exactly that for transmission objects on a
    bright background: the amplitude stays near 1 everywhere, so every pixel
    falls inside the support.

    Note: This is NOT a neural network - it's a deterministic iterative algorithm.
    We wrap it as nn.Module for API consistency with other models.
    """

    def __init__(  # noqa: PLR0913
        self,
        img_size: int,
        wavelength: float,
        pixel_size: float,
        z_distance: float,
        iterations: int = 100,
        numerical_aperture: float | None = None,
    ) -> None:
        """Initialize the GS algorithm.

        Args:
            img_size: Spatial size of input images (assumed square).
            wavelength: Wavelength of light in meters (e.g., 532e-9).
            pixel_size: Pixel pitch in meters (e.g., 4.65e-6).
            z_distance: Propagation distance in meters (e.g., 0.02).
            iterations: Number of GS iterations to run.
            numerical_aperture: NA of the imaging objective of a magnified setup, applied in
                forward propagation, or None for a lensless setup without a pupil.
        """
        super().__init__()
        self.iterations = iterations

        # Physics propagator (same as HoloPASWIN for fairness)
        self.propagator = AngularSpectrumPropagator(
            (img_size, img_size),
            wavelength,
            pixel_size,
            z_distance,
            numerical_aperture=numerical_aperture,
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
            # Initialize: sqrt(hologram) with zero phase at sensor plane
            # Add small epsilon to avoid sqrt(0) issues
            amplitude_sensor: torch.Tensor = torch.sqrt(hologram + 1e-8)  # (B, 1, H, W)
            field_sensor: torch.Tensor = torch.complex(amplitude_sensor, torch.zeros_like(amplitude_sensor))

            # Back-propagate to get initial dirty estimate (for return value)
            dirty_field_complex: torch.Tensor = self.propagator(field_sensor, backward=True)
            dirty_2ch: torch.Tensor = torch.cat([dirty_field_complex.real, dirty_field_complex.imag], dim=1)

            # GS iterations
            current_field: torch.Tensor = field_sensor
            object_field: torch.Tensor
            sensor_field: torch.Tensor
            for _ in range(self.iterations):
                # 1-2. Back-propagate to object plane and apply the object constraint
                object_field = self._constrain_object(self.propagator(current_field, backward=True))

                # 3. Forward-propagate to sensor plane
                sensor_field = self.propagator(object_field, backward=False)

                # 4. Apply sensor constraint: replace amplitude with measured sqrt(H)
                current_field = amplitude_sensor * torch.exp(1j * torch.angle(sensor_field))

            # Final back-propagation, constrained like every iterate
            recovered_object: torch.Tensor = self._constrain_object(self.propagator(current_field, backward=True))

            # Convert to 2-channel format (Real, Imag) for API consistency
            recovered_2ch: torch.Tensor = torch.cat([recovered_object.real, recovered_object.imag], dim=1)

            return recovered_2ch, dirty_2ch

    @staticmethod
    def _constrain_object(object_field: torch.Tensor) -> torch.Tensor:
        """Apply the object-plane constraint to a back-propagated field.

        The field is a transmission function normalised so that the empty
        background is 1. It may carry a constant phase from the propagation
        over z, so a constraint must not depend on the absolute phase.

        A passive sample cannot amplify light, so the transmission amplitude is
        at most 1. The twin image adds fringes of both signs around each object,
        and clipping the bright half of them removes part of the twin at every
        iteration. The phase is kept as it is: a constraint on its sign would
        depend on the constant propagation phase, and objects whose phase wraps
        past pi break it.

        Args:
            object_field: Complex object field of shape (B, 1, H, W).

        Returns:
            The constrained complex field, same shape, with amplitude at most 1.
        """
        amplitude: torch.Tensor = object_field.abs()
        return torch.where(amplitude > 1, object_field / amplitude, object_field)

    def __repr__(self) -> str:
        """String representation."""  # noqa: D401
        return f"GerchbergSaxton(iterations={self.iterations})"
