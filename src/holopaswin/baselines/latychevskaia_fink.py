"""Iterative twin-image removal after Latychevskaia and Fink for inline holography.

Latychevskaia and Fink (Phys. Rev. Lett. 98, 233901 (2007)) remove the twin image
by alternating between the sensor and object planes, as Gerchberg-Saxton does,
with physical constraints on the transmission function of the object. Their
later reconstruction routine (Appl. Opt. 54, 3925 (2015)) adds a support and a
smoothing step; this module follows that routine.
"""

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

from holopaswin.propagator import AngularSpectrumPropagator

# Smoothing kernel of the 2015 routine (their Eq. 6): ones with 4 in the centre.
SMOOTHING_KERNEL: list[list[float]] = [[1.0] * 5, [1.0] * 5, [1.0, 1.0, 4.0, 1.0, 1.0], [1.0] * 5, [1.0] * 5]

# Scale from the median absolute deviation to the standard deviation of normal noise.
MAD_TO_SIGMA: float = 1.4826


class LatychevskaiaFink(nn.Module):
    """Iterative reconstruction with physical constraints on the object.

    The object is described by its transmission function t = (1 - a) exp(-i phi),
    where a is the absorption and phi the phase shift, so t = 1 where there is no
    object. Each iteration:
    1. Back-propagates the sensor field and divides it by the incident wave,
       which gives t.
    2. Sets t to 1 outside a support found in the first reconstruction.
    3. Replaces negative absorption by zero, so |t| is at most 1, keeping phi.
    4. Smooths |t| with the 5 x 5 kernel of the 2015 routine, keeping phi.
    5. Forward-propagates t times the incident wave and replaces the amplitude
       with the measured sqrt(H), keeping the new phase.

    Negative absorption is where the twin image interferes with the incident
    wave and makes the field brighter than the light that reached the object;
    removing it removes part of the twin at every iteration. The support removes
    the twin's fringes away from the objects, and the smoothing keeps noise from
    building up over the iterations.

    Note: This is NOT a neural network - it's a deterministic iterative algorithm.
    We wrap it as nn.Module for API consistency with other models.
    """

    smoothing_kernel: torch.Tensor

    def __init__(  # noqa: PLR0913
        self,
        img_size: int,
        wavelength: float,
        pixel_size: float,
        z_distance: float,
        iterations: int = 100,
        numerical_aperture: float | None = None,
        support_threshold: float = 3.0,
        support_margin: int = 4,
        support_blur: float = 2.0,
    ) -> None:
        """Initialize the algorithm.

        Args:
            img_size: Spatial size of input images (assumed square).
            wavelength: Wavelength of light in meters (e.g., 532e-9).
            pixel_size: Pixel pitch in meters (e.g., 4.65e-6).
            z_distance: Propagation distance in meters (e.g., 0.02).
            iterations: Number of iterations to run.
            numerical_aperture: NA of the imaging objective of a magnified setup, applied in
                forward propagation, or None for a lensless setup without a pupil.
            support_threshold: A pixel belongs to the support when its smoothed
                deviation from the empty background exceeds the median deviation
                by this many robust standard deviations.
            support_margin: Pixels added around the support on every side; the
                2015 routine makes the support 7 pixels wider than the object.
            support_blur: Standard deviation in pixels of the Gaussian that
                softens the edge of the support, 2 pixels in the 2015 routine.
        """
        super().__init__()
        self.iterations = iterations
        self.support_threshold = support_threshold
        self.support_margin = support_margin
        self.support_blur = support_blur

        # Physics propagator (same as HoloPASWIN for fairness)
        self.propagator = AngularSpectrumPropagator(
            (img_size, img_size),
            wavelength,
            pixel_size,
            z_distance,
            numerical_aperture=numerical_aperture,
        )
        kernel: torch.Tensor = torch.tensor(SMOOTHING_KERNEL)
        self.register_buffer("smoothing_kernel", (kernel / kernel.sum())[None, None], persistent=False)

    def forward(self, hologram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the reconstruction on input holograms.

        Args:
            hologram: Normalised hologram intensity tensor of shape (B, 1, H, W).

        Returns:
            Tuple of:
                - recovered_object: Recovered complex object as (B, 2, H, W) [Real, Imag]
                - dirty_field: Initial back-propagation (for API compatibility)
        """
        with torch.no_grad():
            amplitude_sensor: torch.Tensor = torch.sqrt(hologram + 1e-8)
            field_sensor: torch.Tensor = torch.complex(amplitude_sensor, torch.zeros_like(amplitude_sensor))

            # Back-propagating the unscattered light over z multiplies it by
            # exp(-i k z), the zero-frequency value of the conjugated transfer
            # function. Dividing by it turns the field into the transmission t.
            incident: torch.Tensor = torch.conj(self.propagator.H[0, 0]).to(hologram.device)

            dirty_field_complex: torch.Tensor = self.propagator(field_sensor, backward=True)
            dirty_2ch: torch.Tensor = torch.cat([dirty_field_complex.real, dirty_field_complex.imag], dim=1)
            support: torch.Tensor = self.find_support(dirty_field_complex / incident)

            current_field: torch.Tensor = field_sensor
            transmission: torch.Tensor
            sensor_field: torch.Tensor
            for _ in range(self.iterations):
                transmission = self._constrain_object(self.propagator(current_field, backward=True) / incident, support)
                sensor_field = self.propagator(transmission * incident, backward=False)
                current_field = amplitude_sensor * torch.exp(1j * torch.angle(sensor_field))

            # Final back-propagation, constrained like every iterate
            recovered_object: torch.Tensor = (
                self._constrain_object(self.propagator(current_field, backward=True) / incident, support) * incident
            )
            recovered_2ch: torch.Tensor = torch.cat([recovered_object.real, recovered_object.imag], dim=1)
            return recovered_2ch, dirty_2ch

    def _smooth(self, image: torch.Tensor) -> torch.Tensor:
        """Convolve a real image with the smoothing kernel, repeating its edge pixels.

        Args:
            image: Real tensor of shape (B, 1, H, W).

        Returns:
            The smoothed image, same shape.
        """
        padded: torch.Tensor = F.pad(image, (2, 2, 2, 2), mode="replicate")
        return F.conv2d(padded, self.smoothing_kernel.to(image.device, image.dtype))

    def find_support(self, transmission: torch.Tensor) -> torch.Tensor:
        """Find where the objects are in the first reconstruction.

        The deviation |t - 1| is large on the objects and small on the empty
        background, where only noise and the weaker twin fringes remain. It is
        smoothed, thresholded against its median and median absolute deviation,
        widened by the margin and given a soft Gaussian edge.

        Args:
            transmission: Transmission of the first reconstruction, (B, 1, H, W).

        Returns:
            Support weights in [0, 1] of shape (B, 1, H, W), 1 on the objects.
        """
        deviation: torch.Tensor = self._smooth((transmission - 1).abs())
        flat: torch.Tensor = deviation.flatten(1)
        median: torch.Tensor = flat.median(dim=1).values[:, None, None, None]
        spread: torch.Tensor = (flat - median.flatten(1)).abs().median(dim=1).values[:, None, None, None]
        mask: torch.Tensor = (deviation > median + self.support_threshold * MAD_TO_SIGMA * spread).to(deviation.dtype)

        width: int = 2 * self.support_margin + 1
        mask = F.max_pool2d(mask, width, stride=1, padding=self.support_margin)

        radius: int = max(1, round(3 * self.support_blur))
        offsets: torch.Tensor = torch.arange(-radius, radius + 1, device=mask.device, dtype=mask.dtype)
        gaussian: torch.Tensor = torch.exp(-(offsets**2) / (2 * self.support_blur**2))
        gaussian = gaussian / gaussian.sum()
        padded: torch.Tensor = F.pad(mask, (radius, radius, radius, radius), mode="replicate")
        blurred: torch.Tensor = F.conv2d(padded, gaussian.view(1, 1, 1, -1))
        blurred = F.conv2d(blurred, gaussian.view(1, 1, -1, 1))
        return torch.maximum(blurred, mask)

    def _constrain_object(self, transmission: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
        """Apply the support, non-negative absorption and smoothing to t.

        Args:
            transmission: Complex transmission function of shape (B, 1, H, W).
            support: Support weights in [0, 1], same shape.

        Returns:
            The constrained transmission function, same shape.
        """
        transmission = support * transmission + (1 - support)
        amplitude: torch.Tensor = self._smooth(transmission.abs().clamp(max=1.0))
        return amplitude * torch.exp(1j * torch.angle(transmission))

    def __repr__(self) -> str:
        """String representation."""  # noqa: D401
        return f"LatychevskaiaFink(iterations={self.iterations})"
