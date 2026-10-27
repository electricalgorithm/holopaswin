"""Compressive phase retrieval with constrained complex total variation for inline holography.

Gao and Cao (Light Adv. Manuf. 4, 37-53 (2023)) combine the physical constraints of
iterative projection with a sparsity prior: the object field is sought as the
minimiser of an amplitude data term plus the complex total variation, subject to
non-negative absorption. This module follows their constrained complex total
variation (CCTV) model and its accelerated proximal gradient solver.
"""

import torch
from torch import nn

from holopaswin.propagator import AngularSpectrumPropagator


class ConstrainedComplexTV(nn.Module):
    """Constrained complex total variation reconstruction (CCTV).

    The object field x minimises

        1/2 || |A x| - sqrt(H) ||^2 + lam * || D x ||_1   subject to |x| <= 1,

    where A propagates the object to the sensor, D takes the differences between
    neighbouring pixels along both axes (the anisotropic complex total
    variation) and |x| <= 1 is non-negative absorption. Each iteration:
    1. Extrapolates from the last two estimates (Nesterov momentum).
    2. Takes a gradient step on the data term.
    3. Applies the proximal operator of the total variation and the
       constraint, solved in the dual by a few accelerated projection steps.

    The total variation is small for fields made of flat regions with sharp
    edges, as cells on an empty background are, while the twin image spreads
    fringes over the whole field and raises it. The data term keeps the
    estimate consistent with the hologram, so the minimiser keeps the object and
    sheds the twin. The total variation acts on the whole field, so unlike the
    Latychevskaia-Fink routine it needs no support; Gao and Cao also leave the
    support out for extended samples.

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
        regularization: float = 1e-2,
        inner_iterations: int = 5,
    ) -> None:
        """Initialize the algorithm.

        Args:
            img_size: Spatial size of input images (assumed square).
            wavelength: Wavelength of light in meters (e.g., 532e-9).
            pixel_size: Pixel pitch in meters (e.g., 4.65e-6).
            z_distance: Propagation distance in meters (e.g., 0.02).
            iterations: Number of outer (proximal gradient) iterations.
            numerical_aperture: NA of the imaging objective of a magnified setup, applied in
                forward propagation, or None for a lensless setup without a pupil.
            regularization: Weight lam of the total variation, 1e-2 in the
                simulations of Gao and Cao.
            inner_iterations: Dual projection steps per proximal operator; Gao
                and Cao use 5 for lam = 1e-2.
        """
        super().__init__()
        self.iterations = iterations
        self.regularization = regularization
        self.inner_iterations = inner_iterations

        # Physics propagator (same as HoloPASWIN for fairness)
        self.propagator = AngularSpectrumPropagator(
            (img_size, img_size),
            wavelength,
            pixel_size,
            z_distance,
            numerical_aperture=numerical_aperture,
        )

        # The propagator is unitary or, with a pupil, a projection, so the
        # largest eigenvalue of A^H A is at most 1 and the gradient step can be
        # 2. The difference operator has || D^H D || <= 8, which bounds the
        # step of the dual iterations by 1 / (8 step^2).
        self.step: float = 2.0
        self.dual_step: float = 1.0 / (8 * self.step**2)

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

            # The plain back-propagation, as every other model returns it.
            dirty_field_complex: torch.Tensor = self.propagator(field_sensor, backward=True)
            dirty_2ch: torch.Tensor = torch.cat([dirty_field_complex.real, dirty_field_complex.imag], dim=1)

            # Gao and Cao start from the back-propagated amplitude, x0 = A^H sqrt(H).
            current: torch.Tensor = self._adjoint(field_sensor)
            previous: torch.Tensor = current
            extrapolated: torch.Tensor
            sensor_field: torch.Tensor
            gradient: torch.Tensor
            for iteration in range(self.iterations):
                momentum: float = iteration / (iteration + 3)
                extrapolated = current + momentum * (current - previous)
                sensor_field = self.propagator(extrapolated, backward=False)
                gradient = 0.5 * self._adjoint(torch.sgn(sensor_field) * (sensor_field.abs() - amplitude_sensor))
                previous = current
                current = self._proximal(extrapolated - self.step * gradient)

            recovered_2ch: torch.Tensor = torch.cat([current.real, current.imag], dim=1)
            return recovered_2ch, dirty_2ch

    def _adjoint(self, sensor_field: torch.Tensor) -> torch.Tensor:
        """Apply A^H, back-propagation through the pupil when one is set.

        Args:
            sensor_field: Complex field at the sensor, shape (B, 1, H, W).

        Returns:
            The complex field at the object plane, same shape.
        """
        back: torch.Tensor = self.propagator(sensor_field, backward=True)
        if self.propagator.pupil is None:
            return back
        pupil: torch.Tensor = self.propagator.pupil.to(back.device)
        filtered: torch.Tensor = torch.fft.ifft2(torch.fft.fft2(back) * pupil)
        return filtered

    @staticmethod
    def _gradient(field: torch.Tensor) -> torch.Tensor:
        """Differences to the next pixel along both axes, wrapping around like the FFT.

        Args:
            field: Complex field of shape (B, 1, H, W).

        Returns:
            Differences of shape (B, 2, H, W), vertical then horizontal.
        """
        return torch.cat([torch.roll(field, -1, dims=2) - field, torch.roll(field, -1, dims=3) - field], dim=1)

    @staticmethod
    def _gradient_adjoint(differences: torch.Tensor) -> torch.Tensor:
        """Apply D^H, the adjoint of _gradient.

        Args:
            differences: Complex differences of shape (B, 2, H, W).

        Returns:
            The complex field of shape (B, 1, H, W).
        """
        vertical: torch.Tensor = differences[:, 0:1]
        horizontal: torch.Tensor = differences[:, 1:2]
        return (torch.roll(vertical, 1, dims=2) - vertical) + (torch.roll(horizontal, 1, dims=3) - horizontal)

    @staticmethod
    def _constrain(field: torch.Tensor) -> torch.Tensor:
        """Project onto non-negative absorption, |x| <= 1, keeping the phase.

        Args:
            field: Complex field of shape (B, 1, H, W).

        Returns:
            The projected field, same shape.
        """
        amplitude: torch.Tensor = field.abs()
        return torch.where(amplitude > 1, field / amplitude, field)

    def _proximal(self, point: torch.Tensor) -> torch.Tensor:
        """Proximal operator of step * (lam ||D x||_1 + constraint) at a point.

        It has no closed form. Gao and Cao solve its dual with the fast
        gradient projection of Beck and Teboulle (IEEE Trans. Image Process.
        18, 2419 (2009)): accelerated projected gradient steps on a dual
        variable w with |w| <= lam entrywise, after which the primal point is
        x = P(point - step D^H w), with P the projection onto |x| <= 1. A few
        steps give an inexact but sufficient proximal operator.

        Args:
            point: Complex field of shape (B, 1, H, W).

        Returns:
            The proximal point, same shape.
        """
        dual: torch.Tensor = torch.zeros((point.shape[0], 2, *point.shape[2:]), dtype=point.dtype, device=point.device)
        previous: torch.Tensor = dual
        extrapolated: torch.Tensor
        primal: torch.Tensor
        update: torch.Tensor
        for iteration in range(self.inner_iterations):
            momentum: float = iteration / (iteration + 3)
            extrapolated = dual + momentum * (dual - previous)
            primal = self._constrain(point - self.step * self._gradient_adjoint(extrapolated))
            update = extrapolated + self.dual_step * self.step * self._gradient(primal)
            previous = dual
            # Each complex dual entry is projected onto the disc of radius lam.
            dual = update / torch.clamp(update.abs() / self.regularization, min=1.0)
        return self._constrain(point - self.step * self._gradient_adjoint(dual))

    def __repr__(self) -> str:
        """String representation."""  # noqa: D401
        return f"ConstrainedComplexTV(iterations={self.iterations}, regularization={self.regularization})"
