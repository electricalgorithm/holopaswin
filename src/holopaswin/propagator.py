"""Wave propagation module using the Angular Spectrum Method (ASM).

This module provides a differentiable implementation of the Angular Spectrum Method
for wave propagation, which can be used as a layer in neural networks or as part
of loss functions. The propagator can perform both forward and backward propagation.

In a magnified setup the microscope objective only collects light up to its
numerical aperture (NA). Its coherent transfer function is a circular pupil that
passes spatial frequencies up to NA / wavelength (Goodman, Introduction to Fourier
Optics, Sec. 6.2). Forward propagation applies that pupil, so a predicted object
is compared with what the camera can actually record. This mirrors the hologen
simulator, which applies the pupil on the way to the sensor only.
"""

from typing import ClassVar

import numpy as np
import torch
from torch import nn

BATCH_RANK: int = 4  # (B, C, H, W)


class AngularSpectrumPropagator(nn.Module):
    """Differentiable Angular Spectrum Method (ASM) for wave propagation.

    Implements the Angular Spectrum Method for propagating complex wave fields
    through free space. This is a physics-based, differentiable operation that
    can be used as a layer in neural networks or as part of loss functions.

    The propagator precomputes the transfer function for efficiency during training.
    It supports both forward propagation (in the +z direction) and backward
    propagation (in the -z direction) by using the complex conjugate of the
    transfer function. With a numerical aperture, forward propagation also passes
    the field through the objective pupil.
    """

    # Largest numerical aperture of a real objective, an oil immersion objective
    # (n = 1.52). Larger values point to a unit or typing mistake.
    MAX_NUMERICAL_APERTURE: ClassVar[float] = 1.5

    # Buffers registered in __init__, declared here for type checkers.
    H: torch.Tensor
    kz_root: torch.Tensor
    propagating: torch.Tensor
    pupil: torch.Tensor | None

    def __init__(
        self,
        shape: tuple[int, int],
        wavelength: float,
        pixel_size: float,
        z_distance: float,
        numerical_aperture: float | None = None,
    ) -> None:
        """Initialize the Angular Spectrum Propagator.

        Args:
            shape (tuple[int, int]): Height and width of the input field.
            wavelength (float): Wavelength of the light in meters (e.g., 532e-9).
            pixel_size (float): Pixel size in the object plane in meters (e.g.,
                0.24e-6 for 2.4 um sensor pixels behind a 10x objective).
            z_distance (float): Propagation distance in meters (e.g., 20e-6).
            numerical_aperture (float | None): NA of the imaging objective of a
                magnified setup, or None for a lensless setup without a pupil.

        Raises:
            ValueError: If the numerical aperture is outside (0, 1.5].
        """
        super().__init__()
        if numerical_aperture is not None and not 0 < numerical_aperture <= self.MAX_NUMERICAL_APERTURE:
            msg: str = f"numerical_aperture must be in (0, {self.MAX_NUMERICAL_APERTURE}], got {numerical_aperture}."
            raise ValueError(msg)

        self.shape = shape
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.z_distance = z_distance
        self.numerical_aperture = numerical_aperture

        # Since z can change, we cannnot pre-compute transfer function.
        # Instead, we pre-compute whatever we can and register them to buffers.
        root: torch.Tensor
        propagating: torch.Tensor
        root, propagating = self._precompute_axial_wavenumber()
        k: float = 2 * np.pi / self.wavelength
        self.register_buffer("kz_root", k * root, persistent=False)
        self.register_buffer("propagating", propagating, persistent=False)

        # Same operation order as before per-sample distances existed, so H is unchanged bit for bit.
        self.register_buffer("H", torch.exp(1j * (k * self.z_distance * root)) * propagating, persistent=False)
        self.register_buffer(
            "pupil",
            None if numerical_aperture is None else self._precompute_pupil(numerical_aperture),
            persistent=False,
        )

    def _precompute_axial_wavenumber(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Precompute the axial part of every plane wave on the grid.

        The transfer function over a distance z is
        H = exp(i * k * z * sqrt(1 - (lambda * f)^2)),
        where k = 2*pi/lambda is the wavenumber, z is the propagation distance,
        lambda is the wavelength, and f is the spatial frequency.

        Beyond f = 1 / lambda the plane waves are evanescent. They decay instead
        of travelling (Goodman, Introduction to Fourier Optics, Sec. 3.10), so H
        is 0 there. This matches the physical decay to below 1e-12 once f is 0.02
        cycles per um past the cutoff at 20 um, and it keeps back-propagation
        stable, because the exact inverse would amplify these waves. Only grids
        with pixels smaller than lambda / 2 have evanescent frequencies, such as
        the 0.24 um grid of the magnified setup.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: sqrt(1 - (lambda * f)^2), set to 0
                for evanescent waves, and the mask of propagating waves, both of
                shape (M, N).
        """
        m, n = self.shape

        # Spatial frequencies
        fx = torch.fft.fftfreq(m, d=self.pixel_size)
        fy = torch.fft.fftfreq(n, d=self.pixel_size)
        g_fx, g_fy = torch.meshgrid(fx, fy, indexing="ij")

        # Squared magnitude of spatial frequencies
        f_sq = g_fx**2 + g_fy**2

        # Evanescent waves (where 1 - (lambda*f)^2 < 0) do not propagate.
        term_inside_sqrt = 1 - (self.wavelength**2 * f_sq)
        propagating = term_inside_sqrt >= 0
        term_inside_sqrt = torch.clamp(term_inside_sqrt, min=0)  # Avoid NaNs

        return torch.sqrt(term_inside_sqrt), propagating

    def transfer_function(self, z: torch.Tensor) -> torch.Tensor:
        """Build the transfer function for a different distance per sample.

        Args:
            z (torch.Tensor): Propagation distance of each sample in meters,
                shape (B,).

        Returns:
            torch.Tensor: Transfer functions of shape (B, 1, M, N), with the same
                evanescent cut as H, on the device of z.
        """
        kz_root: torch.Tensor = self.kz_root.to(z.device)
        propagating: torch.Tensor = self.propagating.to(z.device)
        phase: torch.Tensor = kz_root[None, None] * z.to(kz_root.dtype)[:, None, None, None]
        return torch.exp(1j * phase) * propagating

    def _precompute_pupil(self, numerical_aperture: float) -> torch.Tensor:
        """Precompute the coherent transfer function of the objective.

        Args:
            numerical_aperture (float): NA of the objective.

        Returns:
            torch.Tensor: 1 inside the circle of spatial frequencies up to
                NA / wavelength and 0 outside, shape (M, N), on the same
                unshifted frequency grid as the transfer function.
        """
        m, n = self.shape
        fx: torch.Tensor = torch.fft.fftfreq(m, d=self.pixel_size)
        fy: torch.Tensor = torch.fft.fftfreq(n, d=self.pixel_size)
        g_fx, g_fy = torch.meshgrid(fx, fy, indexing="ij")
        cutoff: float = numerical_aperture / self.wavelength
        return (g_fx**2 + g_fy**2 <= cutoff**2).to(torch.float32)

    def forward(
        self, complex_field: torch.Tensor, backward: bool = False, z: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Propagate a complex wave field using the Angular Spectrum Method.

        Args:
            complex_field: Input complex field tensor of shape (B, C, H, W) or (H, W).
                         Represents the complex amplitude of the wave field.
            backward: If True, performs backward propagation (conjugate of transfer function).
                     If False, performs forward propagation, through the objective pupil
                     when a numerical aperture is set. Defaults to False.
            z: Propagation distance of each sample in meters, shape (B,), for a
                batch of shape (B, C, H, W) whose samples were recorded at
                different distances. None uses the distance the propagator was
                built with.

        Returns:
            Propagated complex field tensor of the same shape as input.

        Raises:
            ValueError: If z is given and does not hold one distance per sample
                of a (B, C, H, W) batch.
        """
        h: torch.Tensor
        if z is None:
            # Ensure H is on the same device as input
            h = self.H.to(complex_field.device)
        else:
            if complex_field.dim() != BATCH_RANK or z.shape != (complex_field.shape[0],):
                msg: str = (
                    f"z must hold one distance per sample of a (B, C, H, W) batch; got z of shape "
                    f"{tuple(z.shape)} for a field of shape {tuple(complex_field.shape)}."
                )
                raise ValueError(msg)
            h = self.transfer_function(z.to(complex_field.device))

        if backward:
            h = torch.conj(h)
        elif self.pupil is not None:
            h = h * self.pupil.to(complex_field.device)

        # FFT
        u_f = torch.fft.fft2(complex_field)

        # Multiply by Transfer Function
        u_z_f = u_f * h

        # IFFT
        u_z = torch.fft.ifft2(u_z_f)
        return u_z  # type: ignore[no-any-return]
