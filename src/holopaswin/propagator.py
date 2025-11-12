"""Wave propagation module using the Angular Spectrum Method (ASM).

This module provides a differentiable implementation of the Angular Spectrum Method
for wave propagation, which can be used as a layer in neural networks or as part
of loss functions. The propagator can perform both forward and backward propagation.
"""

import numpy as np
import torch
from torch import nn


class AngularSpectrumPropagator(nn.Module):
    """Differentiable Angular Spectrum Method (ASM) for wave propagation.

    Implements the Angular Spectrum Method for propagating complex wave fields
    through free space. This is a physics-based, differentiable operation that
    can be used as a layer in neural networks or as part of loss functions.

    The propagator precomputes the transfer function for efficiency during training.
    It supports both forward propagation (in the +z direction) and backward
    propagation (in the -z direction) by using the complex conjugate of the
    transfer function.
    """

    def __init__(
        self,
        shape: tuple[int, int],
        wavelength: float,
        pixel_size: float,
        z_distance: float,
    ) -> None:
        """Initialize the Angular Spectrum Propagator.

        Args:
            shape: Tuple of (height, width) for the input field.
            wavelength: Wavelength of the light in meters (e.g., 532e-9).
            pixel_size: Physical size of each pixel in meters (e.g., 4.65e-6).
            z_distance: Propagation distance in meters (e.g., 0.02).
        """
        super().__init__()
        self.shape = shape
        self.wavelength = wavelength
        self.pixel_size = pixel_size
        self.z_distance = z_distance

        # Precompute transfer function H to save time during training
        self.H = self._precompute_transfer_function()

    def _precompute_transfer_function(self) -> torch.Tensor:
        """Precompute the transfer function for wave propagation.

        Computes the frequency-domain transfer function H that represents
        the propagation kernel. This is precomputed once during initialization
        for efficiency during training.

        The transfer function is: H = exp(i * k * z * sqrt(1 - (lambda * f)^2))
        where k = 2*pi/lambda is the wavenumber, z is the propagation distance,
        lambda is the wavelength, and f is the spatial frequency.

        Returns:
            Complex tensor of shape (M, N) containing the transfer function.
        """
        m, n = self.shape

        # Spatial frequencies
        fx = torch.fft.fftfreq(m, d=self.pixel_size)
        fy = torch.fft.fftfreq(n, d=self.pixel_size)
        g_fx, g_fy = torch.meshgrid(fx, fy, indexing="ij")

        # Squared magnitude of spatial frequencies
        f_sq = g_fx**2 + g_fy**2

        # Propagation kernel in frequency domain
        # H = exp(i * k * z * sqrt(1 - (lambda * f)^2))
        # Using computationally stable form for backprop
        k = 2 * np.pi / self.wavelength

        # Mask for evanescent waves (where 1 - (lambda*f)^2 < 0)
        term_inside_sqrt = 1 - (self.wavelength**2 * f_sq)
        term_inside_sqrt = torch.clamp(term_inside_sqrt, min=0)  # Avoid NaNs

        phase = k * self.z_distance * torch.sqrt(term_inside_sqrt)
        h = torch.exp(1j * phase)
        return h

    def forward(self, complex_field: torch.Tensor, backward: bool = False) -> torch.Tensor:
        """Propagate a complex wave field using the Angular Spectrum Method.

        Args:
            complex_field: Input complex field tensor of shape (B, C, H, W) or (H, W).
                         Represents the complex amplitude of the wave field.
            backward: If True, performs backward propagation (conjugate of transfer function).
                     If False, performs forward propagation. Defaults to False.

        Returns:
            Propagated complex field tensor of the same shape as input.
        """
        # Ensure H is on the same device as input
        h = self.H.to(complex_field.device)

        if backward:
            h = torch.conj(h)

        # FFT
        u_f = torch.fft.fft2(complex_field)

        # Multiply by Transfer Function
        u_z_f = u_f * h

        # IFFT
        u_z = torch.fft.ifft2(u_z_f)
        return u_z  # type: ignore[no-any-return]
