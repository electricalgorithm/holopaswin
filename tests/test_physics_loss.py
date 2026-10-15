"""Tests for the objective pupil of the angular spectrum propagator.

The geometry is the magnified setup of dataset 2.0: 532 nm, 0.24 um pixels in the
object plane, 20 um defocus and a 10x/0.45 objective, so the pupil passes spatial
frequencies up to 0.45 / 0.532 um = 0.85 cycles per um.
"""

import numpy as np
import numpy.typing as npt
import pytest
import torch

from holopaswin.propagator import AngularSpectrumPropagator

SIZE = 224
WAVELENGTH = 532e-9
PIXEL = 0.24e-6
Z = 20e-6
NA = 0.45


def propagator(numerical_aperture: float | None = NA) -> AngularSpectrumPropagator:
    """Build a propagator with the dataset 2.0 geometry.

    Args:
        numerical_aperture (float | None): Objective NA, or None for no pupil.

    Returns:
        AngularSpectrumPropagator: The propagator.
    """
    return AngularSpectrumPropagator((SIZE, SIZE), WAVELENGTH, PIXEL, Z, numerical_aperture=numerical_aperture)


def absorbing_cells() -> torch.Tensor:
    """Build disks with sharp rims that absorb 7% and shift the phase by 2 rad.

    They resemble red blood cells at 532 nm: their sharp edges scatter light
    beyond the objective NA, and their absorption lowers the mean intensity.

    Returns:
        torch.Tensor: Complex object field, shape (SIZE, SIZE).
    """
    y, x = np.mgrid[:SIZE, :SIZE]
    obj: npt.NDArray[np.complex128] = np.ones((SIZE, SIZE), dtype=np.complex128)
    for cy, cx in [(50, 60), (60, 160), (150, 50), (160, 150), (110, 110)]:
        cell: npt.NDArray[np.bool_] = (y - cy) ** 2 + (x - cx) ** 2 <= 16**2
        obj[cell] = 0.93 * np.exp(2.0j)
    return torch.from_numpy(obj).to(torch.complex64)


# Pupil ------------------------------------------------------------------------


def test_pupil_removes_frequencies_beyond_the_cutoff() -> None:
    """Forward propagation keeps no light beyond NA / wavelength."""
    spectrum: torch.Tensor = torch.fft.fft2(propagator()(absorbing_cells()))
    fx: torch.Tensor = torch.fft.fftfreq(SIZE, d=PIXEL)
    g_fx, g_fy = torch.meshgrid(fx, fx, indexing="ij")
    outside: torch.Tensor = torch.sqrt(g_fx**2 + g_fy**2) > 1.001 * NA / WAVELENGTH
    assert outside.any()
    assert torch.abs(spectrum[outside]).max() < 1e-3 * torch.abs(spectrum).max()


def test_without_aperture_propagation_is_unchanged() -> None:
    """Without a numerical aperture the propagator behaves as before."""
    obj: torch.Tensor = absorbing_cells()
    plain: AngularSpectrumPropagator = AngularSpectrumPropagator((SIZE, SIZE), WAVELENGTH, PIXEL, Z)
    assert plain.pupil is None
    torch.testing.assert_close(propagator(None)(obj), plain(obj))


def test_backward_propagation_ignores_the_pupil() -> None:
    """The model back-propagates the recorded hologram, which already passed the pupil."""
    obj: torch.Tensor = absorbing_cells()
    torch.testing.assert_close(propagator()(obj, backward=True), propagator(None)(obj, backward=True))


@pytest.mark.parametrize("numerical_aperture", [0.0, -0.1, 1.6])
def test_invalid_aperture_raises(numerical_aperture: float) -> None:
    """A numerical aperture outside (0, 1.5] is rejected."""
    with pytest.raises(ValueError, match="numerical_aperture"):
        propagator(numerical_aperture)
