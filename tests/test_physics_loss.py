"""Tests for the propagator (evanescent waves, objective pupil) and the physics loss.

The geometry is the magnified setup of dataset 2.0: 532 nm, 0.24 um pixels in the
object plane, 20 um defocus and a 10x/0.45 objective, so the pupil passes spatial
frequencies up to 0.45 / 0.532 um = 0.85 cycles per um.
"""

import numpy as np
import numpy.typing as npt
import pytest
import torch

from holopaswin.loss import PhysicsLoss
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


def as_2ch(field: torch.Tensor) -> torch.Tensor:
    """Split a complex field into the (1, 2, H, W) real/imaginary layout of the model.

    Args:
        field (torch.Tensor): Complex field, shape (H, W).

    Returns:
        torch.Tensor: Real and imaginary parts, shape (1, 2, H, W).
    """
    return torch.stack([field.real, field.imag]).unsqueeze(0)


def recorded(obj: torch.Tensor) -> torch.Tensor:
    """Record an object through the objective and normalise it by its mean.

    Args:
        obj (torch.Tensor): Complex object field, shape (H, W).

    Returns:
        torch.Tensor: Mean-normalised intensity, shape (1, 1, H, W).
    """
    intensity: torch.Tensor = torch.abs(propagator()(obj)) ** 2
    return (intensity / intensity.mean()).reshape(1, 1, SIZE, SIZE)


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


# Evanescent waves --------------------------------------------------------------


def evanescent(pixel_size: float) -> torch.Tensor:
    """Mark the spatial frequencies beyond 1 / wavelength.

    Args:
        pixel_size (float): Pixel size in meters.

    Returns:
        torch.Tensor: True where the plane wave is evanescent, on the unshifted grid.
    """
    f: torch.Tensor = torch.fft.fftfreq(SIZE, d=pixel_size)
    g_fx, g_fy = torch.meshgrid(f, f, indexing="ij")
    return g_fx**2 + g_fy**2 > (1 / WAVELENGTH) ** 2


def test_evanescent_waves_are_removed() -> None:
    """On the 0.24 um grid, 36% of the frequencies are evanescent and none pass."""
    transfer: torch.Tensor = propagator(None).H
    mask: torch.Tensor = evanescent(PIXEL)
    assert 0.35 < float(mask.float().mean()) < 0.37
    assert float(transfer[mask].abs().max()) == 0.0
    torch.testing.assert_close(transfer[~mask].abs(), torch.ones(int((~mask).sum())))


def test_back_propagation_removes_evanescent_noise() -> None:
    """The network input loses the evanescent part of the noise, which holds no signal."""
    noise: torch.Tensor = torch.randn(SIZE, SIZE, generator=torch.Generator().manual_seed(3)).to(torch.complex64)
    spectrum: torch.Tensor = torch.fft.fft2(propagator(None)(noise, backward=True)).abs()
    assert float(spectrum[evanescent(PIXEL)].max()) < 1e-4 * float(spectrum.max())


def test_lensless_grid_is_unchanged() -> None:
    """With 4.65 um pixels every frequency propagates, as before."""
    lensless: AngularSpectrumPropagator = AngularSpectrumPropagator((SIZE, SIZE), WAVELENGTH, 4.65e-6, 0.02)
    assert not bool(evanescent(4.65e-6).any())
    torch.testing.assert_close(lensless.H.abs(), torch.ones(SIZE, SIZE))


# Scale-invariant physics loss -------------------------------------------------


def test_physics_term_ignores_the_hologram_scale() -> None:
    """Multiplying the input hologram by a constant leaves the physics term unchanged."""
    loss: PhysicsLoss = PhysicsLoss(propagator())
    hologram: torch.Tensor = recorded(absorbing_cells())
    pred: torch.Tensor = torch.abs(propagator()(absorbing_cells() * 0.9)).reshape(1, 1, SIZE, SIZE) ** 2
    reference: torch.Tensor = loss.scale_invariant_physics_loss(pred, hologram)
    for scale in (0.5, 1.023, 3.0):
        torch.testing.assert_close(loss.scale_invariant_physics_loss(pred, scale * hologram), reference)


def test_correct_absorbing_prediction_has_no_physics_penalty() -> None:
    """A perfect prediction of an absorbing sample costs nothing.

    Mean normalisation puts the background of this sample above 1, because the
    cells absorb. Comparing absolute levels would penalise the correct answer.
    """
    obj: torch.Tensor = absorbing_cells()
    hologram: torch.Tensor = recorded(obj)
    loss: PhysicsLoss = PhysicsLoss(propagator(), lambda_physics=1.0)

    total: torch.Tensor = loss(as_2ch(obj), as_2ch(obj), hologram)
    absolute: torch.Tensor = torch.mean(torch.abs(torch.abs(propagator()(obj)) ** 2 - hologram[0, 0]))

    assert float(total) < 1e-5
    assert float(absolute) > 1e-3  # what the old, absolute comparison charged


def test_pupil_removes_the_penalty_for_fine_rims() -> None:
    """Without the pupil, the model predicts rim fringes the camera never records."""
    obj: torch.Tensor = absorbing_cells()
    hologram: torch.Tensor = recorded(obj)
    with_pupil: PhysicsLoss = PhysicsLoss(propagator(), lambda_physics=1.0)
    without_pupil: PhysicsLoss = PhysicsLoss(propagator(None), lambda_physics=1.0)

    matched: torch.Tensor = with_pupil(as_2ch(obj), as_2ch(obj), hologram)
    mismatched: torch.Tensor = without_pupil(as_2ch(obj), as_2ch(obj), hologram)

    assert float(matched) < 1e-5
    assert float(mismatched) > 0.02  # about 8% of the background for these sharp rims
