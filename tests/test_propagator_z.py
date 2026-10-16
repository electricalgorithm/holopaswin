"""Tests for propagation over a different distance for each sample.

During z-augmented training each hologram is recorded at its own distance, and the
physics loss forward-propagates the prediction over that distance. A per-sample
distance must give the same field as a propagator built for that distance.
"""

import pytest
import torch

from holopaswin.loss import PhysicsLoss
from holopaswin.propagator import AngularSpectrumPropagator

SIZE: int = 64
WAVELENGTH: float = 532e-9
PIXEL: float = 0.24e-6
NOMINAL: float = 20e-6
NA: float = 0.45


def random_field(batch: int, seed: int = 0) -> torch.Tensor:
    """Make a pure phase field with an independent random phase per pixel.

    White noise fills the whole spectrum, so it also exercises the evanescent cut.

    Args:
        batch (int): Number of samples.
        seed (int): Seed of the generator.

    Returns:
        torch.Tensor: Complex field of shape (batch, 1, SIZE, SIZE).
    """
    generator: torch.Generator = torch.Generator().manual_seed(seed)
    phase: torch.Tensor = torch.rand(batch, 1, SIZE, SIZE, generator=generator)
    return torch.exp(1j * phase).to(torch.complex64)


@pytest.mark.parametrize("backward", [False, True])
@pytest.mark.parametrize("aperture", [None, NA])
def test_per_sample_distance_matches_fixed_propagators(backward: bool, aperture: float | None) -> None:
    """Each sample's field equals that of a propagator built for its distance."""
    distances: torch.Tensor = torch.tensor([14e-6, 20e-6, 26e-6])
    field: torch.Tensor = random_field(3)
    shared: AngularSpectrumPropagator = AngularSpectrumPropagator(
        (SIZE, SIZE), WAVELENGTH, PIXEL, NOMINAL, numerical_aperture=aperture
    )
    result: torch.Tensor = shared(field, backward=backward, z=distances)
    for index, distance in enumerate(distances.tolist()):
        fixed: AngularSpectrumPropagator = AngularSpectrumPropagator(
            (SIZE, SIZE), WAVELENGTH, PIXEL, distance, numerical_aperture=aperture
        )
        expected: torch.Tensor = fixed(field[index : index + 1], backward=backward)
        torch.testing.assert_close(result[index : index + 1], expected, atol=1e-4, rtol=1e-4)


def test_without_a_distance_the_result_is_unchanged() -> None:
    """z=None uses the precomputed transfer function exactly as before."""
    field: torch.Tensor = random_field(2)
    propagator: AngularSpectrumPropagator = AngularSpectrumPropagator(
        (SIZE, SIZE), WAVELENGTH, PIXEL, NOMINAL, numerical_aperture=NA
    )
    expected: torch.Tensor = torch.fft.ifft2(torch.fft.fft2(field) * propagator.H * propagator.pupil)
    torch.testing.assert_close(propagator(field), expected, atol=0.0, rtol=0.0)


def test_distance_needs_one_value_per_sample() -> None:
    """A distance count that does not match the batch is rejected."""
    propagator: AngularSpectrumPropagator = AngularSpectrumPropagator((SIZE, SIZE), WAVELENGTH, PIXEL, NOMINAL)
    with pytest.raises(ValueError, match="one distance per sample"):
        propagator(random_field(2), z=torch.tensor([20e-6, 21e-6, 22e-6]))
    with pytest.raises(ValueError, match="one distance per sample"):
        propagator(random_field(1)[0, 0], z=torch.tensor([20e-6]))


def test_physics_loss_at_the_nominal_distance_is_unchanged() -> None:
    """Passing the nominal distance as z_true gives the current loss."""
    propagator: AngularSpectrumPropagator = AngularSpectrumPropagator(
        (SIZE, SIZE), WAVELENGTH, PIXEL, NOMINAL, numerical_aperture=NA
    )
    criterion: PhysicsLoss = PhysicsLoss(propagator, lambda_physics=0.1)
    prediction: torch.Tensor = torch.view_as_real(random_field(2, seed=1)[:, 0]).permute(0, 3, 1, 2)
    target: torch.Tensor = torch.view_as_real(random_field(2, seed=2)[:, 0]).permute(0, 3, 1, 2)
    hologram: torch.Tensor = torch.rand(2, 1, SIZE, SIZE) + 0.5
    nominal: torch.Tensor = torch.full((2,), NOMINAL)
    torch.testing.assert_close(
        criterion(prediction, target, hologram, z_true=nominal),
        criterion(prediction, target, hologram),
        atol=1e-6,
        rtol=1e-6,
    )


def test_physics_loss_prefers_the_true_distance() -> None:
    """A hologram recorded at 26 um is explained by its object only at 26 um."""
    propagator: AngularSpectrumPropagator = AngularSpectrumPropagator(
        (SIZE, SIZE), WAVELENGTH, PIXEL, NOMINAL, numerical_aperture=NA
    )
    criterion: PhysicsLoss = PhysicsLoss(propagator, lambda_physics=1.0)
    obj: torch.Tensor = random_field(1, seed=3)
    true_distance: torch.Tensor = torch.tensor([26e-6])
    hologram: torch.Tensor = torch.abs(propagator(obj, z=true_distance)) ** 2
    obj_2ch: torch.Tensor = torch.view_as_real(obj[:, 0]).permute(0, 3, 1, 2)
    at_true: torch.Tensor = criterion(obj_2ch, obj_2ch, hologram, z_true=true_distance)
    at_nominal: torch.Tensor = criterion(obj_2ch, obj_2ch, hologram)
    assert at_true.item() < 1e-5
    assert at_nominal.item() > 10 * at_true.item()
