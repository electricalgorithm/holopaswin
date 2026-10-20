"""Tests that every network and baseline uses the same physics."""

import pytest
import torch
from torch import nn

from holopaswin.baselines.gerchberg_saxton import GerchbergSaxton
from holopaswin.baselines.hrnet import HRNet
from holopaswin.baselines.unet_baseline import UNetBaseline
from holopaswin.model import HoloPASWIN
from holopaswin.resnet_unet import ResNetUNet

SIZE: int = 64
GEOMETRY: tuple[int, float, float, float] = (SIZE, 532e-9, 0.24e-6, 20e-6)
CLASSES: list[type[nn.Module]] = [UNetBaseline, ResNetUNet, HRNet, GerchbergSaxton, HoloPASWIN]


def construct(cls: type[nn.Module], numerical_aperture: float | None) -> nn.Module:
    """Build a network or baseline for the test geometry.

    Args:
        cls (type[nn.Module]): Class to build.
        numerical_aperture (float | None): Objective NA, or None for no pupil.

    Returns:
        nn.Module: The built module.
    """
    extra: dict[str, object] = {"use_pretrained": False} if cls is HoloPASWIN else {}
    return cls(*GEOMETRY, numerical_aperture=numerical_aperture, **extra)


@pytest.mark.parametrize("cls", CLASSES, ids=lambda cls: cls.__name__)
def test_the_objective_pupil_reaches_every_propagator(cls: type[nn.Module]) -> None:
    """Without an NA there is no pupil; with one, the propagator applies it."""
    assert construct(cls, None).propagator.pupil is None
    assert construct(cls, 0.45).propagator.pupil is not None


@pytest.mark.parametrize("cls", [UNetBaseline, ResNetUNet, HRNet, HoloPASWIN], ids=lambda cls: cls.__name__)
def test_every_network_back_propagates_the_same_field(cls: type[nn.Module]) -> None:
    """All networks start from sqrt(I) back-propagated with zero phase."""
    torch.manual_seed(0)
    hologram: torch.Tensor = torch.rand(1, 1, SIZE, SIZE) + 0.5
    module: nn.Module = construct(cls, 0.45).eval()
    amplitude: torch.Tensor = torch.complex(torch.sqrt(hologram + 1e-8), torch.zeros_like(hologram))
    expected_field: torch.Tensor = module.propagator(amplitude, backward=True)
    expected: torch.Tensor = torch.cat([expected_field.real, expected_field.imag], dim=1)
    with torch.no_grad():
        _, dirty = module(hologram)
    torch.testing.assert_close(dirty, expected, atol=1e-5, rtol=1e-5)
