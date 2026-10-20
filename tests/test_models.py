"""Tests that every network and baseline uses the same physics."""

import pytest
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
