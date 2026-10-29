"""Tests that every network and baseline uses the same physics."""

import pathlib

import pytest
import torch
from torch import nn

from holopaswin.baselines.gerchberg_saxton import GerchbergSaxton
from holopaswin.baselines.hrnet import HRNet
from holopaswin.baselines.latychevskaia_fink import LatychevskaiaFink
from holopaswin.baselines.total_variation import ConstrainedComplexTV
from holopaswin.baselines.unet_baseline import UNetBaseline
from holopaswin.model import WIRING_VERSION, HoloPASWIN
from holopaswin.models import (
    ARCHITECTURES,
    BASELINES,
    Geometry,
    build_baseline,
    build_model,
    load_run,
    read_geometry,
    resolve_model,
)
from holopaswin.propagator import AngularSpectrumPropagator
from holopaswin.resnet_unet import ResNetUNet
from holopaswin.training.config import ModelSettings, TrainingConfig, config_from_dict, config_to_toml

SIZE: int = 64
GEOMETRY: tuple[int, float, float, float] = (SIZE, 532e-9, 0.24e-6, 20e-6)
CLASSES: list[type[nn.Module]] = [
    UNetBaseline,
    ResNetUNet,
    HRNet,
    GerchbergSaxton,
    LatychevskaiaFink,
    ConstrainedComplexTV,
    HoloPASWIN,
]


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


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("cls", [UNetBaseline, ResNetUNet, HRNet, HoloPASWIN], ids=lambda cls: cls.__name__)
def test_an_untrained_network_starts_near_the_back_propagated_scale(cls: type[nn.Module], seed: int) -> None:
    """A fresh residual network adds a small correction to the input, so a flat field stays near 1.

    Adding the back-propagated field twice would start every network at an
    amplitude of 2, which training must first undo.

    Args:
        cls (type[nn.Module]): Network to build.
        seed (int): Seed of the initial weights.
    """
    torch.manual_seed(seed)
    module: nn.Module = construct(cls, 0.45).eval()
    with torch.no_grad():
        clean, _ = module(torch.ones(2, 1, SIZE, SIZE))
    assert abs(torch.hypot(clean[:, 0], clean[:, 1]).median().item() - 1) < 0.5


@pytest.mark.parametrize("residual_mode", [True, False])
def test_holopaswin_adds_the_back_propagated_field_once(residual_mode: bool) -> None:
    """With a zero last layer the Swin U-Net predicts no correction.

    The output is then the back-propagated field in residual mode and zero
    otherwise; any skip inside the Swin U-Net would show as an extra copy.

    Args:
        residual_mode (bool): Whether the correction is added to the back-propagated field.
    """
    torch.manual_seed(0)
    module: HoloPASWIN = HoloPASWIN(*GEOMETRY, use_pretrained=False, residual_mode=residual_mode).eval()
    last: nn.Module = module.swin_unet.final_up[-1]
    assert isinstance(last, nn.Conv2d)
    assert last.bias is not None
    nn.init.zeros_(last.weight)
    nn.init.zeros_(last.bias)
    hologram: torch.Tensor = torch.rand(1, 1, SIZE, SIZE) + 0.5
    with torch.no_grad():
        clean, dirty = module(hologram)
    torch.testing.assert_close(clean, dirty if residual_mode else torch.zeros_like(dirty))


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


def test_gerchberg_saxton_improves_on_back_propagation() -> None:
    """GS on a weak transmission object moves away from ASM, depends on the iterations and gets closer to the truth.

    The geometry is lensless with pixels larger than the wavelength, so ASM is
    unitary and an object constraint that does nothing leaves GS equal to ASM.
    """
    size: int = 64
    geometry: tuple[int, float, float, float] = (size, 532e-9, 4.65e-6, 0.02)
    axis: torch.Tensor = torch.arange(size) - size / 2
    disk: torch.Tensor = ((axis[:, None] ** 2 + axis[None, :] ** 2) < 10**2).float()
    truth: torch.Tensor = ((1 - 0.1 * disk) * torch.exp(1j * 0.5 * disk))[None, None]
    propagator: AngularSpectrumPropagator = AngularSpectrumPropagator((size, size), *geometry[1:])
    hologram: torch.Tensor = propagator(truth).abs() ** 2

    outputs: dict[int, torch.Tensor] = {}
    for iterations in (0, 10, 40):
        estimate, _ = GerchbergSaxton(*geometry, iterations=iterations)(hologram)
        outputs[iterations] = torch.complex(estimate[:, :1], estimate[:, 1:])

    assert (outputs[10] - outputs[0]).abs().max() > 1e-3
    assert (outputs[40] - outputs[10]).abs().max() > 1e-4
    assert (outputs[40] - truth).abs().mean() < (outputs[0] - truth).abs().mean()
    torch.testing.assert_close(outputs[40].abs(), torch.ones_like(outputs[40].abs()), atol=1e-5, rtol=0)


def sparse_object(size: int) -> torch.Tensor:
    """Two weak absorbing phase disks on an empty background, as a (1, 1, size, size) field.

    Args:
        size (int): Image size in pixels.

    Returns:
        torch.Tensor: Complex transmission function, 1 away from the disks.
    """
    axis: torch.Tensor = torch.arange(size) - size / 2
    first: torch.Tensor = (((axis[:, None] + 14) ** 2 + (axis[None, :] + 10) ** 2) < 6**2).float()
    second: torch.Tensor = (((axis[:, None] - 12) ** 2 + (axis[None, :] - 16) ** 2) < 5**2).float()
    disks: torch.Tensor = first + second
    return ((1 - 0.1 * disks) * torch.exp(1j * 0.5 * disks))[None, None]


def test_latychevskaia_fink_improves_on_back_propagation() -> None:
    """LF on sparse objects moves away from ASM, depends on the iterations and gets closer to the truth."""
    size: int = 64
    geometry: tuple[int, float, float, float] = (size, 532e-9, 4.65e-6, 0.02)
    truth: torch.Tensor = sparse_object(size)
    propagator: AngularSpectrumPropagator = AngularSpectrumPropagator((size, size), *geometry[1:])
    hologram: torch.Tensor = propagator(truth).abs() ** 2

    # The outputs carry the constant phase exp(-i k z) of the back-propagation; compare without it.
    incident: torch.Tensor = torch.conj(propagator.H[0, 0])
    outputs: dict[int, torch.Tensor] = {}
    for iterations in (0, 10, 40):
        estimate, _ = LatychevskaiaFink(*geometry, iterations=iterations)(hologram)
        outputs[iterations] = torch.complex(estimate[:, :1], estimate[:, 1:]) / incident

    asm, _ = GerchbergSaxton(*geometry, iterations=0)(hologram)
    assert (outputs[10] - torch.complex(asm[:, :1], asm[:, 1:]) / incident).abs().max() > 1e-3
    assert (outputs[40] - outputs[10]).abs().max() > 1e-4
    assert (outputs[40] - truth).abs().mean() < 0.5 * (outputs[0] - truth).abs().mean()
    assert outputs[40].abs().max().item() <= 1 + 1e-5


def test_latychevskaia_fink_constraint_clips_and_smooths() -> None:
    """Inside the support |t| is clipped to 1 and smoothed, the phase kept; outside, t is 1."""
    size: int = 16
    lf: LatychevskaiaFink = LatychevskaiaFink(size, 532e-9, 4.65e-6, 0.02)
    rows: torch.Tensor = torch.arange(size)[:, None]
    columns: torch.Tensor = torch.arange(size)[None, :]
    checkerboard: torch.Tensor = 0.6 + 0.6 * ((rows + columns) % 2).float()
    transmission: torch.Tensor = (checkerboard * torch.exp(1j * torch.full((size, size), 0.3)))[None, None]
    support: torch.Tensor = torch.zeros(1, 1, size, size)
    support[..., 4:12, 4:12] = 1.0
    constrained: torch.Tensor = lf._constrain_object(transmission, support)  # noqa: SLF001
    inside: torch.Tensor = constrained[..., 6:10, 6:10]
    # Clipping turns 1.2 into 1 before the kernel averages 0.6 and 1, so the result lies strictly between them.
    assert inside.abs().max().item() < 1.0
    assert inside.abs().min().item() > 0.6
    torch.testing.assert_close(torch.angle(inside), torch.full_like(inside.real, 0.3), atol=1e-5, rtol=0)
    torch.testing.assert_close(constrained[..., :2, :2], torch.ones(1, 1, 2, 2, dtype=constrained.dtype))


def test_latychevskaia_fink_support_holds_the_objects() -> None:
    """The support found in the first reconstruction covers the disks and leaves most of the field out."""
    size: int = 64
    lf: LatychevskaiaFink = LatychevskaiaFink(size, 532e-9, 4.65e-6, 0.02)
    truth: torch.Tensor = sparse_object(size)
    support: torch.Tensor = lf.find_support(lf.propagator(lf.propagator(truth), backward=True))
    inside: torch.Tensor = (truth - 1).abs() > 1e-3
    assert support[inside].min().item() == 1.0
    assert (support > 0.5).float().mean().item() < 0.5


def test_latychevskaia_fink_margin_is_a_distance() -> None:
    """The 3 um margin is 13 pixels at the magnified 0.24 um pitch and one pixel at a 4.65 um sensor."""
    assert LatychevskaiaFink(32, 532e-9, 0.24e-6, 20e-6).margin_pixels == 13
    assert LatychevskaiaFink(32, 532e-9, 4.65e-6, 0.02).margin_pixels == 1


def disk_hologram(radius: int, phase: float) -> tuple[LatychevskaiaFink, torch.Tensor, torch.Tensor]:
    """Back-propagate the hologram of one phase disk in the magnified setup.

    Args:
        radius (int): Disk radius in pixels.
        phase (float): Phase shift of the disk in radians.

    Returns:
        tuple[LatychevskaiaFink, torch.Tensor, torch.Tensor]: The reconstruction
            for a 128-pixel field at 0.24 um and NA 0.45, the transmission of the
            first reconstruction and the disk mask.
    """
    size: int = 128
    lf: LatychevskaiaFink = LatychevskaiaFink(size, 532e-9, 0.24e-6, 20e-6, numerical_aperture=0.45)
    axis: torch.Tensor = torch.arange(size) - size / 2
    disk: torch.Tensor = (axis[:, None] ** 2 + axis[None, :] ** 2) < radius**2
    hologram: torch.Tensor = lf.propagator(torch.exp(1j * phase * disk.float())[None, None]).abs() ** 2
    back: torch.Tensor = lf.propagator(torch.sqrt(hologram).to(torch.complex64), backward=True)
    return lf, back / torch.conj(lf.propagator.H[0, 0]), disk


def test_latychevskaia_fink_support_holds_a_strong_cell_in_the_magnified_setup() -> None:
    """A strong phase disk larger than the margin lies wholly inside the support, which stays well short of the field."""
    lf, transmission, disk = disk_hologram(radius=20, phase=1.5)
    support: torch.Tensor = lf.find_support(transmission)[0, 0]
    assert support[disk].min().item() == 1.0
    assert (support > 0.5).float().mean().item() < 0.6


def test_latychevskaia_fink_falls_back_to_the_whole_field() -> None:
    """With nothing to tell apart from the background, the whole field is the support, with a warning.

    A disk over half the field puts the median deviation on the disk, so the
    threshold keeps nothing; an empty field has no deviation at all.
    """
    size: int = 64
    lf: LatychevskaiaFink = LatychevskaiaFink(size, 532e-9, 0.24e-6, 20e-6)
    axis: torch.Tensor = torch.arange(size) - size / 2
    large: torch.Tensor = ((axis[:, None] ** 2 + axis[None, :] ** 2) < 26**2).float()
    transmissions: torch.Tensor = torch.stack([torch.exp(1j * large), torch.ones(size, size, dtype=torch.complex64)])
    with pytest.warns(RuntimeWarning, match="No support found in 2 of 2"):
        support: torch.Tensor = lf.find_support(transmissions[:, None])
    torch.testing.assert_close(support, torch.ones_like(support))


@pytest.mark.parametrize("numerical_aperture", [None, 0.45])
def test_total_variation_uses_exact_adjoints(numerical_aperture: float | None) -> None:
    """The solver's back-propagation and difference adjoints satisfy <A x, r> = <x, A^H r>."""
    torch.manual_seed(0)
    tv: ConstrainedComplexTV = ConstrainedComplexTV(*GEOMETRY, numerical_aperture=numerical_aperture)
    field: torch.Tensor = torch.randn(1, 1, SIZE, SIZE, dtype=torch.complex64)
    sensor: torch.Tensor = torch.randn(1, 1, SIZE, SIZE, dtype=torch.complex64)
    differences: torch.Tensor = torch.randn(1, 2, SIZE, SIZE, dtype=torch.complex64)
    forward: torch.Tensor = (tv.propagator(field) * sensor.conj()).sum()
    adjoint: torch.Tensor = (field * tv._adjoint(sensor).conj()).sum()  # noqa: SLF001
    assert (forward - adjoint).abs().item() < 1e-3
    gradient: torch.Tensor = (tv._gradient(field) * differences.conj()).sum()  # noqa: SLF001
    gradient_adjoint: torch.Tensor = (field * tv._gradient_adjoint(differences).conj()).sum()  # noqa: SLF001
    assert (gradient - gradient_adjoint).abs().item() < 1e-3


def test_total_variation_improves_on_back_propagation() -> None:
    """CCTV moves away from ASM, depends on the iterations, gets closer to the truth and absorbs only."""
    size: int = 64
    geometry: tuple[int, float, float, float] = (size, 532e-9, 4.65e-6, 0.02)
    truth: torch.Tensor = sparse_object(size)
    propagator: AngularSpectrumPropagator = AngularSpectrumPropagator((size, size), *geometry[1:])
    hologram: torch.Tensor = propagator(truth).abs() ** 2

    # The outputs carry the constant phase exp(-i k z) of the back-propagation; compare without it.
    incident: torch.Tensor = torch.conj(propagator.H[0, 0])
    outputs: dict[int, torch.Tensor] = {}
    for iterations in (0, 10, 40):
        estimate, _ = ConstrainedComplexTV(*geometry, iterations=iterations)(hologram)
        outputs[iterations] = torch.complex(estimate[:, :1], estimate[:, 1:]) / incident

    assert (outputs[10] - outputs[0]).abs().max() > 1e-3
    assert (outputs[40] - outputs[10]).abs().max() > 1e-4
    assert (outputs[40] - truth).abs().mean() < 0.5 * (outputs[0] - truth).abs().mean()
    assert outputs[40].abs().max().item() <= 1 + 1e-5


def write_recipe(folder: pathlib.Path, size: int = SIZE) -> None:
    """Write the geometry section of a hologen recipe.

    Args:
        folder (pathlib.Path): Dataset or run folder.
        size (int): Image size in pixels.
    """
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "recipe.toml").write_text(
        f"[geometry]\nresolution = {size}\nsensor_pixel_size = 2.4\nmagnification = 10.0\n"
        'numerical_aperture = 0.45\nwavelength = 0.532\nz_distance = 20.0\nz_offset = 0.0\nz_error = 0.0\nnote = "x"\n'
    )


def test_geometry_comes_from_the_recipe_in_si_units(tmp_path: pathlib.Path) -> None:
    """The recipe's um values become metres, and the object-plane pixel is pitch / magnification."""
    write_recipe(tmp_path)
    geometry: Geometry = read_geometry(tmp_path)
    assert geometry.resolution == SIZE
    assert geometry.wavelength == pytest.approx(532e-9)
    assert geometry.pixel_size == pytest.approx(0.24e-6)
    assert geometry.z_distance == pytest.approx(20e-6)
    assert geometry.numerical_aperture == 0.45


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_every_architecture_is_built_for_the_dataset_geometry(tmp_path: pathlib.Path, architecture: str) -> None:
    """The factory gives each network the dataset's size, distance and pupil."""
    write_recipe(tmp_path)
    model: nn.Module = build_model(ModelSettings(architecture=architecture, pretrained=False), read_geometry(tmp_path))
    assert model.propagator.shape == (SIZE, SIZE)
    assert model.propagator.z_distance == pytest.approx(20e-6)
    assert model.propagator.pupil is not None


def test_a_run_folder_rebuilds_the_trained_model(tmp_path: pathlib.Path) -> None:
    """load_run reads config.toml and recipe.toml and restores the weights."""
    write_recipe(tmp_path)
    config: TrainingConfig = config_from_dict({"model": {"architecture": "unet"}})
    (tmp_path / "config.toml").write_text(config_to_toml(config))
    torch.manual_seed(3)
    original: nn.Module = build_model(config.model, read_geometry(tmp_path))
    torch.save(original.state_dict(), tmp_path / "model.pth")
    restored: nn.Module = load_run(tmp_path, torch.device("cpu"))
    assert isinstance(restored, UNetBaseline)
    for name, tensor in original.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], tensor)


def test_a_bare_checkpoint_is_read_as_holopaswin(tmp_path: pathlib.Path) -> None:
    """A .pth file without a run folder is a HoloPASWIN model for the given geometry."""
    write_recipe(tmp_path)
    geometry: Geometry = read_geometry(tmp_path)
    model: nn.Module = build_model(ModelSettings(pretrained=False), geometry)
    torch.save(model.state_dict(), tmp_path / "plain.pth")
    assert isinstance(resolve_model(tmp_path / "plain.pth", geometry, torch.device("cpu")), HoloPASWIN)


@pytest.mark.parametrize("version", [None, WIRING_VERSION - 1])
def test_a_checkpoint_of_another_skip_wiring_is_rejected(tmp_path: pathlib.Path, version: int | None) -> None:
    """Weights trained with another number of skips around the Swin U-Net load without error but reconstruct wrongly.

    The skips have no weights, so only the stored wiring version tells the
    generations apart; a checkpoint without it (None) or with an older one
    must be refused, both bare and in a run folder.

    Args:
        tmp_path (pathlib.Path): Folder for the checkpoints.
        version (int | None): Stored wiring version, or None to leave it out.
    """
    write_recipe(tmp_path / "run")
    (tmp_path / "run" / "config.toml").write_text(config_to_toml(config_from_dict({"model": {"pretrained": False}})))
    geometry: Geometry = read_geometry(tmp_path / "run")
    state: dict[str, torch.Tensor] = build_model(ModelSettings(pretrained=False), geometry).state_dict()
    if version is None:
        del state["wiring_version"]
    else:
        state["wiring_version"] = torch.tensor(version)
    torch.save(state, tmp_path / "run" / "model.pth")
    with pytest.raises(ValueError, match="skip wiring"):
        resolve_model(tmp_path / "run" / "model.pth", geometry, torch.device("cpu"))
    with pytest.raises(ValueError, match="skip wiring"):
        load_run(tmp_path / "run", torch.device("cpu"))


def test_a_checkpoint_records_the_current_skip_wiring() -> None:
    """New HoloPASWIN weights carry the wiring version they were trained with."""
    model: HoloPASWIN = HoloPASWIN(*GEOMETRY, use_pretrained=False)
    assert int(model.state_dict()["wiring_version"]) == WIRING_VERSION


def test_a_run_for_another_image_size_is_rejected(tmp_path: pathlib.Path) -> None:
    """A run trained at one size cannot score test sets of another."""
    write_recipe(tmp_path / "run", size=32)
    (tmp_path / "run" / "config.toml").write_text(config_to_toml(config_from_dict({"model": {"pretrained": False}})))
    torch.save(
        build_model(ModelSettings(pretrained=False), read_geometry(tmp_path / "run")).state_dict(),
        tmp_path / "run" / "model.pth",
    )
    write_recipe(tmp_path / "test")
    with pytest.raises(ValueError, match="32 px"):
        resolve_model(tmp_path / "run", read_geometry(tmp_path / "test"), torch.device("cpu"))


# An empty hologram holds no objects, so Latychevskaia-Fink rightly warns that it found no support.
@pytest.mark.filterwarnings("ignore:No support found:RuntimeWarning")
@pytest.mark.parametrize("name", BASELINES)
def test_baselines_have_no_propagation_phase(tmp_path: pathlib.Path, name: str) -> None:
    """On an empty hologram the classical reconstructions are the flat unit field."""
    write_recipe(tmp_path)
    baseline: nn.Module = build_baseline(name, read_geometry(tmp_path))
    field: torch.Tensor
    field, _ = baseline(torch.ones(1, 1, SIZE, SIZE))
    phase: torch.Tensor = torch.atan2(field[:, 1], field[:, 0])
    assert phase.abs().max().item() < 1e-3
    assert torch.hypot(field[:, 0], field[:, 1]).mean().item() == pytest.approx(1.0, abs=1e-3)
