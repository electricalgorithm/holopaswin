"""Building networks and classical baselines for the geometry of a dataset.

Every network back-propagates the hologram with the angular spectrum method and
corrects the result, so it needs the wavelength, the object-plane pixel size,
the distance and the objective NA of the setup. They always come from the
recipe that hologen saved with the dataset, never from constants. A run folder
written by holopaswin train holds the training configuration and that recipe
next to the weights, so a trained model can be rebuilt from the folder alone.
"""

import dataclasses
import math
from pathlib import Path

import torch
from torch import nn

from holopaswin.baselines.gerchberg_saxton import GerchbergSaxton
from holopaswin.baselines.hrnet import HRNet
from holopaswin.baselines.latychevskaia_fink import LatychevskaiaFink
from holopaswin.baselines.total_variation import ConstrainedComplexTV
from holopaswin.baselines.unet_baseline import UNetBaseline
from holopaswin.dataset import MICROMETER, read_recipe_geometry
from holopaswin.model import WIRING_VERSION, HoloPASWIN
from holopaswin.propagator import AngularSpectrumPropagator
from holopaswin.resnet_unet import ResNetUNet
from holopaswin.training.config import ARCHITECTURES, ModelSettings, TrainingConfig, load_config

# The iterative classical reconstructions; they share one constructor signature.
ClassicalAlgorithm = type[GerchbergSaxton | LatychevskaiaFink | ConstrainedComplexTV]

# Classical reconstructions: the iterative algorithm and its number of iterations, or None for plain
# back-propagation.
BASELINES: dict[str, tuple[ClassicalAlgorithm | None, int]] = {
    "asm": (None, 0),
    "gs50": (GerchbergSaxton, 50),
    "gs100": (GerchbergSaxton, 100),
    "lf50": (LatychevskaiaFink, 50),
    "lf100": (LatychevskaiaFink, 100),
    "tv50": (ConstrainedComplexTV, 50),
    "tv100": (ConstrainedComplexTV, 100),
}

__all__ = [
    "ARCHITECTURES",
    "BASELINES",
    "Geometry",
    "build_baseline",
    "build_model",
    "load_run",
    "propagator_of",
    "read_geometry",
    "resolve_model",
]


@dataclasses.dataclass(frozen=True)
class Geometry:
    """Imaging geometry of a dataset in SI units."""

    resolution: int  # pixels
    wavelength: float  # m
    pixel_size: float  # m, in the object plane
    z_distance: float  # m, the nominal distance
    numerical_aperture: float | None  # None for a lensless setup


def read_geometry(data_dir: Path) -> Geometry:
    """Read the geometry of a dataset or run folder from its recipe.toml.

    Args:
        data_dir (Path): Folder with the recipe that hologen saved.

    Returns:
        Geometry: The geometry in SI units.

    Raises:
        ValueError: If the recipe lacks one of the geometry values.
    """
    values: dict[str, float] = read_recipe_geometry(data_dir)
    missing: list[str] = [
        key
        for key in ("resolution", "wavelength", "sensor_pixel_size", "magnification", "z_distance")
        if key not in values
    ]
    if missing:
        msg: str = f"{data_dir / 'recipe.toml'} lacks the geometry value(s) {missing}."
        raise ValueError(msg)
    return Geometry(
        resolution=int(values["resolution"]),
        wavelength=values["wavelength"] * MICROMETER,
        pixel_size=values["sensor_pixel_size"] / values["magnification"] * MICROMETER,
        z_distance=values["z_distance"] * MICROMETER,
        numerical_aperture=values.get("numerical_aperture"),
    )


def build_model(settings: ModelSettings, geometry: Geometry) -> nn.Module:
    """Build a network for the geometry of a dataset.

    Args:
        settings (ModelSettings): Architecture, residual mode and pretraining.
        geometry (Geometry): Geometry of the dataset.

    Returns:
        nn.Module: The network, returning (clean field, back-propagated field).
    """
    size: int = geometry.resolution
    physics: tuple[float, float, float] = (geometry.wavelength, geometry.pixel_size, geometry.z_distance)
    if settings.architecture == "holopaswin":
        return HoloPASWIN(
            size,
            *physics,
            use_pretrained=settings.pretrained,
            residual_mode=settings.residual_mode,
            numerical_aperture=geometry.numerical_aperture,
        )
    if settings.architecture == "unet":
        return UNetBaseline(
            size, *physics, residual_mode=settings.residual_mode, numerical_aperture=geometry.numerical_aperture
        )
    if settings.architecture == "resnet_unet":
        return ResNetUNet(
            size, *physics, residual_mode=settings.residual_mode, numerical_aperture=geometry.numerical_aperture
        )
    return HRNet(size, *physics, residual_mode=settings.residual_mode, numerical_aperture=geometry.numerical_aperture)


class _BackPropagation(nn.Module):
    """Back-propagation of sqrt(I) with zero phase, the reconstruction without a network."""

    def __init__(self, propagator: AngularSpectrumPropagator) -> None:
        """Keep the propagator.

        Args:
            propagator (AngularSpectrumPropagator): Propagator at the nominal distance.
        """
        super().__init__()
        self.propagator: AngularSpectrumPropagator = propagator

    def forward(self, hologram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Back-propagate the holograms.

        Args:
            hologram (torch.Tensor): Normalised holograms, shape (B, 1, H, W).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The field as (B, 2, H, W), twice.
        """
        amplitude: torch.Tensor = torch.complex(torch.sqrt(hologram + 1e-8), torch.zeros_like(hologram))
        field: torch.Tensor = self.propagator(amplitude, backward=True)
        two_channel: torch.Tensor = torch.cat([field.real, field.imag], dim=1)
        return two_channel, two_channel


class _WithoutPropagationPhase(nn.Module):
    """A classical reconstruction with the constant phase of back-propagation removed.

    Back-propagating over z multiplies the unscattered light by exp(-i k z). The
    ground truth has no such phase, and it is known from z, so it is removed
    exactly. The networks learn this themselves and are left as they are.
    """

    def __init__(self, reconstruction: nn.Module, wavelength: float, z_distance: float) -> None:
        """Keep the reconstruction and the phase to remove.

        Args:
            reconstruction (nn.Module): Module returning (field, back-propagated field).
            wavelength (float): Wavelength in m.
            z_distance (float): Distance of the back-propagation in m.
        """
        super().__init__()
        self.reconstruction: nn.Module = reconstruction
        # k z is reduced modulo 2 pi in double precision before it meets float32 fields.
        self.phase: float = math.fmod(2 * math.pi / wavelength * z_distance, 2 * math.pi)

    def forward(self, hologram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct and remove the propagation phase from both outputs.

        Args:
            hologram (torch.Tensor): Normalised holograms, shape (B, 1, H, W).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Reconstructed and back-propagated
                fields, each (B, 2, H, W), without the propagation phase.
        """
        outputs: tuple[torch.Tensor, torch.Tensor] = self.reconstruction(hologram)
        cosine: float = math.cos(self.phase)
        sine: float = math.sin(self.phase)
        corrected: list[torch.Tensor] = []
        field: torch.Tensor
        for field in outputs:
            real: torch.Tensor = field[:, 0:1] * cosine - field[:, 1:2] * sine
            imag: torch.Tensor = field[:, 0:1] * sine + field[:, 1:2] * cosine
            corrected.append(torch.cat([real, imag], dim=1))
        return corrected[0], corrected[1]


def build_baseline(name: str, geometry: Geometry) -> nn.Module:
    """Build a classical reconstruction for the geometry of a dataset.

    Args:
        name (str): One of BASELINES: asm (back-propagation only), gs50 or gs100
            (Gerchberg-Saxton), lf50 or lf100 (Latychevskaia-Fink), tv50 or
            tv100 (constrained complex total variation), the iterative ones
            with that many iterations.
        geometry (Geometry): Geometry of the dataset.

    Returns:
        nn.Module: The reconstruction, returning (field, back-propagated field)
            without the constant propagation phase.

    Raises:
        ValueError: If the name is unknown.
    """
    if name not in BASELINES:
        msg: str = f"Unknown baseline {name!r}; choose from {', '.join(BASELINES)}."
        raise ValueError(msg)
    size: int = geometry.resolution
    algorithm: ClassicalAlgorithm | None
    iterations: int
    algorithm, iterations = BASELINES[name]
    reconstruction: nn.Module
    if algorithm is None:
        reconstruction = _BackPropagation(
            AngularSpectrumPropagator(
                (size, size),
                geometry.wavelength,
                geometry.pixel_size,
                geometry.z_distance,
                numerical_aperture=geometry.numerical_aperture,
            )
        )
    else:
        reconstruction = algorithm(
            size,
            geometry.wavelength,
            geometry.pixel_size,
            geometry.z_distance,
            iterations=iterations,
            numerical_aperture=geometry.numerical_aperture,
        )
    return _WithoutPropagationPhase(reconstruction, geometry.wavelength, geometry.z_distance)


def propagator_of(model: nn.Module) -> AngularSpectrumPropagator:
    """Return the propagator of a network or baseline.

    Args:
        model (nn.Module): A module built by build_model or build_baseline.

    Returns:
        AngularSpectrumPropagator: Its propagator at the nominal distance.

    Raises:
        TypeError: If the module has no angular spectrum propagator.
    """
    propagator: object = getattr(model, "propagator", None)
    if not isinstance(propagator, AngularSpectrumPropagator):
        msg: str = f"{type(model).__name__} has no angular spectrum propagator."
        raise TypeError(msg)
    return propagator


def load_weights(model: nn.Module, path: Path, device: torch.device) -> None:
    """Load trained weights into a model, refusing HoloPASWIN weights of another skip wiring.

    The skips around the Swin U-Net have no weights, so weights trained with
    another wiring would load without error and reconstruct the wrong field;
    only the stored wiring version tells them apart.

    Args:
        model (nn.Module): The built model.
        path (Path): The .pth file of its weights.
        device (torch.device): Device to load the weights on.

    Raises:
        ValueError: If HoloPASWIN weights lack the current wiring version.
    """
    state: dict[str, torch.Tensor] = torch.load(path, map_location=device, weights_only=True)
    if isinstance(model, HoloPASWIN):
        stored: torch.Tensor | None = state.get("wiring_version")
        if stored is None or int(stored) != WIRING_VERSION:
            found: str = "no wiring version" if stored is None else f"wiring version {int(stored)}"
            msg: str = (
                f"{path} has {found}, but HoloPASWIN now has skip wiring {WIRING_VERSION}: the "
                "back-propagated field is added once, in residual mode only. Its weights would "
                "reconstruct the wrong field, so retrain the model."
            )
            raise ValueError(msg)
    model.load_state_dict(state)


def load_run(folder: Path, device: torch.device) -> nn.Module:
    """Rebuild a trained model from a run folder of holopaswin train.

    Args:
        folder (Path): Run folder with model.pth, config.toml and recipe.toml.
        device (torch.device): Device to place the model on.

    Returns:
        nn.Module: The model in evaluation mode.

    Raises:
        ValueError: If the weights are HoloPASWIN weights of another skip wiring.
    """
    config: TrainingConfig = load_config(folder / "config.toml")
    # The trained weights replace any pretrained ones, so nothing is downloaded.
    settings: ModelSettings = dataclasses.replace(config.model, pretrained=False)
    model: nn.Module = build_model(settings, read_geometry(folder)).to(device)
    load_weights(model, folder / "model.pth", device)
    return model.eval()


def resolve_model(path: Path, geometry: Geometry, device: torch.device) -> nn.Module:
    """Load a model from a run folder, or a bare HoloPASWIN checkpoint.

    Args:
        path (Path): A run folder of holopaswin train, or a .pth file of a
            HoloPASWIN model.
        geometry (Geometry): Geometry of the test sets; a bare checkpoint is
            built for it.
        device (torch.device): Device to place the model on.

    Returns:
        nn.Module: The model in evaluation mode.

    Raises:
        ValueError: If a run folder was trained at another image size than the
            test sets, or the weights are HoloPASWIN weights of another skip wiring.
    """
    if path.is_dir():
        trained: Geometry = read_geometry(path)
        if trained.resolution != geometry.resolution:
            msg: str = (
                f"{path} was trained on {trained.resolution} px holograms and cannot score "
                f"{geometry.resolution} px test sets."
            )
            raise ValueError(msg)
        return load_run(path, device)
    model: nn.Module = build_model(ModelSettings(pretrained=False), geometry).to(device)
    load_weights(model, path, device)
    return model.eval()
