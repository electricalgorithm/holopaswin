"""The training configuration, which holds every setting of a training run.

Each value carries its unit and source, and the configuration can be written to
and read from a TOML file, so that a run is reproducible from one file. Values
missing from a file keep their defaults, and unknown sections or keys are
rejected. The configuration holds no geometry. Wavelength, pixel size,
distance, numerical aperture and image size always come from the recipe that
hologen saved with the training set.
"""

import dataclasses
import json
import pathlib
import tomllib
from typing import Any

from holopaswin.loss import LAMBDA_PHYSICS, LossWeights

DEFAULT_WEIGHTS: LossWeights = LossWeights()

ARCHITECTURES: tuple[str, ...] = ("holopaswin", "unet", "resnet_unet", "hrnet")


def _value(default: Any, unit: str, source: str) -> Any:  # noqa: ANN401
    """Declare a configuration value together with its unit and source.

    Args:
        default (Any): Default value of the field.
        unit (str): Unit of the value, empty if it has none.
        source (str): Reason for the default.

    Returns:
        Any: A dataclass field that keeps the unit and source as metadata.
    """
    return dataclasses.field(default=default, metadata={"unit": unit, "source": source})


@dataclasses.dataclass(frozen=True)
class ModelSettings:
    """The network that is trained."""

    architecture: str = _value("holopaswin", "", "Any of: " + ", ".join(ARCHITECTURES))
    residual_mode: bool = _value(
        default=True, unit="", source="The network predicts a correction added to the back-propagated field"
    )
    pretrained: bool = _value(default=True, unit="", source="ImageNet weights for the Swin encoder (HoloPASWIN only)")

    def __post_init__(self) -> None:
        """Check the architecture name.

        Raises:
            ValueError: If the architecture is unknown.
        """
        if self.architecture not in ARCHITECTURES:
            msg: str = f"Unknown architecture {self.architecture!r}; choose from {', '.join(ARCHITECTURES)}."
            raise ValueError(msg)


@dataclasses.dataclass(frozen=True)
class TrainingSettings:
    """Optimisation, data split and the distance used by the physics loss."""

    epochs: int = _value(30, "epochs", "Training length of the published model")
    batch_size: int = _value(32, "holograms per step", "Batch size of the published model")
    learning_rate: float = _value(1e-4, "", "AdamW learning rate of the published model")
    seed: int = _value(42, "", "Seed of the training and validation split")
    validation_fraction: float = _value(0.2, "", "Share of the training set held out for validation")
    physics_use_true_z: bool = _value(
        default=False,
        unit="",
        source="Let the physics loss use each hologram's recorded distance (z_true in the "
        "dataset metadata), for training on holograms recorded at a wrong distance",
    )

    def __post_init__(self) -> None:
        """Check the training settings.

        Raises:
            ValueError: If the epochs or batch size are not positive, or the
                validation fraction is not between 0 and 1.
        """
        if self.epochs <= 0:
            msg: str = f"epochs must be positive (got {self.epochs})."
            raise ValueError(msg)
        if self.batch_size <= 0:
            msg = f"batch_size must be positive (got {self.batch_size})."
            raise ValueError(msg)
        if not 0 < self.validation_fraction < 1:
            msg = f"validation_fraction must lie between 0 and 1 (got {self.validation_fraction})."
            raise ValueError(msg)


@dataclasses.dataclass(frozen=True)
class LossSettings:
    """Weights of the terms of the physics loss."""

    complex: float = _value(DEFAULT_WEIGHTS.complex, "", "L1 on the real and imaginary parts, a numerical stabiliser")
    amplitude: float = _value(DEFAULT_WEIGHTS.amplitude, "", "L1 on the amplitude, the main driver")
    phase: float = _value(DEFAULT_WEIGHTS.phase, "", "L1 on the phase, a structural guide")
    frequency: float = _value(
        DEFAULT_WEIGHTS.frequency, "", "L1 on the log spectra of amplitude and phase, for sharp edges"
    )
    physics: float = _value(LAMBDA_PHYSICS, "", "Intensity consistency of the forward-propagated prediction")

    def __post_init__(self) -> None:
        """Check that no weight is negative.

        Raises:
            ValueError: If a weight is negative.
        """
        field: dataclasses.Field[Any]
        for field in dataclasses.fields(self):
            if getattr(self, field.name) < 0:
                msg: str = f"The {field.name} weight must not be negative (got {getattr(self, field.name)})."
                raise ValueError(msg)

    def weights(self) -> LossWeights:
        """Collect the weights of the supervised terms.

        Returns:
            LossWeights: The weights in the form PhysicsLoss takes them.
        """
        return LossWeights(complex=self.complex, amplitude=self.amplitude, phase=self.phase, frequency=self.frequency)


@dataclasses.dataclass(frozen=True)
class TrainingConfig:
    """A complete, reproducible description of a training run."""

    model: ModelSettings = dataclasses.field(default_factory=ModelSettings)
    training: TrainingSettings = dataclasses.field(default_factory=TrainingSettings)
    loss: LossSettings = dataclasses.field(default_factory=LossSettings)


def default_config() -> TrainingConfig:
    """Return the configuration of the published HoloPASWIN model.

    Returns:
        TrainingConfig: The configuration with the default values.
    """
    return TrainingConfig()


def config_from_dict(data: dict[str, Any]) -> TrainingConfig:
    """Build a configuration from nested dictionaries such as a parsed TOML file.

    Args:
        data (dict[str, Any]): Values by section and key. Missing values keep
            their defaults.

    Returns:
        TrainingConfig: The configuration with the given values.

    Raises:
        ValueError: If a section or key is unknown, or a value is invalid.
    """
    defaults: TrainingConfig = default_config()
    sections: dict[str, Any] = {}
    section_name: str
    values: dict[str, Any]
    for section_name, values in data.items():
        if section_name not in {field.name for field in dataclasses.fields(TrainingConfig)}:
            msg: str = f"Unknown config section [{section_name}]."
            raise ValueError(msg)
        section: Any = getattr(defaults, section_name)
        known: set[str] = {field.name for field in dataclasses.fields(section)}
        unknown: list[str] = sorted(set(values) - known)
        if unknown:
            msg = f"Unknown key(s) {unknown} in [{section_name}]; expected one of {sorted(known)}."
            raise ValueError(msg)
        sections[section_name] = dataclasses.replace(section, **values)
    return dataclasses.replace(defaults, **sections)


def load_config(path: pathlib.Path) -> TrainingConfig:
    """Read a configuration from a TOML file.

    Args:
        path (pathlib.Path): The TOML file.

    Returns:
        TrainingConfig: The configuration in the file.
    """
    with path.open("rb") as file:
        data: dict[str, Any] = tomllib.load(file)
    return config_from_dict(data)


def _toml_value(value: Any) -> str:  # noqa: ANN401
    """Format a configuration value as a TOML literal.

    Args:
        value (Any): A string, number or boolean.

    Returns:
        str: The value as TOML.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)  # a JSON string is a valid TOML basic string, quotes escaped
    return repr(value)


def config_to_toml(config: TrainingConfig) -> str:
    """Write a configuration as TOML, with the unit and source of every value.

    Args:
        config (TrainingConfig): The configuration to write.

    Returns:
        str: The TOML text, ending with a newline.
    """
    lines: list[str] = [
        "# holopaswin training configuration",
        "# Train with: holopaswin train DATA_DIR --config <this file>",
    ]
    section_field: dataclasses.Field[Any]
    for section_field in dataclasses.fields(config):
        section: Any = getattr(config, section_field.name)
        lines += ["", f"[{section_field.name}]"]
        field: dataclasses.Field[Any]
        for field in dataclasses.fields(section):
            source: str = field.metadata.get("source", "")
            unit: str = field.metadata.get("unit", "")
            if source:
                lines.append(f"# {source}")
            line: str = f"{field.name} = {_toml_value(getattr(section, field.name))}"
            lines.append(f"{line}  # {unit}" if unit else line)
    return "\n".join(lines) + "\n"


def with_overrides(
    config: TrainingConfig,
    architecture: str | None = None,
    epochs: int | None = None,
    batch_size: int | None = None,
    seed: int | None = None,
) -> TrainingConfig:
    """Apply command-line flags to a configuration.

    Args:
        config (TrainingConfig): Configuration from a file or the defaults.
        architecture (str | None): Network to train, or None to keep the config's.
        epochs (int | None): Number of epochs, or None to keep the config's.
        batch_size (int | None): Batch size, or None to keep the config's.
        seed (int | None): Seed of the split, or None to keep the config's.

    Returns:
        TrainingConfig: The configuration with the flags applied.

    Raises:
        ValueError: If a flag gives an invalid value.
    """
    model: ModelSettings = config.model
    training: TrainingSettings = config.training
    return dataclasses.replace(
        config,
        model=dataclasses.replace(model, architecture=model.architecture if architecture is None else architecture),
        training=dataclasses.replace(
            training,
            epochs=training.epochs if epochs is None else epochs,
            batch_size=training.batch_size if batch_size is None else batch_size,
            seed=training.seed if seed is None else seed,
        ),
    )
