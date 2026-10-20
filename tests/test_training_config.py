"""Tests for the training configuration and its TOML file."""

import pathlib
import tomllib

import pytest

from holopaswin.loss import LossWeights
from holopaswin.training.config import (
    TrainingConfig,
    config_from_dict,
    config_to_toml,
    default_config,
    load_config,
)


def test_defaults_are_the_published_protocol() -> None:
    """The defaults are the settings of the published HoloPASWIN model."""
    config: TrainingConfig = default_config()
    assert config.model.architecture == "holopaswin"
    assert (config.training.epochs, config.training.batch_size, config.training.learning_rate) == (30, 32, 1e-4)
    assert (config.training.seed, config.training.validation_fraction) == (42, 0.2)
    assert config.loss.weights() == LossWeights()
    assert config.loss.physics == 0.1
    assert config.training.physics_use_true_z is False


def test_toml_round_trip_keeps_every_value(tmp_path: pathlib.Path) -> None:
    """A config written to TOML and read back is unchanged."""
    config: TrainingConfig = config_from_dict({"model": {"architecture": "unet"}, "loss": {"physics": 0.0}})
    path: pathlib.Path = tmp_path / "config.toml"
    path.write_text(config_to_toml(config))
    assert load_config(path) == config


def test_toml_lists_the_source_of_every_value() -> None:
    """Every value is preceded by a comment that says where it comes from."""
    text: str = config_to_toml(default_config())
    assert "[model]" in text
    assert "[loss]" in text
    assert text.count("\n# ") >= 14
    assert tomllib.loads(text)["training"]["epochs"] == 30


def test_partial_file_keeps_the_other_defaults(tmp_path: pathlib.Path) -> None:
    """Values missing from a file keep their defaults."""
    path: pathlib.Path = tmp_path / "config.toml"
    path.write_text("[training]\nepochs = 3\n")
    config: TrainingConfig = load_config(path)
    assert config.training.epochs == 3
    assert config.model == default_config().model


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"optimizer": {"lr": 1.0}}, "Unknown config section"),
        ({"training": {"epoch": 3}}, "Unknown key"),
        ({"model": {"architecture": "vit"}}, "Unknown architecture"),
        ({"training": {"epochs": 0}}, "epochs must be positive"),
        ({"training": {"batch_size": 0}}, "batch_size must be positive"),
        ({"training": {"validation_fraction": 1.0}}, "validation_fraction"),
        ({"loss": {"amplitude": -0.1}}, "must not be negative"),
    ],
)
def test_invalid_configs_are_rejected(data: dict[str, dict[str, object]], message: str) -> None:
    """Typos and impossible values stop with a clear message."""
    with pytest.raises(ValueError, match=message):
        config_from_dict(data)
