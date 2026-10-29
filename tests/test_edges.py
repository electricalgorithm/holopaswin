"""Tests for error paths and less common branches of the package and the CLI."""

import json
import math
import pathlib

import numpy as np
import pandas as pd
import pytest
import torch
from holo_fixtures import write_dataset
from torch import nn
from typer.testing import CliRunner, Result

from holopaswin.baselines.gerchberg_saxton import GerchbergSaxton
from holopaswin.cli import app, select_device
from holopaswin.dataset import HoloDataset, read_nominal_distance
from holopaswin.evaluation.benchmark import synchronize
from holopaswin.evaluation.metrics import background_to_signal_ratio
from holopaswin.evaluation.report import describe
from holopaswin.evaluation.scoring import write_json
from holopaswin.factory import Geometry, build_model, propagator_of, read_geometry, resolve_model
from holopaswin.training.config import ModelSettings, config_from_dict, config_to_toml

runner: CliRunner = CliRunner()


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["figure", "{data}", "--samples", "a,b", "-b", "asm"], "comma-separated integers"),
        (["figure", "{data}", "--samples", "0", "-b", "asm", "--zoom", "1,2"], "3 comma-separated integers"),
        (["figure", "{data}", "--samples", "0", "-b", "asm", "--zoom", "50,50,30"], "does not fit"),
        (["figure", "{data}", "--samples", "0"], "at least one --model or --baseline"),
        (["evaluate", "{data}", "-b", "asm", "-b", "asm"], "must not repeat"),
        (["benchmark", "--architecture", "unet", "-m", "{data}", "--data", "{data}"], "either --model or"),
    ],
)
def test_bad_options_are_a_clear_error(tmp_path: pathlib.Path, arguments: list[str], message: str) -> None:
    """Every rejected option ends with exit code 2 and says why."""
    write_dataset(tmp_path / "data")
    result: Result = runner.invoke(
        app, [*(part.format(data=tmp_path / "data") for part in arguments), "-o", str(tmp_path / "out" / "x")]
    )
    assert result.exit_code == 2
    assert message in result.output


def test_config_prints_when_no_file_is_given() -> None:
    """Without --output the configuration goes to the terminal as TOML."""
    result: Result = runner.invoke(app, ["config"])
    assert result.exit_code == 0
    assert result.output.startswith("# holopaswin training configuration")


def test_benchmark_without_data_times_the_model_only(tmp_path: pathlib.Path) -> None:
    """A run folder carries its own geometry, so --data is optional."""
    write_dataset(tmp_path / "run")
    (tmp_path / "run" / "config.toml").write_text(config_to_toml(config_from_dict({"model": {"architecture": "unet"}})))
    model: nn.Module = build_model(ModelSettings(architecture="unet"), read_geometry(tmp_path / "run"))
    torch.save(model.state_dict(), tmp_path / "run" / "model.pth")
    output: pathlib.Path = tmp_path / "timings" / "benchmark.json"
    result: Result = runner.invoke(
        app, ["benchmark", "-m", str(tmp_path / "run"), "--iterations", "1", "--warmup", "0", "-o", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert "end_to_end" not in json.loads(output.read_text())


def test_a_matching_run_folder_is_loaded(tmp_path: pathlib.Path) -> None:
    """resolve_model loads a run folder trained at the test set's size."""
    write_dataset(tmp_path / "run")
    (tmp_path / "run" / "config.toml").write_text(config_to_toml(config_from_dict({"model": {"architecture": "unet"}})))
    geometry: Geometry = read_geometry(tmp_path / "run")
    torch.save(build_model(ModelSettings(architecture="unet"), geometry).state_dict(), tmp_path / "run" / "model.pth")
    assert not resolve_model(tmp_path / "run", geometry, torch.device("cpu")).training


def test_a_module_without_a_propagator_is_rejected() -> None:
    """propagator_of names the module that has no angular spectrum propagator."""
    with pytest.raises(TypeError, match="Linear has no angular spectrum propagator"):
        propagator_of(nn.Linear(2, 2))


def test_a_recipe_without_a_distance_is_an_error(tmp_path: pathlib.Path) -> None:
    """read_nominal_distance says which value is missing."""
    (tmp_path / "recipe.toml").write_text("[geometry]\nresolution = 64\n")
    with pytest.raises(KeyError, match="z_distance"):
        read_nominal_distance(tmp_path)


def test_a_missing_metadata_file_gives_empty_columns(tmp_path: pathlib.Path) -> None:
    """Datasets without metadata files still index, with missing values."""
    write_dataset(tmp_path / "data")
    (tmp_path / "data" / "metadata_0000.parquet").unlink()
    labels: pd.DataFrame = HoloDataset(str(tmp_path / "data"), target_size=64, img_dim=64).read_metadata(
        ["object_family"]
    )
    assert labels["object_family"].isna().all()


def test_describe_handles_values_that_are_never_finite() -> None:
    """A method whose every value is infinite has no statistics, not a crash."""
    summary: dict[str, float] = describe(pd.Series([math.inf, math.nan]))
    assert summary["n"] == 0
    assert math.isnan(summary["median"])


def test_summaries_are_strict_json(tmp_path: pathlib.Path) -> None:
    """Infinities and NaN in tuples and dicts are written as null."""
    write_json({"a": (1.0, math.inf), "b": {"c": math.nan}}, tmp_path / "s.json")
    assert json.loads((tmp_path / "s.json").read_text()) == {"a": [1.0, None], "b": {"c": None}}


def test_background_to_signal_ratio_on_an_absorbing_object() -> None:
    """With an absorbing disk the ratio is the background spread over the object contrast."""
    truth: np.ndarray = np.ones((32, 32))
    truth[8:24, 8:24] = 0.5
    prediction: np.ndarray = truth.copy()
    prediction[0, :] = 1.1
    background: np.ndarray = prediction[truth > 0.98]
    expected: float = background.std() / np.mean(np.abs(background.mean() - prediction[truth <= 0.98]))
    assert background_to_signal_ratio(prediction, truth) == pytest.approx(expected)


def test_devices_are_chosen_and_synchronised() -> None:
    """A named device is used as given; the default is an available accelerator or the CPU."""
    assert select_device("cpu") == torch.device("cpu")
    chosen: torch.device = select_device(None)
    synchronize(chosen)
    assert chosen.type in {"cuda", "mps", "cpu"}


def test_gerchberg_saxton_describes_itself() -> None:
    """The baseline's repr names its iterations."""
    assert "50" in repr(GerchbergSaxton(32, 532e-9, 0.24e-6, 20e-6, iterations=50))
