"""End-to-end tests of the holopaswin command-line interface."""

import json
import pathlib

import pandas as pd
import pytest
import torch
from holo_fixtures import COUNT, SIZE, write_dataset
from typer.testing import CliRunner, Result

from holopaswin.cli import app
from holopaswin.model import HoloPASWIN

runner: CliRunner = CliRunner()


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory: pytest.TempPathFactory) -> dict[str, pathlib.Path]:
    """Save two untrained models built for the test geometry.

    Args:
        tmp_path_factory (pytest.TempPathFactory): Source of a shared folder.

    Returns:
        dict[str, pathlib.Path]: Checkpoint path by model name.
    """
    folder: pathlib.Path = tmp_path_factory.mktemp("checkpoints")
    paths: dict[str, pathlib.Path] = {}
    for seed, name in enumerate(("standard", "z_augmented")):
        torch.manual_seed(seed)
        model: HoloPASWIN = HoloPASWIN(SIZE, 532e-9, 0.24e-6, 20e-6, use_pretrained=False, numerical_aperture=0.45)
        paths[name] = folder / f"{name}.pth"
        torch.save(model.state_dict(), paths[name])
    return paths


def test_help_lists_the_sweep_command() -> None:
    """The command has subcommands, like hologen, starting with sweep."""
    result: Result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "sweep" in result.output


def test_sweep_writes_samples_summary_and_figure(tmp_path: pathlib.Path, checkpoints: dict[str, pathlib.Path]) -> None:
    """A sweep over three offsets scores both models and both references."""
    for offset in (-4.0, 0.0, 4.0):
        write_dataset(tmp_path / "sets" / f"offset_{offset:+.0f}", offset)
    output: pathlib.Path = tmp_path / "out"
    result: Result = runner.invoke(
        app,
        [
            "sweep",
            str(tmp_path / "sets"),
            "--model",
            f"standard={checkpoints['standard']}",
            "--model",
            f"z_augmented={checkpoints['z_augmented']}",
            "--output",
            str(output),
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output

    table: pd.DataFrame = pd.read_csv(output / "per_sample.csv")
    assert set(table["method"]) == {"backprop_nominal", "backprop_true_z", "standard", "z_augmented"}
    assert sorted(table["offset_um"].unique()) == [-4.0, 0.0, 4.0]
    assert len(table) == 3 * COUNT * 4
    summary: dict = json.loads((output / "summary.json").read_text())
    assert "phase_psnr" in summary
    assert set(summary["phase_psnr"]) == {"all", "adherent_cells", "red_blood_cells"}
    assert (output / "z_sweep.png").stat().st_size > 0


def test_references_agree_without_a_distance_error(
    tmp_path: pathlib.Path, checkpoints: dict[str, pathlib.Path]
) -> None:
    """At offset 0 the nominal and the true distance are the same, so are the references."""
    write_dataset(tmp_path / "sets" / "offset_+0", 0.0)
    output: pathlib.Path = tmp_path / "out"
    result: Result = runner.invoke(
        app,
        ["sweep", str(tmp_path / "sets"), "--model", f"a={checkpoints['standard']}", "-o", str(output)],
    )
    assert result.exit_code == 0, result.output
    table: pd.DataFrame = pd.read_csv(output / "per_sample.csv")
    nominal: pd.Series = table[table["method"] == "backprop_nominal"]["phase_psnr"].reset_index(drop=True)
    true: pd.Series = table[table["method"] == "backprop_true_z"]["phase_psnr"].reset_index(drop=True)
    pd.testing.assert_series_equal(nominal, true)


@pytest.mark.parametrize(
    ("model", "message"),
    [("standard", "NAME=RUN"), ("backprop_nominal={}", "reserved"), ("a=missing.pth", "not found")],
)
def test_invalid_models_are_a_clear_error(
    tmp_path: pathlib.Path, checkpoints: dict[str, pathlib.Path], model: str, message: str
) -> None:
    """Bad --model values stop with exit code 2 and say why."""
    write_dataset(tmp_path / "sets" / "offset_+0", 0.0)
    result: Result = runner.invoke(
        app,
        ["sweep", str(tmp_path / "sets"), "--model", model.format(checkpoints["standard"]), "-o", str(tmp_path)],
    )
    assert result.exit_code == 2
    assert message in result.output


def test_inconsistent_sets_are_a_clear_error(tmp_path: pathlib.Path, checkpoints: dict[str, pathlib.Path]) -> None:
    """Sets with different geometry stop with exit code 2."""
    write_dataset(tmp_path / "sets" / "a", 0.0)
    write_dataset(tmp_path / "sets" / "b", 2.0, size=32)
    result: Result = runner.invoke(
        app,
        ["sweep", str(tmp_path / "sets"), "--model", f"a={checkpoints['standard']}", "-o", str(tmp_path / "out")],
    )
    assert result.exit_code == 2
    assert "differs" in result.output


def test_sweep_help_hides_the_docstring_details() -> None:
    """The form feed in the docstring keeps Args and Raises out of --help."""
    result: Result = runner.invoke(app, ["sweep", "--help"])
    assert result.exit_code == 0
    assert "wrong distance" in result.output
    assert "Args:" not in result.output
