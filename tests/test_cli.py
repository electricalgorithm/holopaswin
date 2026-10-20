"""End-to-end tests of the holopaswin command-line interface."""

import json
import pathlib

import pandas as pd
import pytest
import torch
from holo_fixtures import COUNT, SIZE, write_dataset
from PIL import Image
from typer.testing import CliRunner, Result

from holopaswin.cli import app
from holopaswin.evaluation.figures import PANEL_PIXELS
from holopaswin.model import HoloPASWIN
from holopaswin.training import trainer
from holopaswin.training.config import default_config, load_config

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


def test_config_writes_the_default_training_configuration(tmp_path: pathlib.Path) -> None:
    """Holopaswin config writes a file that loads as the defaults."""
    path: pathlib.Path = tmp_path / "config.toml"
    result: Result = runner.invoke(app, ["config", "--output", str(path)])
    assert result.exit_code == 0, result.output
    assert load_config(path) == default_config()


def test_train_writes_a_run_and_flags_override_the_config(tmp_path: pathlib.Path) -> None:
    """Flags win over the config file, and the saved config records them."""
    write_dataset(tmp_path / "data", count=8)
    config: pathlib.Path = tmp_path / "config.toml"
    config.write_text('[model]\narchitecture = "unet"\n\n[training]\nepochs = 5\nbatch_size = 2\n')
    run: pathlib.Path = tmp_path / "run"
    result: Result = runner.invoke(
        app,
        ["train", str(tmp_path / "data"), "--config", str(config), "--epochs", "1", "-o", str(run), "--device", "cpu"],
    )
    assert result.exit_code == 0, result.output
    saved = load_config(run / "config.toml")
    assert (saved.model.architecture, saved.training.epochs, saved.training.batch_size) == ("unet", 1, 2)
    assert "Epoch 1/1" in result.output

    resumed: Result = runner.invoke(
        app, ["train", str(tmp_path / "data"), "-o", str(run), "--resume", "--epochs", "2", "--device", "cpu"]
    )
    assert resumed.exit_code == 0, resumed.output
    assert "Epoch 2/2" in resumed.output
    assert pd.read_csv(run / "history.csv")["epoch"].tolist() == [1, 2]


@pytest.mark.parametrize(
    ("arguments", "message"), [(["--architecture", "vit"], "Unknown architecture"), (["--resume"], "nothing to resume")]
)
def test_invalid_training_requests_are_a_clear_error(
    tmp_path: pathlib.Path, arguments: list[str], message: str
) -> None:
    """Bad options stop with exit code 2 and say why."""
    write_dataset(tmp_path / "data", count=8)
    result: Result = runner.invoke(app, ["train", str(tmp_path / "data"), "-o", str(tmp_path / "run"), *arguments])
    assert result.exit_code == 2
    assert message in result.output


def test_evaluate_scores_models_and_baselines_with_breakdowns(
    tmp_path: pathlib.Path, checkpoints: dict[str, pathlib.Path]
) -> None:
    """A checkpoint and two baselines are scored, summarised and compared."""
    write_dataset(tmp_path / "test")
    output: pathlib.Path = tmp_path / "out"
    result: Result = runner.invoke(
        app,
        [
            "evaluate",
            str(tmp_path / "test"),
            "--model",
            f"standard={checkpoints['standard']}",
            "--baseline",
            "asm",
            "--baseline",
            "gs50",
            "-o",
            str(output),
            "--device",
            "cpu",
        ],
    )
    assert result.exit_code == 0, result.output
    table: pd.DataFrame = pd.read_csv(output / "per_sample.csv")
    assert set(table["method"]) == {"standard", "asm", "gs50"}
    assert len(table) == 3 * COUNT
    summary: dict = json.loads((output / "summary.json").read_text())
    psnr: dict = summary["metrics"]["phase_psnr"]
    assert set(psnr["standard"]) == {"all", "object_family", "config_name"}
    assert set(psnr["standard"]["all"]) == {"n", "mean", "std", "median", "ci_low", "ci_high"}
    assert set(psnr["asm"]["config_name"]) == {"camera", "camera_laser_speckle"}
    assert summary["compared"] == ["standard", "asm"]
    assert set(summary["paired_tests"]["phase_psnr"]) == {"all", "object_family", "config_name"}


@pytest.mark.parametrize(
    ("arguments", "message"), [(["--baseline", "gs7"], "Unknown baseline"), ([], "at least one --model or --baseline")]
)
def test_invalid_evaluations_are_a_clear_error(tmp_path: pathlib.Path, arguments: list[str], message: str) -> None:
    """Unknown baselines and an empty method list stop with exit code 2."""
    write_dataset(tmp_path / "test")
    result: Result = runner.invoke(app, ["evaluate", str(tmp_path / "test"), "-o", str(tmp_path / "out"), *arguments])
    assert result.exit_code == 2
    assert message in result.output


@pytest.mark.parametrize(("extra", "rows"), [([], 2), (["--error", "--zoom", "16,16,20"], 4)])
def test_figure_draws_one_row_per_sample(
    tmp_path: pathlib.Path, checkpoints: dict[str, pathlib.Path], extra: list[str], rows: int
) -> None:
    """Two samples with one model and one baseline give a 4-column grid."""
    write_dataset(tmp_path / "test")
    output: pathlib.Path = tmp_path / "figure.png"
    result: Result = runner.invoke(
        app,
        [
            "figure",
            str(tmp_path / "test"),
            "-m",
            f"standard={checkpoints['standard']}",
            "-b",
            "asm",
            "--samples",
            "0,2",
            "-o",
            str(output),
            "--device",
            "cpu",
            *extra,
        ],
    )
    assert result.exit_code == 0, result.output
    width: int
    height: int
    width, height = Image.open(output).size
    assert (width, height) == (4 * PANEL_PIXELS, rows * PANEL_PIXELS)


def test_figure_rejects_samples_outside_the_set(tmp_path: pathlib.Path) -> None:
    """A sample index beyond the test set stops with exit code 2."""
    write_dataset(tmp_path / "test")
    result: Result = runner.invoke(
        app, ["figure", str(tmp_path / "test"), "-b", "asm", "--samples", "0,9", "-o", str(tmp_path / "f.png")]
    )
    assert result.exit_code == 2
    assert "sample 9" in result.output


def test_an_interrupted_training_exits_with_130(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ctrl-C ends the command with the conventional exit code and keeps the weights."""

    def interrupt(*_: object, **__: object) -> float:
        """Stand in for an epoch that the user interrupts.

        Returns:
            float: Never returns.

        Raises:
            KeyboardInterrupt: Always.
        """
        raise KeyboardInterrupt

    monkeypatch.setattr(trainer, "_run_epoch", interrupt)
    write_dataset(tmp_path / "data", count=8)
    result: Result = runner.invoke(
        app, ["train", str(tmp_path / "data"), "-o", str(tmp_path / "run"), "--architecture", "unet", "--device", "cpu"]
    )
    assert result.exit_code == 130
    assert (tmp_path / "run" / "interrupted.pth").is_file()


@pytest.mark.parametrize(
    ("command", "message"),
    [
        (["evaluate", "{data}", "-m", "=x.pth"], "needs a name"),
        (["evaluate", "{data}", "-b", "asm", "-m", "a={data}/missing"], "not found"),
        (["train", "{data}", "--architecture", "unet"], "lacks the geometry value"),
    ],
)
def test_broken_inputs_are_a_clear_error(tmp_path: pathlib.Path, command: list[str], message: str) -> None:
    """Missing names, runs and geometry values stop with exit code 2, not a traceback."""
    write_dataset(tmp_path / "data")
    if command[0] == "train":
        (tmp_path / "data" / "recipe.toml").write_text("[geometry]\nresolution = 64\n")
    arguments: list[str] = [part.format(data=tmp_path / "data") for part in command]
    result: Result = runner.invoke(app, [*arguments, "-o", str(tmp_path / "out")])
    assert result.exit_code == 2
    assert message in result.output
    assert not (tmp_path / "out" / "config.toml").exists()
