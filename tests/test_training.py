"""Tests for training runs and their run folders."""

import pathlib

import pandas as pd
import pytest
import torch
from holo_fixtures import write_dataset
from torch import nn

from holopaswin.dataset import HoloDataset
from holopaswin.models import ARCHITECTURES, load_run
from holopaswin.training import trainer
from holopaswin.training.config import TrainingConfig, config_from_dict, load_config
from holopaswin.training.trainer import train

CPU: torch.device = torch.device("cpu")


def tiny_config(architecture: str = "holopaswin", epochs: int = 1) -> TrainingConfig:
    """Build a configuration small enough for a test.

    Args:
        architecture (str): Network to train.
        epochs (int): Number of epochs.

    Returns:
        TrainingConfig: The configuration, without pretrained weights.
    """
    return config_from_dict(
        {
            "model": {"architecture": architecture, "pretrained": False},
            "training": {"epochs": epochs, "batch_size": 2, "validation_fraction": 0.25},
        }
    )


@pytest.fixture
def data(tmp_path: pathlib.Path) -> pathlib.Path:
    """Write a tiny training set.

    Args:
        tmp_path (pathlib.Path): Folder for the test.

    Returns:
        pathlib.Path: The dataset folder.
    """
    folder: pathlib.Path = tmp_path / "data"
    write_dataset(folder, count=8)
    return folder


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_one_epoch_writes_a_complete_run_folder(data: pathlib.Path, tmp_path: pathlib.Path, architecture: str) -> None:
    """Every architecture trains, and its run folder rebuilds the best model."""
    run: pathlib.Path = tmp_path / "run"
    history: pd.DataFrame = train(data, tiny_config(architecture), run, CPU)
    assert list(history.columns) == ["epoch", "train_loss", "val_loss", "seconds"]
    assert len(history) == 1
    for name in ("model.pth", "last.pth", "config.toml", "recipe.toml", "history.csv"):
        assert (run / name).is_file(), name
    assert load_config(run / "config.toml") == tiny_config(architecture)
    assert (run / "recipe.toml").read_text() == (data / "recipe.toml").read_text()
    model: nn.Module = load_run(run, CPU)
    saved: dict[str, torch.Tensor] = torch.load(run / "model.pth", weights_only=True)
    for name, tensor in saved.items():
        torch.testing.assert_close(model.state_dict()[name], tensor)


def test_resume_continues_the_epoch_count(data: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """A resumed run adds epochs to its history instead of starting over."""
    run: pathlib.Path = tmp_path / "run"
    train(data, tiny_config("unet", epochs=1), run, CPU)
    history: pd.DataFrame = train(data, tiny_config("unet", epochs=3), run, CPU, resume=True)
    assert history["epoch"].tolist() == [1, 2, 3]
    assert pd.read_csv(run / "history.csv")["epoch"].tolist() == [1, 2, 3]


def test_an_existing_run_is_not_overwritten(data: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Training into a folder with a run needs resume=True."""
    run: pathlib.Path = tmp_path / "run"
    train(data, tiny_config("unet"), run, CPU)
    with pytest.raises(ValueError, match="already holds a training run"):
        train(data, tiny_config("unet"), run, CPU)


def test_resume_needs_a_run(data: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Resuming an empty folder is an error, not a fresh start."""
    with pytest.raises(ValueError, match="nothing to resume"):
        train(data, tiny_config("unet"), tmp_path / "empty", CPU, resume=True)


def test_a_resumed_run_equals_an_uninterrupted_one(data: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Weights, optimizer and random state are all restored, so the end result is identical."""
    straight: pathlib.Path = tmp_path / "straight"
    train(data, tiny_config("unet", epochs=2), straight, CPU)
    resumed: pathlib.Path = tmp_path / "resumed"
    train(data, tiny_config("unet", epochs=1), resumed, CPU)
    train(data, tiny_config("unet", epochs=2), resumed, CPU, resume=True)
    expected: dict[str, torch.Tensor] = torch.load(straight / "last.pth", weights_only=True)["model"]
    actual: dict[str, torch.Tensor] = torch.load(resumed / "last.pth", weights_only=True)["model"]
    for name, tensor in expected.items():
        torch.testing.assert_close(actual[name], tensor, atol=0.0, rtol=0.0)
    assert load_config(resumed / "config.toml").training.epochs == 2


def test_a_resume_cannot_change_more_than_the_epochs(data: pathlib.Path, tmp_path: pathlib.Path) -> None:
    """Another seed would change the split, so a resume rejects it."""
    run: pathlib.Path = tmp_path / "run"
    train(data, tiny_config("unet"), run, CPU)
    other_seed: TrainingConfig = config_from_dict(
        {
            "model": {"architecture": "unet", "pretrained": False},
            "training": {"epochs": 2, "batch_size": 2, "validation_fraction": 0.25, "seed": 7},
        }
    )
    with pytest.raises(ValueError, match="only change its number of epochs"):
        train(data, other_seed, run, CPU, resume=True)


def test_an_interruption_saves_the_current_weights(
    data: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Ctrl-C during an epoch leaves interrupted.pth and re-raises."""

    def interrupt(*_: object, **__: object) -> float:
        """Stand in for an epoch that the user interrupts.

        Returns:
            float: Never returns.

        Raises:
            KeyboardInterrupt: Always.
        """
        raise KeyboardInterrupt

    monkeypatch.setattr(trainer, "_run_epoch", interrupt)
    run: pathlib.Path = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        train(data, tiny_config("unet"), run, CPU)
    assert (run / "interrupted.pth").is_file()
    assert not (run / "last.pth").exists()


def test_the_split_is_reproducible_and_follows_the_seed(data: pathlib.Path) -> None:
    """The same seed gives the same validation holograms; another seed other ones."""
    dataset: HoloDataset = HoloDataset(str(data), target_size=64, img_dim=64)
    first: list[int] = trainer._split(dataset, tiny_config())[1].indices  # noqa: SLF001
    again: list[int] = trainer._split(dataset, tiny_config())[1].indices  # noqa: SLF001
    reseeded: TrainingConfig = config_from_dict({"training": {"seed": 3, "validation_fraction": 0.25}})
    other: list[int] = trainer._split(dataset, reseeded)[1].indices  # noqa: SLF001
    assert first == again
    assert len(first) == 2
    assert sorted(first) != sorted(other)
