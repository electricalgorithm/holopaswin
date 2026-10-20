"""Training of HoloPASWIN and the baseline networks under one protocol.

Every architecture is trained with the same loop: the dataset is split into
training and validation sets with the configured seed, the network is optimised
with AdamW against the physics loss, and the model with the lowest validation
loss is kept. A run folder records everything needed to rebuild and continue the
run. It holds the best weights (model.pth), the state for resuming after the
last finished epoch (last.pth, with the history), the configuration
(config.toml), the dataset's geometry (recipe.toml) and a readable copy of the
loss per epoch (history.csv).
"""

import dataclasses
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split

from holopaswin.dataset import HoloDataset
from holopaswin.loss import PhysicsLoss
from holopaswin.models import Geometry, build_model, propagator_of, read_geometry
from holopaswin.training.config import TrainingConfig, config_to_toml, load_config

HISTORY_COLUMNS: list[str] = ["epoch", "train_loss", "val_loss", "seconds"]


def _split(dataset: HoloDataset, config: TrainingConfig) -> tuple[Subset[Any], Subset[Any]]:
    """Split a dataset into training and validation sets with the configured seed.

    Args:
        dataset (HoloDataset): The full training set.
        config (TrainingConfig): Configuration with the seed and validation share.

    Returns:
        tuple[Subset[Any], Subset[Any]]: The training and validation sets.
    """
    validation_size: int = int(config.training.validation_fraction * len(dataset))
    parts: list[Subset[Any]] = random_split(
        dataset,
        [len(dataset) - validation_size, validation_size],
        generator=torch.Generator().manual_seed(config.training.seed),
    )
    return parts[0], parts[1]


def _run_epoch(
    model: nn.Module,
    criterion: PhysicsLoss,
    loader: DataLoader[Any],
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> float:
    """Run one pass over a loader, training when an optimizer is given.

    Args:
        model (nn.Module): The network.
        criterion (PhysicsLoss): The loss.
        loader (DataLoader[Any]): Batches of (hologram, object) or (hologram, object, z).
        device (torch.device): Device of the model.
        optimizer (torch.optim.Optimizer | None): Optimizer for training, or None
            to evaluate without gradients.

    Returns:
        float: Mean loss over the batches.
    """
    model.train(optimizer is not None)
    total: float = 0.0
    batches: int = 0
    batch: list[torch.Tensor]
    with torch.set_grad_enabled(optimizer is not None):
        for batch in loader:
            hologram: torch.Tensor = batch[0].to(device)
            target: torch.Tensor = batch[1].to(device)
            z_true: torch.Tensor | None = batch[2].to(device) if len(batch) > 2 else None  # noqa: PLR2004
            prediction: torch.Tensor
            prediction, _ = model(hologram)
            loss: torch.Tensor = criterion(prediction, target, hologram, z_true=z_true)
            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()  # type: ignore[no-untyped-call]  # untyped in torch's stubs
                optimizer.step()
            total += loss.item()
            batches += 1
    return total / max(batches, 1)


def _save_atomically(value: object, path: Path) -> None:
    """Save with torch.save so that an interruption never leaves a half-written file.

    Args:
        value (object): What to save.
        path (Path): Destination; replaced in one step after the save completes.
    """
    temporary: Path = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _check_resumed_config(config: TrainingConfig, output: Path) -> TrainingConfig:
    """Check that a resumed run keeps its settings, and record a new total of epochs.

    Args:
        config (TrainingConfig): Configuration given for the resumed run.
        output (Path): Run folder with the saved config.toml.

    Returns:
        TrainingConfig: The configuration to continue with.

    Raises:
        ValueError: If anything but the number of epochs differs from the saved
            configuration, which would change the split or the network.
    """
    saved: TrainingConfig = load_config(output / "config.toml")
    same_but_epochs: TrainingConfig = dataclasses.replace(
        config, training=dataclasses.replace(config.training, epochs=saved.training.epochs)
    )
    if same_but_epochs != saved:
        msg: str = (
            f"A resumed run can only change its number of epochs; the configuration given differs from "
            f"{output / 'config.toml'} in other settings."
        )
        raise ValueError(msg)
    (output / "config.toml").write_text(config_to_toml(config))
    return config


def _start(data_dir: Path, config: TrainingConfig, output: Path, resume: bool) -> dict[str, Any] | None:
    """Prepare the run folder, or read the state of the run to resume.

    Args:
        data_dir (Path): Training set folder with recipe.toml.
        config (TrainingConfig): Configuration of the run.
        output (Path): Run folder.
        resume (bool): Whether to continue the run in the folder.

    Returns:
        dict[str, Any] | None: The saved state of the run (model, optimizer,
            random state and history), or None for a new run.

    Raises:
        ValueError: If a new run would overwrite one, there is nothing to resume,
            or a resumed run changes more than its number of epochs.
    """
    if resume:
        if not (output / "last.pth").is_file():
            msg: str = f"{output} holds no training run, so there is nothing to resume."
            raise ValueError(msg)
        _check_resumed_config(config, output)
        state: dict[str, Any] = torch.load(output / "last.pth", map_location="cpu", weights_only=True)
        return state
    if (output / "last.pth").exists() or (output / "model.pth").exists():
        msg = f"{output} already holds a training run; resume it or choose another folder."
        raise ValueError(msg)
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.toml").write_text(config_to_toml(config))
    shutil.copyfile(data_dir / "recipe.toml", output / "recipe.toml")
    return None


def train(  # noqa: PLR0913
    data_dir: Path,
    config: TrainingConfig,
    output: Path,
    device: torch.device,
    resume: bool = False,
    report: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Train a network on a hologen dataset and write its run folder.

    last.pth holds everything needed to continue (weights, optimizer, random
    state and the history) and is replaced in one step after each epoch, so a
    resumed run continues exactly where the last finished epoch ended.
    history.csv is a readable copy of the history.

    Args:
        data_dir (Path): Training set folder, with recipe.toml from hologen.
        config (TrainingConfig): Configuration of the run. When resuming, only
            its number of epochs may differ from the saved one.
        output (Path): Run folder.
        device (torch.device): Device to train on.
        resume (bool): Continue the run in the folder after its last finished epoch.
        report (Callable[[str], None]): Receives one line per finished epoch.

    Returns:
        pd.DataFrame: Loss per epoch, with the HISTORY_COLUMNS.

    Raises:
        KeyboardInterrupt: After saving the current weights to interrupted.pth.
    """
    geometry: Geometry = read_geometry(data_dir)  # checked before anything is written
    state: dict[str, Any] | None = _start(data_dir, config, output, resume)
    rows: list[dict[str, float]] = [] if state is None else list(state["history"])
    dataset: HoloDataset = HoloDataset(
        str(data_dir),
        target_size=geometry.resolution,
        img_dim=geometry.resolution,
        return_z=config.training.physics_use_true_z,
    )
    training_set: Subset[Any]
    validation_set: Subset[Any]
    training_set, validation_set = _split(dataset, config)
    torch.manual_seed(config.training.seed)
    batch_size: int = config.training.batch_size
    training_loader: DataLoader[Any] = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validation_loader: DataLoader[Any] = DataLoader(validation_set, batch_size=batch_size * 2)

    model: nn.Module = build_model(config.model, geometry).to(device)
    criterion: PhysicsLoss = PhysicsLoss(
        propagator_of(model), lambda_physics=config.loss.physics, weights=config.loss.weights()
    ).to(device)
    optimizer: torch.optim.AdamW = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])  # the shuffle continues as in an uninterrupted run
    best: float = min((row["val_loss"] for row in rows), default=float("inf"))

    epoch: int
    for epoch in range(len(rows) + 1, config.training.epochs + 1):
        start: float = time.perf_counter()
        try:
            train_loss: float = _run_epoch(model, criterion, training_loader, device, optimizer)
            val_loss: float = _run_epoch(model, criterion, validation_loader, device, None)
            rows.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "seconds": time.perf_counter() - start}
            )
            if val_loss <= best:
                best = val_loss
                _save_atomically(model.state_dict(), output / "model.pth")
            _save_atomically(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "rng": torch.get_rng_state(),
                    "history": rows,
                },
                output / "last.pth",
            )
        except KeyboardInterrupt:
            _save_atomically(model.state_dict(), output / "interrupted.pth")
            report(f"Interrupted in epoch {epoch}; the current weights are in {output / 'interrupted.pth'}.")
            raise
        pd.DataFrame(rows, columns=HISTORY_COLUMNS).to_csv(output / "history.csv", index=False)
        report(f"Epoch {epoch}/{config.training.epochs}: train {train_loss:.4f}, validation {val_loss:.4f}")
    history: pd.DataFrame = pd.DataFrame(rows, columns=HISTORY_COLUMNS)
    history["epoch"] = history["epoch"].astype(int)
    return history
