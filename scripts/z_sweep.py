"""Distance sweep of reconstruction methods.

Every model reconstructs test holograms recorded at a known distance error, while
it is told only the nominal distance, as a user would. Each test set is a hologen
dataset generated with `--z-offset`, for example one folder per offset in
{0, +-1, +-2, +-4, +-6, +-8, +-10} um under one sweep folder.

Two references need no network. Back-propagation at the nominal distance is what
a user gets without one. Back-propagation at the true distance is an oracle that
separates the error caused by the wrong distance from the twin image.

Every hologram is scored with holopaswin.evaluation.metrics.compute_all_metrics,
the metrics of the paper's tables. Back-propagating over a distance z leaves the
known constant phase -k z on the field, which the ground truth does not have, so
the two references are multiplied by exp(i k z) for the distance they used. The
models are scored as they are.

The script writes per_sample.csv (one row per hologram and method), summary.json
(for every metric, medians with 95 percent bootstrap intervals and a Wilcoxon
signed-rank test between the first two models at each offset) and z_sweep.png.

Example:
    uv run python scripts/z_sweep.py --sweep-root ../sweep --output-dir results/z_sweep
        --model standard=checkpoints/standard.pth --model z_augmented=checkpoints/zaug.pth
"""

import argparse
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from holopaswin.dataset import MICROMETER, HoloDataset
from holopaswin.evaluation.metrics import compute_all_metrics
from holopaswin.evaluation.sweep import REFERENCES, bootstrap_median, find_sweep_sets, summarise
from holopaswin.model import HoloPASWIN
from holopaswin.propagator import AngularSpectrumPropagator

# Columns of per_sample.csv that describe a row instead of scoring it.
IDENTIFIERS: tuple[str, ...] = ("method", "offset_um", "index", "z_true_um", "object_family", "config_name")


def parse_arguments() -> argparse.Namespace:
    """Read the command-line arguments.

    Returns:
        argparse.Namespace: Sweep folder, models, output folder and options.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sweep-root", type=Path, required=True, help="folder with one hologen dataset per offset")
    parser.add_argument(
        "--model", action="append", default=[], help="NAME=CHECKPOINT; repeat for several models, first two are tested"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--train-error", type=float, default=6.0, help="half-width of the training z range, um")
    parser.add_argument("--limit", type=int, default=None, help="holograms per set, for a quick check")
    return parser.parse_args()


def select_device() -> torch.device:
    """Choose CUDA, then MPS, then the CPU.

    Returns:
        torch.device: The fastest available device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_models(specs: list[str], size: int, geometry: dict[str, float], device: torch.device) -> dict[str, HoloPASWIN]:
    """Build every model for the nominal distance and load its weights.

    Args:
        specs (list[str]): NAME=CHECKPOINT strings.
        size (int): Input size of the models in pixels, the size of the sweep sets.
        geometry (dict[str, float]): Recipe geometry of the sweep sets, in um.
        device (torch.device): Device to place the models on.

    Returns:
        dict[str, HoloPASWIN]: Models in evaluation mode, in the order given.

    Raises:
        ValueError: If a spec is not NAME=CHECKPOINT or a name repeats.
    """
    models: dict[str, HoloPASWIN] = {}
    spec: str
    for spec in specs:
        if "=" not in spec:
            msg: str = f"--model expects NAME=CHECKPOINT, got {spec!r}."
            raise ValueError(msg)
        name: str
        checkpoint: str
        name, checkpoint = spec.split("=", 1)
        if name in models or name in REFERENCES:
            msg = f"Model name {name!r} is used twice or is reserved."
            raise ValueError(msg)
        model: HoloPASWIN = HoloPASWIN(
            size,
            geometry["wavelength"] * MICROMETER,
            geometry["sensor_pixel_size"] / geometry["magnification"] * MICROMETER,
            geometry["z_distance"] * MICROMETER,
            use_pretrained=False,
            residual_mode=True,
            numerical_aperture=geometry["numerical_aperture"],
        ).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        models[name] = model.eval()
    return models


def method_rows(name: str, prediction: torch.Tensor, target: torch.Tensor) -> list[dict[str, float | str]]:
    """Score one method on a batch of holograms with the paper's metrics.

    Args:
        name (str): Name of the method.
        prediction (torch.Tensor): Reconstructed field, shape (B, 2, H, W) as real
            and imaginary parts.
        target (torch.Tensor): True field, same shape.

    Returns:
        list[dict[str, float | str]]: One row per hologram with every value of
            compute_all_metrics.
    """
    predicted: np.ndarray = prediction.cpu().numpy().astype(np.float64)
    true: np.ndarray = target.cpu().numpy().astype(np.float64)
    rows: list[dict[str, float | str]] = []
    index: int
    for index in range(len(true)):
        p_real: np.ndarray = predicted[index, 0]
        p_imag: np.ndarray = predicted[index, 1]
        t_real: np.ndarray = true[index, 0]
        t_imag: np.ndarray = true[index, 1]
        row: dict[str, float | str] = {"method": name}
        row.update(
            compute_all_metrics(
                np.hypot(p_real, p_imag),
                np.arctan2(p_imag, p_real),
                np.hypot(t_real, t_imag),
                np.arctan2(t_imag, t_real),
                p_real,
                p_imag,
                t_real,
                t_imag,
            )
        )
        rows.append(row)
    return rows


def back_propagate(
    propagator: AngularSpectrumPropagator, hologram: torch.Tensor, z: torch.Tensor | None
) -> torch.Tensor:
    """Back-propagate holograms and undo the constant phase of the propagation.

    Back-propagating sqrt(I) over z multiplies the unscattered light by exp(-i k z),
    a constant phase that the ground truth does not have. It is known from z, so
    it is removed exactly, without looking at the ground truth.

    Args:
        propagator (AngularSpectrumPropagator): Full-size propagator at the nominal distance.
        hologram (torch.Tensor): Normalised holograms, shape (B, 1, H, W).
        z (torch.Tensor | None): Distance of each sample in meters, shape (B,), or
            None for the nominal distance.

    Returns:
        torch.Tensor: The back-propagated field, shape (B, 2, H, W) as real and
            imaginary parts.
    """
    amplitude: torch.Tensor = torch.complex(torch.sqrt(hologram + 1e-8), torch.zeros_like(hologram))
    field: torch.Tensor = propagator(amplitude, backward=True, z=z)
    k: float = 2 * np.pi / propagator.wavelength
    distance: torch.Tensor = (
        torch.full((len(hologram),), propagator.z_distance, dtype=torch.float64) if z is None else z.cpu().double()
    )
    # k z is about 236 rad, so it is reduced modulo 2 pi in double precision on the
    # CPU (some accelerators lack float64) before it meets the float32 field.
    phase: torch.Tensor = torch.remainder(k * distance, 2 * np.pi).to(device=field.device, dtype=torch.float32)
    field = field * torch.exp(1j * phase)[:, None, None, None]
    return torch.cat([field.real, field.imag], dim=1)


def evaluate_set(  # noqa: PLR0913
    folder: Path,
    offset: float,
    models: dict[str, HoloPASWIN],
    propagator: AngularSpectrumPropagator,
    arguments: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, float | str]]:
    """Reconstruct one test set with every method.

    Args:
        folder (Path): hologen dataset folder of one offset.
        offset (float): Its distance offset in um.
        models (dict[str, HoloPASWIN]): Trained models by name.
        propagator (AngularSpectrumPropagator): Full-size propagator at the
            nominal distance, for the two back-propagation references.
        arguments (argparse.Namespace): Batch size and limit.
        device (torch.device): Device the models run on.

    Returns:
        list[dict[str, float | str]]: One row per hologram and method.
    """
    size: int = propagator.shape[0]
    dataset: HoloDataset = HoloDataset(str(folder), target_size=size, img_dim=size, return_z=True)
    labels: pd.DataFrame = dataset.read_metadata(["object_family", "config_name"])
    labels["config_name"] = labels["config_name"].str.split("/").str[0]
    count: int = len(dataset) if arguments.limit is None else min(arguments.limit, len(dataset))
    loader: DataLoader = DataLoader(torch.utils.data.Subset(dataset, range(count)), batch_size=arguments.batch_size)
    rows: list[dict[str, float | str]] = []
    first: int = 0
    batch: list[torch.Tensor]
    with torch.no_grad():
        for batch in loader:
            hologram: torch.Tensor = batch[0].to(device)
            target: torch.Tensor = batch[1].to(device)
            z_true: torch.Tensor = batch[2].to(device)
            batch_rows: list[list[dict[str, float | str]]] = [
                method_rows("backprop_nominal", back_propagate(propagator, hologram, None), target),
                method_rows("backprop_true_z", back_propagate(propagator, hologram, z_true), target),
            ]
            name: str
            model: HoloPASWIN
            for name, model in models.items():
                prediction: torch.Tensor
                prediction, _ = model(hologram)
                batch_rows.append(method_rows(name, prediction, target))
            for method in batch_rows:
                for position, row in enumerate(method):
                    index: int = first + position
                    row.update(
                        {
                            "offset_um": offset,
                            "index": index,
                            "z_true_um": float(z_true[position]) / MICROMETER,
                            "object_family": labels.loc[index, "object_family"],
                            "config_name": labels.loc[index, "config_name"],
                        }
                    )
                    rows.append(row)
            first += len(hologram)
    return rows


def plot(table: pd.DataFrame, metrics: list[str], methods: list[str], train_error: float, path: Path) -> None:
    """Plot the median of every metric against the distance offset.

    Rows are metrics and columns are object families.

    Args:
        table (pd.DataFrame): Rows of per_sample.csv.
        metrics (list[str]): Metric columns to plot.
        methods (list[str]): Methods to draw, references first.
        train_error (float): Half-width of the training range in um, shaded.
        path (Path): PNG file to write.
    """
    families: list[str] = sorted(table["object_family"].unique())
    figure: plt.Figure
    axes: np.ndarray
    figure, axes = plt.subplots(
        len(metrics), len(families), figsize=(5 * len(families), 3.2 * len(metrics)), squeeze=False
    )
    row: int
    metric: str
    for row, metric in enumerate(metrics):
        column: int
        family: str
        for column, family in enumerate(families):
            axis: plt.Axes = axes[row][column]
            axis.axvspan(-train_error, train_error, color="0.92")
            method: str
            for method in methods:
                rows: pd.DataFrame = table[(table["object_family"] == family) & (table["method"] == method)]
                finite: pd.DataFrame = rows[np.isfinite(rows[metric])]
                grouped: pd.core.groupby.SeriesGroupBy = finite.groupby("offset_um")[metric]
                ci: np.ndarray = np.array([bootstrap_median(values)[1:] for _, values in grouped])
                style: str = "--" if method in REFERENCES else "-"
                axis.plot(grouped.median().index, grouped.median().to_numpy(), style, marker="o", ms=3, label=method)
                if len(ci):
                    axis.fill_between(grouped.median().index, ci[:, 0], ci[:, 1], alpha=0.15)
            axis.set_title(f"{metric}, {family.replace('_', ' ')}", fontsize=9)
            axis.set_xlabel("distance offset dz (um)")
            axis.grid(alpha=0.3)
    axes[0][0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)


def main() -> None:
    """Run the sweep and write per_sample.csv, summary.json and z_sweep.png."""
    arguments: argparse.Namespace = parse_arguments()
    device: torch.device = select_device()
    sets: list[tuple[float, Path, dict[str, float]]] = find_sweep_sets(arguments.sweep_root)
    geometry: dict[str, float] = sets[0][2]
    size: int = int(geometry["resolution"])
    models: dict[str, HoloPASWIN] = load_models(arguments.model, size, geometry, device)
    propagator: AngularSpectrumPropagator = AngularSpectrumPropagator(
        (size, size),
        geometry["wavelength"] * MICROMETER,
        geometry["sensor_pixel_size"] / geometry["magnification"] * MICROMETER,
        geometry["z_distance"] * MICROMETER,
        numerical_aperture=geometry["numerical_aperture"],
    ).to(device)
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float | str]] = []
    offset: float
    folder: Path
    for offset, folder, _ in sets:
        print(f"offset {offset:+.1f} um: {folder}", flush=True)
        rows.extend(evaluate_set(folder, offset, models, propagator, arguments, device))
    table: pd.DataFrame = pd.DataFrame(rows)
    table.to_csv(arguments.output_dir / "per_sample.csv", index=False)
    metrics: list[str] = [column for column in table.columns if column not in IDENTIFIERS]
    summary: dict = summarise(table, metrics, list(models))
    (arguments.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    plot(table, metrics, [*REFERENCES, *models], arguments.train_error, arguments.output_dir / "z_sweep.png")
    print(f"Wrote {arguments.output_dir / 'summary.json'} with metrics {metrics}")


if __name__ == "__main__":
    main()
