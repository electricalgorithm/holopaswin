"""Distance sweep: how reconstructions degrade when the assumed distance is wrong.

Each test set is a hologen dataset recorded at a fixed distance offset from the
nominal distance (hologen generate --z-offset). Every model reconstructs it
while told only the nominal distance, as a user would. Two references need no
network. Back-propagation at the nominal distance is what a user gets without
one, and back-propagation at the true distance is an oracle that separates the
error caused by the wrong distance from the twin image. Back-propagating over z
leaves the known constant phase -k z on the field, which the ground truth does
not have, so both references are multiplied by exp(i k z) for the distance they
used. The models are scored as they are.

Every hologram is scored with holopaswin.evaluation.metrics.compute_all_metrics,
the metrics of the paper's tables. The summary reports, for every metric,
reconstruction method and object family, the median at each offset with a
bootstrap interval, and a paired Wilcoxon signed-rank test between the first two
models at each offset with Holm-adjusted p-values.
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
import torch
from torch import nn

from holopaswin.dataset import read_recipe_geometry
from holopaswin.evaluation.scoring import IDENTIFIERS, REFERENCES, build_propagator, evaluate_set, write_json
from holopaswin.evaluation.statistics import bootstrap_median, paired_tests
from holopaswin.models import Geometry, read_geometry, resolve_model

if TYPE_CHECKING:
    from holopaswin.propagator import AngularSpectrumPropagator

SHARED_GEOMETRY: tuple[str, ...] = (
    "resolution",
    "sensor_pixel_size",
    "magnification",
    "numerical_aperture",
    "wavelength",
    "z_distance",
)


def find_sweep_sets(sweep_root: Path) -> list[tuple[float, Path, dict[str, float]]]:
    """Find the test sets and check that they share one setup.

    Args:
        sweep_root (Path): Folder with one hologen dataset per offset.

    Returns:
        list[tuple[float, Path, dict[str, float]]]: Offset in um, folder and
            geometry of each set, sorted by offset.

    Raises:
        ValueError: If no set is found, if two sets have the same offset, if a
            set has a random distance error, or if the sets differ in geometry.
    """
    sets: list[tuple[float, Path, dict[str, float]]] = []
    folder: Path
    for folder in sorted(path.parent for path in sweep_root.glob("*/recipe.toml")):
        geometry: dict[str, float] = read_recipe_geometry(folder)
        if geometry.get("z_error", 0.0) != 0:
            msg: str = f"{folder} has a random distance error; sweep sets need a fixed z_offset only."
            raise ValueError(msg)
        sets.append((geometry.get("z_offset", 0.0), folder, geometry))
    if not sets:
        msg = f"No hologen datasets (recipe.toml) found under {sweep_root}."
        raise ValueError(msg)
    offsets: list[float] = [offset for offset, _, _ in sets]
    if len(set(offsets)) != len(offsets):
        msg = f"Two sweep sets share an offset: {sorted(offsets)}."
        raise ValueError(msg)
    reference: dict[str, float] = sets[0][2]
    for _, folder, geometry in sets:
        different: list[str] = [key for key in SHARED_GEOMETRY if geometry.get(key) != reference.get(key)]
        if different:
            msg = f"{folder} differs from {sets[0][1]} in {different}."
            raise ValueError(msg)
    return sorted(sets, key=lambda entry: entry[0])


def summarise(table: pd.DataFrame, metrics: list[str], models: list[str]) -> dict[str, Any]:
    """Summarise the sweep per metric, object family, method and offset.

    Args:
        table (pd.DataFrame): Per-sample rows with method, offset_um, index,
            object_family and one column per metric.
        metrics (list[str]): Metric columns to summarise.
        models (list[str]): Model names in the order given; the first two are
            compared at every offset.

    Returns:
        dict[str, Any]: By metric, then by family ("all" and each object family), the
            medians with 95 percent intervals by method and offset, the number
            of non-finite values left out by method, and the paired tests.
    """
    summary: dict[str, Any] = {}
    families: list[str] = ["all", *sorted(table["object_family"].unique())]
    metric: str
    for metric in metrics:
        summary[metric] = {}
        family: str
        for family in families:
            part: pd.DataFrame = table if family == "all" else table[table["object_family"] == family]
            medians: dict[str, dict[str, list[float]]] = {}
            non_finite: dict[str, int] = {}
            method: str
            for method in [*REFERENCES, *models]:
                rows: pd.DataFrame = part[part["method"] == method]
                finite: pd.DataFrame = rows[np.isfinite(rows[metric])]
                non_finite[method] = len(rows) - len(finite)
                medians[method] = {
                    str(offset): list(bootstrap_median(group[metric])) for offset, group in finite.groupby("offset_um")
                }
            summary[metric][family] = {
                "median_with_ci": medians,
                "non_finite": non_finite,
                "wilcoxon_first_vs_second_model": (
                    paired_tests(part, metric, models[0], models[1]) if len(models) >= 2 else {}  # noqa: PLR2004
                ),
            }
    return summary


def run_sweep(
    sweep_root: Path, runs: dict[str, Path], device: torch.device, batch_size: int, limit: int | None
) -> pd.DataFrame:
    """Reconstruct and score every sweep set with every method.

    Args:
        sweep_root (Path): Folder with one hologen dataset per offset.
        runs (dict[str, Path]): Run folder or checkpoint by model name.
        device (torch.device): Device the models run on.
        batch_size (int): Holograms per forward pass.
        limit (int | None): Holograms to use from each set, or None for all.

    Returns:
        pd.DataFrame: One row per hologram and method, with the IDENTIFIERS
            columns and one column per metric.
    """
    sets: list[tuple[float, Path, dict[str, float]]] = find_sweep_sets(sweep_root)
    geometry: Geometry = read_geometry(sets[0][1])
    models: dict[str, nn.Module] = {name: resolve_model(path, geometry, device) for name, path in runs.items()}
    propagator: AngularSpectrumPropagator = build_propagator(geometry).to(device)
    tables: list[pd.DataFrame] = []
    offset: float
    folder: Path
    for offset, folder, _ in sets:
        table: pd.DataFrame = evaluate_set(folder, models, propagator, REFERENCES, batch_size, limit)
        table.insert(1, "offset_um", offset)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def plot_sweep(table: pd.DataFrame, metrics: list[str], methods: list[str], train_error: float, path: Path) -> None:
    """Plot the median of every metric against the distance offset.

    Rows are metrics and columns are object families. The training range is shaded.

    Args:
        table (pd.DataFrame): Rows of per_sample.csv.
        metrics (list[str]): Metric columns to plot.
        methods (list[str]): Methods to draw, references first.
        train_error (float): Half-width of the training range in um.
        path (Path): PNG file to write.
    """
    import matplotlib as mpl  # noqa: PLC0415  (imported here so the CLI starts fast)

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    families: list[str] = sorted(table["object_family"].unique())
    figure, axes = plt.subplots(
        len(metrics), len(families), figsize=(5 * len(families), 3.2 * len(metrics)), squeeze=False
    )
    row: int
    metric: str
    for row, metric in enumerate(metrics):
        column: int
        family: str
        for column, family in enumerate(families):
            axis: Any = axes[row][column]
            axis.axvspan(-train_error, train_error, color="0.92")
            method: str
            for method in methods:
                rows: pd.DataFrame = table[(table["object_family"] == family) & (table["method"] == method)]
                finite: pd.DataFrame = rows[np.isfinite(rows[metric])]
                grouped: Any = finite.groupby("offset_um")[metric]
                ci: npt.NDArray[np.float64] = np.array([bootstrap_median(values)[1:] for _, values in grouped])
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


def write_results(table: pd.DataFrame, models: list[str], train_error: float, output: Path) -> list[str]:
    """Write per_sample.csv, summary.json and z_sweep.png.

    Args:
        table (pd.DataFrame): Output of run_sweep.
        models (list[str]): Model names in the order given.
        train_error (float): Half-width of the training range in um, shaded in the figure.
        output (Path): Folder for the files; created if needed.

    Returns:
        list[str]: The metric columns that were summarised.
    """
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "per_sample.csv", index=False)
    metrics: list[str] = [column for column in table.columns if column not in IDENTIFIERS]
    write_json(summarise(table, metrics, models), output / "summary.json")
    plot_sweep(table, metrics, [*REFERENCES, *models], train_error, output / "z_sweep.png")
    return metrics
