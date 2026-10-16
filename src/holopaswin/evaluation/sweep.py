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

import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
import torch
from scipy import stats  # type: ignore[import-untyped]
from torch.utils.data import DataLoader, Subset

from holopaswin.dataset import MICROMETER, HoloDataset, read_recipe_geometry
from holopaswin.evaluation.metrics import compute_all_metrics
from holopaswin.model import HoloPASWIN
from holopaswin.propagator import AngularSpectrumPropagator

BOOTSTRAP_RESAMPLES: int = 4000
REFERENCES: tuple[str, str] = ("backprop_nominal", "backprop_true_z")
SHARED_GEOMETRY: tuple[str, ...] = (
    "resolution",
    "sensor_pixel_size",
    "magnification",
    "numerical_aperture",
    "wavelength",
    "z_distance",
)
# Columns of per_sample.csv that describe a row instead of scoring it.
IDENTIFIERS: tuple[str, ...] = ("method", "offset_um", "index", "z_true_um", "object_family", "config_name")


def bootstrap_median(
    values: npt.ArrayLike, seed: int = 0, resamples: int = BOOTSTRAP_RESAMPLES
) -> tuple[float, float, float]:
    """Compute the median and its 95 percent percentile-bootstrap interval.

    Args:
        values (npt.ArrayLike): One value per hologram.
        seed (int): Seed of the resampling, so that a report can be reproduced.
        resamples (int): Number of bootstrap resamples.

    Returns:
        tuple[float, float, float]: The median, and the 2.5 and 97.5 percentiles
            of the resampled medians.
    """
    data: npt.NDArray[np.float64] = np.asarray(values, dtype=np.float64)
    rng: np.random.Generator = np.random.default_rng(seed)
    medians: npt.NDArray[np.float64] = np.median(rng.choice(data, size=(resamples, data.size)), axis=1)
    return float(np.median(data)), float(np.percentile(medians, 2.5)), float(np.percentile(medians, 97.5))


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


def holm(p_values: list[float]) -> list[float]:
    """Adjust p-values for multiple tests with Holm's step-down method.

    Args:
        p_values (list[float]): Raw p-values.

    Returns:
        list[float]: Adjusted p-values in the same order, each at most 1.
    """
    count: int = len(p_values)
    adjusted: list[float] = [0.0] * count
    running: float = 0.0
    rank: int
    position: int
    for rank, position in enumerate(np.argsort(p_values)):
        running = max(running, min(1.0, (count - rank) * p_values[position]))
        adjusted[position] = running
    return adjusted


def paired_tests(part: pd.DataFrame, metric: str, first: str, second: str) -> dict[str, dict[str, float]]:
    """Compare two models on the same holograms at every offset.

    Holograms whose value is not finite (for example an infinite PSNR) are left
    out. Pairs with equal values carry no sign, so the Wilcoxon test drops them,
    and n_used counts the pairs it kept.

    Args:
        part (pd.DataFrame): Per-sample rows of one object family.
        metric (str): Column to compare.
        first (str): Name of the first model.
        second (str): Name of the second model.

    Returns:
        dict[str, dict[str, float]]: By offset, the median difference (first minus
            second), the number of pairs used, and the raw and Holm-adjusted
            p-values. The Holm adjustment runs over the offsets of this metric.
    """
    tests: dict[str, dict[str, float]] = {}
    offset: float
    group: pd.DataFrame
    for offset, group in part.groupby("offset_um"):
        a: pd.Series = group[group["method"] == first].set_index("index")[metric]
        b: pd.Series = group[group["method"] == second].set_index("index")[metric]
        pairs: pd.DataFrame = pd.concat([a, b], axis=1, keys=["a", "b"], join="inner")
        pairs = pairs[np.isfinite(pairs).all(axis=1)]
        difference: pd.Series = pairs["a"] - pairs["b"]
        nonzero: int = int((difference != 0).sum())
        tests[str(offset)] = {
            "median_difference": float(difference.median()) if len(difference) else float("nan"),
            "n_used": nonzero,
            "p": float(stats.wilcoxon(pairs["a"], pairs["b"]).pvalue) if nonzero else 1.0,
        }
    keys: list[str] = list(tests)
    key: str
    adjusted_value: float
    for key, adjusted_value in zip(keys, holm([tests[k]["p"] for k in keys]), strict=True):
        tests[key]["p_holm"] = adjusted_value
    return tests


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


def parse_model_specs(specs: list[str]) -> dict[str, Path]:
    """Read NAME=CHECKPOINT model specifications.

    Args:
        specs (list[str]): Specifications in the order given.

    Returns:
        dict[str, Path]: Checkpoint by model name, in the order given.

    Raises:
        ValueError: If a specification is not NAME=CHECKPOINT, a name repeats or
            is reserved for a reference, or a checkpoint does not exist.
    """
    checkpoints: dict[str, Path] = {}
    spec: str
    for spec in specs:
        if "=" not in spec:
            msg: str = f"--model expects NAME=CHECKPOINT, got {spec!r}."
            raise ValueError(msg)
        name: str
        path: str
        name, path = spec.split("=", 1)
        if name in checkpoints or name in REFERENCES:
            msg = f"Model name {name!r} is used twice or is reserved for a reference."
            raise ValueError(msg)
        if not Path(path).is_file():
            msg = f"Checkpoint {path} not found."
            raise ValueError(msg)
        checkpoints[name] = Path(path)
    return checkpoints


def build_propagator(geometry: dict[str, float]) -> AngularSpectrumPropagator:
    """Build a full-size propagator at the nominal distance of the sweep sets.

    Args:
        geometry (dict[str, float]): Recipe geometry of the sweep sets, in um.

    Returns:
        AngularSpectrumPropagator: The propagator, with the objective pupil.
    """
    size: int = int(geometry["resolution"])
    return AngularSpectrumPropagator(
        (size, size),
        geometry["wavelength"] * MICROMETER,
        geometry["sensor_pixel_size"] / geometry["magnification"] * MICROMETER,
        geometry["z_distance"] * MICROMETER,
        numerical_aperture=geometry["numerical_aperture"],
    )


def load_models(
    checkpoints: dict[str, Path], geometry: dict[str, float], device: torch.device
) -> dict[str, HoloPASWIN]:
    """Build every model for the nominal distance and load its weights.

    Args:
        checkpoints (dict[str, Path]): Checkpoint by model name.
        geometry (dict[str, float]): Recipe geometry of the sweep sets, in um.
        device (torch.device): Device to place the models on.

    Returns:
        dict[str, HoloPASWIN]: Models in evaluation mode, in the order given.
    """
    models: dict[str, HoloPASWIN] = {}
    name: str
    path: Path
    for name, path in checkpoints.items():
        model: HoloPASWIN = HoloPASWIN(
            int(geometry["resolution"]),
            geometry["wavelength"] * MICROMETER,
            geometry["sensor_pixel_size"] / geometry["magnification"] * MICROMETER,
            geometry["z_distance"] * MICROMETER,
            use_pretrained=False,
            residual_mode=True,
            numerical_aperture=geometry["numerical_aperture"],
        ).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
        models[name] = model.eval()
    return models


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


def score_batch(method: str, prediction: torch.Tensor, target: torch.Tensor) -> list[dict[str, float | str]]:
    """Score one method on a batch of holograms with the paper's metrics.

    Args:
        method (str): Name of the method.
        prediction (torch.Tensor): Reconstructed field, shape (B, 2, H, W) as real
            and imaginary parts.
        target (torch.Tensor): True field, same shape.

    Returns:
        list[dict[str, float | str]]: One row per hologram with every value of
            compute_all_metrics.
    """
    predicted: npt.NDArray[np.float64] = prediction.cpu().numpy().astype(np.float64)
    true: npt.NDArray[np.float64] = target.cpu().numpy().astype(np.float64)
    rows: list[dict[str, float | str]] = []
    index: int
    for index in range(len(true)):
        p_real: npt.NDArray[np.float64] = predicted[index, 0]
        p_imag: npt.NDArray[np.float64] = predicted[index, 1]
        t_real: npt.NDArray[np.float64] = true[index, 0]
        t_imag: npt.NDArray[np.float64] = true[index, 1]
        row: dict[str, float | str] = {"method": method}
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


def evaluate_set(  # noqa: PLR0913
    folder: Path,
    offset: float,
    models: dict[str, HoloPASWIN],
    propagator: AngularSpectrumPropagator,
    batch_size: int,
    limit: int | None,
) -> list[dict[str, float | str]]:
    """Reconstruct one test set with every method and score it.

    Args:
        folder (Path): hologen dataset folder of one offset.
        offset (float): Its distance offset in um.
        models (dict[str, HoloPASWIN]): Trained models by name.
        propagator (AngularSpectrumPropagator): Full-size propagator at the
            nominal distance, on the models' device.
        batch_size (int): Holograms per forward pass.
        limit (int | None): Holograms to use from the set, or None for all.

    Returns:
        list[dict[str, float | str]]: One row per hologram and method.
    """
    size: int = propagator.shape[0]
    device: torch.device = propagator.H.device
    dataset: HoloDataset = HoloDataset(str(folder), target_size=size, img_dim=size, return_z=True)
    labels: pd.DataFrame = dataset.read_metadata(["object_family", "config_name"])
    labels["config_name"] = labels["config_name"].str.split("/").str[0]
    count: int = len(dataset) if limit is None else min(limit, len(dataset))
    loader: DataLoader[tuple[torch.Tensor, ...]] = DataLoader(Subset(dataset, range(count)), batch_size=batch_size)
    rows: list[dict[str, float | str]] = []
    first: int = 0
    batch: list[torch.Tensor]
    with torch.no_grad():
        for batch in loader:
            hologram: torch.Tensor = batch[0].to(device)
            target: torch.Tensor = batch[1].to(device)
            z_true: torch.Tensor = batch[2].to(device)
            scored: list[list[dict[str, float | str]]] = [
                score_batch("backprop_nominal", back_propagate(propagator, hologram, None), target),
                score_batch("backprop_true_z", back_propagate(propagator, hologram, z_true), target),
            ]
            name: str
            model: HoloPASWIN
            for name, model in models.items():
                prediction: torch.Tensor
                prediction, _ = model(hologram)
                scored.append(score_batch(name, prediction, target))
            method_rows: list[dict[str, float | str]]
            for method_rows in scored:
                position: int
                row: dict[str, float | str]
                for position, row in enumerate(method_rows):
                    index: int = first + position
                    row.update(
                        {
                            "offset_um": offset,
                            "index": index,
                            "z_true_um": float(z_true[position]) / MICROMETER,
                            "object_family": str(labels.loc[index, "object_family"]),
                            "config_name": str(labels.loc[index, "config_name"]),
                        }
                    )
                    rows.append(row)
            first += len(hologram)
    return rows


def run_sweep(
    sweep_root: Path, checkpoints: dict[str, Path], device: torch.device, batch_size: int, limit: int | None
) -> pd.DataFrame:
    """Reconstruct and score every sweep set with every method.

    Args:
        sweep_root (Path): Folder with one hologen dataset per offset.
        checkpoints (dict[str, Path]): Checkpoint by model name.
        device (torch.device): Device the models run on.
        batch_size (int): Holograms per forward pass.
        limit (int | None): Holograms to use from each set, or None for all.

    Returns:
        pd.DataFrame: One row per hologram and method, with the IDENTIFIERS
            columns and one column per metric.
    """
    sets: list[tuple[float, Path, dict[str, float]]] = find_sweep_sets(sweep_root)
    geometry: dict[str, float] = sets[0][2]
    models: dict[str, HoloPASWIN] = load_models(checkpoints, geometry, device)
    propagator: AngularSpectrumPropagator = build_propagator(geometry).to(device)
    rows: list[dict[str, float | str]] = []
    offset: float
    folder: Path
    for offset, folder, _ in sets:
        rows.extend(evaluate_set(folder, offset, models, propagator, batch_size, limit))
    return pd.DataFrame(rows)


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
    (output / "summary.json").write_text(json.dumps(summarise(table, metrics, models), indent=2))
    plot_sweep(table, metrics, [*REFERENCES, *models], train_error, output / "z_sweep.png")
    return metrics
