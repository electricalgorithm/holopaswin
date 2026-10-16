"""Finding and summarising the test sets of a distance sweep.

Each test set is a hologen dataset recorded at a fixed distance offset from the
nominal distance. Every hologram is scored with
holopaswin.evaluation.metrics.compute_all_metrics, the metrics of the paper's
tables. The summary reports, for every metric, reconstruction method and object
family, the median at each offset with a bootstrap interval, and a paired
Wilcoxon signed-rank test between the first two models at each offset with
Holm-adjusted p-values.
"""

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
from scipy import stats  # type: ignore[import-untyped]

from holopaswin.dataset import read_recipe_geometry

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
