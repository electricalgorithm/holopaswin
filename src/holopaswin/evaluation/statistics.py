"""Summary statistics over per-hologram scores.

Medians come with a percentile-bootstrap interval. Two methods scored on the same
holograms are compared with a Wilcoxon signed-rank test, and several such tests
are corrected with Holm's step-down method.
"""

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
from scipy import stats  # type: ignore[import-untyped]

BOOTSTRAP_RESAMPLES: int = 4000


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


def paired_tests(
    part: pd.DataFrame, metric: str, first: str, second: str, by: str = "offset_um"
) -> dict[str, dict[str, float]]:
    """Compare two methods on the same holograms in every group.

    Holograms whose value is not finite (for example an infinite PSNR) are left
    out. Pairs with equal values carry no sign, so the Wilcoxon test drops them,
    and n_used counts the pairs it kept.

    Args:
        part (pd.DataFrame): Per-sample rows with method, index and the metric.
        metric (str): Column to compare.
        first (str): Name of the first method.
        second (str): Name of the second method.
        by (str): Column that defines the groups, e.g. offset_um; the holograms of
            a group are paired by their index.

    Returns:
        dict[str, dict[str, float]]: By group, the median difference (first minus
            second), the number of pairs used, and the raw and Holm-adjusted
            p-values. The Holm adjustment runs over the groups.
    """
    tests: dict[str, dict[str, float]] = {}
    key: object
    group: pd.DataFrame
    for key, group in part.groupby(by):
        a: pd.Series = group[group["method"] == first].set_index("index")[metric]
        b: pd.Series = group[group["method"] == second].set_index("index")[metric]
        pairs: pd.DataFrame = pd.concat([a, b], axis=1, keys=["a", "b"], join="inner")
        pairs = pairs[np.isfinite(pairs).all(axis=1)]
        difference: pd.Series = pairs["a"] - pairs["b"]
        nonzero: int = int((difference != 0).sum())
        tests[str(key)] = {
            "median_difference": float(difference.median()) if len(difference) else float("nan"),
            "n_used": nonzero,
            "p": float(stats.wilcoxon(pairs["a"], pairs["b"]).pvalue) if nonzero else 1.0,
        }
    names: list[str] = list(tests)
    name: str
    adjusted_value: float
    for name, adjusted_value in zip(names, holm([tests[k]["p"] for k in names]), strict=True):
        tests[name]["p_holm"] = adjusted_value
    return tests
