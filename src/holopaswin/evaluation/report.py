"""Evaluation of trained networks and classical baselines on one test set.

Every method reconstructs the same holograms, which are scored with the paper's
metrics (holopaswin.evaluation.scoring). The summary gives, for every metric and
method, the mean and standard deviation (the form of the paper's tables) and the
median with a bootstrap interval, overall, per object family and per noise
configuration. The first two methods are compared on the same holograms with
paired Wilcoxon tests.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
from torch import nn

from holopaswin.evaluation.scoring import build_propagator, evaluate_set, metric_columns, write_json
from holopaswin.evaluation.statistics import bootstrap_median, paired_tests
from holopaswin.factory import Geometry, build_baseline, read_geometry, resolve_model

# Groupings of the summary: every hologram, per object family, per noise configuration.
GROUPINGS: tuple[str, ...] = ("all", "object_family", "config_name")


def run_evaluation(  # noqa: PLR0913
    test_dir: Path,
    runs: dict[str, Path],
    baselines: list[str],
    device: torch.device,
    batch_size: int,
    limit: int | None,
) -> pd.DataFrame:
    """Reconstruct a test set with every model and baseline and score it.

    Args:
        test_dir (Path): hologen test set, with its recipe.toml.
        runs (dict[str, Path]): Run folder or checkpoint by model name.
        baselines (list[str]): Classical baselines, any of holopaswin.factory.BASELINES.
        device (torch.device): Device the methods run on.
        batch_size (int): Holograms per forward pass.
        limit (int | None): Holograms to use, or None for all.

    Returns:
        pd.DataFrame: One row per hologram and method, models first in the order
            given, then the baselines.

    Raises:
        ValueError: If no method is given, or a baseline is unknown or repeated.
    """
    if not runs and not baselines:
        msg: str = "Give at least one --model or --baseline."
        raise ValueError(msg)
    if len(set(baselines)) != len(baselines):
        msg = f"Baselines must not repeat (got {baselines})."
        raise ValueError(msg)
    geometry: Geometry = read_geometry(test_dir)
    methods: dict[str, nn.Module] = {name: resolve_model(path, geometry, device) for name, path in runs.items()}
    name: str
    for name in baselines:
        methods[name] = build_baseline(name, geometry).to(device).eval()
    return evaluate_set(
        test_dir, methods, build_propagator(geometry).to(device), references=(), batch_size=batch_size, limit=limit
    )


def describe(values: pd.Series) -> dict[str, float]:
    """Summarise one method's values of one metric.

    Values that are not finite, such as an infinite PSNR, are left out.

    Args:
        values (pd.Series): One value per hologram.

    Returns:
        dict[str, float]: The number of finite values, their mean and standard
            deviation, and their median with a 95 percent bootstrap interval.
    """
    finite: pd.Series = values[np.isfinite(values)]
    if finite.empty:
        missing: float = float("nan")
        return {"n": 0, "mean": missing, "std": missing, "median": missing, "ci_low": missing, "ci_high": missing}
    median: tuple[float, float, float] = bootstrap_median(finite)
    return {
        "n": len(finite),
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
        "median": median[0],
        "ci_low": median[1],
        "ci_high": median[2],
    }


def summarise_evaluation(table: pd.DataFrame, metrics: list[str], methods: list[str]) -> dict[str, Any]:
    """Summarise an evaluation per metric, method and grouping.

    Args:
        table (pd.DataFrame): Output of run_evaluation.
        metrics (list[str]): Metric columns to summarise.
        methods (list[str]): Models in the order given, then baselines; the first
            two are compared.

    Returns:
        dict[str, Any]: Under "metrics", by metric and method, the description of
            every hologram ("all") and per object family and noise configuration.
            Under "compared" the two methods of the paired tests, and under
            "paired_tests", by metric and grouping, the tests of holopaswin.
            evaluation.statistics.paired_tests. The Holm correction runs over the
            groups of one grouping and metric, not across metrics or groupings.
    """
    grouped: pd.DataFrame = table.assign(all="all")
    summary: dict[str, Any] = {"metrics": {}, "compared": methods[:2], "paired_tests": {}}
    metric: str
    for metric in metrics:
        per_method: dict[str, Any] = {}
        method: str
        for method in methods:
            rows: pd.DataFrame = grouped[grouped["method"] == method]
            entry: dict[str, Any] = {"all": describe(rows[metric])}
            grouping: str
            for grouping in GROUPINGS[1:]:
                entry[grouping] = {str(key): describe(group[metric]) for key, group in rows.groupby(grouping)}
            per_method[method] = entry
        summary["metrics"][metric] = per_method
        if len(methods) >= 2:  # noqa: PLR2004
            summary["paired_tests"][metric] = {
                grouping: paired_tests(grouped, metric, methods[0], methods[1], by=grouping) for grouping in GROUPINGS
            }
    return summary


def write_evaluation(table: pd.DataFrame, methods: list[str], output: Path) -> list[str]:
    """Write per_sample.csv and summary.json.

    Args:
        table (pd.DataFrame): Output of run_evaluation.
        methods (list[str]): Methods in the order given.
        output (Path): Folder for the files; created if needed.

    Returns:
        list[str]: The metric columns that were summarised.
    """
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "per_sample.csv", index=False)
    metrics: list[str] = metric_columns(list(table.columns))
    write_json(summarise_evaluation(table, metrics, methods), output / "summary.json")
    return metrics
