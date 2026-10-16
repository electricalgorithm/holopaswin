"""Tests for finding the sweep sets and summarising the distance sweep."""

import pathlib

import numpy as np
import pandas as pd
import pytest

from holopaswin.evaluation.sweep import bootstrap_median, find_sweep_sets, holm, summarise


def write_recipe(folder: pathlib.Path, offset: float, error: float = 0.0, resolution: int = 64) -> None:
    """Write the geometry section of a hologen recipe.

    Args:
        folder (pathlib.Path): Dataset folder to create.
        offset (float): z_offset in um.
        error (float): z_error in um.
        resolution (int): Image size in pixels.
    """
    folder.mkdir(parents=True)
    (folder / "recipe.toml").write_text(
        f"[geometry]\nresolution = {resolution}\nz_distance = 20.0  # um\nz_offset = {offset}\nz_error = {error}\n"
    )


def sweep_table(rng: np.random.Generator) -> pd.DataFrame:
    """Build per-sample rows where model b loses 1 dB of phase PSNR per um away from dz = 0.

    Args:
        rng (np.random.Generator): Source of small per-hologram variations.

    Returns:
        pd.DataFrame: Rows like per_sample.csv, two families, five offsets.
    """
    rows: list[dict[str, float | str | int]] = []
    for offset in (-4.0, -2.0, 0.0, 2.0, 4.0):
        for index in range(20):
            family: str = "cells" if index % 2 else "discs"
            base: float = 30.0 + 0.1 * rng.standard_normal()
            for method, slope in (("backprop_nominal", 2.0), ("backprop_true_z", 0.0), ("a", 0.0), ("b", 1.0)):
                rows.append(
                    {
                        "method": method,
                        "offset_um": offset,
                        "index": index,
                        "object_family": family,
                        "phase_psnr": base - slope * abs(offset),
                        "phase_ssim": 0.9,
                    }
                )
    table: pd.DataFrame = pd.DataFrame(rows)
    return table.sample(frac=1.0, random_state=0).reset_index(drop=True)  # out of order on purpose


def test_sweep_sets_are_sorted_by_offset(tmp_path: pathlib.Path) -> None:
    """Sets are found by their recipe and ordered by offset."""
    for name, offset in (("p4", 4.0), ("m4", -4.0), ("zero", 0.0)):
        write_recipe(tmp_path / name, offset)
    assert [offset for offset, _, _ in find_sweep_sets(tmp_path)] == [-4.0, 0.0, 4.0]


@pytest.mark.parametrize(
    ("sets", "message"),
    [
        ([("a", 0.0, 0.0, 64), ("b", 0.0, 0.0, 64)], "share an offset"),
        ([("a", 0.0, 0.0, 64), ("b", 2.0, 0.0, 96)], "differs"),
        ([("a", 0.0, 6.0, 64)], "random distance error"),
        ([], "No hologen datasets"),
    ],
)
def test_inconsistent_sweep_sets_are_rejected(
    tmp_path: pathlib.Path, sets: list[tuple[str, float, float, int]], message: str
) -> None:
    """Duplicated offsets, mixed geometry, random errors and empty folders are errors."""
    for name, offset, error, resolution in sets:
        write_recipe(tmp_path / name, offset, error, resolution)
    with pytest.raises(ValueError, match=message):
        find_sweep_sets(tmp_path)


def test_summary_reports_every_metric_and_pairs_holograms() -> None:
    """Medians per metric and offset; the paired test sees a's 4 dB advantage at 4 um."""
    summary: dict = summarise(sweep_table(np.random.default_rng(0)), ["phase_psnr", "phase_ssim"], ["a", "b"])
    assert set(summary) == {"phase_psnr", "phase_ssim"}
    assert set(summary["phase_psnr"]) == {"all", "cells", "discs"}
    psnr: dict = summary["phase_psnr"]["all"]
    assert psnr["median_with_ci"]["b"]["4.0"][0] == pytest.approx(26.0, abs=0.2)
    assert psnr["median_with_ci"]["a"]["4.0"][0] == pytest.approx(30.0, abs=0.2)
    at_four: dict = psnr["wilcoxon_first_vs_second_model"]["4.0"]
    assert at_four["median_difference"] == pytest.approx(4.0, abs=1e-9)
    assert at_four["n_used"] == 20
    at_zero: dict = psnr["wilcoxon_first_vs_second_model"]["0.0"]
    assert at_zero["n_used"] == 0
    assert at_zero["p"] == 1.0


def test_infinite_values_are_left_out_and_counted() -> None:
    """An infinite PSNR (a perfect match) is excluded from medians and tests, and counted."""
    table: pd.DataFrame = sweep_table(np.random.default_rng(1))
    table.loc[(table["method"] == "a") & (table["index"] == 3), "phase_psnr"] = np.inf
    psnr: dict = summarise(table, ["phase_psnr"], ["a", "b"])["phase_psnr"]["all"]
    assert psnr["non_finite"]["a"] == 5
    assert psnr["wilcoxon_first_vs_second_model"]["4.0"]["n_used"] == 19


def test_holm_adjustment_is_monotone_and_capped() -> None:
    """Holm's step-down adjustment keeps the order and never exceeds 1."""
    adjusted: list[float] = holm([0.01, 0.04, 0.03, 0.5])
    assert adjusted == pytest.approx([0.04, 0.09, 0.09, 0.5])
    assert holm([]) == []


def test_bootstrap_median_brackets_the_median() -> None:
    """The interval contains the median and narrows with more samples."""
    rng: np.random.Generator = np.random.default_rng(1)
    small: tuple[float, float, float] = bootstrap_median(rng.normal(size=50), seed=0)
    large: tuple[float, float, float] = bootstrap_median(rng.normal(size=5000), seed=0)
    assert small[1] <= small[0] <= small[2]
    assert large[2] - large[1] < small[2] - small[1]
    assert bootstrap_median(np.arange(9.0), seed=3) == bootstrap_median(np.arange(9.0), seed=3)
