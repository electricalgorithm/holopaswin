"""Known-answer tests for the reconstruction metrics and the scoring of a batch.

Each metric is checked against a value worked out by hand, so a test fails when a
metric returns a plausible but wrong number, not only when it crashes.
"""

import math
import pathlib
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import pytest
import torch
from holo_fixtures import write_dataset
from torch import nn

from holopaswin.evaluation.metrics import (
    AMPLITUDE_DATA_RANGE,
    PHASE_DATA_RANGE,
    amplitude_mse,
    amplitude_psnr,
    amplitude_ssim,
    background_amplitude_ratio,
    background_to_signal_ratio,
    complex_mse,
    compute_all_metrics,
    frequency_ssim,
    phase_mse,
    phase_psnr,
    phase_ssim,
)
from holopaswin.evaluation.scoring import (
    BACKGROUND_TOLERANCE,
    BACKGROUND_WARMUP,
    DIAGNOSTICS,
    IDENTIFIERS,
    REFERENCES,
    BackgroundCheck,
    back_propagate,
    build_propagator,
    check_background,
    evaluate_set,
    metric_columns,
    score_batch,
)
from holopaswin.models import read_geometry
from holopaswin.propagator import AngularSpectrumPropagator

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]

SIZE: int = 96
CELL_AMPLITUDE: float = 0.95
CELL_PHASE: float = 0.3


def cell_field() -> npt.NDArray[np.complex128]:
    """Build a field of weakly absorbing phase cells on a background of exactly 1.

    The cells sit below the background threshold of the B/S ratio and the
    background above it, so every metric sees both regions.

    Returns:
        npt.NDArray[np.complex128]: The field, shape (SIZE, SIZE).
    """
    y: npt.NDArray[np.int64]
    x: npt.NDArray[np.int64]
    y, x = np.mgrid[:SIZE, :SIZE]
    field: npt.NDArray[np.complex128] = np.ones((SIZE, SIZE), dtype=np.complex128)
    cy: int
    cx: int
    radius: int
    for cy, cx, radius in [(25, 30, 9), (60, 50, 12), (35, 75, 7)]:
        cell: npt.NDArray[np.bool_] = (y - cy) ** 2 + (x - cx) ** 2 <= radius**2
        field[cell] = CELL_AMPLITUDE * np.exp(1j * CELL_PHASE)
    return field


def noisy(field: npt.NDArray[np.complex128], sigma: float = 0.01, seed: int = 0) -> npt.NDArray[np.complex128]:
    """Add complex Gaussian noise, a stand-in for a good but imperfect reconstruction.

    Args:
        field (npt.NDArray[np.complex128]): The clean field.
        sigma (float): Standard deviation of the real and of the imaginary noise.
        seed (int): Seed of the noise.

    Returns:
        npt.NDArray[np.complex128]: The noisy field.
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    noise: npt.NDArray[np.complex128] = rng.standard_normal(field.shape) + 1j * rng.standard_normal(field.shape)
    return field + sigma * noise


def metrics_of(prediction: npt.NDArray[np.complex128], truth: npt.NDArray[np.complex128]) -> dict[str, float]:
    """Score a complex prediction with every metric, as score_batch does.

    Args:
        prediction (npt.NDArray[np.complex128]): Predicted field.
        truth (npt.NDArray[np.complex128]): True field.

    Returns:
        dict[str, float]: The values of compute_all_metrics.
    """
    return compute_all_metrics(
        np.abs(prediction),
        np.angle(prediction),
        np.abs(truth),
        np.angle(truth),
        prediction.real,
        prediction.imag,
        truth.real,
        truth.imag,
    )


# PSNR -----------------------------------------------------------------------


def test_psnr_of_identical_images_is_infinite() -> None:
    """A perfect reconstruction has no error, so its PSNR is infinite."""
    truth: npt.NDArray[np.complex128] = cell_field()
    assert math.isinf(phase_psnr(np.angle(truth), np.angle(truth)))
    assert math.isinf(amplitude_psnr(np.abs(truth), np.abs(truth)))


@pytest.mark.parametrize("offset", [0.01, 0.1, 0.5])
def test_phase_psnr_of_a_constant_offset(offset: float) -> None:
    """An offset d everywhere has MSE d^2, so PSNR = 10 log10((2 pi)^2 / d^2)."""
    phase: npt.NDArray[np.float64] = np.angle(cell_field()).astype(np.float64)
    expected: float = 10 * math.log10(PHASE_DATA_RANGE**2 / offset**2)
    assert phase_psnr(phase + offset, phase) == pytest.approx(expected, rel=1e-9)


@pytest.mark.parametrize("offset", [0.01, 0.1, 0.5])
def test_amplitude_psnr_of_a_constant_offset(offset: float) -> None:
    """An offset d everywhere has MSE d^2, so PSNR = 10 log10(1.2^2 / d^2)."""
    amplitude: npt.NDArray[np.float64] = np.abs(cell_field())
    expected: float = 10 * math.log10(AMPLITUDE_DATA_RANGE**2 / offset**2)
    assert amplitude_psnr(amplitude + offset, amplitude) == pytest.approx(expected, rel=1e-9)


def test_doubled_amplitude_gives_an_amplitude_psnr_near_1_6_db() -> None:
    """Doubling a field of background 1 gives an amplitude MSE near 1, about 1.6 dB.

    The phase is untouched, so its PSNR stays infinite. An amplitude PSNR near
    1.6 dB next to a high phase PSNR therefore points to a global amplitude scale
    of about 2 between prediction and truth, such as a network output that adds
    the back-propagated field to a field that already contains it.
    """
    truth: npt.NDArray[np.complex128] = cell_field()
    doubled: dict[str, float] = metrics_of(2 * truth, truth)
    mse: float = float(np.mean(np.abs(truth) ** 2))
    assert doubled["amp_psnr"] == pytest.approx(10 * math.log10(AMPLITUDE_DATA_RANGE**2 / mse), rel=1e-9)
    assert 1.5 < doubled["amp_psnr"] < 1.7
    assert math.isinf(doubled["phase_psnr"])


def test_phase_psnr_ignores_the_wrap_at_pi() -> None:
    """Phases just above -pi and just below pi differ by 0.02 rad, not by 2 pi."""
    truth: npt.NDArray[np.float64] = np.full((SIZE, SIZE), math.pi - 0.01)
    prediction: npt.NDArray[np.float64] = np.full((SIZE, SIZE), -math.pi + 0.01)
    assert phase_psnr(prediction, truth) == pytest.approx(10 * math.log10(PHASE_DATA_RANGE**2 / 0.02**2))


def test_phase_psnr_matches_plain_psnr_away_from_the_wrap() -> None:
    """Without a crossing of +-pi, the wrapped error is the plain difference."""
    truth: npt.NDArray[np.complex128] = cell_field()
    prediction: npt.NDArray[np.complex128] = noisy(truth, sigma=0.05)
    error: npt.NDArray[np.float64] = np.angle(prediction) - np.angle(truth)
    expected: float = 10 * math.log10(PHASE_DATA_RANGE**2 / float(np.mean(error**2)))
    assert phase_psnr(np.angle(prediction), np.angle(truth)) == pytest.approx(expected, rel=1e-12)


# SSIM -----------------------------------------------------------------------


def test_ssim_of_identical_images_is_one() -> None:
    """SSIM reaches its maximum of 1 only for identical images."""
    truth: npt.NDArray[np.complex128] = cell_field()
    assert phase_ssim(np.angle(truth), np.angle(truth)) == pytest.approx(1.0)
    assert amplitude_ssim(np.abs(truth), np.abs(truth)) == pytest.approx(1.0)


def test_ssim_falls_with_noise() -> None:
    """More noise on the same field must give a lower SSIM."""
    truth: npt.NDArray[np.complex128] = cell_field()
    low: npt.NDArray[np.complex128] = noisy(truth, sigma=0.01)
    high: npt.NDArray[np.complex128] = noisy(truth, sigma=0.1)
    assert amplitude_ssim(np.abs(high), np.abs(truth)) < amplitude_ssim(np.abs(low), np.abs(truth)) < 1.0
    assert phase_ssim(np.angle(high), np.angle(truth)) < phase_ssim(np.angle(low), np.angle(truth)) < 1.0


# Frequency SSIM -------------------------------------------------------------


def test_frequency_ssim_of_identical_images_is_one() -> None:
    """Identical amplitudes have identical spectra."""
    amplitude: npt.NDArray[np.float64] = np.abs(cell_field())
    assert frequency_ssim(amplitude, amplitude) == pytest.approx(1.0)


def test_frequency_ssim_ignores_where_the_objects_are() -> None:
    """A circular shift changes only the phase of the spectrum, so Freq SSIM stays 1 where amplitude SSIM falls.

    This holds only for the magnitude of the full-image spectrum, so it pins
    that Freq SSIM compares |FFT| and not the image itself.
    """
    truth: npt.NDArray[np.float64] = np.abs(cell_field())
    shifted: npt.NDArray[np.float64] = np.roll(truth, (17, -9), axis=(0, 1))
    assert frequency_ssim(shifted, truth) == pytest.approx(1.0, abs=1e-9)
    # The weak cells (0.95 on 1) move, which costs the image SSIM about 0.07.
    assert amplitude_ssim(shifted, truth) < 0.95


def test_frequency_ssim_falls_with_noise() -> None:
    """Noise spreads power over all frequencies, so the spectra drift apart."""
    truth: npt.NDArray[np.float64] = np.abs(cell_field())
    low: float = frequency_ssim(np.abs(noisy(cell_field(), sigma=0.01)), truth)
    high: float = frequency_ssim(np.abs(noisy(cell_field(), sigma=0.1)), truth)
    assert high < low < 1.0


# MSE ------------------------------------------------------------------------


@pytest.mark.parametrize("offset", [0.01, 0.1, 0.5])
def test_amplitude_mse_of_a_constant_offset(offset: float) -> None:
    """An offset d everywhere has an MSE of d^2.

    Args:
        offset (float): Constant amplitude error.
    """
    amplitude: npt.NDArray[np.float64] = np.abs(cell_field())
    assert amplitude_mse(amplitude + offset, amplitude) == pytest.approx(offset**2, rel=1e-9)


def test_phase_mse_uses_the_wrapped_error() -> None:
    """Phases of pi - 0.01 and -pi + 0.01 are 0.02 rad apart, an MSE of 0.0004."""
    truth: npt.NDArray[np.float64] = np.full((SIZE, SIZE), math.pi - 0.01)
    prediction: npt.NDArray[np.float64] = np.full((SIZE, SIZE), -math.pi + 0.01)
    assert phase_mse(prediction, truth) == pytest.approx(0.02**2, rel=1e-9)


def test_mse_is_kept_for_a_perfect_sample() -> None:
    """A perfect sample has an infinite PSNR but a finite MSE of 0, so both MSEs are reported."""
    truth: npt.NDArray[np.complex128] = cell_field()
    perfect: dict[str, float] = metrics_of(truth, truth)
    assert perfect["amp_mse"] == 0.0
    assert perfect["phase_mse"] == 0.0
    assert math.isinf(perfect["amp_psnr"])


# Complex MSE ----------------------------------------------------------------


def test_complex_mse_is_the_mean_squared_modulus_of_the_error() -> None:
    """The real and imaginary MSEs add up to mean(|prediction - truth|^2)."""
    truth: npt.NDArray[np.complex128] = cell_field()
    prediction: npt.NDArray[np.complex128] = noisy(truth, sigma=0.05)
    expected: float = float(np.mean(np.abs(prediction - truth) ** 2))
    value: float = complex_mse(prediction.real, prediction.imag, truth.real, truth.imag)
    assert value == pytest.approx(expected, rel=1e-12)


def test_complex_mse_sees_a_global_phase() -> None:
    """Unlike the amplitude metrics, complex MSE punishes a constant phase: |e^(i a) - 1|^2 = 2 - 2 cos a."""
    truth: npt.NDArray[np.complex128] = np.ones((SIZE, SIZE), dtype=np.complex128)
    angle: float = 0.5
    value: float = complex_mse(*_parts(truth * np.exp(1j * angle)), *_parts(truth))
    assert value == pytest.approx(2 - 2 * math.cos(angle), rel=1e-12)


def _parts(field: npt.NDArray[np.complex128]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Split a complex field into its real and imaginary parts.

    Args:
        field (npt.NDArray[np.complex128]): The field.

    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: Real and imaginary parts.
    """
    return field.real, field.imag


# B/S ratio ------------------------------------------------------------------


def test_bs_ratio_of_a_perfect_reconstruction_is_zero() -> None:
    """A flat predicted background has no fluctuation, so B/S is 0."""
    amplitude: npt.NDArray[np.float64] = np.abs(cell_field())
    assert background_to_signal_ratio(amplitude, amplitude) == pytest.approx(0.0, abs=1e-12)


def test_bs_ratio_by_hand() -> None:
    """B/S = std(background) / mean(|background mean - signal|), on a hand-built image.

    The left half is background and the right half object. The predicted
    background alternates rows of 1.1 and 0.9, an even number of each, so its
    mean is 1 and its standard deviation 0.1 exactly; the object sits at 0.5,
    so the contrast is 0.5 and B/S = 0.1 / 0.5 = 0.2.
    """
    truth: npt.NDArray[np.float64] = np.ones((SIZE, SIZE))
    truth[:, SIZE // 2 :] = 0.5
    prediction: npt.NDArray[np.float64] = np.full((SIZE, SIZE), 0.5)
    prediction[:, : SIZE // 2] = np.where(np.arange(SIZE) % 2 == 0, 1.1, 0.9)[:, None]
    # The 1e-8 guard in the denominator shifts the ratio by 2e-8 relative.
    assert background_to_signal_ratio(prediction, truth) == pytest.approx(0.2, rel=1e-7)


@pytest.mark.parametrize("level", [0.5, 1.0])
def test_bs_ratio_is_undefined_without_both_regions(level: float) -> None:
    """A field with no background (0.5) or no object (1.0) cannot be scored, so B/S must not report a perfect 0.

    Args:
        level (float): Amplitude of the uniform true field.
    """
    truth: npt.NDArray[np.float64] = np.full((SIZE, SIZE), level)
    prediction: npt.NDArray[np.float64] = noisy(truth.astype(np.complex128), sigma=0.1).real
    assert math.isnan(background_to_signal_ratio(prediction, truth))


# Background amplitude -------------------------------------------------------


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_background_ratio_measures_a_global_scale(scale: float) -> None:
    """Scaling a reconstruction by s scales its background, so the ratio is s.

    Args:
        scale (float): Global amplitude scale of the prediction.
    """
    truth: npt.NDArray[np.complex128] = cell_field()
    assert metrics_of(scale * truth, truth)["bg_amp_ratio"] == pytest.approx(scale, rel=1e-12)


def test_background_ratio_is_robust_to_twin_image_ripples() -> None:
    """Symmetric ripples around the true background leave the median at 1."""
    truth: npt.NDArray[np.float64] = np.abs(cell_field())
    ripples: npt.NDArray[np.float64] = 0.2 * np.sin(np.arange(SIZE) / 3.0)[None, :] * np.ones((SIZE, 1))
    assert background_amplitude_ratio(truth + ripples, truth) == pytest.approx(1.0, abs=0.02)


def test_background_ratio_is_undefined_without_a_background() -> None:
    """A field that is all object has no background to compare."""
    truth: npt.NDArray[np.float64] = np.full((SIZE, SIZE), 0.5)
    assert math.isnan(background_amplitude_ratio(truth, truth))


# Invariances ----------------------------------------------------------------


@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_scale_errors_are_visible_only_in_amplitude_metrics(scale: float) -> None:
    """A global amplitude scale leaves phase metrics and B/S untouched.

    arg(s f) = arg(f), and B/S divides a standard deviation by a contrast, so
    both cancel s. Only the amplitude metrics and complex MSE can catch a scale
    error; a table without them can look healthy for a reconstruction on the
    wrong scale.

    Args:
        scale (float): Global amplitude scale applied to a good reconstruction.
    """
    truth: npt.NDArray[np.complex128] = cell_field()
    good: dict[str, float] = metrics_of(noisy(truth), truth)
    scaled: dict[str, float] = metrics_of(scale * noisy(truth), truth)
    name: str
    for name in ("phase_psnr", "phase_ssim"):
        assert scaled[name] == pytest.approx(good[name], rel=1e-12), name
    # The 1e-8 guard in the denominator of B/S keeps the cancellation from being exact.
    assert scaled["bs_ratio"] == pytest.approx(good["bs_ratio"], rel=1e-6)
    # Even the milder error, s = 0.5, costs tens of dB, far beyond the noise of a good fit.
    assert scaled["amp_psnr"] < good["amp_psnr"] - 20
    assert scaled["amp_ssim"] < good["amp_ssim"]
    assert scaled["complex_mse"] > good["complex_mse"]


# Scoring a batch ------------------------------------------------------------


def as_batch(field: npt.NDArray[np.complex128]) -> torch.Tensor:
    """Stack a field as a batch of one in the (B, 2, H, W) real and imaginary layout.

    Args:
        field (npt.NDArray[np.complex128]): The field.

    Returns:
        torch.Tensor: Shape (1, 2, H, W), float32.
    """
    return torch.from_numpy(np.stack([field.real, field.imag])[None]).float()


def test_score_batch_matches_compute_all_metrics() -> None:
    """score_batch must convert (B, 2, H, W) to amplitude and phase exactly as the metrics expect."""
    truth: npt.NDArray[np.complex128] = cell_field()
    prediction: npt.NDArray[np.complex128] = noisy(truth, sigma=0.05)
    rows: list[dict[str, float | str]] = score_batch("method", as_batch(prediction), as_batch(truth))
    expected: dict[str, float] = metrics_of(
        prediction.astype(np.complex64).astype(np.complex128), truth.astype(np.complex64).astype(np.complex128)
    )
    assert len(rows) == 1
    assert rows[0]["method"] == "method"
    name: str
    for name, value in expected.items():
        assert rows[0][name] == pytest.approx(value, rel=1e-6), name


def test_score_batch_keeps_samples_apart() -> None:
    """Each row scores its own hologram; a perfect and a doubled sample must not be averaged."""
    truth: npt.NDArray[np.complex128] = cell_field()
    target: torch.Tensor = torch.cat([as_batch(truth), as_batch(truth)])
    prediction: torch.Tensor = torch.cat([as_batch(truth), as_batch(2 * truth)])
    rows: list[dict[str, float | str]] = score_batch("method", prediction, target)
    assert math.isinf(float(rows[0]["amp_psnr"]))
    assert 1.5 < float(rows[1]["amp_psnr"]) < 1.7


@pytest.mark.parametrize("scale", [0.8, 1.2, 2.0])
def test_check_background_rejects_a_scaled_method(scale: float) -> None:
    """A method whose background is off by more than the tolerance stops the evaluation.

    Args:
        scale (float): Global amplitude scale of the prediction.
    """
    truth: npt.NDArray[np.complex128] = cell_field()
    rows: list[dict[str, float | str]] = score_batch("scaled", as_batch(scale * truth), as_batch(truth))
    with pytest.raises(ValueError, match="scaled"):
        check_background("scaled", rows, BACKGROUND_TOLERANCE)


def test_check_background_accepts_a_good_method_and_skips_undefined_samples() -> None:
    """A small scale within the tolerance passes, and samples without a background are ignored."""
    truth: npt.NDArray[np.complex128] = cell_field()
    rows: list[dict[str, float | str]] = score_batch("good", as_batch(1.05 * noisy(truth)), as_batch(truth))
    rows.append({"method": "good", "bg_amp_ratio": float("nan")})
    check_background("good", rows, BACKGROUND_TOLERANCE)


def test_check_background_warns_when_nothing_can_be_checked() -> None:
    """A method whose samples all lack a background passes, but not silently."""
    with pytest.warns(RuntimeWarning, match="not checked"):
        check_background("blind", ratio_rows("blind", [float("nan")] * 3), BACKGROUND_TOLERANCE)


def ratio_rows(method: str, ratios: list[float]) -> list[dict[str, float | str]]:
    """Build scored rows that hold only a background amplitude ratio.

    Args:
        method (str): Name of the method.
        ratios (list[float]): Ratio of each sample.

    Returns:
        list[dict[str, float | str]]: One row per sample.
    """
    return [{"method": method, "bg_amp_ratio": ratio} for ratio in ratios]


def test_background_check_tolerates_a_hard_sample() -> None:
    """The median of the warm-up decides, so one sample on another scale does not stop a good method."""
    check: BackgroundCheck = BackgroundCheck()
    check.add(ratio_rows("good", [2.0]))
    check.add(ratio_rows("good", [1.0] * (BACKGROUND_WARMUP - 1)))
    check.finish()


def test_background_check_stops_as_soon_as_the_warm_up_is_full() -> None:
    """A method on the wrong scale stops at its warm-up sample, before the rest of the set is reconstructed."""
    check: BackgroundCheck = BackgroundCheck()
    check.add(ratio_rows("doubled", [2.0] * (BACKGROUND_WARMUP - 1)))
    with pytest.raises(ValueError, match="doubled"):
        check.add(ratio_rows("doubled", [2.0]))


def test_background_check_of_a_small_set_happens_at_the_end() -> None:
    """A set smaller than the warm-up is checked once all of it is scored."""
    check: BackgroundCheck = BackgroundCheck()
    check.add(ratio_rows("doubled", [2.0] * 3))
    with pytest.raises(ValueError, match="doubled"):
        check.finish()


def test_background_check_rechecks_the_whole_set_at_the_end() -> None:
    """A scale error in the later samples passes the warm-up but not the median over the whole set."""
    check: BackgroundCheck = BackgroundCheck()
    check.add(ratio_rows("late", [1.0] * BACKGROUND_WARMUP))
    check.add(ratio_rows("late", [2.0] * (BACKGROUND_WARMUP + 8)))
    with pytest.raises(ValueError, match=f"late.*of {2 * BACKGROUND_WARMUP + 8} samples"):
        check.finish()


def test_background_check_warns_once_for_a_method_without_background() -> None:
    """The warm-up and the final check see the same missing background, but report it once."""
    check: BackgroundCheck = BackgroundCheck()
    check.add(ratio_rows("blind", [float("nan")] * (BACKGROUND_WARMUP + 4)))
    with pytest.warns(RuntimeWarning, match="not checked") as warned:
        check.finish()
    assert len(warned) == 1


def test_background_check_leaves_the_references_alone() -> None:
    """Back-propagation is what the methods are compared against, not a method under test."""
    check: BackgroundCheck = BackgroundCheck()
    name: str
    for name in REFERENCES:
        check.add(ratio_rows(name, [2.0] * (2 * BACKGROUND_WARMUP)))
    check.finish()


def test_diagnostics_are_not_summarised_as_metrics() -> None:
    """The background ratio checks the scale; it is not a quality to average or test."""
    truth: npt.NDArray[np.complex128] = cell_field()
    rows: list[dict[str, float | str]] = score_batch("method", as_batch(truth), as_batch(truth))
    row: dict[str, float | str] = {**rows[0], "index": 0, "z_true_um": 20.0}
    columns: list[str] = metric_columns(list(row))
    assert set(DIAGNOSTICS).isdisjoint(columns)
    assert set(IDENTIFIERS).isdisjoint(columns)
    assert {"amp_psnr", "phase_psnr", "freq_ssim", "bs_ratio", "amp_mse", "phase_mse"} <= set(columns)


class _Doubled(nn.Module):
    """Back-propagation with its amplitude doubled, a method on the wrong scale."""

    def __init__(self, propagator: AngularSpectrumPropagator) -> None:
        """Keep the propagator.

        Args:
            propagator (AngularSpectrumPropagator): Full-size propagator at the nominal distance.
        """
        super().__init__()
        self.propagator: AngularSpectrumPropagator = propagator

    def forward(self, hologram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return twice the back-propagated field.

        Args:
            hologram (torch.Tensor): Normalised holograms, shape (B, 1, H, W).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The doubled field and the plain one.
        """
        field: torch.Tensor = back_propagate(self.propagator, hologram, None)
        return 2 * field, field


def test_evaluate_set_stops_on_a_method_with_the_wrong_scale(tmp_path: pathlib.Path) -> None:
    """A doubled method stops the run, and turning the check off scores it at a ratio of 2.

    The fixture holograms are flat noise, so only the scale of the method's
    output is meaningful here, not the quality of its reconstruction.

    Args:
        tmp_path (pathlib.Path): Folder for the dataset.
    """
    write_dataset(tmp_path / "set")
    propagator: AngularSpectrumPropagator = build_propagator(read_geometry(tmp_path / "set"))
    table: pd.DataFrame = evaluate_set(tmp_path / "set", {}, propagator)
    assert set(table["method"]) == set(REFERENCES)
    with pytest.raises(ValueError, match="doubled"):
        evaluate_set(tmp_path / "set", {"doubled": _Doubled(propagator)}, propagator)
    unchecked: pd.DataFrame = evaluate_set(
        tmp_path / "set", {"doubled": _Doubled(propagator)}, propagator, background_tolerance=None
    )
    assert unchecked[unchecked["method"] == "doubled"]["bg_amp_ratio"].median() == pytest.approx(2.0, rel=0.05)


@pytest.mark.parametrize("name", REFERENCES)
def test_evaluate_set_rejects_a_method_named_like_a_reference(tmp_path: pathlib.Path, name: str) -> None:
    """A method under a reference's name would skip the background check and mix with the reference rows.

    Args:
        tmp_path (pathlib.Path): Folder for the dataset.
        name (str): Name of a reference.
    """
    write_dataset(tmp_path / "set")
    propagator: AngularSpectrumPropagator = build_propagator(read_geometry(tmp_path / "set"))
    with pytest.raises(ValueError, match=name):
        evaluate_set(tmp_path / "set", {name: _Doubled(propagator)}, propagator)


def test_back_propagation_of_a_flat_hologram_is_the_flat_field() -> None:
    """No object gives a hologram of 1, which must back-propagate to exactly 1 + 0i.

    The propagation phase exp(-i k z) is undone, so the result is real and needs
    no rescaling: any global amplitude or phase error would show here.
    """
    propagator: AngularSpectrumPropagator = AngularSpectrumPropagator(
        (SIZE, SIZE), 532e-9, 0.24e-6, 20e-6, numerical_aperture=0.45
    )
    hologram: torch.Tensor = torch.ones((2, 1, SIZE, SIZE))
    nominal: torch.Tensor = back_propagate(propagator, hologram, None)
    shifted: torch.Tensor = back_propagate(propagator, hologram, torch.tensor([20e-6, 23e-6]))
    field: torch.Tensor
    for field in (nominal, shifted):
        torch.testing.assert_close(field[:, 0], torch.ones((2, SIZE, SIZE)), atol=1e-4, rtol=0)
        torch.testing.assert_close(field[:, 1], torch.zeros((2, SIZE, SIZE)), atol=1e-4, rtol=0)
