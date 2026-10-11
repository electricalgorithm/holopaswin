"""Tests for flat-field hologram normalisation.

The camera model below mirrors hologen's recording chain (EMVA 1288): 4000
photo-electrons at the background, Poisson shot noise, 7.1 dark electrons, 10 e-
read noise and a 12-bit ADC whose gain maps the 11,000 e- saturation capacity
to 4095 DN. The background therefore lands at 1489 DN, not at 1000.
"""

from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import torch

from holopaswin.dataset import HoloDataset
from holopaswin.normalization import normalize_hologram
from holopaswin.propagator import AngularSpectrumPropagator

BACKGROUND_ELECTRONS = 4000.0
DARK_ELECTRONS = 7.1
READ_NOISE = 10.0
GAIN = 4095 / 11000  # DN per electron
SIZE = 224


def record(intensity: npt.NDArray[np.float64], seed: int = 0) -> npt.NDArray[np.float64]:
    """Record a relative intensity (background = 1) with the camera model, in DN."""
    rng = np.random.default_rng(seed)
    electrons = rng.poisson(intensity * BACKGROUND_ELECTRONS) + rng.poisson(DARK_ELECTRONS, intensity.shape)
    electrons = electrons + rng.normal(0.0, READ_NOISE, intensity.shape)
    return np.round(np.clip(electrons * GAIN, 0, 4095))


def ideal_hologram() -> tuple[npt.NDArray[np.float64], AngularSpectrumPropagator]:
    """Return |ASM(O)|^2 of a weak phase object with holopaswin's own propagator."""
    y, x = np.mgrid[:SIZE, :SIZE]
    obj = np.ones((SIZE, SIZE), dtype=np.complex128)
    for cy, cx, radius in [(60, 70, 12), (150, 120, 9), (110, 180, 15)]:
        cell = (y - cy) ** 2 + (x - cx) ** 2 <= radius**2
        obj[cell] = 0.95 * np.exp(0.3j)
    propagator = AngularSpectrumPropagator((SIZE, SIZE), 532e-9, 4.65e-6, 0.02)
    field = propagator(torch.from_numpy(obj).to(torch.complex64), backward=False)
    return np.abs(field.numpy()).astype(np.float64) ** 2, propagator


# Unit behaviour -------------------------------------------------------------


def test_flat_field_becomes_one() -> None:
    """A uniform recording is the background, so it normalises to exactly 1."""
    np.testing.assert_allclose(normalize_hologram(np.full((32, 32), 1489.0)), 1.0)


@pytest.mark.parametrize("scale", [0.2, 1.0, 7.5])
def test_result_does_not_depend_on_camera_scale(scale: float) -> None:
    """Gain or exposure only scale the recording, so they must not change H."""
    hologram, _ = ideal_hologram()
    recorded = record(hologram)
    np.testing.assert_allclose(normalize_hologram(scale * recorded), normalize_hologram(recorded), rtol=1e-6)


def test_dark_level_is_subtracted_before_scaling() -> None:
    """With a known dark offset the shape of the hologram is recovered exactly."""
    hologram, _ = ideal_hologram()
    recorded = 1489.0 * hologram + 30.0
    expected = hologram / hologram.mean()
    np.testing.assert_allclose(normalize_hologram(recorded, dark_level=30.0), expected, rtol=1e-9)


@pytest.mark.parametrize("hologram", [np.zeros((8, 8)), np.full((8, 8), 5.0)])
def test_no_signal_above_dark_raises(hologram: npt.NDArray[np.float64]) -> None:
    """Without light above the dark level there is nothing to normalise by."""
    with pytest.raises(ValueError, match="no signal"):
        normalize_hologram(hologram, dark_level=5.0)


def test_input_is_not_modified_and_dtype_is_kept() -> None:
    """The dataset feeds float32 arrays; they must stay float32 and untouched."""
    recorded = np.full((16, 16), 1489.0, dtype=np.float32)
    before = recorded.copy()
    result = normalize_hologram(recorded)
    np.testing.assert_array_equal(recorded, before)
    assert result.dtype == np.float32


# Physics: the scale L_phy relies on (HOLO-001 AC#4) -------------------------


def test_normalised_recording_matches_the_asm_intensity_scale() -> None:
    """Normalised camera output must equal |ASM(O)|^2, the scale L_phy compares with.

    No bias at the background and an L1 error at the shot noise level (about
    1.6% at 4000 e-).
    """
    hologram, _ = ideal_hologram()
    normalised = normalize_hologram(record(hologram))
    background = np.abs(hologram - 1.0) < 0.01

    assert np.mean(normalised[background]) == pytest.approx(1.0, abs=0.005)
    assert np.mean(np.abs(normalised - hologram)) < 0.02


def test_fixed_divisor_breaks_the_asm_scale() -> None:
    """Regression test: the former fixed division by 1000 breaks the ASM scale.

    hologen now records the background at 1489 DN, so dividing by 1000 leaves it
    at 1.49 and L_phy would see a 49% offset.
    """
    hologram, _ = ideal_hologram()
    legacy = record(hologram) / 1000.0
    assert np.mean(np.abs(legacy - hologram)) > 0.4


# Integration: HoloDataset reads hologen's parquet layout --------------------


def write_hologen_parquet(directory: Path, holograms: list[npt.NDArray[np.float64]]) -> None:
    """Write holograms and empty-field ground truths in hologen's parquet layout."""
    pixels = holograms[0].size
    ids = np.repeat(np.arange(len(holograms)), pixels)
    pd.DataFrame({"sample_id": ids, "intensity": np.concatenate([h.ravel() for h in holograms])}).to_parquet(
        directory / "hologram_0000.parquet"
    )
    ones = np.ones(pixels * len(holograms))
    pd.DataFrame({"sample_id": ids, "real": ones, "imag": 0.0 * ones}).to_parquet(
        directory / "ground_truth_0000.parquet"
    )


def test_dataset_returns_normalised_holograms(tmp_path: Path) -> None:
    """Holograms recorded at different exposures all come out with background 1."""
    hologram, _ = ideal_hologram()
    recorded = record(hologram)
    write_hologen_parquet(tmp_path, [recorded, 0.25 * recorded])

    dataset = HoloDataset(str(tmp_path), target_size=SIZE, img_dim=SIZE)
    first, _ = dataset[0]
    second, _ = dataset[1]

    assert len(dataset) == 2
    assert first.mean().item() == pytest.approx(1.0, abs=1e-4)
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-5)
