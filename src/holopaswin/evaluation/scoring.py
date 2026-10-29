"""Scoring reconstructions of a hologen test set with the paper's metrics.

Every method reconstructs the same holograms, and every reconstruction is scored
with holopaswin.evaluation.metrics.compute_all_metrics, the metrics of the
paper's tables. Methods are trained networks, classical baselines, and the
back-propagation references. Back-propagating over z leaves the known constant
phase -k z on the field, which the ground truth does not have, so the references
and baselines are corrected by exp(i k z) for the distance they used. Trained
networks are scored as they are.
"""

import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore[import-untyped]
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from holopaswin.dataset import MICROMETER, HoloDataset
from holopaswin.evaluation.metrics import compute_all_metrics
from holopaswin.factory import BASELINES, Geometry
from holopaswin.propagator import AngularSpectrumPropagator

# Back-propagation at the nominal distance, and at the true distance as an oracle.
REFERENCES: tuple[str, str] = ("backprop_nominal", "backprop_true_z")
# Columns of a per-sample table that describe a row instead of scoring it.
IDENTIFIERS: tuple[str, ...] = ("method", "offset_um", "index", "z_true_um", "object_family", "config_name")
# Largest accepted distance of a method's median background amplitude ratio from 1.
# Twin-image ripples move the median by a few percent; a wrong normalisation moves
# it by tens of percent (dividing a 1489 DN background by 1000 gives 1.22).
BACKGROUND_TOLERANCE: float = 0.1
# Samples of a method whose median background ratio is checked; the median of this
# many tolerates a few hard holograms, and a wrong scale still stops a run early.
BACKGROUND_WARMUP: int = 32
# Columns of a per-sample table that check the evaluation instead of scoring a method.
DIAGNOSTICS: tuple[str, ...] = ("bg_amp_ratio",)


def metric_columns(columns: list[str]) -> list[str]:
    """Select the columns of a per-sample table that score the methods.

    Args:
        columns (list[str]): Columns of the table, in order.

    Returns:
        list[str]: The columns that are neither IDENTIFIERS nor DIAGNOSTICS, in order.
    """
    return [column for column in columns if column not in IDENTIFIERS and column not in DIAGNOSTICS]


def parse_model_specs(specs: list[str]) -> dict[str, Path]:
    """Read NAME=RUN model specifications.

    RUN is a run folder of holopaswin train or a bare HoloPASWIN checkpoint.

    Args:
        specs (list[str]): Specifications in the order given.

    Returns:
        dict[str, Path]: Run folder or checkpoint by model name, in the order given.

    Raises:
        ValueError: If a specification is not NAME=RUN, a name repeats or is
            reserved for a reference, or the run does not exist.
    """
    runs: dict[str, Path] = {}
    spec: str
    for spec in specs:
        if "=" not in spec:
            msg: str = f"--model expects NAME=RUN (a run folder or checkpoint), got {spec!r}."
            raise ValueError(msg)
        name: str
        path: str
        name, path = spec.split("=", 1)
        if not name:
            msg = f"--model needs a name before '=', got {spec!r}."
            raise ValueError(msg)
        if name in runs or name in REFERENCES or name in BASELINES:
            msg = f"Model name {name!r} is used twice or is reserved for a reference or baseline."
            raise ValueError(msg)
        if not Path(path).exists():
            msg = f"Run or checkpoint {path} not found."
            raise ValueError(msg)
        runs[name] = Path(path)
    return runs


def build_propagator(geometry: Geometry) -> AngularSpectrumPropagator:
    """Build a full-size propagator at the nominal distance of a test set.

    Args:
        geometry (Geometry): Geometry of the test set.

    Returns:
        AngularSpectrumPropagator: The propagator, with the objective pupil.
    """
    return AngularSpectrumPropagator(
        (geometry.resolution, geometry.resolution),
        geometry.wavelength,
        geometry.pixel_size,
        geometry.z_distance,
        numerical_aperture=geometry.numerical_aperture,
    )


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


def check_background(method: str, rows: list[dict[str, float | str]], tolerance: float, *, warn: bool = True) -> None:
    """Stop when a method reconstructs on a different amplitude scale than the truth.

    A global amplitude scale leaves the phase metrics and the B/S ratio
    unchanged, so a wrong normalisation would otherwise only show as a low
    amplitude PSNR. The median of the background amplitude ratio over the rows
    must lie within the tolerance of 1; rows without a background are skipped,
    with a warning when none is left.

    Args:
        method (str): Name of the method, for the message.
        rows (list[dict[str, float | str]]): Scored rows of the method, from score_batch.
        tolerance (float): Largest accepted distance of the median ratio from 1.
        warn (bool): Whether to warn when no row has a background.

    Raises:
        ValueError: If the median ratio is further than the tolerance from 1.
    """
    ratios: list[float] = [float(row["bg_amp_ratio"]) for row in rows if math.isfinite(float(row["bg_amp_ratio"]))]
    if not ratios:
        if not warn:
            return
        warnings.warn(
            f"None of the {len(rows)} samples of {method} has a background, so its amplitude scale is not checked.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    level: float = float(np.median(ratios))
    if abs(level - 1) > tolerance:
        msg: str = (
            f"{method} reconstructs a background amplitude {level:.3f} times the true one (median of "
            f"{len(ratios)} of {len(rows)} samples), outside 1 +- {tolerance}; the hologram normalisation or "
            "the output scale of the method does not match the truth."
        )
        raise ValueError(msg)


class BackgroundCheck:
    """Check the amplitude scale of each method early on its first samples, and again on all of them.

    Each method is checked on the median of its first BACKGROUND_WARMUP
    samples as soon as they are in, so a method on the wrong scale stops an
    evaluation early, and by finish on the median of all its samples, so a
    scale error in later samples is found too. Only the method and the
    background ratio of each sample are kept. The back-propagation references
    are what the methods are compared against, so they are not checked.
    """

    def __init__(self, tolerance: float = BACKGROUND_TOLERANCE, warmup: int = BACKGROUND_WARMUP) -> None:
        """Start with no rows.

        Args:
            tolerance (float): Tolerance of check_background.
            warmup (int): Samples of a method to collect before its early check.
        """
        self.tolerance: float = tolerance
        self.warmup: int = warmup
        self.ratios: dict[str, list[dict[str, float | str]]] = {}

    def add(self, rows: list[dict[str, float | str]]) -> None:
        """Collect scored rows and check every method whose warm-up has just filled.

        A method without any background sample is reported by finish, not here.

        Args:
            rows (list[dict[str, float | str]]): Scored rows, from score_batch.

        Raises:
            ValueError: If a method fails check_background on its warm-up.
        """
        row: dict[str, float | str]
        for row in rows:
            method: str = str(row["method"])
            if method in REFERENCES:
                continue
            collected: list[dict[str, float | str]] = self.ratios.setdefault(method, [])
            collected.append({"method": method, "bg_amp_ratio": row["bg_amp_ratio"]})
            if len(collected) == self.warmup:
                check_background(method, collected, self.tolerance, warn=False)

    def finish(self) -> None:
        """Check every method on the median of all its samples.

        Raises:
            ValueError: If a method fails check_background.
        """
        method: str
        collected: list[dict[str, float | str]]
        for method, collected in self.ratios.items():
            check_background(method, collected, self.tolerance)


def evaluate_set(  # noqa: PLR0913
    folder: Path,
    methods: dict[str, nn.Module],
    propagator: AngularSpectrumPropagator,
    references: tuple[str, ...] = REFERENCES,
    batch_size: int = 4,
    limit: int | None = None,
    background_tolerance: float | None = BACKGROUND_TOLERANCE,
) -> pd.DataFrame:
    """Reconstruct one test set with every method and score every hologram.

    Every method passes a BackgroundCheck, so an evaluation on the wrong
    amplitude scale stops after the method's first BACKGROUND_WARMUP samples,
    or at the end of the set when only its later samples are on the wrong scale.

    Args:
        folder (Path): hologen dataset folder.
        methods (dict[str, nn.Module]): Networks and baselines by name, each
            returning (reconstructed field, back-propagated field).
        propagator (AngularSpectrumPropagator): Full-size propagator at the
            nominal distance, on the methods' device.
        references (tuple[str, ...]): Back-propagation references to add, any of REFERENCES.
        batch_size (int): Holograms per forward pass.
        limit (int | None): Holograms to use from the set, or None for all.
        background_tolerance (float | None): Tolerance of the BackgroundCheck,
            or None to score without it.

    Returns:
        pd.DataFrame: One row per hologram and method, with method, index,
            z_true_um, object_family, config_name and one column per metric.

    Raises:
        ValueError: If a method is named like a reference, or fails the BackgroundCheck.
    """
    clashes: list[str] = [name for name in methods if name in REFERENCES]
    if clashes:
        msg: str = f"Methods {clashes} have the names of back-propagation references; rename them."
        raise ValueError(msg)
    check: BackgroundCheck | None = None if background_tolerance is None else BackgroundCheck(background_tolerance)
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
            scored: list[list[dict[str, float | str]]] = []
            if "backprop_nominal" in references:
                scored.append(score_batch("backprop_nominal", back_propagate(propagator, hologram, None), target))
            if "backprop_true_z" in references:
                scored.append(score_batch("backprop_true_z", back_propagate(propagator, hologram, z_true), target))
            name: str
            method: nn.Module
            for name, method in methods.items():
                prediction: torch.Tensor
                prediction, _ = method(hologram)
                scored.append(score_batch(name, prediction, target))
            method_rows: list[dict[str, float | str]]
            for method_rows in scored:
                if check is not None:
                    check.add(method_rows)
                position: int
                row: dict[str, float | str]
                for position, row in enumerate(method_rows):
                    index: int = first + position
                    row.update(
                        {
                            "index": index,
                            "z_true_um": float(z_true[position]) / MICROMETER,
                            "object_family": str(labels.loc[index, "object_family"]),
                            "config_name": str(labels.loc[index, "config_name"]),
                        }
                    )
                    rows.append(row)
            first += len(hologram)
    if check is not None:
        check.finish()
    return pd.DataFrame(rows)


def _without_nan(value: Any) -> Any:  # noqa: ANN401
    """Replace non-finite floats by None, recursively.

    Args:
        value (Any): A JSON-like value of dicts, lists, strings and numbers.

    Returns:
        Any: The same value with NaN and infinities replaced by None.
    """
    if isinstance(value, dict):
        return {key: _without_nan(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_without_nan(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(data: dict[str, Any], path: Path) -> None:
    """Write a summary as strict JSON, with missing values as null.

    Args:
        data (dict[str, Any]): The summary.
        path (Path): File to write.
    """
    path.write_text(json.dumps(_without_nan(data), indent=2, allow_nan=False))
