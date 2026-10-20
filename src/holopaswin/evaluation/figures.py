"""Figures that compare reconstructions of the same test holograms.

Each sample gets a row with the hologram, the phase of every method and the true
phase, on one colour scale per row so the methods can be compared directly. An
optional second row per sample shows the absolute phase error of every method,
and an optional zoom adds an enlarged inset of one region to every panel.
"""

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from holopaswin.dataset import HoloDataset
from holopaswin.models import Geometry, build_baseline, read_geometry, resolve_model

if TYPE_CHECKING:
    from matplotlib.axes import Axes

PANEL_INCHES: float = 2.0
DPI: int = 100
PANEL_PIXELS: int = int(PANEL_INCHES * DPI)  # width and height of one panel in the PNG
ERROR_RANGE: float = 1.0  # rad, the top of the error colour scale
PHASE_FLOOR: float = -0.3  # rad, the bottom of the phase colour scale, just below the empty background


def check_zoom(zoom: tuple[int, int, int] | None, size: int) -> None:
    """Check that a zoom region lies inside the images.

    Args:
        zoom (tuple[int, int, int] | None): Column, row and size in pixels, or None.
        size (int): Image size in pixels.

    Raises:
        ValueError: If the region does not fit the images.
    """
    if zoom is not None and (min(zoom) < 0 or zoom[2] == 0 or max(zoom[0], zoom[1]) + zoom[2] > size):
        msg: str = f"The zoom region {zoom} does not fit {size} px images."
        raise ValueError(msg)


def _phase(field: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Phase of a (2, H, W) real and imaginary field.

    Args:
        field (npt.NDArray[np.float32]): The field.

    Returns:
        npt.NDArray[np.float32]: Its phase in rad, shape (H, W).
    """
    phase: npt.NDArray[np.float32] = np.arctan2(field[1], field[0])
    return phase


def reconstruct_samples(
    test_dir: Path,
    runs: dict[str, Path],
    baselines: list[str],
    samples: list[int],
    device: torch.device,
) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32], dict[str, npt.NDArray[np.float32]]]:
    """Reconstruct selected holograms of a test set with every method.

    Args:
        test_dir (Path): hologen test set, with its recipe.toml.
        runs (dict[str, Path]): Run folder or checkpoint by model name.
        baselines (list[str]): Classical baselines, any of holopaswin.models.BASELINES.
        samples (list[int]): Indices of the holograms to reconstruct.
        device (torch.device): Device the methods run on.

    Returns:
        tuple: The holograms (N, H, W), the true fields (N, 2, H, W) and the
            reconstructed fields (N, 2, H, W) by method name.

    Raises:
        ValueError: If no method is given or a sample index is outside the set.
    """
    if not runs and not baselines:
        msg: str = "Give at least one --model or --baseline."
        raise ValueError(msg)
    geometry: Geometry = read_geometry(test_dir)
    dataset: HoloDataset = HoloDataset(str(test_dir), target_size=geometry.resolution, img_dim=geometry.resolution)
    index: int
    for index in samples:
        if not 0 <= index < len(dataset):
            msg = f"There is no sample {index} in {test_dir}, which holds {len(dataset)} holograms."
            raise ValueError(msg)
    items: list[tuple[torch.Tensor, ...]] = [dataset[index] for index in samples]
    holograms: torch.Tensor = torch.stack([item[0] for item in items]).to(device)
    targets: torch.Tensor = torch.stack([item[1] for item in items])
    methods: dict[str, nn.Module] = {name: resolve_model(path, geometry, device) for name, path in runs.items()}
    name: str
    for name in baselines:
        methods[name] = build_baseline(name, geometry).to(device).eval()
    fields: dict[str, npt.NDArray[np.float32]] = {}
    method: nn.Module
    with torch.no_grad():
        for name, method in methods.items():
            prediction: torch.Tensor
            prediction, _ = method(holograms)
            fields[name] = prediction.cpu().numpy().astype(np.float32)
    return holograms[:, 0].cpu().numpy().astype(np.float32), targets.numpy().astype(np.float32), fields


def _show(
    axis: "Axes",
    image: npt.NDArray[np.float32],
    cmap: str,
    limits: tuple[float, float],
    zoom: tuple[int, int, int] | None,
) -> None:
    """Draw one panel, with an enlarged inset of the zoom region if given.

    Args:
        axis (Axes): Matplotlib axes of the panel.
        image (npt.NDArray[np.float32]): Image to draw, shape (H, W).
        cmap (str): Colour map.
        limits (tuple[float, float]): Lower and upper end of the colour scale.
        zoom (tuple[int, int, int] | None): Column, row and size in pixels of the
            zoom region, or None.
    """
    axis.imshow(image, cmap=cmap, vmin=limits[0], vmax=limits[1], interpolation="nearest")
    axis.set_axis_off()
    if zoom is None:
        return
    column: int
    row: int
    size: int
    column, row, size = zoom
    from matplotlib.patches import Rectangle  # noqa: PLC0415  (matplotlib loads only for figures)

    axis.add_patch(Rectangle((column - 0.5, row - 0.5), size, size, fill=False, edgecolor="white", linewidth=0.8))
    inset: Axes = axis.inset_axes((0.58, 0.58, 0.4, 0.4))
    inset.imshow(
        image[row : row + size, column : column + size],
        cmap=cmap,
        vmin=limits[0],
        vmax=limits[1],
        interpolation="nearest",
    )
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor("white")


def _label(axis: "Axes", text: str) -> None:
    """Write a label in the lower left corner of a panel, readable on any image.

    Args:
        axis (Axes): Matplotlib axes of the panel.
        text (str): The label.
    """
    axis.text(
        0.03,
        0.04,
        text,
        transform=axis.transAxes,
        fontsize=8,
        color="white",
        bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none", "pad": 1.5},
    )


def draw_figure(  # noqa: PLR0913
    holograms: npt.NDArray[np.float32],
    targets: npt.NDArray[np.float32],
    fields: dict[str, npt.NDArray[np.float32]],
    output: Path,
    zoom: tuple[int, int, int] | None = None,
    error: bool = False,
) -> None:
    """Draw the comparison figure and save it as a PNG.

    The PNG is PANEL_PIXELS wide per column (hologram, each method, truth) and
    PANEL_PIXELS high per row (one per sample, two with error rows).

    Args:
        holograms (npt.NDArray[np.float32]): Holograms, shape (N, H, W).
        targets (npt.NDArray[np.float32]): True fields, shape (N, 2, H, W).
        fields (dict[str, npt.NDArray[np.float32]]): Reconstructed fields by method.
        output (Path): PNG file to write.
        zoom (tuple[int, int, int] | None): Column, row and size in pixels of a
            region to enlarge in every panel, or None.
        error (bool): Whether to add a row with the absolute phase error per sample.

    Raises:
        ValueError: If the zoom region does not fit the images.
    """
    import matplotlib as mpl  # noqa: PLC0415  (imported here so the CLI starts fast)

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    check_zoom(zoom, holograms.shape[-1])
    methods: list[str] = list(fields)
    columns: int = len(methods) + 2
    rows_per_sample: int = 2 if error else 1
    figure, axes = plt.subplots(
        len(holograms) * rows_per_sample,
        columns,
        figsize=(columns * PANEL_INCHES, len(holograms) * rows_per_sample * PANEL_INCHES),
        dpi=DPI,
        squeeze=False,
    )
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.03, hspace=0.03)
    sample: int
    for sample in range(len(holograms)):
        truth: npt.NDArray[np.float32] = _phase(targets[sample])
        limits: tuple[float, float] = (PHASE_FLOOR, float(np.clip(np.percentile(truth, 99.5), 1.0, np.pi)))
        row: int = sample * rows_per_sample
        hologram: npt.NDArray[np.float32] = holograms[sample]
        _show(axes[row][0], hologram, "gray", (float(hologram.min()), float(hologram.max())), zoom)
        _label(axes[row][0], "hologram")
        column: int
        name: str
        for column, name in enumerate(methods, start=1):
            phase: npt.NDArray[np.float32] = _phase(fields[name][sample])
            _show(axes[row][column], phase, "cividis", limits, zoom)
            _label(axes[row][column], name)
            if error:
                wrapped: npt.NDArray[np.float32] = np.abs(np.angle(np.exp(1j * (phase - truth)))).astype(np.float32)
                _show(axes[row + 1][column], wrapped, "magma", (0.0, ERROR_RANGE), zoom)
                _label(axes[row + 1][column], f"{name} |error|")
        _show(axes[row][columns - 1], truth, "cividis", limits, zoom)
        _label(axes[row][columns - 1], "true phase")
        if error:
            axes[row + 1][0].set_axis_off()
            axes[row + 1][columns - 1].set_axis_off()
    figure.savefig(output, dpi=DPI)
    plt.close(figure)
