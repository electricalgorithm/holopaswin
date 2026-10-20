"""Timing of reconstructions, for the network alone and for the whole path.

The network is timed on random holograms of the dataset size for several batch
sizes, after a warm-up, with the device synchronised before each clock reading so
that queued GPU work is counted. The end-to-end path starts from a recorded
frame on disk, as a camera would save it: loading the frame, normalising it,
moving it to the device and reconstructing it. That is what a user waits for
with a recorded hologram, unlike the isolated forward pass.
"""

import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from holopaswin.dataset import HoloDataset
from holopaswin.models import Geometry, build_model, propagator_of, read_geometry, resolve_model
from holopaswin.normalization import normalize_hologram
from holopaswin.training.config import ModelSettings


def synchronize(device: torch.device) -> None:
    """Wait until the device has finished its queued work.

    Args:
        device (torch.device): The device to wait for.
    """
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def time_model(  # noqa: PLR0913
    model: nn.Module,
    size: int,
    batch_sizes: list[int],
    iterations: int,
    warmup: int,
    device: torch.device,
) -> list[dict[str, float]]:
    """Time the forward pass of a network for several batch sizes.

    Args:
        model (nn.Module): Network in evaluation mode on the device.
        size (int): Image size in pixels.
        batch_sizes (list[int]): Batch sizes to time.
        iterations (int): Timed forward passes per batch size.
        warmup (int): Untimed forward passes before timing.
        device (torch.device): Device of the network.

    Returns:
        list[dict[str, float]]: Per batch size, the median latency of one pass in
            ms and the throughput in holograms per second.
    """
    rows: list[dict[str, float]] = []
    batch_size: int
    with torch.no_grad():
        for batch_size in batch_sizes:
            hologram: torch.Tensor = torch.rand(batch_size, 1, size, size, device=device) + 0.5
            timings: list[float] = []
            iteration: int
            for iteration in range(warmup + iterations):
                synchronize(device)
                start: float = time.perf_counter()
                model(hologram)
                synchronize(device)
                if iteration >= warmup:
                    timings.append(time.perf_counter() - start)
            latency: float = float(np.median(timings))
            rows.append(
                {"batch_size": batch_size, "latency_ms": latency * 1e3, "holograms_per_second": batch_size / latency}
            )
    return rows


def time_end_to_end(model: nn.Module, data_dir: Path, holograms: int, device: torch.device) -> dict[str, float]:
    """Time loading, normalising and reconstructing recorded frames one by one.

    The holograms of the dataset are first written, untimed, as 16-bit frames in
    the camera's digital numbers, one .npy file each, as a camera would record
    them. Each frame is then loaded, normalised, moved to the device and
    reconstructed.

    Args:
        model (nn.Module): Network in evaluation mode on the device.
        data_dir (Path): hologen dataset that provides the frames.
        holograms (int): Number of frames to time (at most the dataset size).
        device (torch.device): Device of the network.

    Returns:
        dict[str, float]: The number of frames timed and the median time per
            frame in ms for loading and normalising, for the network, and in total.
    """
    size: int = propagator_of(model).shape[0]
    dataset: HoloDataset = HoloDataset(str(data_dir), target_size=size, img_dim=size)
    count: int = min(holograms, len(dataset))
    load: list[float] = []
    reconstruct: list[float] = []
    index: int
    with tempfile.TemporaryDirectory() as folder:
        frames: list[Path] = []
        for index in range(count):
            frames.append(Path(folder) / f"frame_{index:05d}.npy")
            np.save(frames[-1], dataset.read_recorded(index).astype(np.uint16))
        with torch.no_grad():
            model(torch.ones(1, 1, size, size, device=device))  # warm-up, so first-call setup is not timed
            frame: Path
            for frame in frames:
                start: float = time.perf_counter()
                recorded: npt.NDArray[np.float32] = np.load(frame).astype(np.float32)
                hologram: torch.Tensor = torch.from_numpy(normalize_hologram(recorded))[None, None].to(device)
                synchronize(device)
                loaded: float = time.perf_counter()
                model(hologram)
                synchronize(device)
                load.append(loaded - start)
                reconstruct.append(time.perf_counter() - loaded)
    return {
        "holograms": count,
        "load_ms": float(np.median(load)) * 1e3,
        "model_ms": float(np.median(reconstruct)) * 1e3,
        "total_ms": float(np.median(np.add(load, reconstruct))) * 1e3,
    }


def run_benchmark(  # noqa: PLR0913
    model: Path | None,
    architecture: str | None,
    data: Path | None,
    batch_sizes: list[int],
    iterations: int,
    warmup: int,
    device: torch.device,
) -> dict[str, Any]:
    """Build the network to time and time it.

    Args:
        model (Path | None): Run folder of holopaswin train or a HoloPASWIN checkpoint.
        architecture (str | None): Untrained network to time instead of model.
        data (Path | None): Dataset for the geometry and the end-to-end timing.
        batch_sizes (list[int]): Batch sizes to time.
        iterations (int): Timed passes per batch size (and holograms end to end).
        warmup (int): Untimed passes before timing.
        device (torch.device): Device to run on.

    Returns:
        dict[str, Any]: The device, the image size, the per-batch timings under
            "model", and with data the end-to-end timing under "end_to_end".

    Raises:
        ValueError: If neither or both of model and architecture are given, or no
            geometry is available.
    """
    if (model is None) == (architecture is None):
        msg: str = "Give either --model or --architecture."
        raise ValueError(msg)
    source: Path | None = model if model is not None and model.is_dir() else data
    if source is None:
        msg = "The geometry comes from a run folder or --data; give --data for a checkpoint or --architecture."
        raise ValueError(msg)
    geometry: Geometry = read_geometry(source)
    network: nn.Module = (
        resolve_model(model, geometry, device)
        if model is not None
        else build_model(ModelSettings(architecture=str(architecture), pretrained=False), geometry).to(device).eval()
    )
    report: dict[str, Any] = {
        "device": str(device),
        "size": geometry.resolution,
        "model": time_model(network, geometry.resolution, batch_sizes, iterations, warmup, device),
    }
    if data is not None:
        report["end_to_end"] = time_end_to_end(network, data, iterations, device)
    return report
