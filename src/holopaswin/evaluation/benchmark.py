"""Timing of reconstructions with one protocol for every method.

Networks and classical reconstructions are timed the same way: on the same
holograms of a test set, on one device, for the same batch sizes, after the same
number of untimed passes, with the device synchronised before every clock reading
so that queued device work is counted where it belongs. The methods take turns:
every timed pass runs each method once, in the order given, so a change of the
device's speed during the run, such as thermal throttling, affects all of them
alike. The timed region is the call of the method, from the normalised hologram
on the device to the reconstructed field, which includes the back-propagation
every method starts with. The end-to-end path starts from a recorded frame on
disk, as a camera would save it: loading the frame, normalising it, moving it to
the device and reconstructing it.
"""

import platform
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from holopaswin.dataset import HoloDataset
from holopaswin.factory import Geometry, build_baseline, build_model, read_geometry, resolve_model
from holopaswin.normalization import normalize_hologram
from holopaswin.training.config import ModelSettings

# What a timing covers, recorded with the timings.
TIMED_REGION: str = "normalised hologram on the device to reconstructed field"


def synchronize(device: torch.device) -> None:
    """Wait until the device has finished its queued work.

    Args:
        device (torch.device): The device to wait for.
    """
    if device.type == "cuda":  # pragma: no cover  (no CUDA on the test machine)
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def device_name(device: torch.device) -> str:
    """Name the hardware a device runs on, for the record of a benchmark.

    Args:
        device (torch.device): The device.

    Returns:
        str: The GPU name for CUDA, otherwise the processor name.
    """
    if device.type == "cuda":  # pragma: no cover  (no CUDA on the test machine)
        return torch.cuda.get_device_name(device)
    if platform.system() == "Darwin":
        brand: subprocess.CompletedProcess[str] = subprocess.run(
            ["/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, check=False
        )
        if brand.stdout.strip():
            return brand.stdout.strip()
    return platform.processor() or platform.machine()  # pragma: no cover  (the test machine has a brand string)


def describe_environment(device: torch.device) -> dict[str, Any]:
    """Record the hardware and numerics that the timings depend on.

    Args:
        device (torch.device): The device the methods run on.

    Returns:
        dict[str, Any]: Device name, platform, torch version, dtype and CPU
            threads, and on CUDA whether TF32 and cuDNN autotuning were on.
    """
    environment: dict[str, Any] = {
        "device_name": device_name(device),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "dtype": "float32",
        "cpu_threads": torch.get_num_threads(),
    }
    if device.type == "cuda":  # pragma: no cover  (no CUDA on the test machine)
        environment["cuda_tf32"] = torch.backends.cuda.matmul.allow_tf32
        environment["cudnn_tf32"] = torch.backends.cudnn.allow_tf32
        environment["cudnn_benchmark"] = torch.backends.cudnn.benchmark
    return environment


def read_holograms(test_dir: Path, size: int, count: int) -> torch.Tensor:
    """Read the first holograms of a test set, normalised as the methods receive them.

    Args:
        test_dir (Path): hologen test set.
        size (int): Image size in pixels.
        count (int): Holograms to read, at most the size of the set.

    Returns:
        torch.Tensor: Holograms of shape (N, 1, size, size) on the CPU.

    Raises:
        ValueError: If the test set holds no holograms.
    """
    dataset: HoloDataset = HoloDataset(str(test_dir), target_size=size, img_dim=size)
    if len(dataset) == 0:
        msg: str = f"{test_dir} holds no holograms to time."
        raise ValueError(msg)
    return torch.stack([dataset[index][0] for index in range(min(count, len(dataset)))])


def write_frames(test_dir: Path, size: int, count: int, folder: Path) -> list[Path]:
    """Write the first recorded holograms of a test set as 16-bit frames, one .npy file each.

    The frames hold the camera's digital numbers, as a camera would save them,
    so loading and normalising them is part of the end-to-end timing.

    Args:
        test_dir (Path): hologen test set.
        size (int): Image size in pixels.
        count (int): Frames to write, at most the size of the set.
        folder (Path): Folder for the frames; created if needed.

    Returns:
        list[Path]: The frame files, in the order of the test set.
    """
    dataset: HoloDataset = HoloDataset(str(test_dir), target_size=size, img_dim=size)
    folder.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    index: int
    for index in range(min(count, len(dataset))):
        frames.append(folder / f"frame_{index:05d}.npy")
        np.save(frames[-1], dataset.read_recorded(index).astype(np.uint16))
    return frames


def count_parameters(model: nn.Module) -> int:
    """Count the parameters of a method; buffers such as transfer functions are not counted.

    Args:
        model (nn.Module): A network or classical reconstruction.

    Returns:
        int: The number of parameters, 0 for a classical reconstruction.
    """
    return sum(parameter.numel() for parameter in model.parameters())


def timed_call(model: nn.Module, batch: torch.Tensor, device: torch.device) -> float:
    """Time one call of a method, with the device synchronised before each clock reading.

    Args:
        model (nn.Module): Network or classical reconstruction on the device.
        batch (torch.Tensor): Normalised holograms on the device.
        device (torch.device): Device of the method.

    Returns:
        float: Duration of the call in seconds.
    """
    synchronize(device)
    start: float = perf_counter()
    model(batch)
    synchronize(device)
    return perf_counter() - start


def warm_up(methods: dict[str, nn.Module], batch: torch.Tensor, passes: int, device: torch.device) -> None:
    """Run untimed passes of every method, then wait until the device has finished them.

    Without the final synchronisation, queued warm-up work would be counted in
    the first timing that follows.

    Args:
        methods (dict[str, nn.Module]): Methods by name.
        batch (torch.Tensor): Normalised holograms on the device.
        passes (int): Untimed passes per method.
        device (torch.device): Device of the methods.
    """
    _: int
    for _ in range(passes):
        method: nn.Module
        for method in methods.values():
            method(batch)
    synchronize(device)


def time_methods(  # noqa: PLR0913
    methods: dict[str, nn.Module],
    holograms: torch.Tensor,
    batch_sizes: list[int],
    iterations: int,
    warmup: int,
    device: torch.device,
) -> dict[str, list[dict[str, float]]]:
    """Time the call of every method for several batch sizes, taking turns on every pass.

    A batch holds the given holograms in order, repeated when the batch is
    larger than their number, so every method and pass sees the same input.

    Args:
        methods (dict[str, nn.Module]): Methods in evaluation mode on the device, by name.
        holograms (torch.Tensor): Normalised holograms of shape (N, 1, H, W).
        batch_sizes (list[int]): Batch sizes to time.
        iterations (int): Timed passes per method and batch size.
        warmup (int): Untimed passes per method before timing a batch size.
        device (torch.device): Device of the methods.

    Returns:
        dict[str, list[dict[str, float]]]: By method and batch size, the median
            latency of one pass in ms and the throughput in holograms per second.
    """
    rows: dict[str, list[dict[str, float]]] = {name: [] for name in methods}
    batch_size: int
    with torch.no_grad():
        for batch_size in batch_sizes:
            batch: torch.Tensor = holograms[[index % len(holograms) for index in range(batch_size)]].to(device)
            warm_up(methods, batch, warmup, device)
            timings: dict[str, list[float]] = {name: [] for name in methods}
            _: int
            for _ in range(iterations):
                name: str
                method: nn.Module
                for name, method in methods.items():
                    timings[name].append(timed_call(method, batch, device))
            for name in methods:
                latency: float = float(np.median(timings[name]))
                rows[name].append(
                    {
                        "batch_size": batch_size,
                        "latency_ms": latency * 1e3,
                        "holograms_per_second": batch_size / latency,
                    }
                )
    return rows


def timed_frame(model: nn.Module, frame: Path, device: torch.device) -> tuple[float, float]:
    """Time loading one stored frame and reconstructing it.

    Args:
        model (nn.Module): Network or classical reconstruction on the device.
        frame (Path): A 16-bit frame written by write_frames.
        device (torch.device): Device of the method.

    Returns:
        tuple[float, float]: Seconds for loading, normalising and moving the
            frame to the device, and seconds for the reconstruction.
    """
    synchronize(device)
    start: float = perf_counter()
    recorded: npt.NDArray[np.float32] = np.load(frame).astype(np.float32)
    hologram: torch.Tensor = torch.from_numpy(normalize_hologram(recorded))[None, None].to(device)
    synchronize(device)
    loaded: float = perf_counter()
    model(hologram)
    synchronize(device)
    return loaded - start, perf_counter() - loaded


def time_end_to_end(
    methods: dict[str, nn.Module], frames: list[Path], holograms: torch.Tensor, warmup: int, device: torch.device
) -> dict[str, dict[str, float]]:
    """Time loading, normalising and reconstructing the same stored frames with every method, taking turns.

    Args:
        methods (dict[str, nn.Module]): Methods in evaluation mode on the device, by name.
        frames (list[Path]): Frames written by write_frames.
        holograms (torch.Tensor): Normalised holograms; the first one is used
            for the untimed warm-up passes.
        warmup (int): Untimed reconstructions per method before timing.
        device (torch.device): Device of the methods.

    Returns:
        dict[str, dict[str, float]]: By method, the number of frames timed and
            the median time per frame in ms for loading, normalising and moving
            to the device, for the reconstruction, and for both together. The
            last is the median of the per-frame sums, not the sum of the medians.
    """
    load: dict[str, list[float]] = {name: [] for name in methods}
    reconstruct: dict[str, list[float]] = {name: [] for name in methods}
    with torch.no_grad():
        warm_up(methods, holograms[:1].to(device), warmup, device)
        frame: Path
        for frame in frames:
            name: str
            method: nn.Module
            for name, method in methods.items():
                loading: float
                reconstruction: float
                loading, reconstruction = timed_frame(method, frame, device)
                load[name].append(loading)
                reconstruct[name].append(reconstruction)
    return {
        name: {
            "holograms": len(frames),
            "load_ms": float(np.median(load[name])) * 1e3,
            "model_ms": float(np.median(reconstruct[name])) * 1e3,
            "total_ms": float(np.median(np.add(load[name], reconstruct[name]))) * 1e3,
        }
        for name in methods
    }


def build_methods(
    geometry: Geometry, models: dict[str, Path], architectures: list[str], baselines: list[str], device: torch.device
) -> dict[str, nn.Module]:
    """Build every method to time for the geometry of a test set.

    Args:
        geometry (Geometry): Geometry of the test set.
        models (dict[str, Path]): Run folder or HoloPASWIN checkpoint by model name.
        architectures (list[str]): Untrained networks to time; the weights do not change the work.
        baselines (list[str]): Classical reconstructions to time, any of BASELINES.
        device (torch.device): Device to run on.

    Returns:
        dict[str, nn.Module]: The methods in evaluation mode on the device, models
            first, then networks, then baselines.

    Raises:
        ValueError: If no method is given, a name is given twice, or an
            architecture or baseline is unknown.
    """
    methods: dict[str, nn.Module] = {}
    names: list[str] = [*models, *architectures, *baselines]
    repeated: list[str] = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        msg: str = f"Methods {repeated} are given twice; their timings would overwrite each other."
        raise ValueError(msg)
    name: str
    path: Path
    for name, path in models.items():
        methods[name] = resolve_model(path, geometry, device)
    for name in architectures:
        settings: ModelSettings = ModelSettings(architecture=name, pretrained=False)
        methods[name] = build_model(settings, geometry).to(device).eval()
    for name in baselines:
        methods[name] = build_baseline(name, geometry).to(device).eval()
    if not methods:
        msg = "Give at least one --model, --architecture or --baseline to time."
        raise ValueError(msg)
    return methods


def run_benchmark(  # noqa: PLR0913
    test_dir: Path,
    models: dict[str, Path],
    architectures: list[str],
    baselines: list[str],
    batch_sizes: list[int],
    iterations: int,
    warmup: int,
    device: torch.device,
) -> dict[str, Any]:
    """Time every method on the same holograms of a test set with one protocol.

    Args:
        test_dir (Path): hologen test set for the geometry, the timed holograms
            and the end-to-end frames.
        models (dict[str, Path]): Run folder or HoloPASWIN checkpoint by model name.
        architectures (list[str]): Untrained networks to time.
        baselines (list[str]): Classical reconstructions to time.
        batch_sizes (list[int]): Batch sizes to time, each at least 1.
        iterations (int): Timed passes per batch size, and frames timed end to
            end, at least 1.
        warmup (int): Untimed passes before timing.
        device (torch.device): Device to run on.

    Returns:
        dict[str, Any]: The protocol and environment, and by method its
            parameter count, the timings per batch size under "model" and the
            end-to-end timing.

    Raises:
        ValueError: If a batch size or the number of iterations is below 1, no
            method is given, a name is given twice, an architecture or baseline
            is unknown, or the test set holds no holograms.
    """
    if min(batch_sizes) < 1:
        msg: str = f"Every batch size must be at least 1, got {batch_sizes}."
        raise ValueError(msg)
    if iterations < 1:
        msg = f"At least one timed iteration is needed, got {iterations}."
        raise ValueError(msg)
    geometry: Geometry = read_geometry(test_dir)
    methods: dict[str, nn.Module] = build_methods(geometry, models, architectures, baselines, device)
    size: int = geometry.resolution
    holograms: torch.Tensor = read_holograms(test_dir, size, max(batch_sizes))
    report: dict[str, Any] = {
        "protocol": {
            "device": str(device),
            "size": size,
            "batch_sizes": batch_sizes,
            "warmup": warmup,
            "iterations": iterations,
            "holograms": len(holograms),
            "timed_region": TIMED_REGION,
            "order": list(methods),
            **describe_environment(device),
        },
        "methods": {},
    }
    model_timings: dict[str, list[dict[str, float]]] = time_methods(
        methods, holograms, batch_sizes, iterations, warmup, device
    )
    with tempfile.TemporaryDirectory() as folder:
        frames: list[Path] = write_frames(test_dir, size, iterations, Path(folder))
        end_to_end: dict[str, dict[str, float]] = time_end_to_end(methods, frames, holograms, warmup, device)
    name: str
    method: nn.Module
    for name, method in methods.items():
        report["methods"][name] = {
            "parameters": count_parameters(method),
            "model": model_timings[name],
            "end_to_end": end_to_end[name],
        }
    return report
