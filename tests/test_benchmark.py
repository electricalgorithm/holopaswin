"""Tests for timing every method with one protocol."""

import pathlib
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
import torch
from holo_fixtures import COUNT, SIZE, write_dataset
from torch import nn

from holopaswin.dataset import HoloDataset
from holopaswin.evaluation import benchmark
from holopaswin.evaluation.benchmark import (
    build_methods,
    read_holograms,
    run_benchmark,
    time_end_to_end,
    time_methods,
    write_frames,
)
from holopaswin.factory import read_geometry
from holopaswin.normalization import normalize_hologram

if TYPE_CHECKING:
    from collections.abc import Iterator


class Recorder(nn.Module):
    """A method that records its calls in a shared event log and returns its input as the field."""

    def __init__(self, name: str, events: list[str]) -> None:
        """Keep the name and the log.

        Args:
            name (str): Name written to the log on every call.
            events (list[str]): Log shared with the patched clock and synchronisation.
        """
        super().__init__()
        self.name: str = name
        self.events: list[str] = events
        self.inputs: list[torch.Tensor] = []

    def forward(self, hologram: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Record the call and the batch.

        Args:
            hologram (torch.Tensor): Holograms, shape (B, 1, H, W).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: The holograms twice, as field and back-propagated field.
        """
        self.events.append(f"call {self.name}")
        self.inputs.append(hologram.clone())
        return hologram, hologram


def log_clock_and_sync(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> None:
    """Replace the benchmark's clock and synchronisation by entries in an event log.

    Args:
        monkeypatch (pytest.MonkeyPatch): Patches the benchmark module.
        events (list[str]): The log.
    """
    ticks: Iterator[int] = iter(range(10**6))

    def clock() -> float:
        """Log a clock reading.

        Returns:
            float: A strictly increasing time.
        """
        events.append("clock")
        return float(next(ticks))

    monkeypatch.setattr(benchmark, "perf_counter", clock)
    monkeypatch.setattr(benchmark, "synchronize", lambda device: events.append("sync"))  # noqa: ARG005


def test_a_batch_cycles_through_the_given_holograms() -> None:
    """Every pass of a batch size sees the same holograms, repeated in order when the batch is larger than the set."""
    holograms: torch.Tensor = torch.arange(3, dtype=torch.float32).reshape(3, 1, 1, 1).expand(3, 1, 4, 4).clone()
    recorder: Recorder = Recorder("r", [])
    rows: dict[str, list[dict[str, float]]] = time_methods(
        {"r": recorder}, holograms, [1, 5], iterations=2, warmup=1, device=torch.device("cpu")
    )
    assert [row["batch_size"] for row in rows["r"]] == [1, 5]
    assert len(recorder.inputs) == 2 * (1 + 2)
    assert [float(batch[0, 0, 0, 0]) for batch in recorder.inputs[:3]] == [0.0, 0.0, 0.0]
    batch: torch.Tensor
    for batch in recorder.inputs[3:]:
        assert [float(value) for value in batch[:, 0, 0, 0]] == [0.0, 1.0, 2.0, 0.0, 1.0]


def test_the_latency_is_the_median_of_the_timed_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    """With known clock ticks, warm-up passes are left out, every timed pass is kept, and the latency is their median.

    The warm-up pass reads no clock, so the six readings belong to the three
    timed passes, which take 1, 2 and 10 s: the median is 2 s and the
    throughput 0.5 holograms per second.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the clock.
    """
    readings: Iterator[float] = iter([0.0, 1.0, 1.0, 3.0, 3.0, 13.0])
    monkeypatch.setattr(benchmark, "perf_counter", lambda: next(readings))
    recorder: Recorder = Recorder("r", [])
    rows: dict[str, list[dict[str, Any]]] = time_methods(
        {"r": recorder}, torch.ones(1, 1, 4, 4), [1], iterations=3, warmup=2, device=torch.device("cpu")
    )
    assert rows["r"] == [
        {"batch_size": 1, "latency_ms": 2000.0, "holograms_per_second": 0.5, "passes_ms": [1000.0, 2000.0, 10000.0]}
    ]
    assert len(recorder.inputs) == 2 + 3


def test_every_clock_reading_follows_a_synchronisation(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Queued device work, also of an untimed warm-up, must finish before the clock is read, or it is booked to the next timing.

    Args:
        tmp_path (pathlib.Path): Folder for the dataset and the frames.
        monkeypatch (pytest.MonkeyPatch): Replaces the clock and synchronisation by a log.
    """
    write_dataset(tmp_path / "test")
    events: list[str] = []
    log_clock_and_sync(monkeypatch, events)
    methods: dict[str, nn.Module] = {"a": Recorder("a", events), "b": Recorder("b", events)}
    holograms: torch.Tensor = read_holograms(tmp_path / "test", SIZE, 2)
    frames: list[pathlib.Path] = write_frames(tmp_path / "test", SIZE, 2, tmp_path / "frames")
    time_methods(methods, holograms, [1, 2], iterations=2, warmup=2, device=torch.device("cpu"))
    time_end_to_end(methods, frames, holograms, warmup=2, device=torch.device("cpu"))
    position: int
    event: str
    for position, event in enumerate(events):
        if event == "clock":
            assert events[position - 1] == "sync", events[max(0, position - 4) : position + 1]


def test_methods_take_turns_on_every_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Round robin: every timed pass runs each method once in the order given, so drift of the device affects all alike.

    Args:
        monkeypatch (pytest.MonkeyPatch): Replaces the clock and synchronisation by a log.
    """
    events: list[str] = []
    log_clock_and_sync(monkeypatch, events)
    methods: dict[str, nn.Module] = {"a": Recorder("a", events), "b": Recorder("b", events)}
    time_methods(methods, torch.ones(1, 1, 4, 4), [1], iterations=3, warmup=1, device=torch.device("cpu"))
    calls: list[str] = [event for event in events if event.startswith("call")]
    assert calls == ["call a", "call b"] * (1 + 3)


def test_the_holograms_are_the_normalised_recordings(tmp_path: pathlib.Path) -> None:
    """The timed holograms are the recorded frames divided by their background, as evaluation sees them, not random noise.

    Args:
        tmp_path (pathlib.Path): Folder for the dataset.
    """
    write_dataset(tmp_path / "test")
    holograms: torch.Tensor = read_holograms(tmp_path / "test", SIZE, 2)
    dataset: HoloDataset = HoloDataset(str(tmp_path / "test"), target_size=SIZE, img_dim=SIZE)
    assert holograms.shape == (2, 1, SIZE, SIZE)
    index: int
    for index in range(2):
        expected: np.ndarray = normalize_hologram(dataset.read_recorded(index).astype(np.float32))
        np.testing.assert_allclose(holograms[index, 0].numpy(), expected, rtol=1e-6)
        assert float(holograms[index].mean()) == pytest.approx(1.0, abs=1e-6)
    assert read_holograms(tmp_path / "test", SIZE, 100).shape[0] == COUNT


def test_the_frames_keep_the_recorded_values(tmp_path: pathlib.Path) -> None:
    """The 16-bit frames hold exactly the camera's digital numbers of the test set.

    Args:
        tmp_path (pathlib.Path): Folder for the dataset and the frames.
    """
    write_dataset(tmp_path / "test")
    frames: list[pathlib.Path] = write_frames(tmp_path / "test", SIZE, 3, tmp_path / "frames")
    dataset: HoloDataset = HoloDataset(str(tmp_path / "test"), target_size=SIZE, img_dim=SIZE)
    assert len(frames) == 3
    index: int
    frame: pathlib.Path
    for index, frame in enumerate(frames):
        stored: np.ndarray = np.load(frame)
        assert stored.dtype == np.uint16
        np.testing.assert_array_equal(stored, dataset.read_recorded(index))


def test_every_method_is_timed_with_the_same_protocol(tmp_path: pathlib.Path) -> None:
    """Networks and classical baselines share one device, batch sizes, warm-up and iteration count.

    Args:
        tmp_path (pathlib.Path): Folder for the dataset.
    """
    write_dataset(tmp_path / "test")
    report: dict[str, Any] = run_benchmark(
        tmp_path / "test",
        models={},
        architectures=["unet"],
        baselines=["asm", "gs50"],
        batch_sizes=[1, 2],
        iterations=2,
        warmup=1,
        device=torch.device("cpu"),
    )
    protocol: dict[str, Any] = report["protocol"]
    assert {key: protocol[key] for key in ("device", "size", "batch_sizes", "warmup", "iterations", "holograms")} == {
        "device": "cpu",
        "size": SIZE,
        "batch_sizes": [1, 2],
        "warmup": 1,
        "iterations": 2,
        "holograms": 2,
    }
    assert protocol["timed_region"] == benchmark.TIMED_REGION
    assert protocol["order"] == ["unet", "asm", "gs50"]
    assert protocol["dtype"] == "float32"
    assert protocol["torch"] == torch.__version__
    assert protocol["device_name"]
    assert list(report["methods"]) == ["unet", "asm", "gs50"]
    name: str
    method: dict[str, Any]
    for name, method in report["methods"].items():
        assert [row["batch_size"] for row in method["model"]] == [1, 2], name
        assert all(row["latency_ms"] > 0 for row in method["model"]), name
        assert all(len(row["passes_ms"]) == 2 for row in method["model"]), name
        end_to_end: dict[str, Any] = method["end_to_end"]
        assert set(end_to_end) == {
            "holograms",
            "load_ms",
            "model_ms",
            "total_ms",
            "load_ms_per_frame",
            "model_ms_per_frame",
        }, name
        assert len(end_to_end["load_ms_per_frame"]) == len(end_to_end["model_ms_per_frame"]) == 2, name
    assert report["methods"]["unet"]["parameters"] > 0
    assert report["methods"]["asm"]["parameters"] == 0


def test_a_benchmark_needs_a_method(tmp_path: pathlib.Path) -> None:
    """Without a model, network or baseline there is nothing to time.

    Args:
        tmp_path (pathlib.Path): Folder for the dataset.
    """
    write_dataset(tmp_path / "test")
    with pytest.raises(ValueError, match="at least one"):
        run_benchmark(tmp_path / "test", {}, [], [], [1], 1, 0, torch.device("cpu"))


@pytest.mark.parametrize(
    ("models", "architectures", "baselines", "message"),
    [
        ({}, ["unet", "unet"], [], "unet"),
        ({}, [], ["gs50", "gs50"], "gs50"),
        ({"unet": pathlib.Path("unet.pth")}, ["unet"], [], "unet"),
        ({}, ["holo"], [], "Unknown architecture"),
    ],
)
def test_methods_must_be_known_and_named_once(
    tmp_path: pathlib.Path,
    models: dict[str, pathlib.Path],
    architectures: list[str],
    baselines: list[str],
    message: str,
) -> None:
    """Two methods under one name would overwrite each other's timings.

    Args:
        tmp_path (pathlib.Path): Folder for the dataset.
        models (dict[str, pathlib.Path]): Trained models to time.
        architectures (list[str]): Untrained networks to time.
        baselines (list[str]): Classical reconstructions to time.
        message (str): Expected part of the error.
    """
    write_dataset(tmp_path / "test")
    with pytest.raises(ValueError, match=message):
        build_methods(read_geometry(tmp_path / "test"), models, architectures, baselines, torch.device("cpu"))


@pytest.mark.parametrize(
    ("batch_sizes", "iterations", "message"),
    [([0], 1, "batch size"), ([1, -2], 1, "batch size"), ([1], 0, "iteration")],
)
def test_the_protocol_must_time_something(
    tmp_path: pathlib.Path, batch_sizes: list[int], iterations: int, message: str
) -> None:
    """An empty batch or no timed pass would give no or meaningless timings.

    Args:
        tmp_path (pathlib.Path): Folder for the dataset.
        batch_sizes (list[int]): Batch sizes to time.
        iterations (int): Timed passes.
        message (str): Expected part of the error.
    """
    write_dataset(tmp_path / "test")
    with pytest.raises(ValueError, match=message):
        run_benchmark(tmp_path / "test", {}, [], ["asm"], batch_sizes, iterations, 0, torch.device("cpu"))


def test_an_empty_test_set_is_a_clear_error(tmp_path: pathlib.Path) -> None:
    """A test set without holograms gives nothing to time.

    Args:
        tmp_path (pathlib.Path): Folder for the dataset.
    """
    write_dataset(tmp_path / "test")
    image: pathlib.Path
    for image in [*(tmp_path / "test").glob("hologram_*.parquet"), *(tmp_path / "test").glob("ground_truth_*.parquet")]:
        image.unlink()
    with pytest.raises(ValueError, match="no holograms"):
        read_holograms(tmp_path / "test", SIZE, 1)
