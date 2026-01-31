"""Inference benchmark script for HoloPASWIN."""

import time

import numpy as np
import torch

from holopaswin.model import HoloPASWIN

# Config
IMG_SIZE = 224
WAVELENGTH = 532e-9
PIXEL_SIZE = 4.65e-6
Z_DIST = 0.02
MODEL_PATH = "results/experiment9/holopaswin_exp9.pth"
NUM_ITER = 100
WARMUP = 20


def benchmark() -> None:
    """Run inference benchmark on selected device."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    print(f"Benchmarking on {device}...")

    # Init model
    model = HoloPASWIN(IMG_SIZE, WAVELENGTH, PIXEL_SIZE, Z_DIST).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:  # noqa: BLE001
        print(f"Warning: Could not load model from {MODEL_PATH}. Benchmarking with random weights. Error: {e}")
    model.eval()

    # Dummy input
    dummy_input = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(device)

    # Warmup
    print(f"Warmup ({WARMUP} iterations)...")
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(dummy_input)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()

    # Benchmark
    print(f"Benchmark ({NUM_ITER} iterations)...")
    times = []
    with torch.no_grad():
        for _ in range(NUM_ITER):
            start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    avg_time = np.mean(times) * 1000  # ms
    std_time = np.std(times) * 1000  # ms
    fps = 1.0 / np.mean(times)

    print("-" * 30)
    print(f"Avg Inference Time: {avg_time:.2f} ms")
    print(f"Std Dev:           {std_time:.2f} ms")
    print(f"Throughput:        {fps:.2f} FPS")
    print("-" * 30)

    print("\nDraft sentence for article:")
    print(
        f"The inference time per $224 \\times 224$ hologram is approximately {avg_time:.1f} ms on an Apple M2 Pro, corresponding to a throughput of {fps:.1f} frames per second (FPS), making it suitable for real-time applications."
    )


if __name__ == "__main__":
    benchmark()
