"""Inference and visualization module for holographic reconstruction.

This module provides functionality to load a trained model and visualize
reconstruction results on test samples, showing the input hologram, dirty
backpropagation result, cleaned output, and ground truth.
"""

import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms

from holopaswin.model import HoloPASWIN

# Point this to dataset folder
DATA_DIR = "./data_train"
MODEL_PATH = "best_swin_holo.pth"

# Dataset/Model Config
IMG_SIZE = 224
WAVELENGTH = 532e-9
PIXEL_SIZE = 4.65e-6
Z_DIST = 0.02


def visualize_inference(model_path: str, data_dir: str, num_samples: int = 3) -> None:
    """Load a trained model and visualize reconstruction results on random samples.

    Creates a visualization showing four images per sample:
    1. Input Hologram
    2. Dirty Backprop (with twin image artifacts)
    3. Swin Output (cleaned reconstruction)
    4. Ground Truth

    The visualization is saved as 'swin_reconstruction_results.png'.

    Args:
        model_path: Path to the saved model checkpoint (.pth file).
        data_dir: Directory containing .npz test files.
        num_samples: Number of random samples to visualize. Defaults to 3.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {model_path} to {device}...")

    # Load Model
    model = HoloPASWIN(IMG_SIZE, WAVELENGTH, PIXEL_SIZE, Z_DIST).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Get Files
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    if not files:
        print("No .npz files found in data directory!")
        return

    # Pick random samples
    indices = np.random.choice(len(files), num_samples, replace=False)

    # Resizer
    resize = transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR)

    # Setup Plot
    _, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]

    cols = [
        "Input Hologram",
        "Dirty Backprop (Twin)",
        "Swin Output (Clean)",
        "Ground Truth",
    ]
    for ax, col in zip(axes[0], cols, strict=False):
        ax.set_title(col, fontsize=14, fontweight="bold")

    print("Running Inference...")

    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Load Data
            data = np.load(files[idx])
            obj_np = data["object"].astype(np.float32)
            holo_np = data["hologram"].astype(np.float32)

            # Preprocess
            holo_t = torch.from_numpy(holo_np).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            obj_t = torch.from_numpy(obj_np).unsqueeze(0).unsqueeze(0)

            # Resize
            holo_t = resize(holo_t).to(device)
            obj_t = resize(obj_t).to(device)

            # Inference
            clean_pred, dirty_intermediate = model(holo_t)

            # Convert to CPU Numpy for plotting
            img_holo = holo_t.squeeze().cpu().numpy()
            img_dirty = dirty_intermediate.squeeze().cpu().numpy()
            img_pred = clean_pred.squeeze().cpu().numpy()
            img_gt = obj_t.squeeze().cpu().numpy()

            # Plotting
            axes[i, 0].imshow(img_holo, cmap="gray")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(img_dirty, cmap="gray")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(img_pred, cmap="gray")
            axes[i, 2].axis("off")

            axes[i, 3].imshow(img_gt, cmap="gray")
            axes[i, 3].axis("off")

    plt.tight_layout()
    plt.savefig("swin_reconstruction_results.png", dpi=150)
    plt.show()
    print("Inference Complete. Saved result to 'swin_reconstruction_results.png'")


if __name__ == "__main__":
    visualize_inference(MODEL_PATH, DATA_DIR, num_samples=3)
