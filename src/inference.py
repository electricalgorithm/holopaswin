"""Inference and visualization module for holographic reconstruction.

This module provides functionality to load a trained model and visualize
reconstruction results on test samples, showing the input hologram, dirty
backpropagation result, cleaned output, and ground truth.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data

from holopaswin.dataset import HoloDataset
from holopaswin.model import HoloPASWIN

# Point this to dataset folder
DATA_DIR = "../hologen/inline-digital-holography-v2"
MODEL_PATH = "results/experiment3/holopaswin_exp3.pth"

# Dataset/Model Config
IMG_SIZE = 224
WAVELENGTH = 532e-9
PIXEL_SIZE = 4.65e-6
Z_DIST = 0.02


def visualize_inference(model_path: str, data_dir: str, num_samples: int = 3) -> None:  # noqa: PLR0915
    """Load a trained model and visualize reconstruction results on random samples.

    Creates a visualization showing four images per sample:
    1. Input Hologram
    2. Dirty Backprop (with twin image artifacts)
    3. Swin Output (Cleaned Amplitude)
    4. Swin Output (Cleaned Phase)
    5. Ground Truth (Amplitude)
    6. Ground Truth (Phase)

    The visualization is saved as 'exp3_inference_results.png'.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Loading model from {model_path} to {device}...")

    # Load Model
    model = HoloPASWIN(IMG_SIZE, WAVELENGTH, PIXEL_SIZE, Z_DIST).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return
    except Exception as e:  # noqa: BLE001
        print(f"Error loading weights: {e}")
        return

    model.eval()

    # Load Dataset
    print(f"Loading dataset from {data_dir}...")
    try:
        dataset = HoloDataset(data_dir, target_size=IMG_SIZE)
    except Exception as e:  # noqa: BLE001
        print(f"Error loading dataset: {e}")
        return

    if len(dataset) == 0:
        print("Dataset is empty!")
        return

    # Pick random samples
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    # Setup Plot (Rows=Samples, Cols=6)
    # 1. Holo, 2. Dirty Amp, 3. Pred Amp, 4. Pred Phase, 5. GT Amp, 6. GT Phase
    cols = ["Input Hologram", "Dirty Backprop", "Pred Amplitude", "Pred Phase", "GT Amplitude", "GT Phase"]

    _fig, axes = plt.subplots(num_samples, 6, figsize=(20, 3.5 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]

    for ax, col in zip(axes[0], cols, strict=False):
        ax.set_title(col, fontsize=12, fontweight="bold")

    print(f"Running Inference on indices {indices}...")

    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Get Item from Dataset
            holo_t, gt_obj_t = dataset[idx]  # holo: (1,H,W), gt: (2,H,W) [real, imag]

            # Prepare Batch
            # Dataset already normalizes by 1000.0
            holo_in = holo_t.unsqueeze(0).to(device)  # (1,1,H,W)

            # Inference
            # Forward returns: (clean_complex_2ch, dirty_complex_2ch)
            # dirty_complex is the backpropagated field
            clean_pred, dirty_intermediate = model(holo_in)  # (1,2,H,W)

            # --- PROCESS FOR DISPLAY ---

            # 1. Input Hologram
            img_holo = holo_t.squeeze().numpy()

            # 2. Dirty Backprop (Amplitude)
            # dirty_intermediate shape is (B, 2, H, W)
            dirty_real = dirty_intermediate[:, 0, :, :]
            dirty_imag = dirty_intermediate[:, 1, :, :]
            dirty_complex = torch.complex(dirty_real, dirty_imag)
            img_dirty = torch.abs(dirty_complex).squeeze().cpu().numpy()

            # 3/4. Prediction (Amp/Phase)
            pred_real = clean_pred[:, 0, :, :]
            pred_imag = clean_pred[:, 1, :, :]
            pred_c = torch.complex(pred_real, pred_imag)

            img_pred_amp = torch.abs(pred_c).squeeze().cpu().numpy()
            img_pred_phase = torch.angle(pred_c).squeeze().cpu().numpy()

            # 5/6. Ground Truth (Amp/Phase)
            gt_real = gt_obj_t[0]
            gt_imag = gt_obj_t[1]
            gt_c = torch.complex(gt_real, gt_imag)

            img_gt_amp = torch.abs(gt_c).numpy()
            img_gt_phase = torch.angle(gt_c).numpy()

            # DEBUG STATS
            print(f"\n--- Sample {i} ---")
            print(f"Holo  : min={img_holo.min():.4f}, max={img_holo.max():.4f}, mean={img_holo.mean():.4f}")
            print(f"Dirty : min={img_dirty.min():.4f}, max={img_dirty.max():.4f}, mean={img_dirty.mean():.4f}")
            print(f"Pred  : min={img_pred_amp.min():.4f}, max={img_pred_amp.max():.4f}, mean={img_pred_amp.mean():.4f}")
            print(f"GT    : min={img_gt_amp.min():.4f}, max={img_gt_amp.max():.4f}, mean={img_gt_amp.mean():.4f}")

            # --- PLOTTING ---
            # Helper to clear axis
            for ax in axes[i]:
                ax.axis("off")

            # Amplitude plots: fixed range [0, 1.2] to handle potential overshoot > 1.0
            axes[i, 0].imshow(img_holo, cmap="gray", vmin=0, vmax=1.2)
            axes[i, 1].imshow(img_dirty, cmap="gray", vmin=0, vmax=1.2)
            axes[i, 2].imshow(img_pred_amp, cmap="inferno", vmin=0, vmax=1.2)
            axes[i, 3].imshow(img_pred_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
            axes[i, 4].imshow(img_gt_amp, cmap="inferno", vmin=0, vmax=1.2)
            axes[i, 5].imshow(img_gt_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)

    plt.tight_layout()
    output_file = "exp3_inference_results.png"
    plt.savefig(output_file, dpi=150)
    print(f"Inference Complete. Saved result to '{output_file}'")
    plt.show()


if __name__ == "__main__":
    visualize_inference(MODEL_PATH, DATA_DIR, num_samples=3)
