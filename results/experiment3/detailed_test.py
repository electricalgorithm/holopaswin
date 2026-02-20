"""Detailed evaluation script for Phase and Magnitude analysis.

This script performs a deep-dive analysis on a small subset of the test dataset (10 samples).
It computes and reports SSIM and PSNR separately for the Magnitude (Amplitude) and Phase
domains of the reconstructed complex field. It also generates a comparative visualization.
"""  # noqa: INP001

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from holopaswin.dataset import HoloDataset
from holopaswin.model import HoloPASWIN

# Configuration
IMG_SIZE = 224
WAVELENGTH = 532e-9
PIXEL_SIZE = 4.65e-6
Z_DIST = 0.02
BATCH_SIZE = 1

MODEL_PATH = "results/experiment3/holopaswin_exp3.pth"
DATA_DIR = "../hologen/inline-digital-holography-v2"
OUTPUT_IMG = "detailed_test_comparison.png"
NUM_SAMPLES = 10


def evaluate_detailed_metrics(model_path: str, data_dir: str) -> None:  # noqa: PLR0915
    """Run detailed evaluation on phase and magnitude."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"--- Starting Detailed Evaluation on {device} ---")

    # 1. Load Model
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return

    model = HoloPASWIN(IMG_SIZE, WAVELENGTH, PIXEL_SIZE, Z_DIST).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:  # noqa: BLE001
        print(f"Error loading weights: {e}")
        return
    model.eval()

    # 2. Load Dataset
    print(f"Loading dataset from {data_dir}...")
    try:
        dataset = HoloDataset(data_dir, target_size=IMG_SIZE)
    except Exception as e:  # noqa: BLE001
        print(f"Error loading dataset: {e}")
        return

    if len(dataset) == 0:
        print("Dataset is empty!")
        return

    # Select random samples
    indices = np.random.choice(len(dataset), NUM_SAMPLES, replace=False)
    print(f"Selected indices: {indices}")

    # Prepare plotting
    _fig, axes = plt.subplots(NUM_SAMPLES, 5, figsize=(20, 4 * NUM_SAMPLES))
    cols = ["Hologram", "Pred Amp", "GT Amp", "Pred Phase", "GT Phase"]
    for ax, col in zip(axes[0], cols, strict=False):
        ax.set_title(col, fontsize=12, fontweight="bold")

    print("\n" + "=" * 80)
    print(f"{'Sample':<8} | {'Amp SSIM':<10} | {'Amp PSNR':<10} | {'Phase SSIM':<10} | {'Phase PSNR':<10}")
    print("-" * 80)

    avg_amp_ssim = 0.0
    avg_amp_psnr = 0.0
    avg_phase_ssim = 0.0
    avg_phase_psnr = 0.0

    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Load Data
            holo_t, gt_obj_t = dataset[idx]  # holo: (1,H,W), gt: (2,H,W)
            holo_in = holo_t.unsqueeze(0).to(device)

            # Inference
            clean_pred, _ = model(holo_in)

            # --- PROCESS OUTPUTS ---
            # 1. Prediction Complex Field
            pred_c = torch.complex(clean_pred[:, 0, :, :], clean_pred[:, 1, :, :])
            pred_amp = torch.abs(pred_c).squeeze().cpu().numpy()
            pred_phase = torch.angle(pred_c).squeeze().cpu().numpy()

            # 2. Ground Truth Complex Field
            gt_c = torch.complex(gt_obj_t[0], gt_obj_t[1])
            gt_amp = torch.abs(gt_c).numpy()
            gt_phase = torch.angle(gt_c).numpy()

            # 3. Hologram
            img_holo = holo_t.squeeze().numpy()

            # --- METRICS CALCULATION ---

            # Amplitude Metrics
            # Range: [0, 1.2] roughly. We assume physics range 1.0 for standard metric scaling,
            # or we can use dynamic range. Using 1.0 is standard if data is mostly [0,1].
            # However, phase is [-pi, pi], range ~ 6.28.

            data_range_amp = 1.2
            data_range_phase = 2 * np.pi

            score_amp_ssim = ssim(gt_amp, pred_amp, data_range=data_range_amp)  # type: ignore[no-untyped-call]
            score_amp_psnr = psnr(gt_amp, pred_amp, data_range=data_range_amp)  # type: ignore[no-untyped-call]

            score_phase_ssim = ssim(gt_phase, pred_phase, data_range=data_range_phase)  # type: ignore[no-untyped-call]
            score_phase_psnr = psnr(gt_phase, pred_phase, data_range=data_range_phase)  # type: ignore[no-untyped-call]

            # Accumulate
            avg_amp_ssim += score_amp_ssim
            avg_amp_psnr += score_amp_psnr
            avg_phase_ssim += score_phase_ssim
            avg_phase_psnr += score_phase_psnr

            print(
                f"{i:<8} | {score_amp_ssim:.4f}     | {score_amp_psnr:.2f} dB    | {score_phase_ssim:.4f}       | {score_phase_psnr:.2f} dB"
            )

            # --- PLOTTING ---
            ax_row = axes[i]

            # Hologram
            ax_row[0].imshow(img_holo, cmap="gray")
            ax_row[0].axis("off")

            # Amplitude
            ax_row[1].imshow(pred_amp, cmap="inferno", vmin=0, vmax=1.2)
            ax_row[1].axis("off")
            ax_row[2].imshow(gt_amp, cmap="inferno", vmin=0, vmax=1.2)
            ax_row[2].axis("off")

            # Phase
            ax_row[3].imshow(pred_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
            ax_row[3].axis("off")
            ax_row[4].imshow(gt_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
            ax_row[4].axis("off")

    # Averages
    avg_amp_ssim /= NUM_SAMPLES
    avg_amp_psnr /= NUM_SAMPLES
    avg_phase_ssim /= NUM_SAMPLES
    avg_phase_psnr /= NUM_SAMPLES

    print("-" * 80)
    print(
        f"{'AVERAGE':<8} | {avg_amp_ssim:.4f}     | {avg_amp_psnr:.2f} dB    | {avg_phase_ssim:.4f}       | {avg_phase_psnr:.2f} dB"
    )
    print("=" * 80)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=150)
    print(f"\nSaved comparison figure to {OUTPUT_IMG}")
    plt.show()


if __name__ == "__main__":
    evaluate_detailed_metrics(MODEL_PATH, DATA_DIR)
