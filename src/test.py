"""Model evaluation and testing module.

This module provides functionality to evaluate a trained model on a test dataset,
computing standard image quality metrics including SSIM, MSE, and PSNR.
"""

import os

import numpy as np
import torch
import torch.utils.data
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

from holopaswin.dataset import HoloDataset
from holopaswin.model import HoloPASWIN

# Model/Dataset Configuration
IMG_SIZE = 224
WAVELENGTH = 532e-9
PIXEL_SIZE = 4.65e-6
Z_DIST = 0.02

# Define paths
MODEL_PATH = "results/experiment3/holopaswin_exp3.pth"
TEST_DATA_DIR = "../hologen/inline-digital-holography-v2"

# Number of samples to test
SAMPLES_TO_TEST = 500


def calculate_metrics_on_dataset(model_path: str, data_dir: str, num_samples: int = 100) -> dict[str, float] | None:  # noqa: PLR0915
    """Evaluate a trained model on a test dataset and compute image quality metrics.

    Args:
        model_path: Path to the saved model checkpoint (.pth file).
        data_dir: Directory containing parquet files.
        num_samples: Number of random samples to evaluate. Defaults to 100.

    Returns:
        A dictionary containing aggregated metrics.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"--- Starting Evaluation on {device} ---")

    # 1. Load Model
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return None

    model = HoloPASWIN(IMG_SIZE, WAVELENGTH, PIXEL_SIZE, Z_DIST).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:  # noqa: BLE001
        print(f"Error loading model weights: {e}")
        return None
    model.eval()

    # 2. Load Dataset
    print(f"Loading dataset from {data_dir}...")
    try:
        dataset = HoloDataset(data_dir, target_size=IMG_SIZE)
    except Exception as e:  # noqa: BLE001
        print(f"Error loading dataset: {e}")
        return None

    if len(dataset) == 0:
        print("Dataset is empty!")
        return None

    # Select random samples
    total_samples = len(dataset)
    num_samples = min(num_samples, total_samples)
    selected_indices = np.random.choice(total_samples, num_samples, replace=False)

    print(f"Evaluating on {num_samples} random samples...")

    # Storage for metrics (Amplitude based)
    mse_scores = []
    ssim_scores = []
    psnr_scores = []

    # 3. Evaluation Loop
    with torch.no_grad():
        for i, idx in enumerate(selected_indices):
            # Load Data
            holo_t, gt_obj_t = dataset[idx]  # holo: (1,H,W), gt: (2,H,W) [real, imag]

            # Prepare Batch (Normalize / 1000.0 if not done by dataset)
            # Dataset DOES normalize by 1000.0, so we just unsqueeze
            # The previous inference script had a bug where it normalized twice or once depending on context.
            # The dataset definition sets holo = h_np / 1000.0.
            holo_in = holo_t.unsqueeze(0).to(device)

            # Run Inference
            clean_pred, _ = model(holo_in)

            # --- Convert to Numpy for Metrics ---
            # We calculate metrics on AMPLITUDE for standard comparisons

            # Predict Amplitude
            pred_c = torch.complex(clean_pred[:, 0, :, :], clean_pred[:, 1, :, :])
            pred_amp = torch.abs(pred_c).squeeze().cpu().numpy()

            # GT Amplitude
            gt_c = torch.complex(gt_obj_t[0], gt_obj_t[1])
            gt_amp = torch.abs(gt_c).numpy()

            # Ensure range / Clipping for stability
            # GT is typically [0, 1] for amplitude
            # Pred might overshoot
            pred_np = np.clip(pred_amp, 0, 1.2)  # Clip slightly above 1
            gt_np = np.clip(gt_amp, 0, 1.2)  # Clip slightly above 1

            # Normalize to 0-1 for SSIM/PSNR standard calculation if max > 1
            # But here we treat values as absolute physics quantities.
            # SSIM data_range needs to be specified.

            # --- Calculate Metrics ---
            # 1. MSE: Lower is better. (0 = perfect match)
            val_mse = mse(gt_np, pred_np)  # type: ignore[no-untyped-call]

            # 2. SSIM: Higher is better. (1.0 = perfect match)
            val_ssim = ssim(gt_np, pred_np, data_range=1.2)  # type: ignore[no-untyped-call]

            # 3. PSNR: Higher is better.
            val_psnr = psnr(gt_np, pred_np, data_range=1.2)  # type: ignore[no-untyped-call]

            mse_scores.append(val_mse)
            ssim_scores.append(val_ssim)
            psnr_scores.append(val_psnr)

            if (i + 1) % 50 == 0:
                print(f"Sample {i + 1}/{num_samples} -> SSIM: {val_ssim:.4f} | PSNR: {val_psnr:.2f}dB")

    # 4. Aggregation
    results = {
        "count": num_samples,
        "mse_mean": np.mean(mse_scores),
        "mse_std": np.std(mse_scores),
        "ssim_mean": np.mean(ssim_scores),
        "ssim_std": np.std(ssim_scores),
        "psnr_mean": np.mean(psnr_scores),
        "psnr_std": np.std(psnr_scores),
    }

    print("\n" + "=" * 40)
    print("   FINAL MODEL ACCURACY REPORT (Exp 3)   ")
    print("=" * 40)
    print(f"Evaluated on: {results['count']} samples")
    print("-" * 40)
    print(f"MSE (Mean Squared Error):     {results['mse_mean']:.6f} (±{results['mse_std']:.6f})")
    print(f"SSIM (Structural Similarity): {results['ssim_mean']:.4f}   (±{results['ssim_std']:.4f})")
    print(f"PSNR (Peak Signal-Noise):     {results['psnr_mean']:.2f} dB  (±{results['psnr_std']:.2f})")
    print("-" * 40)
    print("Interpretation:")
    print(" - SSIM > 0.9 is considered excellent.")
    print(" - PSNR > 30 dB is considered excellent.")
    print("=" * 40)

    return results


if __name__ == "__main__":
    calculate_metrics_on_dataset(MODEL_PATH, TEST_DATA_DIR, SAMPLES_TO_TEST)
