"""Model evaluation and testing module.

This module provides functionality to evaluate a trained model on a test dataset,
computing standard image quality metrics including SSIM, MSE, and PSNR.
"""

import glob
import os
import random

import numpy as np
import torch
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms

from holopaswin.model import HoloPASWIN

# Model/Dataset Configuration
IMG_SIZE = 224
WAVELENGTH = 532e-9
PIXEL_SIZE = 4.65e-6
Z_DIST = 0.02

# Define paths
MODEL_PATH = "best_swin_holo.pth"
TEST_DATA_DIR = "./data_test"

# Number of samples to test
SAMPLES_TO_TEST = 2000


def calculate_metrics_on_dataset(model_path: str, data_dir: str, num_samples: int = 100) -> dict[str, float] | None:  # noqa: PLR0915
    """Evaluate a trained model on a test dataset and compute image quality metrics.

    Loads the model, randomly selects samples from the test directory,
    computes SSIM, MSE, and PSNR for each sample, and returns aggregated
    statistics (mean and standard deviation).

    Args:
        model_path: Path to the saved model checkpoint (.pth file).
        data_dir: Directory containing .npz test files.
        num_samples: Number of random samples to evaluate. Defaults to 100.

    Returns:
        A dictionary containing:
            - count: Number of samples evaluated
            - mse_mean: Mean MSE across all samples
            - mse_std: Standard deviation of MSE
            - ssim_mean: Mean SSIM across all samples
            - ssim_std: Standard deviation of SSIM
            - psnr_mean: Mean PSNR across all samples (in dB)
            - psnr_std: Standard deviation of PSNR
        Returns None if no files are found in the data directory.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Starting Evaluation on {device} ---")

    # 1. Load Model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")

    model = HoloPASWIN(IMG_SIZE, WAVELENGTH, PIXEL_SIZE, Z_DIST).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Load Data List
    files = glob.glob(os.path.join(data_dir, "*.npz"))
    if len(files) == 0:
        print(f"Error: No .npz files found in {data_dir}")
        return None

    # Select samples
    num_samples = min(num_samples, len(files))
    selected_indices = random.sample(range(len(files)), num_samples)
    print(f"Evaluating on {num_samples} random samples...")

    # Resizing transform (Must match training!)
    resizer = transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=transforms.InterpolationMode.BILINEAR)

    # Storage for metrics
    mse_scores = []
    ssim_scores = []
    psnr_scores = []

    # 3. Evaluation Loop
    with torch.no_grad():
        for i, idx in enumerate(selected_indices):
            # Load Data
            data = np.load(files[idx])

            # Prepare Ground Truth (Object)
            obj_raw = data["object"].astype(np.float32)
            obj_tensor = torch.from_numpy(obj_raw).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            obj_tensor = resizer(obj_tensor).to(device)

            # Prepare Input (Hologram)
            holo_raw = data["hologram"].astype(np.float32)
            holo_tensor = torch.from_numpy(holo_raw).unsqueeze(0).unsqueeze(0)
            holo_tensor = resizer(holo_tensor).to(device)

            # Run Inference
            clean_pred, _ = model(holo_tensor)

            # --- Convert to Numpy for Metrics ---
            # We squeeze to (H, W) because sk-image metrics expect 2D arrays for grayscale
            pred_np = clean_pred.squeeze().cpu().numpy()
            gt_np = obj_tensor.squeeze().cpu().numpy()

            # Ensure range is [0, 1] for SSIM stability
            pred_np = np.clip(pred_np, 0, 1)
            gt_np = np.clip(gt_np, 0, 1)

            # --- Calculate Metrics ---
            # 1. MSE: Lower is better. (0 = perfect match)
            val_mse = mse(gt_np, pred_np)  # type: ignore[no-untyped-call]

            # 2. SSIM: Higher is better. (1.0 = perfect match)
            # data_range=1.0 is crucial because our images are float 0-1
            val_ssim = ssim(gt_np, pred_np, data_range=1.0)  # type: ignore[no-untyped-call]

            # 3. PSNR: Higher is better. (Infinity = perfect match, >30 is usually excellent)
            val_psnr = psnr(gt_np, pred_np, data_range=1.0)  # type: ignore[no-untyped-call]

            mse_scores.append(val_mse)
            ssim_scores.append(val_ssim)
            psnr_scores.append(val_psnr)

            if i % 10 == 0:
                print(f"Sample {i}/{num_samples} -> SSIM: {val_ssim:.4f} | PSNR: {val_psnr:.2f}dB")

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
    print("   FINAL MODEL ACCURACY REPORT   ")
    print("=" * 40)
    print(f"Evaluated on: {results['count']} samples")
    print("-" * 40)
    print(f"MSE (Mean Squared Error):     {results['mse_mean']:.6f} (±{results['mse_std']:.6f})")
    print(f"SSIM (Structural Similarity): {results['ssim_mean']:.4f}   (±{results['ssim_std']:.4f})")
    print(f"PSNR (Peak Signal-Noise):     {results['psnr_mean']:.2f} dB  (±{results['psnr_std']:.2f})")
    print("-" * 40)
    print("Interpretation:")
    print(" - SSIM > 0.9 is considered excellent structural fidelity.")
    print(" - PSNR > 30 dB typically indicates high quality reconstruction.")
    print("=" * 40)

    return results


if __name__ == "__main__":
    try:
        metrics = calculate_metrics_on_dataset(MODEL_PATH, TEST_DATA_DIR, SAMPLES_TO_TEST)
    except FileNotFoundError as e:
        print(f"\nValidation Failed: {e}")
        print("Ensure you have downloaded data into ./data_test folder.")
