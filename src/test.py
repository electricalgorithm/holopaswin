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
# Should point to the dataset directory relative to where script is run
# EXP 8
MODEL_PATH = "results/experiment8/holopaswin_exp8.pth"
TEST_DATA_DIR = "../hologen/test-dataset-224"

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
    print(f"Loading dataset from: {TEST_DATA_DIR}")
    try:
        # EXP 8 uses 224x224 native dataset
        dataset = HoloDataset(data_dir=TEST_DATA_DIR, target_size=IMG_SIZE, img_dim=IMG_SIZE)
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

    # Storage for metrics
    amp_mse_list = []
    amp_ssim_list = []
    amp_psnr_list = []

    phase_mse_list = []
    phase_ssim_list = []
    phase_psnr_list = []

    # Complex MSE as a single scalar: mean(|pred - gt|^2)
    complex_mse_list = []

    # 3. Evaluation Loop
    with torch.no_grad():
        for i, idx in enumerate(selected_indices):
            # Load Data
            holo_t, gt_obj_t = dataset[idx]  # holo: (1,H,W), gt: (2,H,W) [real, imag]

            # Prepare Batch
            holo_in = holo_t.unsqueeze(0).to(device)

            # Run Inference
            clean_pred, _ = model(holo_in)

            # --- PREPARE DATA ---
            # Prediction
            pred_c = torch.complex(clean_pred[:, 0, :, :], clean_pred[:, 1, :, :])
            pred_amp = torch.abs(pred_c).squeeze().cpu().numpy()
            pred_phase = torch.angle(pred_c).squeeze().cpu().numpy()
            
            # Ground Truth
            gt_c = torch.complex(gt_obj_t[0], gt_obj_t[1])
            gt_amp = torch.abs(gt_c).numpy()
            gt_phase = torch.angle(gt_c).numpy()
            
            # --- 1. AMPLITUDE METRICS ---
            # Clip for stability
            pred_amp_c = np.clip(pred_amp, 0, 1.2)
            gt_amp_c = np.clip(gt_amp, 0, 1.2)
            
            val_amp_mse = mse(gt_amp_c, pred_amp_c)  # type: ignore[no-untyped-call]
            val_amp_ssim = ssim(gt_amp_c, pred_amp_c, data_range=1.2)  # type: ignore[no-untyped-call]
            val_amp_psnr = psnr(gt_amp_c, pred_amp_c, data_range=1.2)  # type: ignore[no-untyped-call]
            
            amp_mse_list.append(val_amp_mse)
            amp_ssim_list.append(val_amp_ssim)
            amp_psnr_list.append(val_amp_psnr)

            # --- 2. PHASE METRICS ---
            # Phase is naturally [-pi, pi].
            # Direct comparison is tricky due to wrapping, but standard metrics assume linear scale.
            # We use data_range = 2 * pi.
            data_range_phase = 2 * np.pi
            
            val_phase_mse = mse(gt_phase, pred_phase)  # type: ignore[no-untyped-call]
            val_phase_ssim = ssim(gt_phase, pred_phase, data_range=data_range_phase)  # type: ignore[no-untyped-call]
            val_phase_psnr = psnr(gt_phase, pred_phase, data_range=data_range_phase)  # type: ignore[no-untyped-call]
            
            phase_mse_list.append(val_phase_mse)
            phase_ssim_list.append(val_phase_ssim)
            phase_psnr_list.append(val_phase_psnr)

            # --- 3. COMPLEX METRICS ---
            # Complex MSE: mean(|z_pred - z_gt|^2)
            diff = pred_c.cpu().numpy().squeeze() - gt_c.numpy()
            val_complex_mse = np.mean(np.abs(diff) ** 2)
            complex_mse_list.append(val_complex_mse)

            if (i + 1) % 50 == 0:
                print(f"Sample {i + 1}/{num_samples} -> Amp SSIM: {val_amp_ssim:.4f} | Phase SSIM: {val_phase_ssim:.4f}")

    # 4. Aggregation
    results = {
        "count": num_samples,
        "amp": {
            "mse": (np.mean(amp_mse_list), np.std(amp_mse_list)),
            "ssim": (np.mean(amp_ssim_list), np.std(amp_ssim_list)),
            "psnr": (np.mean(amp_psnr_list), np.std(amp_psnr_list)),
        },
        "phase": {
            "mse": (np.mean(phase_mse_list), np.std(phase_mse_list)),
            "ssim": (np.mean(phase_ssim_list), np.std(phase_ssim_list)),
            "psnr": (np.mean(phase_psnr_list), np.std(phase_psnr_list)),
        },
        "complex": {
            "mse": (np.mean(complex_mse_list), np.std(complex_mse_list)),
        }
    }

    print("\n" + "=" * 60)
    print("   FINAL MODEL ACCURACY REPORT   ")
    print("=" * 60)
    print(f"Evaluated on: {results['count']} samples")
    print("-" * 60)
    print("AMPLITUDE DOMAIN (Absorption/Objects)")
    print(f"  MSE:  {results['amp']['mse'][0]:.6f} (±{results['amp']['mse'][1]:.6f})")
    print(f"  SSIM: {results['amp']['ssim'][0]:.4f}   (±{results['amp']['ssim'][1]:.4f})")
    print(f"  PSNR: {results['amp']['psnr'][0]:.2f} dB  (±{results['amp']['psnr'][1]:.2f})")
    print("-" * 60)
    print("PHASE DOMAIN (Thickness/Refractive Index)")
    print(f"  MSE:  {results['phase']['mse'][0]:.6f} (±{results['phase']['mse'][1]:.6f})")
    print(f"  SSIM: {results['phase']['ssim'][0]:.4f}   (±{results['phase']['ssim'][1]:.4f})")
    print(f"  PSNR: {results['phase']['psnr'][0]:.2f} dB  (±{results['phase']['psnr'][1]:.2f})")
    print("-" * 60)
    print("COMPLEX DOMAIN (Overall Fidelity)")
    print(f"  MSE:  {results['complex']['mse'][0]:.6f} (±{results['complex']['mse'][1]:.6f})")
    print("=" * 60)

    return results


if __name__ == "__main__":
    calculate_metrics_on_dataset(MODEL_PATH, TEST_DATA_DIR, SAMPLES_TO_TEST)
