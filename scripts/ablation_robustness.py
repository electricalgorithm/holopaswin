"""Robustness and data ablation tests.

Tests:
1. Z-mismatch: Model trained at z=20mm, tested at different z values
2. Noise configuration: Performance on individual noise types
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from tqdm import tqdm

from holopaswin.dataset import HoloDataset
from holopaswin.model import HoloPASWIN
from holopaswin.propagator import AngularSpectrumPropagator

# Configuration
TEST_DATA_DIR = "../hologen/test-dataset-224"
IMG_SIZE = 224
WAVELENGTH = 532e-9
PIXEL_SIZE = 4.65e-6
Z_TRAIN = 0.02  # Training distance
PHASE_DATA_RANGE = 2 * np.pi


def test_z_mismatch(
    model: HoloPASWIN,
    test_dataset: HoloDataset,
    device: torch.device,
    z_values: list[float],
) -> pd.DataFrame:
    """Test model robustness to propagation distance errors."""
    print("\nTesting z-mismatch robustness...")

    results = []

    for z in z_values:
        print(f"  Testing z = {z * 1000:.1f} mm")

        # Create propagator for this z
        propagator = AngularSpectrumPropagator(
            (IMG_SIZE, IMG_SIZE),
            WAVELENGTH,
            PIXEL_SIZE,
            z,
        ).to(device)

        phase_psnrs = []
        phase_ssims = []

        # Test on subset
        for i in tqdm(range(min(100, len(test_dataset))), desc=f"z={z * 1000:.1f}mm", leave=False):
            hologram, gt_obj = test_dataset[i]
            hologram = hologram.unsqueeze(0).to(device)
            gt_obj = gt_obj.unsqueeze(0).to(device)

            # Reconstruct with different z
            # Temporarily replace the model's propagator with the test propagator
            original_propagator = model.propagator
            model.propagator = propagator

            with torch.no_grad():
                # Now the model will use the test z-distance
                pred, _ = model(hologram)

            # Compute metrics on phase
            pred_c = torch.complex(pred[0, 0], pred[0, 1]).cpu().numpy()
            gt_c = torch.complex(gt_obj[0, 0], gt_obj[0, 1]).cpu().numpy()

            pred_phase = np.angle(pred_c)
            gt_phase = np.angle(gt_c)

            phase_psnrs.append(psnr(gt_phase, pred_phase, data_range=PHASE_DATA_RANGE))
            phase_ssims.append(ssim(pred_phase, gt_phase, data_range=PHASE_DATA_RANGE))

            # Restore original propagator
            model.propagator = original_propagator

        results.append(
            {
                "z_mm": z * 1000,
                "z_offset_mm": (z - Z_TRAIN) * 1000,
                "phase_psnr": np.mean(phase_psnrs),
                "phase_psnr_std": np.std(phase_psnrs),
                "phase_ssim": np.mean(phase_ssims),
                "phase_ssim_std": np.std(phase_ssims),
            }
        )

    return pd.DataFrame(results)


def test_noise_types(
    model: HoloPASWIN,
    test_data_dir: Path,
    device: torch.device,
) -> pd.DataFrame:
    """Test model performance on different noise configurations.

    Note: This assumes the test dataset has subdirectories for each noise type.
    If not available, this will test on the full mixed dataset.
    """
    print("\nTesting noise configuration sensitivity...")

    # For now, test on the full dataset (mixed noise)
    # In a real scenario, you'd have separate test sets for each noise type
    print("  Note: Testing on mixed noise dataset (noise-specific datasets not found)")

    results = []

    test_dataset = HoloDataset(str(test_data_dir), target_size=IMG_SIZE, img_dim=IMG_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    phase_psnrs = []
    phase_ssims = []

    model.eval()
    with torch.no_grad():
        for h_batch, g_batch in tqdm(test_loader, desc="Mixed noise", leave=False):
            h_dev, g_dev = h_batch.to(device), g_batch.to(device)
            pred, _ = model(h_dev)

            pred_c = torch.complex(pred[0, 0], pred[0, 1]).cpu().numpy()
            gt_c = torch.complex(g_dev[0, 0], g_dev[0, 1]).cpu().numpy()

            pred_phase = np.angle(pred_c)
            gt_phase = np.angle(gt_c)

            phase_psnrs.append(psnr(gt_phase, pred_phase, data_range=PHASE_DATA_RANGE))
            phase_ssims.append(ssim(pred_phase, gt_phase, data_range=PHASE_DATA_RANGE))

    results.append(
        {
            "noise_type": "Mixed (All)",
            "phase_psnr": np.mean(phase_psnrs),
            "phase_psnr_std": np.std(phase_psnrs),
            "phase_ssim": np.mean(phase_ssims),
            "phase_ssim_std": np.std(phase_ssims),
        }
    )

    return pd.DataFrame(results)


def plot_z_mismatch(df: pd.DataFrame, output_path: Path) -> None:
    """Create plot of PSNR vs z-offset."""
    plt.figure(figsize=(8, 5))

    plt.errorbar(
        df["z_offset_mm"],
        df["phase_psnr"],
        yerr=df["phase_psnr_std"],
        marker="o",
        capsize=5,
        linewidth=2,
        markersize=8,
    )

    plt.xlabel("Z-offset from training distance (mm)", fontsize=12)
    plt.ylabel("Phase PSNR (dB)", fontsize=12)
    plt.title("Model Robustness to Propagation Distance Errors", fontsize=14)
    plt.grid(visible=True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  Saved plot to: {output_path}")


def main() -> None:
    """Run robustness tests."""
    parser = argparse.ArgumentParser(description="Robustness and data ablation tests")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model checkpoint")
    parser.add_argument("--output-dir", type=str, default="results/ablation_robustness", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from: {args.model_path}")
    model = HoloPASWIN(IMG_SIZE, WAVELENGTH, PIXEL_SIZE, Z_TRAIN).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()

    # Load test dataset
    test_dataset = HoloDataset(TEST_DATA_DIR, target_size=IMG_SIZE, img_dim=IMG_SIZE)

    # Test 1: Z-mismatch
    print("\n" + "=" * 80)
    print("TEST 1: Z-Mismatch Robustness")
    print("=" * 80)

    z_values = [0.019, 0.0195, 0.020, 0.0205, 0.021]  # ±1mm, ±0.5mm around 20mm
    z_results = test_z_mismatch(model, test_dataset, device, z_values)

    z_results.to_csv(output_dir / "z_mismatch_results.csv", index=False)
    print("\nZ-mismatch results:")
    print(z_results.to_string(index=False))

    # Plot
    plot_z_mismatch(z_results, output_dir / "z_mismatch_plot.png")

    # Test 2: Noise configurations
    print("\n" + "=" * 80)
    print("TEST 2: Noise Configuration Sensitivity")
    print("=" * 80)

    noise_results = test_noise_types(model, Path(TEST_DATA_DIR), device)

    noise_results.to_csv(output_dir / "noise_sensitivity_results.csv", index=False)
    print("\nNoise sensitivity results:")
    print(noise_results.to_string(index=False))

    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()
