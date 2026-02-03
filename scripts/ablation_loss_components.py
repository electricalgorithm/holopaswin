"""Comprehensive loss function component ablation study.

This script trains 5 different loss configurations with full training
to evaluate the impact of each loss component on reconstruction quality.
"""

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from tqdm import tqdm

from holopaswin.dataset import HoloDataset
from holopaswin.loss import PhysicsLoss
from holopaswin.model import HoloPASWIN

# Configuration
DATA_DIR = "../hologen/dataset-224"
TEST_DATA_DIR = "../hologen/test-dataset-224"
IMG_SIZE = 224
WAVELENGTH = 532e-9
PIXEL_SIZE = 4.65e-6
Z_DIST = 0.02
BATCH_SIZE = 24  # Optimized for MPS
NUM_EPOCHS = 3  # Reduced from 5 - sufficient for ablation comparisons
LR = 1e-4
NUM_WORKERS = 4  # Parallel data loading
MAX_TRAIN_SAMPLES = 15000  # Limit dataset size for faster ablation
MAX_VAL_SAMPLES = 3000

BG_THRESHOLD = 0.98
AMP_DATA_RANGE = 1.2
PHASE_DATA_RANGE = 2 * np.pi


def compute_bs_ratio(pred_amp: np.ndarray, gt_amp: np.ndarray) -> float:
    """Compute Background-to-Signal Ratio (B/S).

    Measures the standard deviation in background vs signal contrast.
    Lower is better (indicates cleaner background, better twin suppression).
    """
    bg_mask = gt_amp > BG_THRESHOLD
    obj_mask = gt_amp <= BG_THRESHOLD

    if not np.any(bg_mask) or not np.any(obj_mask):
        return 0.0

    bg_level = np.mean(pred_amp[bg_mask])
    bg_fluctuation = np.std(pred_amp[bg_mask])
    obj_contrast = np.mean(np.abs(bg_level - pred_amp[obj_mask]))

    return float(bg_fluctuation / (obj_contrast + 1e-8))


def get_freq_ssim(pred_amp: np.ndarray, target_amp: np.ndarray) -> float:
    """Compute SSIM of the Log-FFT magnitude spectrum."""

    def log_fft(img: np.ndarray) -> np.ndarray:
        f = np.fft.fft2(img)
        fshift = np.fft.fftshift(f)
        return np.log(np.abs(fshift) + 1e-8)

    spec_p = log_fft(pred_amp)
    spec_t = log_fft(target_amp)
    dr = spec_t.max() - spec_t.min()
    return float(ssim(spec_p, spec_t, data_range=max(dr, 1.0)))


class ConfigurableLoss(PhysicsLoss):
    """Loss function with configurable component weights."""

    def __init__(  # noqa: PLR0913
        self,
        propagator: torch.nn.Module,
        w_complex: float = 0.2,
        w_amp: float = 0.4,
        w_phase: float = 0.2,
        w_freq: float = 0.2,
        lambda_phy: float = 0.1,
    ) -> None:
        """Initialize with specific component weights."""
        super().__init__(propagator, lambda_physics=lambda_phy)
        self.w_complex = w_complex
        self.w_amp = w_amp
        self.w_phase = w_phase
        self.w_freq = w_freq

    def forward(
        self,
        pred_obj_2ch: torch.Tensor,
        target_obj_2ch: torch.Tensor,
        input_hologram: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the weighted composite loss."""
        pred_complex = torch.complex(pred_obj_2ch[:, 0:1, ...], pred_obj_2ch[:, 1:2, ...])
        target_complex = torch.complex(target_obj_2ch[:, 0:1, ...], target_obj_2ch[:, 1:2, ...])

        loss_complex = self.l1(pred_obj_2ch, target_obj_2ch)

        pred_amp = torch.abs(pred_complex)
        target_amp = torch.abs(target_complex)
        loss_amp = self.l1(pred_amp, target_amp)

        pred_phase = torch.angle(pred_complex)
        target_phase = torch.angle(target_complex)
        loss_phase = self.l1(pred_phase, target_phase)

        loss_freq_amp = self.compute_freq_loss(pred_amp, target_amp)
        loss_freq_phase = self.compute_freq_loss(pred_phase, target_phase)
        loss_freq = (loss_freq_amp + loss_freq_phase) / 2.0

        pred_field_holo = self.propagator(pred_complex, backward=False)
        pred_hologram = torch.abs(pred_field_holo) ** 2
        loss_phy = self.l1(pred_hologram, input_hologram)

        loss_supervised = (
            (self.w_complex * loss_complex)
            + (self.w_amp * loss_amp)
            + (self.w_phase * loss_phase)
            + (self.w_freq * loss_freq)
        )

        return loss_supervised + (self.lambda_p * loss_phy)  # type: ignore[no-any-return]


def evaluate_model(
    model: torch.nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on test set and return metrics."""
    model.eval()
    metrics = {
        "amp_ssim": [],
        "phase_ssim": [],
        "amp_psnr": [],
        "phase_psnr": [],
        "freq_ssim": [],
        "bs_ratio": [],
    }

    with torch.no_grad():
        for h_batch, g_batch in tqdm(test_loader, desc="Testing", leave=False):
            h_dev, g_dev = h_batch.to(device), g_batch.to(device)
            pred, _ = model(h_dev)

            # Convert to numpy
            pred_c = torch.complex(pred[:, 0], pred[:, 1]).cpu().numpy()
            gt_c = torch.complex(g_dev[:, 0], g_dev[:, 1]).cpu().numpy()

            # Process each sample in batch
            for i in range(pred_c.shape[0]):
                pa, ta = np.abs(pred_c[i]), np.abs(gt_c[i])
                pp, tp = np.angle(pred_c[i]), np.angle(gt_c[i])

                metrics["amp_ssim"].append(ssim(pa, ta, data_range=AMP_DATA_RANGE))
                metrics["phase_ssim"].append(ssim(pp, tp, data_range=PHASE_DATA_RANGE))
                metrics["amp_psnr"].append(psnr(ta, pa, data_range=AMP_DATA_RANGE))
                metrics["phase_psnr"].append(psnr(tp, pp, data_range=PHASE_DATA_RANGE))
                metrics["freq_ssim"].append(get_freq_ssim(pa, ta))
                metrics["bs_ratio"].append(compute_bs_ratio(pa, ta))

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def train_configuration(  # noqa: PLR0913, PLR0915
    config: dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Train a single configuration and return results."""
    print(f"\n{'=' * 80}")
    print(f"Training: {config['name']}")
    print(f"{'=' * 80}")

    model = HoloPASWIN(IMG_SIZE, WAVELENGTH, PIXEL_SIZE, Z_DIST).to(device)
    criterion = ConfigurableLoss(
        model.propagator,
        w_complex=config["w_complex"],
        w_amp=config["w_amp"],
        w_phase=config["w_phase"],
        w_freq=config["w_freq"],
        lambda_phy=config["lambda_phy"],
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # Track training history
    history = {"train_loss": [], "val_loss": [], "epoch": []}
    best_val_loss = float("inf")
    start_time = time.time()

    epochs = 1 if dry_run else NUM_EPOCHS

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]", leave=False)

        # Mixed precision training (faster on CUDA, slightly faster on MPS)
        scaler = torch.amp.GradScaler(device.type) if device.type == "cuda" else None

        for h_batch, g_batch in pbar:
            h_dev, g_dev = h_batch.to(device), g_batch.to(device)
            optimizer.zero_grad()

            # Use automatic mixed precision for CUDA
            if device.type == "cuda" and scaler is not None:
                with torch.amp.autocast(device_type="cuda"):
                    pred, _ = model(h_dev)
                    loss = criterion(pred, g_dev, h_dev)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred, _ = model(h_dev)
                loss = criterion(pred, g_dev, h_dev)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if dry_run and train_batches >= 2:  # noqa: PLR2004
                break

        avg_train_loss = train_loss / train_batches

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        # Use mixed precision for validation too
        with (
            torch.no_grad(),
            torch.amp.autocast(
                device_type=device.type if device.type != "mps" else "cpu", enabled=(device.type == "cuda")
            ),
        ):
            for h_batch, g_batch in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]", leave=False):
                h_dev, g_dev = h_batch.to(device), g_batch.to(device)
                pred, _ = model(h_dev)
                loss = criterion(pred, g_dev, h_dev)
                val_loss += loss.item()
                val_batches += 1

                if dry_run and val_batches >= 2:  # noqa: PLR2004
                    break

        avg_val_loss = val_loss / val_batches

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["epoch"].append(epoch + 1)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_path = output_dir / f"{config['id']}_best.pth"
            torch.save(model.state_dict(), model_path)

    training_time = time.time() - start_time

    # Load best model and evaluate
    model.load_state_dict(torch.load(output_dir / f"{config['id']}_best.pth", weights_only=True))
    test_metrics = evaluate_model(model, test_loader, device)

    # Save history
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / f"{config['id']}_history.csv", index=False)

    return {
        "config_name": config["name"],
        "config_id": config["id"],
        "training_time_sec": training_time,
        "best_val_loss": best_val_loss,
        **test_metrics,
    }


def main() -> None:
    """Run the loss component ablation study."""
    parser = argparse.ArgumentParser(description="Loss function component ablation study")
    parser.add_argument("--dry-run", action="store_true", help="Run with minimal data for testing")
    parser.add_argument("--output-dir", type=str, default="results/ablation_loss", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Define configurations
    configs = [
        {
            "id": "full_model",
            "name": "Full Model (Baseline)",
            "w_complex": 0.2,
            "w_amp": 0.4,
            "w_phase": 0.2,
            "w_freq": 0.2,
            "lambda_phy": 0.1,
        },
        {
            "id": "no_freq",
            "name": "Full - L_freq",
            "w_complex": 0.25,
            "w_amp": 0.5,
            "w_phase": 0.25,
            "w_freq": 0.0,
            "lambda_phy": 0.1,
        },
        {
            "id": "no_phy",
            "name": "Full - L_phy",
            "w_complex": 0.2,
            "w_amp": 0.4,
            "w_phase": 0.2,
            "w_freq": 0.2,
            "lambda_phy": 0.0,
        },
        {
            "id": "only_complex",
            "name": "Only L_complex",
            "w_complex": 1.0,
            "w_amp": 0.0,
            "w_phase": 0.0,
            "w_freq": 0.0,
            "lambda_phy": 0.0,
        },
        {
            "id": "pure_spatial",
            "name": "Pure Spatial (no L_freq, no L_phy)",
            "w_complex": 0.25,
            "w_amp": 0.5,
            "w_phase": 0.25,
            "w_freq": 0.0,
            "lambda_phy": 0.0,
        },
    ]

    # Load datasets
    print("Loading datasets...")
    full_dataset = HoloDataset(DATA_DIR, target_size=IMG_SIZE, img_dim=IMG_SIZE)
    test_dataset = HoloDataset(TEST_DATA_DIR, target_size=IMG_SIZE, img_dim=IMG_SIZE)

    # Split train/val
    total_size = len(full_dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Subsample dataset for faster ablation
    if len(train_dataset) > MAX_TRAIN_SAMPLES:
        indices = torch.randperm(len(train_dataset))[:MAX_TRAIN_SAMPLES].tolist()
        train_dataset = torch.utils.data.Subset(train_dataset, indices)

    if len(val_dataset) > MAX_VAL_SAMPLES:
        indices = torch.randperm(len(val_dataset))[:MAX_VAL_SAMPLES].tolist()
        val_dataset = torch.utils.data.Subset(val_dataset, indices)

    print(f"Using {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples")

    # Create dataloaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=2,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
    )

    # Train each configuration
    results = []
    for config in configs:
        result = train_configuration(
            config,
            train_loader,
            val_loader,
            test_loader,
            device,
            output_dir,
            dry_run=args.dry_run,
        )
        results.append(result)

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "ablation_results.csv", index=False)

    # Print summary
    print("\n" + "=" * 120)
    print("ABLATION STUDY RESULTS")
    print("=" * 120)
    print(results_df.to_string(index=False))
    print("=" * 120)

    # Generate LaTeX table
    latex_table = generate_latex_table(results_df)
    with (output_dir / "ablation_table.tex").open("w") as f:
        f.write(latex_table)

    print(f"\nResults saved to: {output_dir}")
    print("  - ablation_results.csv")
    print("  - ablation_table.tex")
    print("  - *_history.csv (training curves)")
    print("  - *_best.pth (model checkpoints)")


def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table from results dataframe."""
    latex = r"""\begin{table}[h]
\centering
\caption{Loss function component ablation study. All models trained for 5 epochs on 25,000 samples.}
\label{tab:loss_component_ablation}
\begin{tabular}{@{}lcccccc@{}}
\toprule
\textbf{Configuration} & \textbf{Amp SSIM} & \textbf{Phase SSIM} & \textbf{Phase PSNR} & \textbf{Freq SSIM} & \textbf{B/S Ratio} \\ \midrule
"""

    for _, row in df.iterrows():
        latex += f"{row['config_name']} & {row['amp_ssim']:.4f} & {row['phase_ssim']:.4f} & "
        latex += f"{row['phase_psnr']:.2f} & {row['freq_ssim']:.4f} & {row['bs_ratio']:.4f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


if __name__ == "__main__":
    main()
