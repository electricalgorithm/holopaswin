"""Architecture ablation study.

Compares:
1. Swin-Tiny (pretrained) vs ResNet-18 U-Net
2. Residual vs Direct reconstruction
3. Swin-Tiny (pretrained) vs Swin-Tiny (from scratch)
"""

import argparse

# Import evaluation function
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from holopaswin.dataset import HoloDataset
from holopaswin.loss import PhysicsLoss
from holopaswin.model import HoloPASWIN
from holopaswin.resnet_unet import ResNetUNet

sys.path.insert(0, str(Path(__file__).parent))
from ablation_loss_components import evaluate_model

# Configuration
DATA_DIR = "../hologen/dataset-224"
TEST_DATA_DIR = "../hologen/test-dataset-224"
IMG_SIZE = 224
WAVELENGTH = 532e-9
PIXEL_SIZE = 4.65e-6
Z_DIST = 0.02
BATCH_SIZE = 24
NUM_EPOCHS = 3  # Reduced from 5 for faster ablation
LR = 1e-4
MAX_TRAIN_SAMPLES = 15000
MAX_VAL_SAMPLES = 3000


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_inference_time(model: torch.nn.Module, device: torch.device, num_runs: int = 100) -> float:
    """Measure average inference time in milliseconds."""
    model.eval()
    dummy_input = torch.randn(1, 1, IMG_SIZE, IMG_SIZE).to(device)

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # Measure
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            start = time.time()
            _ = model(dummy_input)
            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
            times.append((time.time() - start) * 1000)  # Convert to ms

    return float(sum(times) / len(times))


def train_model(  # noqa: PLR0913
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    model_name: str,
    output_dir: Path,
) -> dict:
    """Train a model and return training info."""
    print(f"\nTraining: {model_name}")

    criterion = PhysicsLoss(model.propagator, lambda_physics=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0

        for h_batch, g_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", leave=False):
            h_dev, g_dev = h_batch.to(device), g_batch.to(device)
            optimizer.zero_grad()
            pred, _ = model(h_dev)
            loss = criterion(pred, g_dev, h_dev)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1  # noqa: SIM113

        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for h_batch, g_batch in val_loader:
                h_dev, g_dev = h_batch.to(device), g_batch.to(device)
                pred, _ = model(h_dev)
                loss = criterion(pred, g_dev, h_dev)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS} - Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_dir / f"{model_name}_best.pth")

    training_time = time.time() - start_time

    return {
        "training_time_sec": training_time,
        "best_val_loss": best_val_loss,
    }


def main() -> None:  # noqa: PLR0915
    """Run architecture ablation study."""
    parser = argparse.ArgumentParser(description="Architecture ablation study")
    parser.add_argument("--output-dir", type=str, default="results/ablation_architecture", help="Output directory")
    parser.add_argument("--dry-run", action="store_true", help="Skip training, only test inference")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
    if not args.dry_run:
        print("Loading datasets...")
        full_dataset = HoloDataset(DATA_DIR, target_size=IMG_SIZE, img_dim=IMG_SIZE)
        total_size = len(full_dataset)
        val_size = int(0.2 * total_size)
        train_size = total_size - val_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        # Subsample for faster ablation
        if len(train_dataset) > MAX_TRAIN_SAMPLES:
            indices = torch.randperm(len(train_dataset))[:MAX_TRAIN_SAMPLES].tolist()
            train_dataset = torch.utils.data.Subset(train_dataset, indices)

        if len(val_dataset) > MAX_VAL_SAMPLES:
            indices = torch.randperm(len(val_dataset))[:MAX_VAL_SAMPLES].tolist()
            val_dataset = torch.utils.data.Subset(val_dataset, indices)

        print(f"Using {len(train_dataset)} train, {len(val_dataset)} val samples")

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE * 2,
            shuffle=False,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
            prefetch_factor=2,
        )

    test_dataset = HoloDataset(TEST_DATA_DIR, target_size=IMG_SIZE, img_dim=IMG_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    # Define models
    models_to_test = [
        {
            "name": "HoloPASWIN (Swin-Tiny, Pretrained, Residual)",
            "id": "swin_pretrained_residual",
            "model": HoloPASWIN(IMG_SIZE, WAVELENGTH, PIXEL_SIZE, Z_DIST, use_pretrained=True, residual_mode=True),
        },
        {
            "name": "ResNet-18 U-Net (Pretrained, Residual)",
            "id": "resnet18_pretrained_residual",
            "model": ResNetUNet(IMG_SIZE, WAVELENGTH, PIXEL_SIZE, Z_DIST, residual_mode=True),
        },
        {
            "name": "HoloPASWIN (Swin-Tiny, Pretrained, Direct)",
            "id": "swin_pretrained_direct",
            "model": HoloPASWIN(IMG_SIZE, WAVELENGTH, PIXEL_SIZE, Z_DIST, use_pretrained=True, residual_mode=False),
        },
        {
            "name": "HoloPASWIN (Swin-Tiny, From Scratch, Residual)",
            "id": "swin_scratch_residual",
            "model": HoloPASWIN(IMG_SIZE, WAVELENGTH, PIXEL_SIZE, Z_DIST, use_pretrained=False, residual_mode=True),
        },
    ]

    results = []

    for model_config in models_to_test:
        print(f"\n{'=' * 80}")
        print(f"Testing: {model_config['name']}")
        print(f"{'=' * 80}")

        model = model_config["model"].to(device)

        # Count parameters
        num_params = count_parameters(model)

        # Train (unless dry-run)
        if not args.dry_run:
            train_info = train_model(
                model,
                train_loader,
                val_loader,
                device,
                model_config["id"],
                output_dir,
            )

            # Load best model
            model.load_state_dict(torch.load(output_dir / f"{model_config['id']}_best.pth", weights_only=True))
        else:
            train_info = {"training_time_sec": 0, "best_val_loss": 0}

        # Evaluate
        test_metrics = evaluate_model(model, test_loader, device)

        # Measure inference time
        inf_time = measure_inference_time(model, device)

        results.append(
            {
                "model": model_config["name"],
                "model_id": model_config["id"],
                "num_parameters": num_params,
                "inference_time_ms": inf_time,
                **train_info,
                **test_metrics,
            }
        )

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_dir / "architecture_ablation.csv", index=False)

    print("\n" + "=" * 120)
    print("ARCHITECTURE ABLATION RESULTS")
    print("=" * 120)
    print(results_df.to_string(index=False))
    print("=" * 120)

    # Generate LaTeX table
    latex_table = generate_latex_table(results_df)
    with (output_dir / "architecture_table.tex").open("w") as f:
        f.write(latex_table)

    print(f"\nResults saved to: {output_dir}")


def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate LaTeX table from results."""
    latex = r"""\begin{table}[h]
\centering
\caption{Architecture ablation study comparing Swin Transformer vs ResNet-18, residual vs direct reconstruction, and pretrained vs from-scratch training.}
\label{tab:architecture_ablation}
\scriptsize
\begin{tabular}{@{}lcccccc@{}}
\toprule
\textbf{Model} & \textbf{Params (M)} & \textbf{Inf. Time (ms)} & \textbf{Phase PSNR} & \textbf{Phase SSIM} & \textbf{B/S Ratio} \\ \midrule
"""

    for _, row in df.iterrows():
        params_m = row["num_parameters"] / 1e6
        latex += f"{row['model']} & {params_m:.2f} & {row['inference_time_ms']:.1f} & "
        latex += f"{row['phase_psnr']:.2f} & {row['phase_ssim']:.4f} & {row['bs_ratio']:.4f} \\\\\n"

    latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
    return latex


if __name__ == "__main__":
    main()
