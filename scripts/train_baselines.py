"""Training script for baseline models (U-Net and ResNet-U-Net).

This script trains the CNN baseline models using the same protocol as HoloPASWIN:
- Same dataset split (20k train / 5k val, seed=42)
- Same optimizer (AdamW, LR=1e-4)
- Same physics-aware loss
- Same optics configuration

Usage:
    python scripts/train_baselines.py --model unet --epochs 5 --output-dir results/baseline_comparison
    python scripts/train_baselines.py --model resnet_unet --epochs 5 --output-dir results/baseline_comparison
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from holopaswin.baselines.hrnet import HRNet
from holopaswin.baselines.unet_baseline import UNetBaseline
from holopaswin.dataset import HoloDataset
from holopaswin.loss import PhysicsLoss
from holopaswin.resnet_unet import ResNetUNet

# Optics configuration (same as HoloPASWIN)
IMG_SIZE = 224
WAVELENGTH = 532e-9  # 532 nm
PIXEL_SIZE = 4.65e-6  # 4.65 µm
Z_DIST = 0.02  # 20 mm

# Training configuration
BATCH_SIZE = 32
LR = 1e-4
DEFAULT_EPOCHS = 5


def get_model(model_name: str) -> torch.nn.Module:
    """Create model by name."""
    if model_name == "unet":
        return UNetBaseline(
            img_size=IMG_SIZE,
            wavelength=WAVELENGTH,
            pixel_size=PIXEL_SIZE,
            z_distance=Z_DIST,
            residual_mode=True,
        )
    if model_name == "resnet_unet":
        return ResNetUNet(
            img_size=IMG_SIZE,
            wavelength=WAVELENGTH,
            pixel_size=PIXEL_SIZE,
            z_distance=Z_DIST,
            residual_mode=True,
        )
    if model_name == "hrnet":
        return HRNet(
            img_size=IMG_SIZE,
            wavelength=WAVELENGTH,
            pixel_size=PIXEL_SIZE,
            z_distance=Z_DIST,
            residual_mode=True,
        )
    raise ValueError(f"Unknown model: {model_name}")


def train(  # noqa: PLR0913
    model: torch.nn.Module,
    train_loader: DataLoader,  # type: ignore[type-arg]
    val_loader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
    num_epochs: int,
    output_dir: Path,
    model_name: str,
) -> dict[str, float]:
    """Train a model.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        device: Device to use.
        num_epochs: Number of epochs.
        output_dir: Directory to save checkpoints.
        model_name: Name for checkpoint files.

    Returns:
        Dictionary with training statistics.
    """
    model = model.to(device)

    # Use the same physics-aware loss as HoloPASWIN
    criterion = PhysicsLoss(model.propagator, lambda_physics=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_val_loss = float("inf")
    training_log: list[dict[str, float]] = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Train]")
        for hologram, gt_object in progress:
            hologram = hologram.to(device)  # noqa: PLW2901
            gt_object = gt_object.to(device)  # noqa: PLW2901

            optimizer.zero_grad()
            pred_clean, _ = model(hologram)
            loss = criterion(pred_clean, gt_object, hologram)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / train_batches

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for hologram, gt_object in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{num_epochs} [Val]", leave=False):
                hologram = hologram.to(device)  # noqa: PLW2901
                gt_object = gt_object.to(device)  # noqa: PLW2901

                pred_clean, _ = model(hologram)
                loss = criterion(pred_clean, gt_object, hologram)
                val_loss += loss.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches

        print(f"Epoch {epoch + 1}/{num_epochs} - Train: {avg_train_loss:.4f}, Val: {avg_val_loss:.4f}")

        training_log.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
            }
        )

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = output_dir / f"{model_name}_best.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  --> Saved best model to {checkpoint_path}")

    # Save final model
    final_path = output_dir / f"{model_name}_final.pth"
    torch.save(model.state_dict(), final_path)

    # Save training log
    import json  # noqa: PLC0415

    log_path = output_dir / f"{model_name}_training_log.json"
    with log_path.open("w") as f:
        json.dump(training_log, f, indent=2)

    return {"best_val_loss": best_val_loss, "final_val_loss": avg_val_loss}


def main() -> None:
    """Main training function."""  # noqa: D401
    parser = argparse.ArgumentParser(description="Train baseline models")
    parser.add_argument(
        "--model",
        type=str,
        choices=["unet", "resnet_unet", "hrnet"],
        required=True,
        help="Model to train: 'unet', 'resnet_unet', or 'hrnet'",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Number of training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="../hologen/dataset-224",
        help="Path to training dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/baseline_comparison",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"Batch size (default: {BATCH_SIZE})",
    )
    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load dataset
    print(f"Loading dataset from {args.data_dir}...")
    try:
        full_dataset = HoloDataset(data_dir=args.data_dir, target_size=IMG_SIZE, img_dim=IMG_SIZE)
    except Exception as e:  # noqa: BLE001
        print(f"Error loading dataset: {e}")
        return

    # Split: 80% Train, 20% Validation (same as HoloPASWIN)
    total_size = len(full_dataset)
    val_size = int(0.2 * total_size)
    train_size = total_size - val_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),  # Same seed as HoloPASWIN
    )

    print(f"Training: {len(train_dataset)} samples, Validation: {len(val_dataset)} samples")

    # Create data loaders
    # Note: num_workers=0 is required for LRU cache to work (cache is per-process)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Single process for cache efficiency
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=0,  # Single process for cache efficiency
        pin_memory=True,
    )

    # Create and train model
    print(f"\nTraining {args.model} for {args.epochs} epochs...")
    model = get_model(args.model)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params / 1e6:.2f}M")

    results = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_epochs=args.epochs,
        output_dir=output_dir,
        model_name=args.model,
    )

    print("\nTraining complete!")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Checkpoints saved to: {output_dir}")


if __name__ == "__main__":
    main()
