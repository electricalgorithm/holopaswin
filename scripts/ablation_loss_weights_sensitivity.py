"""Loss weight sensitivity analysis.

Tests sensitivity to different loss weight configurations and λ_phy values.
"""

import argparse

# Import the configurable loss from ablation_loss_components
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from holopaswin.dataset import HoloDataset
from holopaswin.model import HoloPASWIN

sys.path.insert(0, str(Path(__file__).parent))
from ablation_loss_components import ConfigurableLoss, evaluate_model

# Configuration
DATA_DIR = "../hologen/dataset-224"
TEST_DATA_DIR = "../hologen/test-dataset-224"
IMG_SIZE = 224
WAVELENGTH = 532e-9
PIXEL_SIZE = 4.65e-6
Z_DIST = 0.02
BATCH_SIZE = 24
SHORT_EPOCHS = 2  # For weight sensitivity
FULL_EPOCHS = 3  # Reduced from 5 for lambda_phy tests
LR = 1e-4
MAX_TRAIN_SAMPLES = 15000
MAX_VAL_SAMPLES = 3000


def train_short(
    config: dict,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = SHORT_EPOCHS,
) -> dict:
    """Train for a few epochs to see trends."""
    model = HoloPASWIN(IMG_SIZE, WAVELENGTH, PIXEL_SIZE, Z_DIST).to(device)
    criterion = ConfigurableLoss(
        model.propagator,
        w_complex=config["w_complex"],
        w_amp=config["w_amp"],
        w_phase=config["w_phase"],
        w_freq=config["w_freq"],
        lambda_phy=config.get("lambda_phy", 0.1),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    history = {"train_loss": [], "val_loss": [], "epoch": []}

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0

        for h_batch, g_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            h_dev, g_dev = h_batch.to(device), g_batch.to(device)
            optimizer.zero_grad()
            pred, _ = model(h_dev)
            loss = criterion(pred, g_dev, h_dev)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches

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
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["epoch"].append(epoch + 1)

    return {
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "history": history,
    }


def main() -> None:  # noqa: PLR0915
    """Run weight sensitivity analysis."""
    parser = argparse.ArgumentParser(description="Loss weight sensitivity analysis")
    parser.add_argument("--output-dir", type=str, default="results/ablation_weights", help="Output directory")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load datasets
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

    # Part 1: Weight sensitivity (short training)
    print("\n" + "=" * 80)
    print("PART 1: Loss Weight Sensitivity (2 epochs)")
    print("=" * 80)

    weight_configs = [
        {
            "name": "Amp-heavy (0.5/0.2/0.15/0.15)",
            "w_amp": 0.5,
            "w_phase": 0.2,
            "w_complex": 0.15,
            "w_freq": 0.15,
            "lambda_phy": 0.1,
        },
        {
            "name": "Balanced (0.4/0.2/0.2/0.2) [Current]",
            "w_amp": 0.4,
            "w_phase": 0.2,
            "w_complex": 0.2,
            "w_freq": 0.2,
            "lambda_phy": 0.1,
        },
        {
            "name": "Phase-heavy (0.3/0.3/0.2/0.2)",
            "w_amp": 0.3,
            "w_phase": 0.3,
            "w_complex": 0.2,
            "w_freq": 0.2,
            "lambda_phy": 0.1,
        },
    ]

    weight_results = []
    for config in weight_configs:
        print(f"\nTesting: {config['name']}")
        result = train_short(config, train_loader, val_loader, device, epochs=SHORT_EPOCHS)
        weight_results.append(
            {
                "config": config["name"],
                "final_val_loss": result["final_val_loss"],
            }
        )

    weight_df = pd.DataFrame(weight_results)
    weight_df.to_csv(output_dir / "weight_sensitivity.csv", index=False)
    print("\nWeight Sensitivity Results:")
    print(weight_df.to_string(index=False))

    # Part 2: Lambda_phy sensitivity (3 epochs)
    print("\n" + "=" * 80)
    print("PART 2: Lambda_phy Sensitivity (3 epochs)")
    print("=" * 80)

    test_dataset = HoloDataset(TEST_DATA_DIR, target_size=IMG_SIZE, img_dim=IMG_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    lambda_configs = [
        {"name": "No Physics (λ=0.0)", "lambda_phy": 0.0},
        {"name": "Current (λ=0.1)", "lambda_phy": 0.1},
        {"name": "High Physics (λ=0.2)", "lambda_phy": 0.2},
    ]

    lambda_results = []
    for config in lambda_configs:
        print(f"\nTraining: {config['name']}")

        # Full training
        model = HoloPASWIN(IMG_SIZE, WAVELENGTH, PIXEL_SIZE, Z_DIST).to(device)
        criterion = ConfigurableLoss(
            model.propagator,
            w_complex=0.2,
            w_amp=0.4,
            w_phase=0.2,
            w_freq=0.2,
            lambda_phy=config["lambda_phy"],
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

        best_val_loss = float("inf")
        for epoch in range(FULL_EPOCHS):
            model.train()
            train_loss = 0.0
            train_batches = 0

            for h_batch, g_batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{FULL_EPOCHS}", leave=False):
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
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), output_dir / f"lambda_{config['lambda_phy']:.1f}_best.pth")

        # Evaluate
        model.load_state_dict(torch.load(output_dir / f"lambda_{config['lambda_phy']:.1f}_best.pth", weights_only=True))
        metrics = evaluate_model(model, test_loader, device)

        lambda_results.append(
            {
                "config": config["name"],
                "lambda_phy": config["lambda_phy"],
                **metrics,
            }
        )

    lambda_df = pd.DataFrame(lambda_results)
    lambda_df.to_csv(output_dir / "lambda_sensitivity.csv", index=False)
    print("\nLambda_phy Sensitivity Results:")
    print(lambda_df.to_string(index=False))

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
