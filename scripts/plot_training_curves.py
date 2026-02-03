"""Track and plot training/validation curves for the main model.

This script can be run during or after training to visualize loss curves.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_training_curves(history_csv: Path, output_path: Path) -> None:
    """Plot training and validation loss curves."""
    df = pd.read_csv(history_csv)

    plt.figure(figsize=(10, 6))

    plt.plot(df["epoch"], df["train_loss"], label="Training Loss", linewidth=2, marker="o")
    plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", linewidth=2, marker="s")

    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("HoloPASWIN Training and Validation Loss", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(visible=True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Saved training curves to: {output_path}")


def main() -> None:
    """Plot training curves from history CSV."""
    parser = argparse.ArgumentParser(description="Plot training curves")
    parser.add_argument("--history-csv", type=str, required=True, help="Path to training history CSV")
    parser.add_argument("--output", type=str, default="training_curves.png", help="Output plot path")
    args = parser.parse_args()

    history_path = Path(args.history_csv)
    output_path = Path(args.output)

    if not history_path.exists():
        print(f"Error: History file not found: {history_path}")
        return

    plot_training_curves(history_path, output_path)


if __name__ == "__main__":
    main()
