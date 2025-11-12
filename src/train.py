"""Training module for the HoloPASWIN model.

This module provides the main training loop for training the physics-aware
Swin Transformer model on holographic reconstruction tasks. It includes
data loading, model initialization, training and validation loops, and
checkpoint saving.
"""

import torch
import tqdm
from torch.utils.data import DataLoader

from holopaswin.dataset import HoloDataset
from holopaswin.loss import PhysicsLoss
from holopaswin.model import HoloPASWIN

# Dataset/Model Configuration
# huggingface.co/datasets/gokhankocmarli/inline-digital-holography
IMG_SIZE = 224
WAVELENGTH = 532e-9
PIXEL_SIZE = 4.65e-6
Z_DIST = 0.02

# Training Configuration
BATCH_SIZE = 8
LR = 1e-4
NUM_EPOCHS = 5


def main() -> None:
    """Train the HoloPASWIN model.

    This function orchestrates the complete training pipeline:
    1. Initializes the model, loss function, and optimizer
    2. Loads and splits the dataset into train/validation sets (80/20 split)
    3. Runs training and validation loops for the specified number of epochs
    4. Saves the best model checkpoint based on validation loss

    The training uses physics-constrained loss combining structural L1 loss
    with physics consistency loss. The best model (lowest validation loss)
    is saved as 'best_swin_holo.pth' in the current directory.

    Training configuration is set via module-level constants:
        - IMG_SIZE: Image spatial dimension (224)
        - WAVELENGTH: Light wavelength in meters (532e-9)
        - PIXEL_SIZE: Physical pixel size in meters (4.65e-6)
        - Z_DIST: Propagation distance in meters (0.02)
        - BATCH_SIZE: Training batch size (8)
        - LR: Learning rate (1e-4)
        - NUM_EPOCHS: Number of training epochs (5)
    """
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Model
    model = HoloPASWIN(IMG_SIZE, WAVELENGTH, PIXEL_SIZE, Z_DIST).to(device)

    # Physics Loss Wrapper
    # We share the propagator from the model to ensure parameters match
    criterion = PhysicsLoss(model.propagator, lambda_physics=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # --- Data Loading & Splitting ---
    full_dataset = HoloDataset(data_dir="./data_train", target_size=IMG_SIZE)

    # Calculate split sizes
    total_size = len(full_dataset)
    val_size = int(0.2 * total_size)  # 20% for validation
    train_size = total_size - val_size

    # Perform the split
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),  # Fixed seed for reproducibility
    )

    print(f"Total Images: {total_size}")
    print(f"Training Set: {len(train_dataset)} images")
    print(f"Validation Set: {len(val_dataset)} images")

    # Create Loaders
    # Validation loader doesn't need shuffle, and can often handle larger batch size (no gradients stored)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False)

    # --- Training Loop ---
    for epoch in range(NUM_EPOCHS):
        print(f"\n=== Epoch {epoch + 1}/{NUM_EPOCHS} ===")

        # 1. TRAINING PHASE
        model.train()  # Set model to training mode (enables Dropout, BatchNorm updates)
        train_loss = 0.0
        progress_bar = tqdm.tqdm(train_loader, desc="Training", leave=False)
        for holo, gt_obj in progress_bar:
            holo, gt_obj = holo.to(device), gt_obj.to(device)  # noqa: PLW2901
            optimizer.zero_grad()
            pred, _ = model(holo)
            loss = criterion(pred, gt_obj, holo)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            progress_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)

        # 2. VALIDATION PHASE
        model.eval()  # Set model to evaluation mode (freezes BatchNorm stats, disables Dropout)
        val_loss = 0.0
        with torch.no_grad():  # Disable gradient calculation (saves huge memory/time)
            for holo, gt_obj in tqdm.tqdm(val_loader, desc="Validating", leave=False):
                holo, gt_obj = holo.to(device), gt_obj.to(device)  # noqa: PLW2901
                pred, _ = model(holo)
                loss = criterion(pred, gt_obj, holo)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        print(f"Done. Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Simple Early Stopping / Checkpointing logic
        # If this is the best model so far, save it.
        if epoch == 0:
            best_val_loss = avg_val_loss
        if avg_val_loss <= best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_swin_holo.pth")
            print("--> Best model saved!")


if __name__ == "__main__":
    main()
