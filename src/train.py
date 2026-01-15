"""Training module for the HoloPASWIN model.

This module provides the main training loop for training the physics-aware
Swin Transformer model on holographic reconstruction tasks. It includes
data loading, model initialization, training and validation loops, and
checkpoint saving.
"""

from pathlib import Path

import torch
import tqdm
from torch.utils.data import DataLoader

from holopaswin.dataset import HoloDataset
from holopaswin.loss import PhysicsLoss
from holopaswin.model import HoloPASWIN

# Dataset/Model Configuration
# Should point to the dataset directory relative to where script is run
# Dataset/Model Configuration
# Should point to the dataset directory relative to where script is run
DATA_DIR = "../hologen/dataset-224"
IMG_SIZE = 224
WAVELENGTH = 532e-9
PIXEL_SIZE = 4.65e-6
Z_DIST = 0.02

# Training Configuration
BATCH_SIZE = 32
LR = 1e-4
NUM_EPOCHS = 5  # Adjusted based on convergence speed
ENABLE_DEMO_MODE = False
DEMO_BATCH_LIMIT = 20
EXP_DIR = "results/experiment9"
MODEL_SAVE_PATH = f"{EXP_DIR}/holopaswin_exp9.pth"


def main() -> None:  # noqa: C901, PLR0915, PLR0912
    """Train the HoloPASWIN model.

    This function orchestrates the complete training pipeline:
    1. Initializes the model, loss function, and optimizer
    2. Loads and splits the dataset into train/validation sets (80/20 split)
    3. Runs training and validation loops for the specified number of epochs
    4. Saves the best model checkpoint based on validation loss

    The training uses physics-constrained loss combining structural L1,
    Phase, Amplitude, and Frequency loss. The best model (lowest validation loss)
    is saved in 'results/experiment9/holopaswin_exp9.pth'.
    """
    # Create experiment directory
    Path(EXP_DIR).mkdir(parents=True, exist_ok=True)
    # Setup Device (MPS for Mac, CUDA for Colab, else CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Initialize Model
    model = HoloPASWIN(IMG_SIZE, WAVELENGTH, PIXEL_SIZE, Z_DIST).to(device)

    # Physics Loss Wrapper
    # We share the propagator from the model to ensure parameters match
    criterion = PhysicsLoss(model.propagator, lambda_physics=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # --- Data Loading & Splitting ---
    print(f"Loading dataset from: {DATA_DIR}")
    try:
        full_dataset = HoloDataset(data_dir=DATA_DIR, target_size=IMG_SIZE, img_dim=IMG_SIZE)
    except Exception as e:  # noqa: BLE001
        print(f"Error loading dataset: {e}")
        # Identify if path issue
        print(f"Current CWD: {Path.cwd()}")
        return

    # Calculate split sizes
    total_size = len(full_dataset)
    if total_size == 0:
        print("Dataset is empty. Exiting.")
        return

    # Calculate split sizes
    total_size = len(full_dataset)
    if total_size == 0:
        print("Dataset is empty. Exiting.")
        return

    # Split: 80% Train, 20% Validation (No test set, as external test set exists)
    val_size = int(0.2 * total_size)
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
    num_workers = 0  # safe default for simple script
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=num_workers)

    # --- Training Loop ---
    start_epoch = 0
    best_val_loss = float("inf")

    # Resume Logic (Minimal)
    load_checkpoint = "interrupted_swin_holo.pth"
    if Path(load_checkpoint).exists():
        print(f"Loading checkpoint from {load_checkpoint}...")
        model.load_state_dict(torch.load(load_checkpoint, map_location=device))
        start_epoch = 1  # Resume from epoch 2
        best_val_loss = 0.2479
        print(f"Resuming from Epoch {start_epoch + 1}, Best Val Loss: {best_val_loss}")

    try:
        for epoch in range(start_epoch, NUM_EPOCHS):
            print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")

            # 1. TRAINING PHASE
            model.train()  # Set model to training mode (enables Dropout, BatchNorm updates)
            train_loss = 0.0
            progress_bar = tqdm.tqdm(train_loader, desc="Training", leave=False)

            batch_count = 0
            for holo, gt_obj in progress_bar:
                holo_in = holo.to(device)
                gt_obj_in = gt_obj.to(device)

                optimizer.zero_grad()

                # Forward pass returns (clean_complex, dirty_complex)
                pred_clean, _ = model(holo_in)

                # Loss expects (pred_2ch, target_2ch, input_1ch)
                loss = criterion(pred_clean, gt_obj_in, holo_in)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                progress_bar.set_postfix({"Batch Loss": f"{loss.item():.4f}"})

                # For demo: limit to 20 batches to show it runs
                batch_count += 1
                if ENABLE_DEMO_MODE and batch_count >= DEMO_BATCH_LIMIT:
                    print(f" (Demo: Breaking early after {DEMO_BATCH_LIMIT} batches)")
                    break

            avg_train_loss = train_loss / batch_count if batch_count > 0 else 0

            # 2. VALIDATION PHASE
            model.eval()  # Set model to evaluation mode (freezes BatchNorm stats, disables Dropout)
            val_loss = 0.0

            val_batch_count = 0
            with torch.no_grad():  # Disable gradient calculation (saves huge memory/time)
                for holo, gt_obj in tqdm.tqdm(val_loader, desc="Validating", leave=False):
                    holo_in = holo.to(device)
                    gt_obj_in = gt_obj.to(device)

                    pred, _ = model(holo_in)
                    loss = criterion(pred, gt_obj_in, holo_in)
                    val_loss += loss.item()

                    val_batch_count += 1
                    if ENABLE_DEMO_MODE and val_batch_count >= DEMO_BATCH_LIMIT:  # Limit validation for demo too
                        break

            avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0

            print(f"Done. Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

            # Simple Early Stopping / Checkpointing logic
            # If this is the best model so far, save it.
            if avg_val_loss <= best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), MODEL_SAVE_PATH)
                print(f"--> Best model saved to {MODEL_SAVE_PATH}!")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving current checkpoint...")
        torch.save(model.state_dict(), "interrupted_swin_holo.pth")
        print("--> Checkpoint saved to 'interrupted_swin_holo.pth'")


if __name__ == "__main__":
    main()
