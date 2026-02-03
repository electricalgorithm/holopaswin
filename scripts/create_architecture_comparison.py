"""Generate professional architecture ablation comparison figure."""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.holopaswin.dataset import HoloDataset
from src.holopaswin.model import HoloPASWIN
from src.holopaswin.resnet_unet import ResNetUNet

# Configuration
IMG_SIZE = 224
WAVELENGTH = 632.8e-9
PIXEL_SIZE = 3.45e-6
Z_DISTANCE = 0.02
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# Model configurations
MODELS = [
    {
        "name": "Ground Truth",
        "path": None,
        "type": "gt",
        "bs_ratio": 0.0,
    },
    {
        "name": "Swin (Pre, Res)",
        "path": "results/ablation_study/architecture/swin_pretrained_residual_best.pth",
        "type": "swin",
        "bs_ratio": 0.682,
    },
    {
        "name": "ResNet-18 (Pre, Res)",
        "path": "results/ablation_study/architecture/resnet18_pretrained_residual_best.pth",
        "type": "resnet",
        "bs_ratio": 0.710,
    },
    {
        "name": "Swin (Pre, Dir)",
        "path": "results/ablation_study/architecture/swin_pretrained_direct_best.pth",
        "type": "swin_direct",
        "bs_ratio": 0.566,
    },
    {
        "name": "Swin (Scr, Res)",
        "path": "results/ablation_study/architecture/swin_scratch_residual_best.pth",
        "type": "swin",
        "bs_ratio": 1.212,
    },
]


def load_model(model_config):
    """Load a model from checkpoint."""
    if model_config["type"] == "gt":
        return None
    if model_config["type"] == "resnet":
        model = ResNetUNet(
            img_size=IMG_SIZE,
            wavelength=WAVELENGTH,
            pixel_size=PIXEL_SIZE,
            z_distance=Z_DISTANCE,
            residual_mode=True,
        )
    else:
        model = HoloPASWIN(
            img_size=IMG_SIZE,
            wavelength=WAVELENGTH,
            pixel_size=PIXEL_SIZE,
            z_dist=Z_DISTANCE,
            use_pretrained="scratch" not in model_config["name"].lower(),
            residual_mode="Dir" not in model_config["name"],
        )

    checkpoint = torch.load(model_config["path"], map_location=DEVICE)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()
    return model


def compute_bs_ratio(amplitude):
    """Compute background-to-signal ratio."""
    threshold = np.percentile(amplitude, 75)
    background_mask = amplitude < threshold
    object_mask = amplitude >= threshold

    if background_mask.sum() == 0 or object_mask.sum() == 0:
        return 0.0

    bg_mean = amplitude[background_mask].mean()
    obj_mean = amplitude[object_mask].mean()

    return float(bg_mean / obj_mean) if obj_mean > 0 else 0.0


def create_comparison_figure():
    """Create professional architecture comparison figure."""
    print(f"Using device: {DEVICE}")

    # Load test dataset
    test_data_dir = Path("../hologen/test-dataset-224")
    test_dataset = HoloDataset(str(test_data_dir), target_size=IMG_SIZE, img_dim=IMG_SIZE)
    print(f"Loaded {len(test_dataset)} test samples")

    # Select a representative sample
    sample_idx = 42
    hologram, gt_obj = test_dataset[sample_idx]
    hologram = hologram.unsqueeze(0).to(DEVICE)
    gt_obj = gt_obj.unsqueeze(0).to(DEVICE)

    # Create figure with 2 rows (amplitude, phase) and 5 columns (models)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # Remove spacing
    plt.subplots_adjust(wspace=0.02, hspace=0.15, left=0.02, right=0.98, top=0.92, bottom=0.08)

    # Get ground truth for normalization
    gt_complex = torch.complex(gt_obj[0, 0], gt_obj[0, 1]).cpu().numpy()
    gt_amp = np.abs(gt_complex)
    gt_phase = np.angle(gt_complex)
    
    # Use percentile-based normalization for better contrast
    vmin_amp = np.percentile(gt_amp, 1)
    vmax_amp = np.percentile(gt_amp, 99)
    vmin_phase, vmax_phase = -np.pi, np.pi

    # Process each model
    for col_idx, model_config in enumerate(MODELS):
        print(f"Processing: {model_config['name']}")

        if model_config["type"] == "gt":
            # Ground truth
            pred_amp = gt_amp
            pred_phase = gt_phase
            bs_ratio = compute_bs_ratio(gt_amp)
        else:
            # Load model and predict
            model = load_model(model_config)
            with torch.no_grad():
                pred, _ = model(hologram)

            pred_complex = torch.complex(pred[0, 0], pred[0, 1]).cpu().numpy()
            pred_amp = np.abs(pred_complex)
            pred_phase = np.angle(pred_complex)
            bs_ratio = compute_bs_ratio(pred_amp)

        # Plot amplitude (row 0) - normalize each image independently
        ax_amp = axes[0, col_idx]
        # Use independent normalization for each image
        amp_vmin = np.percentile(pred_amp, 1)
        amp_vmax = np.percentile(pred_amp, 99)
        im_amp = ax_amp.imshow(pred_amp, cmap="gray", vmin=amp_vmin, vmax=amp_vmax)
        ax_amp.axis("off")
        
        # Add model name as column title
        ax_amp.set_title(model_config["name"], fontsize=11, pad=5)

        # Plot phase (row 1) - use full range (phase is circular, percentile doesn't work)
        ax_phase = axes[1, col_idx]
        im_phase = ax_phase.imshow(pred_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
        ax_phase.axis("off")
        
        # Add B/S ratio as text below phase for all models
        ax_phase.text(
            0.5, -0.08,
            f"B/S: {bs_ratio:.3f}",
            transform=ax_phase.transAxes,
            ha="center",
            fontsize=9,
            family="monospace"
        )

    # Add row labels on the left
    fig.text(0.01, 0.72, "Amplitude", rotation=90, va="center", fontsize=12, fontweight="bold")
    fig.text(0.01, 0.28, "Phase", rotation=90, va="center", fontsize=12, fontweight="bold")

    # Add phase colorbar (amplitude uses independent normalization, phase uses full range)
    cbar_ax_phase = fig.add_axes([0.99, 0.10, 0.01, 0.80])
    cbar_phase = fig.colorbar(im_phase, cax=cbar_ax_phase)
    cbar_phase.set_label("Phase (rad)", fontsize=10)
    cbar_phase.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cbar_phase.set_ticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])

    # Save figure
    output_path = Path("../article/src/figs/architecture_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\n✓ Saved figure to: {output_path}")
    plt.close()


if __name__ == "__main__":
    create_comparison_figure()
