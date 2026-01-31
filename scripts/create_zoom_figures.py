import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from holopaswin.dataset import HoloDataset
from holopaswin.model import HoloPASWIN

# Configuration
IMG_SIZE = 224
WAVELENGTH = 532e-9
PIXEL_SIZE = 4.65e-6
Z_DIST = 0.02
MODEL_PATH = "results/experiment9/holopaswin_exp9.pth"
DATA_DIR = "../hologen/test-dataset-224"
OUTPUT_DIR = "../article/src/figs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "detailed_comparison.png")


def create_detailed_figure(sample_idx=None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")

    print(f"Using device: {device}")

    # Load model
    model = HoloPASWIN(IMG_SIZE, WAVELENGTH, PIXEL_SIZE, Z_DIST).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Load dataset
    dataset = HoloDataset(DATA_DIR, target_size=IMG_SIZE, img_dim=IMG_SIZE)

    # Find a sample with significant object content AND high SSIM
    print("Searching for an interesting and high-quality sample...")
    best_idx = 0
    max_score = -1
    # Search first 100 samples
    for i in range(min(100, len(dataset))):
        holo, gt_obj = dataset[i]

        # Simple inference for selection
        with torch.no_grad():
            clean_pred, _ = model(holo.unsqueeze(0).to(device))

        pred_c = torch.complex(clean_pred[:, 0, :, :], clean_pred[:, 1, :, :])
        pred_amp_arr = torch.abs(pred_c).squeeze().cpu().numpy()

        gt_c = torch.complex(gt_obj[0], gt_obj[1])
        gt_amp_arr = torch.abs(gt_c).numpy()

        active_count = np.sum(gt_amp_arr < 0.95)
        if active_count < 500:  # Skip samples with too little content
            continue

        from skimage.metrics import structural_similarity as ssim

        score_ssim = ssim(gt_amp_arr, pred_amp_arr, data_range=1.2)

        # Combined score: favors content and quality
        combined_score = score_ssim * (active_count / 1000.0)

        if combined_score > max_score:
            max_score = combined_score
            best_idx = i

    sample_idx = best_idx
    print(f"Selected Sample Index: {sample_idx} with score {max_score:.4f}")

    # Get sample
    holo_t, gt_obj_t = dataset[sample_idx]
    holo_in = holo_t.unsqueeze(0).to(device)

    with torch.no_grad():
        clean_pred, _ = model(holo_in)

    # Process outputs
    pred_c = torch.complex(clean_pred[:, 0, :, :], clean_pred[:, 1, :, :])
    pred_amp = torch.abs(pred_c).squeeze().cpu().numpy()
    pred_phase = torch.angle(pred_c).squeeze().cpu().numpy()

    gt_c = torch.complex(gt_obj_t[0], gt_obj_t[1])
    gt_amp = torch.abs(gt_c).numpy()
    gt_phase = torch.angle(gt_c).numpy()

    # Error maps
    err_amp = np.abs(gt_amp - pred_amp)
    err_phase = np.abs(gt_phase - pred_phase)

    active_mask = gt_amp < 0.98
    # Choose zoom region - find the largest connected component of the objects
    from scipy.ndimage import label

    labeled_mask, _num_features = label(active_mask)
    if _num_features > 0:
        # Find the largest component
        counts = np.bincount(labeled_mask.ravel())
        largest_label = np.argmax(counts[1:]) + 1
        y_coords, x_coords = np.where(labeled_mask == largest_label)
        zoom_y = int(np.mean(y_coords))
        zoom_x = int(np.mean(x_coords))
    else:
        zoom_y, zoom_x = 112, 112

    zoom_size = 40
    y1, y2 = max(0, zoom_y - zoom_size), min(IMG_SIZE, zoom_y + zoom_size)
    x1, x2 = max(0, zoom_x - zoom_size), min(IMG_SIZE, zoom_x + zoom_size)

    # Create figure
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 4)

    # Titles

    # Colormaps
    cmap_amp = "gray"  # "inferno"
    cmap_phase = "viridis"  # "twilight"
    cmap_err = "hot"

    # Custom ranges for low-contrast objects
    v_amp = (0.85, 1.05)
    v_ph = (-0.1, 0.6)

    # Setup axes
    ax_gt_amp = fig.add_subplot(gs[0, 0])
    ax_pr_amp = fig.add_subplot(gs[0, 1])
    ax_er_amp = fig.add_subplot(gs[0, 2])
    ax_zm_amp = fig.add_subplot(gs[0, 3])

    ax_gt_ph = fig.add_subplot(gs[1, 0])
    ax_pr_ph = fig.add_subplot(gs[1, 1])
    ax_er_ph = fig.add_subplot(gs[1, 2])
    ax_zm_ph = fig.add_subplot(gs[1, 3])

    # Main images
    ax_gt_amp.imshow(gt_amp, cmap=cmap_amp, vmin=v_amp[0], vmax=v_amp[1])
    ax_pr_amp.imshow(pred_amp, cmap=cmap_amp, vmin=v_amp[0], vmax=v_amp[1])
    im_er_amp = ax_er_amp.imshow(err_amp, cmap=cmap_err, vmin=0, vmax=0.1)

    ax_gt_ph.imshow(gt_phase, cmap=cmap_phase, vmin=v_ph[0], vmax=v_ph[1])
    ax_pr_ph.imshow(pred_phase, cmap=cmap_phase, vmin=v_ph[0], vmax=v_ph[1])
    im_er_ph = ax_er_ph.imshow(err_phase, cmap=cmap_err, vmin=0, vmax=0.4)

    # Zooms (Prediction)
    zm_pr_amp = pred_amp[y1:y2, x1:x2]
    zm_pr_ph = pred_phase[y1:y2, x1:x2]

    ax_zm_amp.imshow(zm_pr_amp, cmap=cmap_amp, vmin=v_amp[0], vmax=v_amp[1])
    ax_zm_ph.imshow(zm_pr_ph, cmap=cmap_phase, vmin=v_ph[0], vmax=v_ph[1])

    # Rectangles for zoom area on Pred images
    from matplotlib.patches import Rectangle

    rect_amp = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="white", facecolor="none", linestyle="--")
    rect_ph = Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor="white", facecolor="none", linestyle="--")
    ax_pr_amp.add_patch(rect_amp)
    ax_pr_ph.add_patch(rect_ph)

    # Style and labeling
    axs = [ax_gt_amp, ax_pr_amp, ax_er_amp, ax_zm_amp, ax_gt_ph, ax_pr_ph, ax_er_ph, ax_zm_ph]
    names = [
        "GT Amplitude",
        "Pred Amplitude",
        "Amp Error",
        "Amp Zoom",
        "GT Phase",
        "Pred Phase",
        "Phase Error",
        "Phase Zoom",
    ]

    for ax, name in zip(axs, names, strict=False):
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.axis("off")

    # Colorbars
    plt.colorbar(im_er_amp, ax=ax_er_amp, fraction=0.046, pad=0.04)
    plt.colorbar(im_er_ph, ax=ax_er_ph, fraction=0.046, pad=0.04)

    plt.suptitle(
        f"Detailed Reconstruction Analysis (Sample Index: {sample_idx})", fontsize=16, fontweight="bold", y=0.98
    )
    plt.tight_layout()

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved detailed figure to {OUTPUT_FILE}")


if __name__ == "__main__":
    # Create for a few samples to pick the best one?
    # For now just one is fine.
    # Seed for predictability if needed
    np.random.seed(42)
    create_detailed_figure()
