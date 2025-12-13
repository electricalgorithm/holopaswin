import numpy as np
import torch
import torch.utils.data

import matplotlib.pyplot as plt
from holopaswin.dataset import HoloDataset
from holopaswin.model import HoloPASWIN

DATA_DIR = "./results/experiment5/test-dataset"
IMG_SIZE = 224

MODEL_PATH = "results/experiment6/holopaswin_exp6.pth"
# Model Config
WAVELENGTH = 532e-9
PIXEL_SIZE = 4.65e-6
Z_DIST = 0.02

def check_gt_stats():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        
    print(f"Loading dataset from {DATA_DIR}...")
    dataset = HoloDataset(DATA_DIR, target_size=IMG_SIZE)
    
    print(f"Loading model from {MODEL_PATH}...")
    model = HoloPASWIN(IMG_SIZE, WAVELENGTH, PIXEL_SIZE, Z_DIST).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    model.eval()
    
    # Pick random samples
    indices = np.random.choice(len(dataset), 50, replace=False)
    
    print("\nChecking GT vs PRED Amplitude & Phase stats for 50 random samples:")
    print("-" * 130)
    print(f"{'Sample':<6} | {'GT Amp (Min/Max)':<20} | {'Pred Amp (Min/Max)':<20} | {'GT Pha (Min/Max)':<20} | {'Pred Pha (Min/Max)':<20}")
    print("-" * 130)
    
    global_gt_amp_max = -float('inf')
    global_pred_amp_max = -float('inf')
    
    global_gt_phase_min = float('inf')
    global_gt_phase_max = -float('inf')
    global_pred_phase_min = float('inf')
    global_pred_phase_max = -float('inf')
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            holo_t, gt_obj_t = dataset[idx]  # holo: (1,H,W), gt: (2,H,W)
            
            # --- Ground Truth Stats ---
            gt_c = torch.complex(gt_obj_t[0], gt_obj_t[1])
            gt_amp = torch.abs(gt_c).numpy()
            gt_phase = torch.angle(gt_c).numpy()
            
            # --- Prediction Stats ---
            holo_in = holo_t.unsqueeze(0).to(device)
            clean_pred, _ = model(holo_in)
            
            pred_c = torch.complex(clean_pred[:, 0, :, :], clean_pred[:, 1, :, :])
            pred_amp = torch.abs(pred_c).squeeze().cpu().numpy()
            pred_phase = torch.angle(pred_c).squeeze().cpu().numpy()
            
            # Update Globals
            global_gt_amp_max = max(global_gt_amp_max, gt_amp.max())
            global_pred_amp_max = max(global_pred_amp_max, pred_amp.max())
            
            global_gt_phase_min = min(global_gt_phase_min, gt_phase.min())
            global_gt_phase_max = max(global_gt_phase_max, gt_phase.max())
            global_pred_phase_min = min(global_pred_phase_min, pred_phase.min())
            global_pred_phase_max = max(global_pred_phase_max, pred_phase.max())
            
            if i < 15: # Print first 15
                print(f"{idx:<6} | {gt_amp.min():.4f} / {gt_amp.max():.4f}    | {pred_amp.min():.4f} / {pred_amp.max():.4f}      | {gt_phase.min():.4f} / {gt_phase.max():.4f}    | {pred_phase.min():.4f} / {pred_phase.max():.4f}")
            
    print("-" * 130)
    print(f"Global GT   Amp Max: {global_gt_amp_max:.4f}")
    print(f"Global Pred Amp Max: {global_pred_amp_max:.4f}")
    print(f"Global GT   Phase Range: [{global_gt_phase_min:.4f}, {global_gt_phase_max:.4f}]")
    print(f"Global Pred Phase Range: [{global_pred_phase_min:.4f}, {global_pred_phase_max:.4f}]")

    # --- PLOTTING ---
    
    # Plot first 5 samples for visual inspection of ranges
    num_samples_to_plot = 5
    num_rows_per_sample = 2 # 1 for Spatial, 1 for Frequency (FFT)
    
    fig, axes = plt.subplots(num_samples_to_plot * num_rows_per_sample, 5, figsize=(20, 4 * num_samples_to_plot * num_rows_per_sample))
    
    cols = ["Hologram", "GT Amp", "Pred Amp", "GT Phase", "Pred Phase"]
    
    # Set titles for the first row only
    for ax, col in zip(axes[0], cols, strict=False):
        ax.set_title(col + " (Spatial)", fontsize=12, fontweight="bold")
    
    # Set titles for the FFT rows? Or just side labels?
    
    for i in range(num_samples_to_plot):
        idx = indices[i]
        
        # Spatial Row Index
        row_spatial = i * 2
        row_fft = i * 2 + 1
        
        # Get Data again (inefficient but cleaner logic for script)
        holo_t, gt_obj_t = dataset[idx]
        holo_in = holo_t.unsqueeze(0).to(device)
        clean_pred, _ = model(holo_in)
        
        # GT
        gt_c = torch.complex(gt_obj_t[0], gt_obj_t[1])
        gt_amp = torch.abs(gt_c).squeeze().numpy()
        gt_phase = torch.angle(gt_c).squeeze().numpy()
        
        # Pred
        pred_c = torch.complex(clean_pred[:, 0, :, :], clean_pred[:, 1, :, :])
        pred_amp = torch.abs(pred_c).squeeze().detach().cpu().numpy()
        pred_phase = torch.angle(pred_c).squeeze().detach().cpu().numpy()
        
        # Holo
        img_holo = holo_t.squeeze().numpy()
        
        # --- SPATIAL PLOTS ---
        # Plot Holo
        axes[row_spatial, 0].imshow(img_holo, cmap="gray")
        axes[row_spatial, 0].axis("off")
        
        # Plot Amps (Dynamic)
        im_gt_amp = axes[row_spatial, 1].imshow(gt_amp, cmap="inferno")
        axes[row_spatial, 1].axis("off")
        plt.colorbar(im_gt_amp, ax=axes[row_spatial, 1], fraction=0.046, pad=0.04)
        
        im_pred_amp = axes[row_spatial, 2].imshow(pred_amp, cmap="inferno")
        axes[row_spatial, 2].axis("off")
        plt.colorbar(im_pred_amp, ax=axes[row_spatial, 2], fraction=0.046, pad=0.04)
        
        # Plot Phases (Dynamic)
        im_gt_ph = axes[row_spatial, 3].imshow(gt_phase, cmap="twilight")
        axes[row_spatial, 3].axis("off")
        plt.colorbar(im_gt_ph, ax=axes[row_spatial, 3], fraction=0.046, pad=0.04)
        
        im_pred_ph = axes[row_spatial, 4].imshow(pred_phase, cmap="twilight")
        axes[row_spatial, 4].axis("off")
        plt.colorbar(im_pred_ph, ax=axes[row_spatial, 4], fraction=0.046, pad=0.04)
        
        # --- FFT PLOTS ---
        def compute_log_fft(img):
            f = np.fft.fft2(img)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-8)
            return magnitude_spectrum

        fft_holo = compute_log_fft(img_holo)
        fft_gt_amp = compute_log_fft(gt_amp)
        fft_pred_amp = compute_log_fft(pred_amp)
        fft_gt_phase = compute_log_fft(gt_phase)
        fft_pred_phase = compute_log_fft(pred_phase)
        
        # Add labels
        axes[row_fft, 0].set_ylabel("FFT", fontsize=10, fontweight="bold")
        
        axes[row_fft, 0].imshow(fft_holo, cmap="gray")
        axes[row_fft, 0].axis("off")
        
        axes[row_fft, 1].imshow(fft_gt_amp, cmap="gray")
        axes[row_fft, 1].axis("off")
        
        axes[row_fft, 2].imshow(fft_pred_amp, cmap="gray")
        axes[row_fft, 2].axis("off")
        
        axes[row_fft, 3].imshow(fft_gt_phase, cmap="gray")
        axes[row_fft, 3].axis("off")
        
        axes[row_fft, 4].imshow(fft_pred_phase, cmap="gray")
        axes[row_fft, 4].axis("off")

    plt.tight_layout()
    plt.savefig("stats_comparison_fft.png", dpi=150)
    print("\nSaved visualization to 'stats_comparison_fft.png'")

if __name__ == "__main__":
    check_gt_stats()
