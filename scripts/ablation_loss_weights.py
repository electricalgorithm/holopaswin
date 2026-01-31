"""Ablation study script for loss function coefficients."""

import time

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader, Subset

from holopaswin.dataset import HoloDataset
from holopaswin.loss import PhysicsLoss
from holopaswin.model import HoloPASWIN

# Config
DATA_DIR = "../hologen/dataset-224"
TEST_DATA_DIR = "../hologen/test-dataset-224"
IMG_SIZE = 224
WAVELENGTH = 532e-9
PIXEL_SIZE = 4.65e-6
Z_DIST = 0.02
BATCH_SIZE = 16
STEPS_PER_RUN = 100  # Short run to see trends
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


BG_THRESHOLD = 0.98
AMP_DATA_RANGE = 1.2
PHASE_DATA_RANGE = 2 * np.pi


def compute_bs_ratio(pred_amp: np.ndarray, gt_amp: np.ndarray) -> float:
    """Compute Background-to-Signal Ratio (B/S).

    Measures the standard deviation in background vs signal contrast.
    Uses the predicted background level to be scale-invariant.
    Lower is better (indicates cleaner background).
    """
    # Background is where gt_amp > 0.98
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
        mag = np.log(np.abs(fshift) + 1e-8)
        return mag

    spec_p = log_fft(pred_amp)
    spec_t = log_fft(target_amp)
    # Normalize for SSIM [0, 1] roughly or just use dynamic range
    dr = spec_t.max() - spec_t.min()
    return float(ssim(spec_p, spec_t, data_range=max(dr, 1.0)))


class AblationLoss(PhysicsLoss):
    """Custom loss for ablation that allows overriding weights."""

    def __init__(self, propagator: torch.nn.Module, weights: dict, lambda_p: float = 0.0) -> None:
        """Initialize AblationLoss with specific component weights."""
        super().__init__(propagator, lambda_physics=lambda_p)
        self.w_complex = weights.get("complex", 0.2)
        self.w_amp = weights.get("amp", 0.4)
        self.w_phase = weights.get("phase", 0.2)
        self.w_freq = weights.get("freq", 0.2)

    def forward(
        self, pred_obj_2ch: torch.Tensor, target_obj_2ch: torch.Tensor, input_hologram: torch.Tensor
    ) -> torch.Tensor:
        """Compute the weighted composite loss."""
        # Implementation copied from PhysicsLoss but with dynamic weights
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

        return loss_supervised + (self.lambda_p * loss_phy)


def run_ablation() -> None:
    """Run the loss component ablation study."""
    print(f"Starting Ablation Study on {DEVICE}...")

    # Configurations
    configs = [
        {
            "name": "L1 Only (Complex)",
            "weights": {"complex": 1.0, "amp": 0.0, "phase": 0.0, "freq": 0.0},
            "lambda": 0.0,
        },
        {
            "name": "Structural (Amp+Phase)",
            "weights": {"complex": 0.2, "amp": 0.6, "phase": 0.2, "freq": 0.0},
            "lambda": 0.0,
        },
        {
            "name": "HoloPASWIN (Full Sup)",
            "weights": {"complex": 0.2, "amp": 0.4, "phase": 0.2, "freq": 0.2},
            "lambda": 0.0,
        },
        {
            "name": "HoloPASWIN + Phys",
            "weights": {"complex": 0.2, "amp": 0.4, "phase": 0.2, "freq": 0.2},
            "lambda": 0.1,
        },
    ]

    # Load Data
    full_dataset = HoloDataset(DATA_DIR, target_size=IMG_SIZE, img_dim=IMG_SIZE)
    # Subset for faster iteration
    train_indices = list(range(min(2000, len(full_dataset))))
    train_loader = DataLoader(Subset(full_dataset, train_indices), batch_size=BATCH_SIZE, shuffle=True)

    test_dataset = HoloDataset(TEST_DATA_DIR, target_size=IMG_SIZE, img_dim=IMG_SIZE)
    test_indices = list(range(min(50, len(test_dataset))))
    test_loader = DataLoader(Subset(test_dataset, test_indices), batch_size=1, shuffle=False)

    results = []

    for cfg in configs:
        print(f"\nEvaluating: {cfg['name']}")
        model = HoloPASWIN(IMG_SIZE, WAVELENGTH, PIXEL_SIZE, Z_DIST).to(DEVICE)
        criterion = AblationLoss(model.propagator, cfg["weights"], lambda_p=cfg["lambda"]).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Training
        model.train()
        start_time = time.time()

        for step, (h_batch, g_batch) in enumerate(train_loader):
            if step >= STEPS_PER_RUN:
                break

            h_dev, g_dev = h_batch.to(DEVICE), g_batch.to(DEVICE)
            optimizer.zero_grad()
            pred, _ = model(h_dev)
            loss = criterion(pred, g_dev, h_dev)
            loss.backward()
            optimizer.step()

            if (step + 1) % 20 == 0:
                print(f"  Step {step + 1}/{STEPS_PER_RUN} - Loss: {loss.item():.4f}")

        elapsed = time.time() - start_time

        # Testing
        model.eval()
        m_amp_ssim, m_phase_ssim, m_freq_ssim, m_bs_ratio = [], [], [], []

        with torch.no_grad():
            for h_batch, g_batch in test_loader:
                h_dev, g_dev = h_batch.to(DEVICE), g_batch.to(DEVICE)
                pred, _ = model(h_dev)

                # Conversion to NumPy
                pred_c = torch.complex(pred[:, 0], pred[:, 1]).squeeze().cpu().numpy()
                gt_c = torch.complex(g_dev[:, 0], g_dev[:, 1]).squeeze().cpu().numpy()

                pa, ta = np.abs(pred_c), np.abs(gt_c)
                pp, tp = np.angle(pred_c), np.angle(gt_c)

                m_amp_ssim.append(ssim(pa, ta, data_range=AMP_DATA_RANGE))
                m_phase_ssim.append(ssim(pp, tp, data_range=PHASE_DATA_RANGE))
                m_freq_ssim.append(get_freq_ssim(pa, ta))
                m_bs_ratio.append(compute_bs_ratio(pa, ta))

        res = {
            "name": cfg["name"],
            "amp_ssim": np.mean(m_amp_ssim),
            "phase_ssim": np.mean(m_phase_ssim),
            "freq_ssim": np.mean(m_freq_ssim),
            "bs_ratio": np.mean(m_bs_ratio),
            "time": elapsed,
        }
        results.append(res)
        print(
            f"  Results: Amp SSIM: {res['amp_ssim']:.4f}, Phase SSIM: {res['phase_ssim']:.4f}, Freq SSIM: {res['freq_ssim']:.4f}, B/S: {res['bs_ratio']:.4f}"
        )

    # Report
    print("\n" + "=" * 95)
    print(f"{'Configuration':<25} | {'Amp SSIM':<10} | {'Phase SSIM':<10} | {'Freq SSIM':<10} | {'B/S Ratio':<10}")
    print("-" * 95)
    for r in results:
        print(
            f"{r['name']:<25} | {r['amp_ssim']:<10.4f} | {r['phase_ssim']:<10.4f} | {r['freq_ssim']:<10.4f} | {r['bs_ratio']:<10.4f}"
        )
    print("=" * 95)


if __name__ == "__main__":
    run_ablation()
