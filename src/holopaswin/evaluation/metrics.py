"""Metrics module for holographic reconstruction evaluation.

Provides standardized metrics computation for comparing reconstruction methods:
- Phase PSNR and SSIM
- Amplitude SSIM
- Background-to-Signal (B/S) ratio for twin-image suppression
- Complex MSE
"""

import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


# Default ranges for holography domain
PHASE_DATA_RANGE = 2 * np.pi  # Phase in [-pi, pi]
AMPLITUDE_DATA_RANGE = 1.2     # Object amplitude typically < 1.2
BG_THRESHOLD = 0.98            # Background is where GT amplitude > threshold


def phase_psnr(pred_phase: np.ndarray, gt_phase: np.ndarray) -> float:
    """Compute PSNR for phase images.

    Args:
        pred_phase: Predicted phase array in [-pi, pi].
        gt_phase: Ground truth phase array in [-pi, pi].

    Returns:
        PSNR value in dB.
    """
    return float(psnr(gt_phase, pred_phase, data_range=PHASE_DATA_RANGE))  # type: ignore[no-untyped-call]


def phase_ssim(pred_phase: np.ndarray, gt_phase: np.ndarray) -> float:
    """Compute SSIM for phase images.

    Args:
        pred_phase: Predicted phase array in [-pi, pi].
        gt_phase: Ground truth phase array in [-pi, pi].

    Returns:
        SSIM value in [0, 1].
    """
    return float(ssim(gt_phase, pred_phase, data_range=PHASE_DATA_RANGE))  # type: ignore[no-untyped-call]


def amplitude_ssim(pred_amp: np.ndarray, gt_amp: np.ndarray) -> float:
    """Compute SSIM for amplitude images.

    Args:
        pred_amp: Predicted amplitude array.
        gt_amp: Ground truth amplitude array.

    Returns:
        SSIM value in [0, 1].
    """
    return float(ssim(gt_amp, pred_amp, data_range=AMPLITUDE_DATA_RANGE))  # type: ignore[no-untyped-call]


def amplitude_psnr(pred_amp: np.ndarray, gt_amp: np.ndarray) -> float:
    """Compute PSNR for amplitude images.

    Args:
        pred_amp: Predicted amplitude array.
        gt_amp: Ground truth amplitude array.

    Returns:
        PSNR value in dB.
    """
    return float(psnr(gt_amp, pred_amp, data_range=AMPLITUDE_DATA_RANGE))  # type: ignore[no-untyped-call]


def complex_mse(pred_real: np.ndarray, pred_imag: np.ndarray,
                gt_real: np.ndarray, gt_imag: np.ndarray) -> float:
    """Compute MSE over complex field (Real + Imag channels).

    Args:
        pred_real: Predicted real part.
        pred_imag: Predicted imaginary part.
        gt_real: Ground truth real part.
        gt_imag: Ground truth imaginary part.

    Returns:
        MSE value.
    """
    mse_real = np.mean((pred_real - gt_real) ** 2)
    mse_imag = np.mean((pred_imag - gt_imag) ** 2)
    return float(mse_real + mse_imag)


def background_to_signal_ratio(
    pred_amp: np.ndarray,
    gt_amp: np.ndarray,
    bg_threshold: float = BG_THRESHOLD,
) -> float:
    """Compute Background-to-Signal (B/S) ratio.

    Measures twin-image suppression quality:
    - Lower B/S = better suppression (cleaner background)

    The metric computes:
    - Background region: where GT amplitude > threshold (uniform region)
    - Signal region: where GT amplitude <= threshold (object)
    - B/S = std(pred_amp in background) / contrast(signal vs background)

    Args:
        pred_amp: Predicted amplitude array.
        gt_amp: Ground truth amplitude array (used only for mask definition).
        bg_threshold: Threshold for background definition.

    Returns:
        B/S ratio (lower is better).
    """
    # Define masks
    bg_mask = gt_amp > bg_threshold
    obj_mask = gt_amp <= bg_threshold

    # Check if masks are valid
    if not np.any(bg_mask) or not np.any(obj_mask):
        return 0.0

    # Background statistics
    bg_mean = np.mean(pred_amp[bg_mask])
    bg_std = np.std(pred_amp[bg_mask])

    # Signal contrast
    obj_contrast = np.mean(np.abs(bg_mean - pred_amp[obj_mask]))

    # B/S ratio
    return float(bg_std / (obj_contrast + 1e-8))


def compute_all_metrics(
    pred_amp: np.ndarray,
    pred_phase: np.ndarray,
    gt_amp: np.ndarray,
    gt_phase: np.ndarray,
    pred_real: np.ndarray | None = None,
    pred_imag: np.ndarray | None = None,
    gt_real: np.ndarray | None = None,
    gt_imag: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute all metrics for a single sample.

    Args:
        pred_amp: Predicted amplitude.
        pred_phase: Predicted phase.
        gt_amp: Ground truth amplitude.
        gt_phase: Ground truth phase.
        pred_real: Optional predicted real part (for complex MSE).
        pred_imag: Optional predicted imaginary part (for complex MSE).
        gt_real: Optional ground truth real part (for complex MSE).
        gt_imag: Optional ground truth imaginary part (for complex MSE).

    Returns:
        Dictionary with all metric values.
    """
    metrics = {
        "phase_psnr": phase_psnr(pred_phase, gt_phase),
        "phase_ssim": phase_ssim(pred_phase, gt_phase),
        "amp_ssim": amplitude_ssim(pred_amp, gt_amp),
        "amp_psnr": amplitude_psnr(pred_amp, gt_amp),
        "bs_ratio": background_to_signal_ratio(pred_amp, gt_amp),
    }

    # Add complex MSE if real/imag parts provided
    if all(x is not None for x in [pred_real, pred_imag, gt_real, gt_imag]):
        metrics["complex_mse"] = complex_mse(pred_real, pred_imag, gt_real, gt_imag)  # type: ignore[arg-type]

    return metrics
