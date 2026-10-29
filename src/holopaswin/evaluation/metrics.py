"""Metrics module for holographic reconstruction evaluation.

Provides standardized metrics computation for comparing reconstruction methods:
- Phase PSNR and SSIM
- Amplitude PSNR and SSIM
- Amplitude and wrapped phase MSE
- Frequency SSIM of the amplitude spectra
- Background-to-Signal (B/S) ratio for twin-image suppression
- Complex MSE
- Background amplitude ratio, a check of the global amplitude scale
"""

import numpy as np
from skimage.metrics import structural_similarity as ssim

# Default ranges for holography domain
PHASE_DATA_RANGE = 2 * np.pi  # Phase in [-pi, pi]
AMPLITUDE_DATA_RANGE = 1.2  # Object amplitude typically < 1.2
BG_THRESHOLD = 0.98  # Background is where GT amplitude > threshold


def psnr_of(mse: float, data_range: float) -> float:
    """Convert a mean squared error to a PSNR.

    Args:
        mse: Mean squared error.
        data_range: Peak value R of the PSNR.

    Returns:
        10 log10(R^2 / MSE) in dB, infinite for an MSE of 0.
    """
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(data_range**2 / mse))


def phase_mse(pred_phase: np.ndarray, gt_phase: np.ndarray) -> float:
    """Compute the mean squared wrapped phase error.

    Phase is defined modulo 2 pi, so the error of each pixel is wrapped into
    [-pi, pi] before it is squared: phases of pi - 0.01 and -pi + 0.01 differ by
    0.02 rad, not by 2 pi - 0.02.

    Args:
        pred_phase: Predicted phase array in [-pi, pi].
        gt_phase: Ground truth phase array in [-pi, pi].

    Returns:
        MSE in rad^2, at most pi^2.
    """
    error: np.ndarray = np.angle(np.exp(1j * (pred_phase - gt_phase)))
    return float(np.mean(error**2))


def phase_psnr(pred_phase: np.ndarray, gt_phase: np.ndarray) -> float:
    """Compute PSNR for phase images from the wrapped phase error.

    PSNR = 10 log10((2 pi)^2 / phase_mse). The wrapped error is at most pi, so
    the PSNR is at least 10 log10(4) = 6.02 dB.

    Args:
        pred_phase: Predicted phase array in [-pi, pi].
        gt_phase: Ground truth phase array in [-pi, pi].

    Returns:
        PSNR value in dB, infinite for identical phases.
    """
    return psnr_of(phase_mse(pred_phase, gt_phase), PHASE_DATA_RANGE)


def phase_ssim(pred_phase: np.ndarray, gt_phase: np.ndarray) -> float:
    """Compute SSIM for phase images.

    SSIM compares local means and contrasts of the two images, so unlike
    phase_psnr it works on the wrapped phases as they are; an object whose phase
    crosses +-pi lowers it.

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


def amplitude_mse(pred_amp: np.ndarray, gt_amp: np.ndarray) -> float:
    """Compute the mean squared amplitude error.

    Args:
        pred_amp: Predicted amplitude array.
        gt_amp: Ground truth amplitude array.

    Returns:
        MSE value.
    """
    return float(np.mean((pred_amp - gt_amp) ** 2))


def amplitude_psnr(pred_amp: np.ndarray, gt_amp: np.ndarray) -> float:
    """Compute PSNR for amplitude images.

    PSNR = 10 log10(1.2^2 / amplitude_mse); 1.2 bounds the object amplitude.

    Args:
        pred_amp: Predicted amplitude array.
        gt_amp: Ground truth amplitude array.

    Returns:
        PSNR value in dB, infinite for identical amplitudes.
    """
    return psnr_of(amplitude_mse(pred_amp, gt_amp), AMPLITUDE_DATA_RANGE)


def log_spectrum(image: np.ndarray) -> np.ndarray:
    """Compute the centred log-magnitude spectrum of an image.

    Args:
        image: Real image.

    Returns:
        log(|fftshift(fft2(image))| + 1e-8), same shape as the image.
    """
    spectrum: np.ndarray = np.log(np.abs(np.fft.fftshift(np.fft.fft2(image))) + 1e-8)
    return spectrum


def frequency_ssim(pred_amp: np.ndarray, gt_amp: np.ndarray) -> float:
    """Compute SSIM between the log-magnitude spectra of two amplitude images.

    The logarithm keeps the strong low frequencies from hiding the weak high
    ones, so the score reflects how well fine detail is restored. The data range
    is the range of the true spectrum, but at least 1, so a nearly flat true
    spectrum does not inflate small differences.

    Args:
        pred_amp: Predicted amplitude array.
        gt_amp: Ground truth amplitude array.

    Returns:
        SSIM value, at most 1.
    """
    true_spectrum: np.ndarray = log_spectrum(gt_amp)
    data_range: float = max(float(true_spectrum.max() - true_spectrum.min()), 1.0)
    return float(ssim(true_spectrum, log_spectrum(pred_amp), data_range=data_range))  # type: ignore[no-untyped-call]


def complex_mse(pred_real: np.ndarray, pred_imag: np.ndarray, gt_real: np.ndarray, gt_imag: np.ndarray) -> float:
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
        B/S ratio (lower is better), or NaN when the ground truth has no
        background or no object, so that such a sample is left out of averages
        instead of counting as a perfect 0.
    """
    # Define masks
    bg_mask = gt_amp > bg_threshold
    obj_mask = gt_amp <= bg_threshold

    # Check if masks are valid
    if not np.any(bg_mask) or not np.any(obj_mask):
        return float("nan")

    # Background statistics
    bg_mean = np.mean(pred_amp[bg_mask])
    bg_std = np.std(pred_amp[bg_mask])

    # Signal contrast
    obj_contrast = np.mean(np.abs(bg_mean - pred_amp[obj_mask]))

    # B/S ratio
    return float(bg_std / (obj_contrast + 1e-8))


def background_amplitude_ratio(
    pred_amp: np.ndarray,
    gt_amp: np.ndarray,
    bg_threshold: float = BG_THRESHOLD,
) -> float:
    """Compare the predicted and the true amplitude where the ground truth is background.

    Phase metrics and the B/S ratio cannot see a global amplitude scale, so this
    ratio records it: median(pred_amp) / median(gt_amp) over the background
    region. The median ignores the symmetric ripples of the twin image, so a
    reconstruction on the right scale gives about 1, and one whose amplitude is
    doubled gives 2.

    Args:
        pred_amp: Predicted amplitude array.
        gt_amp: Ground truth amplitude array (used for the mask and the reference level).
        bg_threshold: Threshold for background definition.

    Returns:
        The ratio, or NaN when the ground truth has no background.
    """
    bg_mask: np.ndarray = gt_amp > bg_threshold
    if not np.any(bg_mask):
        return float("nan")
    return float(np.median(pred_amp[bg_mask]) / np.median(gt_amp[bg_mask]))


def compute_all_metrics(  # noqa: PLR0913
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
        "amp_mse": amplitude_mse(pred_amp, gt_amp),
        "phase_mse": phase_mse(pred_phase, gt_phase),
        "freq_ssim": frequency_ssim(pred_amp, gt_amp),
        "bs_ratio": background_to_signal_ratio(pred_amp, gt_amp),
        "bg_amp_ratio": background_amplitude_ratio(pred_amp, gt_amp),
    }

    # Add complex MSE if real/imag parts provided
    if all(x is not None for x in [pred_real, pred_imag, gt_real, gt_imag]):
        metrics["complex_mse"] = complex_mse(pred_real, pred_imag, gt_real, gt_imag)  # type: ignore[arg-type]

    return metrics
