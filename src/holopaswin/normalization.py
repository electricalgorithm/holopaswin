"""Hologram normalisation module.

A camera records a hologram in digital numbers (DN) whose scale depends on the
sensor gain, bit depth and exposure. The reconstruction physics instead works on
intensity relative to the illumination, where the background (|U|^2 of the
unscattered wave) is 1: the network back-propagates sqrt(H), and the physics loss
compares H with |ASM(O)|^2 of an object whose background amplitude is 1.

Flat-field normalisation converts between the two:

    H = (I - dark) / mean(I - dark)

The mean is used as the reference level because angular spectrum propagation
conserves the total power, so the mean intensity at the sensor equals the
illumination times the mean object transmission, minus the light the objective
does not collect. It works for synthetic and real holograms alike, without a
separate reference frame.

For transparent samples the result is the illumination. Absorbing samples lower
the mean, so their normalised background lies slightly above 1. For red blood
cells at 532 nm it lies about 2.3% above (1.95% from haemoglobin absorption, 0.32% from light
scattered outside an NA 0.45 objective, measured with hologen). The physics loss
compares shapes after normalising both sides by their mean, so this offset does
not penalise a correct prediction (holopaswin.loss).
"""

import numpy as np
import numpy.typing as npt


def normalize_hologram(hologram: npt.NDArray[np.floating], dark_level: float = 0.0) -> npt.NDArray[np.floating]:
    """Normalise a recorded hologram so that its background intensity is 1.

    Args:
        hologram: Recorded intensity in camera units (e.g. digital numbers).
            It is not modified.
        dark_level: Signal recorded without light (dark offset) in the same
            units, e.g. the mean of a dark frame. Defaults to 0.

    Returns:
        The normalised hologram with the same shape and dtype.

    Raises:
        ValueError: If the hologram has no signal above the dark level.
    """
    signal = hologram - dark_level
    reference = float(signal.mean())
    if not np.isfinite(reference) or reference <= 0:
        msg = f"hologram has no signal above the dark level (mean {reference})"
        raise ValueError(msg)
    return (signal / reference).astype(hologram.dtype, copy=False)
