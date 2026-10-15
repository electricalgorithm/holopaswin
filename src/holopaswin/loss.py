"""Loss function module for physics-constrained holographic reconstruction.

This module provides the PhysicsLoss class which combines structural loss
(supervised) with physics consistency loss (unsupervised constraint) to ensure
that the predicted object, when forward-propagated, matches the input hologram.

The physics term compares shapes, not absolute levels. The input hologram is
normalised by its own mean (holopaswin.normalization), which is not exactly the
illumination when the sample absorbs. Red blood cells at 532 nm lower the mean
by about 2%, so their normalised background sits near 1.02. The forward-propagated
prediction is therefore also divided by its own mean before the two are
compared. A correct prediction then matches the input whatever the absorption,
and no blank reference frame is needed.
"""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from holopaswin.propagator import AngularSpectrumPropagator


class PhysicsLoss(torch.nn.Module):
    """Physics-constrained loss function for holographic reconstruction.

    Combines two loss terms:
    1. Structural Loss (L1): Supervised loss comparing predicted object to ground truth (Complex domain).
    2. Physics Consistency Loss: Unsupervised constraint ensuring predicted object,
       when forward-propagated, matches the input hologram intensity. Both are
       divided by their own mean first (see scale_invariant_physics_loss).

    The total loss is: L = L_structural + lambda_physics * L_physics
    """

    def __init__(self, propagator: "AngularSpectrumPropagator", lambda_physics: float = 0.1) -> None:
        """Initialize the physics loss function."""
        super().__init__()
        self.propagator = propagator
        self.l1 = torch.nn.L1Loss()
        self.lambda_p = lambda_physics

    def compute_freq_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute L1 loss in the frequency domain (Spectral Loss)."""
        # FFT2
        pred_fft = torch.fft.fft2(pred)
        target_fft = torch.fft.fft2(target)

        # Magnitude Spectrum (add epsilon for log stability)
        pred_mag = torch.log(torch.abs(pred_fft) + 1e-8)
        target_mag = torch.log(torch.abs(target_fft) + 1e-8)

        return self.l1(pred_mag, target_mag)  # type: ignore[no-any-return]

    def scale_invariant_physics_loss(self, pred_hologram: torch.Tensor, input_hologram: torch.Tensor) -> torch.Tensor:
        """Compare two holograms after dividing each by its own mean.

        Args:
            pred_hologram (torch.Tensor): Forward-propagated intensity
                |ASM(O)|^2 of the prediction, shape (B, 1, H, W).
            input_hologram (torch.Tensor): Mean-normalised input hologram, shape
                (B, 1, H, W).

        Returns:
            torch.Tensor: L1 between the two, each normalised per sample to a
                mean of 1, as a scalar tensor.
        """
        eps: float = 1e-8
        pred_mean: torch.Tensor = pred_hologram.mean(dim=(-3, -2, -1), keepdim=True)
        input_mean: torch.Tensor = input_hologram.mean(dim=(-3, -2, -1), keepdim=True)
        return torch.nn.functional.l1_loss(pred_hologram / (pred_mean + eps), input_hologram / (input_mean + eps))

    def forward(
        self,
        pred_obj_2ch: torch.Tensor,
        target_obj_2ch: torch.Tensor,
        input_hologram: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the physics-constrained loss with Frequency awareness.

        Args:
            pred_obj_2ch: Predicted object tensor of shape (B, 2, H, W).
                          Channel 0: Real, Channel 1: Imag.
            target_obj_2ch: Ground truth object tensor of shape (B, 2, H, W).
                            Channel 0: Real, Channel 1: Imag.
            input_hologram: Input hologram intensity tensor of shape (B, 1, H, W).

        Returns:
            Total loss value as a scalar tensor.
        """
        # Construct Complex Tensors
        pred_complex = torch.complex(pred_obj_2ch[:, 0:1, ...], pred_obj_2ch[:, 1:2, ...])
        target_complex = torch.complex(target_obj_2ch[:, 0:1, ...], target_obj_2ch[:, 1:2, ...])

        # 1. Complex Structural Loss (Supervised) - Baseline
        loss_complex = self.l1(pred_obj_2ch, target_obj_2ch)

        # 2. Amplitude Loss (Supervised) - Priority for Object Presence
        pred_amp = torch.abs(pred_complex)
        target_amp = torch.abs(target_complex)
        loss_amp = self.l1(pred_amp, target_amp)

        # 3. Phase Loss (Supervised) - Priority for Thickness
        pred_phase = torch.angle(pred_complex)
        target_phase = torch.angle(target_complex)
        loss_phase = self.l1(pred_phase, target_phase)

        # 4. Frequency Loss (Supervised) - Priority for Edges/Texture
        # We compute this on the Amplitude map to ensure sharp object boundaries
        loss_freq_amp = self.compute_freq_loss(pred_amp, target_amp)
        # And optionally on Phase map if needed, but Amp edges are most critical for now.
        # Let's add Phase Freq loss too for completeness to fix the "blur".
        loss_freq_phase = self.compute_freq_loss(pred_phase, target_phase)

        loss_freq = (loss_freq_amp + loss_freq_phase) / 2.0

        # 5. Physics Consistency Loss (Unsupervised constraint)
        # Propagate +z (Forward to Hologram Plane), through the objective pupil
        # when the propagator has a numerical aperture.
        pred_field_holo = self.propagator(pred_complex, backward=False)
        pred_hologram = torch.abs(pred_field_holo) ** 2
        loss_phy = self.scale_invariant_physics_loss(pred_hologram, input_hologram)

        # Total Loss Combination
        # New Balanced Strategy:
        # - Amp (0.4): Still the main driver.
        # - Phase (0.2): Structural guide.
        # - Freq (0.2): Edge/Sharpness enforcer.
        # - Complex (0.2): Numerical stabilizer.
        loss_supervised = (0.2 * loss_complex) + (0.4 * loss_amp) + (0.2 * loss_phase) + (0.2 * loss_freq)

        return loss_supervised + (self.lambda_p * loss_phy)  # type: ignore[no-any-return]
