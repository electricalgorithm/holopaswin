"""Loss function module for physics-constrained holographic reconstruction.

This module provides the PhysicsLoss class which combines structural loss
(supervised) with physics consistency loss (unsupervised constraint) to ensure
that the predicted object, when forward-propagated, matches the input hologram.
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
       when forward-propagated, matches the input hologram intensity.

    The total loss is: L = L_structural + lambda_physics * L_physics
    """

    def __init__(self, propagator: "AngularSpectrumPropagator", lambda_physics: float = 0.1) -> None:
        """Initialize the physics loss function."""
        super().__init__()
        self.propagator = propagator
        self.l1 = torch.nn.L1Loss()
        self.lambda_p = lambda_physics

    def forward(
        self,
        pred_obj_2ch: torch.Tensor,
        target_obj_2ch: torch.Tensor,
        input_hologram: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the physics-constrained loss.

        Args:
            pred_obj_2ch: Predicted object tensor of shape (B, 2, H, W).
                          Channel 0: Real, Channel 1: Imag.
            target_obj_2ch: Ground truth object tensor of shape (B, 2, H, W).
                            Channel 0: Real, Channel 1: Imag.
            input_hologram: Input hologram intensity tensor of shape (B, 1, H, W).

        Returns:
            Total loss value as a scalar tensor.
        """
        pred_complex = torch.complex(pred_obj_2ch[:, 0:1, ...], pred_obj_2ch[:, 1:2, ...])
        target_complex = torch.complex(target_obj_2ch[:, 0:1, ...], target_obj_2ch[:, 1:2, ...])

        # 1. Complex Structural Loss (Supervised)
        # L1 on Real/Imag parts separately
        loss_complex = self.l1(pred_obj_2ch, target_obj_2ch)

        # 2. Amplitude Loss (Supervised)
        # Explicitly penalize errors in magnitude to prevent "empty" predictions
        pred_amp = torch.abs(pred_complex)
        target_amp = torch.abs(target_complex)
        loss_amp = self.l1(pred_amp, target_amp)

        # 3. Phase Loss (Supervised)
        # Explicitly penalize errors in phase.
        # around the core object, direct L1 on angle is a strong starting point.
        pred_phase = torch.angle(pred_complex)
        target_phase = torch.angle(target_complex)
        loss_phase = self.l1(pred_phase, target_phase)

        # 4. Physics Consistency Loss (Unsupervised constraint)
        # Propagate +z (Forward to Hologram Plane)
        pred_field_holo = self.propagator(pred_complex, backward=False)
        pred_hologram = torch.abs(pred_field_holo) ** 2
        loss_phy = self.l1(pred_hologram, input_hologram)

        # Total Loss Combination
        # Normalized Weights (Sum of supervised terms = 1.0):
        # - Amplitude (0.5): Highest priority to force particle detection (fix "empty" output).
        # - Complex (0.25): Baseline numerical stability constraint.
        # - Phase (0.25): Refinement for internal structure/thickness.
        # - Physics (lambda): Independent consistency constraint.
        loss_supervised = (0.25 * loss_complex) + (0.5 * loss_amp) + (0.25 * loss_phase)

        return loss_supervised + (self.lambda_p * loss_phy)  # type: ignore[no-any-return]
