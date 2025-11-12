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
    1. Structural Loss (L1): Supervised loss comparing predicted object to ground truth.
    2. Physics Consistency Loss: Unsupervised constraint ensuring predicted object,
       when forward-propagated, matches the input hologram.

    The total loss is: L = L_structural + lambda_physics * L_physics
    """

    def __init__(self, propagator: "AngularSpectrumPropagator", lambda_physics: float = 0.1) -> None:
        """Initialize the physics loss function.

        Args:
            propagator: AngularSpectrumPropagator instance for forward propagation.
            lambda_physics: Weight for the physics consistency loss term.
                          Defaults to 0.1.
        """
        super().__init__()
        self.propagator = propagator
        self.l1 = torch.nn.L1Loss()
        self.lambda_p = lambda_physics

    def forward(
        self,
        pred_obj: torch.Tensor,
        target_obj: torch.Tensor,
        input_hologram: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the physics-constrained loss.

        Args:
            pred_obj: Predicted object amplitude tensor of shape (B, C, H, W).
            target_obj: Ground truth object amplitude tensor of shape (B, C, H, W).
            input_hologram: Input hologram intensity tensor of shape (B, C, H, W).

        Returns:
            Total loss value as a scalar tensor.
        """
        # 1. Structural Loss (Supervised)
        # Compare cleaned output with ground truth object
        loss_img = self.l1(pred_obj, target_obj)

        # 2. Physics Consistency Loss (Unsupervised/Constraint)
        # Forward propagate the PREDICTED clean object to the hologram plane
        # Assume object phase is flat (or learnt). If amplitude only, assume phase=0.
        pred_field_obj = torch.complex(pred_obj, torch.zeros_like(pred_obj))

        # Propagate +z
        pred_field_holo = self.propagator(pred_field_obj, backward=False)

        # Calculate Intensity (Hologram)
        pred_hologram = torch.abs(pred_field_holo) ** 2

        # Compare predicted hologram with real input hologram
        loss_phy = self.l1(pred_hologram, input_hologram)

        return loss_img + (self.lambda_p * loss_phy)  # type: ignore[no-any-return]
