"""HoloPASWIN: Physics-Aware Swin Transformer for Holographic Reconstruction.

This package provides a deep learning solution for eliminating twin-image artifacts
in in-line holography using a physics-aware Swin Transformer architecture.

The main components include:
    - HoloPASWIN: The complete end-to-end model combining physics-based propagation
      with deep learning refinement
    - HoloDataset: Dataset loader for holographic data
    - PhysicsLoss: Physics-constrained loss function
    - AngularSpectrumPropagator: Differentiable wave propagation module
    - SwinTransformerSys: U-Net architecture with Swin Transformer encoder
"""

from holopaswin.dataset import HoloDataset
from holopaswin.loss import PhysicsLoss
from holopaswin.model import HoloPASWIN
from holopaswin.propagator import AngularSpectrumPropagator
from holopaswin.swin_transformer import SwinTransformerSys

__all__ = [
    "AngularSpectrumPropagator",
    "HoloDataset",
    "HoloPASWIN",
    "PhysicsLoss",
    "SwinTransformerSys",
]

__version__ = "0.1.0"
