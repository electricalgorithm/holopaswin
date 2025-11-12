"""Dataset module for loading holographic data.

This module provides the HoloDataset class for loading and preprocessing
holographic data from .npz files containing object, hologram, and reconstruction data.
"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class HoloDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset for loading holographic data from .npz files.

    This dataset loads holographic data pairs (hologram, object) from .npz files,
    applies resizing transformations, and returns them as PyTorch tensors.
    The dataset assumes each .npz file contains at least 'object' and 'hologram' keys.
    """

    def __init__(self, data_dir: str, target_size: int = 224) -> None:
        """Initialize the dataset.

        Args:
            data_dir: Path to folder with .npz files.
                      Keys assumed: 'object', 'hologram', 'reconstruction'
            target_size: The spatial dimension to resize images to (e.g., 224).
                        Defaults to 224.
        """
        self.files = glob.glob(os.path.join(data_dir, "*.npz"))
        self.target_size = target_size

        # Define resizing transform
        # Interpolation: Bilinear is usually fine for holograms,
        # though Bicubic might preserve interference fringes slightly better.
        self.resize = transforms.Compose(
            [
                transforms.Resize(
                    (target_size, target_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                )
            ]
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset.

        Returns:
            Number of .npz files in the data directory.
        """
        return len(self.files)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single data sample from the dataset.

        Args:
            idx: Index of the sample to retrieve.

        Returns:
            A tuple containing:
                - hologram: Tensor of shape (1, H, W) containing the input hologram.
                - object: Tensor of shape (1, H, W) containing the ground truth object.
        """
        # Load file
        data = np.load(self.files[idx])

        # Extract components (normalize to 0-1 if not already)
        obj = data["object"].astype(np.float32)
        holo = data["hologram"].astype(np.float32)

        # If dataset doesn't have reconstruction, we can compute it in the model,
        # but the prompt says it has it. We might use it for debugging or
        # forcing the input to be exactly what's in the pickle.
        # For this model, we only strictly need 'hologram' (input) and 'object' (target).

        # Add channel dimension (C, H, W)
        obj = torch.from_numpy(obj).unsqueeze(0)
        holo = torch.from_numpy(holo).unsqueeze(0)

        # Apply Resize
        obj = self.resize(obj)
        holo = self.resize(holo)

        return holo, obj
