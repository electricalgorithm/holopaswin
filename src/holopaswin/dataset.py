"""Dataset module for loading holographic data.

This module provides the HoloDataset class for loading and preprocessing
holographic data from Parquet files containing object (Real/Imag) and hologram (Intensity) data.
"""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class HoloDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """PyTorch Dataset for loading holographic data from Parquet files.

    This dataset loads holographic data pairs (hologram, object) from .parquet files.
    - Hologram: Intensity (1 channel)
    - Object: Real + Imag parts (2 channels)

    The dataset assumes the directory structure from `inline-digital-holography-v2`.
    """

    def __init__(self, data_dir: str, target_size: int = 224, img_dim: int = 512) -> None:
        """Initialize the dataset.

        Args:
            data_dir: Path to folder with .parquet files (hologen/inline-digital-holography-v2).
            target_size: The spatial dimension to resize images to (e.g., 224).
                        Defaults to 224.
            img_dim: The original dimension of the images in the parquet files.
                     Defaults to 512.
        """
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.img_dim = img_dim

        # Find all hologram files
        self.holo_files = sorted(self.data_dir.glob("hologram_*.parquet"))

        print(f"Found {len(self.holo_files)} hologram parquet files.")

        # Pixels per image (img_dim x img_dim)
        # We hardcode this because the dataset structure (1 pixel per row) implies it.
        # If it changes, this breaks, but we verified it's 262144 rows per image for 512x512.
        self.rows_per_sample = self.img_dim * self.img_dim

        self.samples_index = []

        print("Indexing dataset...")
        for h_path in self.holo_files:
            gt_path = h_path.parent / h_path.name.replace("hologram", "ground_truth")
            if not gt_path.exists():
                continue

            # Use pyarrow to get metadata
            pf = pq.ParquetFile(h_path)
            num_rows = pf.metadata.num_rows

            # Calculate number of images in this file
            num_samples = num_rows // self.rows_per_sample

            self.samples_index.append(
                {
                    "holo_path": str(h_path),
                    "gt_path": str(gt_path),
                    "count": num_samples,
                }
            )

        self.cumulative_counts = np.cumsum([x["count"] for x in self.samples_index])
        self.total_samples = int(self.cumulative_counts[-1]) if len(self.cumulative_counts) > 0 else 0
        print(f"Indexed {self.total_samples} samples.")

        # Define resizing transform
        # We use CenterCrop instead of Resize to preserve pixel pitch (physics).
        # Resizing would change the effective pixel size and break propagation.
        self.resize = transforms.Compose(
            [
                transforms.CenterCrop(target_size)
            ]
        )

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.total_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a single data sample from the dataset.

        Args:
            idx: Global index of the sample.

        Returns:
            A tuple containing:
                - hologram: Tensor of shape (1, H, W) (Intensity).
                - object: Tensor of shape (2, H, W) (Real, Imag).
        """
        # Find which file contains this index
        file_idx = np.searchsorted(self.cumulative_counts, idx, side="right")

        local_idx = idx if file_idx == 0 else idx - self.cumulative_counts[file_idx - 1]

        file_info = self.samples_index[file_idx]

        # Calculate row offset
        # Each image is img_dim*img_dim consecutive rows
        row_start = local_idx * self.rows_per_sample

        # Load Hologram (Intensity)
        # We read the table with specific columns.
        # Note: 'read_table' might read the full column into memory then slice.
        # This is memory intensive but simple.
        # Given batch size is small, it might pass.
        htable = pq.read_table(file_info["holo_path"], columns=["intensity"])
        h_slice = htable.slice(offset=row_start, length=self.rows_per_sample)

        # Convert to numpy and reshape
        # Note: .to_numpy() on a ChunkedArray (from slice) is efficient
        h_flat = h_slice["intensity"].to_numpy()  # (262144,)
        h_np = h_flat.reshape(self.img_dim, self.img_dim).astype(np.float32)

        # Load Ground Truth (Real, Imag)
        gtable = pq.read_table(file_info["gt_path"], columns=["real", "imag"])
        g_slice = gtable.slice(offset=row_start, length=self.rows_per_sample)

        r_np = g_slice["real"].to_numpy().reshape(self.img_dim, self.img_dim).astype(np.float32)
        i_np = g_slice["imag"].to_numpy().reshape(self.img_dim, self.img_dim).astype(np.float32)

        # Convert to Tensor
        # NORMALIZE: Raw 12-bit intensity is around 1000.
        # We value consistency with the physics model where |Object|=1 => Intensity=1.
        # So we scale the input by 1000.0 to bring it to ~1.0 range.
        holo_np = h_np / 1000.0
        holo = torch.from_numpy(holo_np).unsqueeze(0)

        # Object: (2, H, W) -> [Real, Imag]
        obj = torch.stack([torch.from_numpy(r_np), torch.from_numpy(i_np)], dim=0)

        # Resize
        holo = self.resize(holo)
        obj = self.resize(obj)

        return holo, obj
