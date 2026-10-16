"""Dataset module for loading holographic data.

This module provides the HoloDataset class for loading and preprocessing
holographic data from Parquet files containing object (Real/Imag) and hologram (Intensity) data.
"""

import tomllib
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from holopaswin.normalization import normalize_hologram

MICROMETER: float = 1e-6  # hologen stores distances in um


def read_recipe_geometry(data_dir: Path) -> dict[str, float]:
    """Read the geometry section of the recipe hologen saved with a dataset.

    Args:
        data_dir (Path): Dataset folder with recipe.toml.

    Returns:
        dict[str, float]: The numeric geometry values as written by hologen, so
            distances and pixel sizes are in um.

    Raises:
        FileNotFoundError: If the folder has no recipe.toml.
    """
    recipe: Path = data_dir / "recipe.toml"
    if not recipe.exists():
        msg: str = f"{recipe} not found; it holds the geometry of the dataset."
        raise FileNotFoundError(msg)
    with recipe.open("rb") as file:
        section: dict[str, object] = tomllib.load(file).get("geometry", {})
    return {
        key: float(value)
        for key, value in section.items()
        if isinstance(value, int | float) and not isinstance(value, bool)
    }


def read_nominal_distance(data_dir: Path) -> float:
    """Read the nominal recording distance of a dataset.

    Args:
        data_dir (Path): Dataset folder with recipe.toml.

    Returns:
        float: The geometry's z_distance in meters.

    Raises:
        KeyError: If the recipe has no z_distance in its geometry section.
    """
    geometry: dict[str, float] = read_recipe_geometry(data_dir)
    if "z_distance" not in geometry:
        msg: str = f"{data_dir / 'recipe.toml'} has no z_distance in its [geometry] section."
        raise KeyError(msg)
    return geometry["z_distance"] * MICROMETER


class HoloDataset(Dataset[tuple[torch.Tensor, ...]]):
    """PyTorch Dataset for loading holographic data from Parquet files.

    This dataset loads holographic data pairs (hologram, object) from .parquet files.
    - Hologram: Intensity (1 channel)
    - Object: Real + Imag parts (2 channels)

    The dataset assumes the directory structure from `inline-digital-holography-v2`.
    With return_z, each item also carries the distance at which its hologram was
    recorded, for the physics loss of z-augmented training.
    """

    def __init__(self, data_dir: str, target_size: int = 224, img_dim: int = 512, return_z: bool = False) -> None:
        """Initialize the dataset.

        Args:
            data_dir: Path to folder with .parquet files (hologen/inline-digital-holography-v2).
            target_size: The spatial dimension to resize images to (e.g., 224).
                        Defaults to 224.
            img_dim: The original dimension of the images in the parquet files.
                     Defaults to 512.
            return_z: If True, items are (hologram, object, z_true), where z_true is
                the recording distance in meters from the metadata column z_true.
                Datasets made before hologen recorded the distance use the nominal
                distance of their recipe.toml.
        """
        self.data_dir = Path(data_dir)
        self.target_size = target_size
        self.img_dim = img_dim
        self.return_z = return_z

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

        self.z_true: np.ndarray | None = self._read_distances() if return_z else None

        self.cumulative_counts = np.cumsum([x["count"] for x in self.samples_index])
        self.total_samples = int(self.cumulative_counts[-1]) if len(self.cumulative_counts) > 0 else 0
        print(f"Indexed {self.total_samples} samples.")

        # Define resizing transform
        # We use CenterCrop instead of Resize to preserve pixel pitch (physics).
        # Resizing would change the effective pixel size and break propagation.
        self.resize = transforms.Compose([transforms.CenterCrop(target_size)])

    def read_metadata(self, columns: list[str]) -> pd.DataFrame:
        """Read per-hologram metadata aligned with the samples of this dataset.

        The rows come from the same files and counts as the samples, so row i
        describes sample i. A column that a metadata file lacks, such as z_true
        in datasets made before hologen recorded it, is filled with missing
        values.

        Args:
            columns (list[str]): Metadata columns to read, e.g. object_family.

        Returns:
            pd.DataFrame: One row per sample with the requested columns.

        Raises:
            ValueError: If a metadata file has fewer rows than its hologram file
                has holograms, which would shift every later row.
        """
        frames: list[pd.DataFrame] = []
        file_info: dict[str, str | int]
        for file_info in self.samples_index:
            hologram_path: Path = Path(str(file_info["holo_path"]))
            metadata_path: Path = hologram_path.parent / hologram_path.name.replace("hologram", "metadata")
            count: int = int(file_info["count"])
            frame: pd.DataFrame = pd.DataFrame(index=range(count))
            if metadata_path.exists():
                table: pq.ParquetFile = pq.ParquetFile(metadata_path)
                if table.metadata.num_rows < count:
                    msg: str = f"{metadata_path.name} has {table.metadata.num_rows} rows for {count} holograms."
                    raise ValueError(msg)
                present: list[str] = [column for column in columns if column in table.schema_arrow.names]
                frame = table.read(columns=present).to_pandas().iloc[:count].reset_index(drop=True)
            frames.append(frame.reindex(columns=columns))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)

    def _read_distances(self) -> np.ndarray:
        """Collect the recording distance of every indexed hologram.

        Returns:
            np.ndarray: Distance of each sample in meters, in dataset order.
                Samples without a recorded distance get the nominal one.
        """
        distances: np.ndarray = self.read_metadata(["z_true"])["z_true"].to_numpy(dtype=np.float64) * MICROMETER
        missing: np.ndarray = np.isnan(distances)
        if missing.any():
            distances[missing] = read_nominal_distance(self.data_dir)
        return distances

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.total_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        """Get a single data sample from the dataset.

        Args:
            idx: Global index of the sample.

        Returns:
            A tuple containing:
                - hologram: Tensor of shape (1, H, W) (Intensity).
                - object: Tensor of shape (2, H, W) (Real, Imag).
                - z_true: Recording distance in meters, a float32 scalar tensor,
                  only when the dataset was made with return_z.
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

        # Convert the camera's digital numbers to intensity relative to the
        # illumination (background = 1), as used by the physics model and L_phy.
        holo_np = normalize_hologram(h_np)
        holo = torch.from_numpy(holo_np).unsqueeze(0)

        # Object: (2, H, W) -> [Real, Imag]
        obj = torch.stack([torch.from_numpy(r_np), torch.from_numpy(i_np)], dim=0)

        # Resize
        holo = self.resize(holo)
        obj = self.resize(obj)

        if self.z_true is not None:
            return holo, obj, torch.tensor(self.z_true[idx], dtype=torch.float32)
        return holo, obj
