"""Dataset module for loading holographic data.

This module provides the HoloDataset class for loading and preprocessing
holographic data from Parquet files containing object (Real/Imag) and hologram (Intensity) data.
"""

import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from holopaswin.normalization import normalize_hologram

MICROMETER: float = 1e-6  # hologen stores distances in um
HOLOGEN_FORMAT: bytes = b"2"  # value of hologen.format in the image files this module reads


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

    The folder is a dataset written by hologen in format 2, one hologram per row
    and row group, so each item reads only its own hologram and ground truth.
    With return_z, each item also carries the distance at which its hologram was
    recorded, for the physics loss of z-augmented training.
    """

    def __init__(self, data_dir: str, target_size: int = 224, img_dim: int = 512, return_z: bool = False) -> None:
        """Initialize the dataset.

        Args:
            data_dir: Path to a dataset folder written by hologen.
            target_size: Size of the central crop returned for each item (e.g., 224).
                        Defaults to 224.
            img_dim: Size of the stored images, checked against the files.
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

        self.holo_files = sorted(self.data_dir.glob("hologram_*.parquet"))
        print(f"Found {len(self.holo_files)} hologram parquet files.")
        self.samples_index: list[dict[str, Any]] = []
        h_path: Path
        for h_path in self.holo_files:
            gt_path: Path = h_path.parent / h_path.name.replace("hologram", "ground_truth")
            if not gt_path.exists():
                continue
            hologram_file: pq.ParquetFile
            shape: tuple[int, int]
            hologram_file, shape = self._open_image_file(h_path)
            self._open_image_file(gt_path)
            if shape != (self.img_dim, self.img_dim):
                msg: str = f"{h_path} holds {shape[0]} x {shape[1]} images, not {self.img_dim} x {self.img_dim}."
                raise ValueError(msg)
            self.samples_index.append(
                {"holo_path": str(h_path), "gt_path": str(gt_path), "count": hologram_file.metadata.num_rows}
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

    @staticmethod
    def _open_image_file(path: Path) -> tuple[pq.ParquetFile, tuple[int, int]]:
        """Open an image file written by hologen and check its format.

        hologen format 2 stores one hologram per row, and per row group, with the
        images as 2-D lists; the Parquet metadata names the format and the shape.

        Args:
            path (Path): hologram_XXXX.parquet or ground_truth_XXXX.parquet file.

        Returns:
            tuple[pq.ParquetFile, tuple[int, int]]: The open file and the image shape.

        Raises:
            ValueError: If the file was written before hologen format 2.
        """
        file: pq.ParquetFile = pq.ParquetFile(path)
        metadata: dict[bytes, bytes] = file.schema_arrow.metadata or {}
        if metadata.get(b"hologen.format") != HOLOGEN_FORMAT:
            msg: str = (
                f"{path} was written before hologen format 2 (one pixel per row); regenerate the dataset "
                "with the current hologen."
            )
            raise ValueError(msg)
        height: str
        width: str
        height, width = metadata[b"hologen.shape"].decode().split(",")
        return file, (int(height), int(width))

    @staticmethod
    def _read_image(file: pq.ParquetFile, row: int, column: str, shape: tuple[int, int]) -> np.ndarray:
        """Read one image of one hologram, reading only that hologram's row group.

        Args:
            file (pq.ParquetFile): An open format 2 file.
            row (int): Row, and row group, of the hologram in the file.
            column (str): Column to read, e.g. intensity or real.
            shape (tuple[int, int]): Image shape.

        Returns:
            np.ndarray: The image as float32, shape (H, W).
        """
        images = file.read_row_group(row, columns=[column]).column(column).combine_chunks()
        image: np.ndarray = images.flatten().flatten().to_numpy(zero_copy_only=False).reshape(shape).astype(np.float32)
        return image

    def _locate(self, idx: int) -> tuple[dict[str, Any], int]:
        """Find the file and the row of a sample.

        Args:
            idx: Global index of the sample.

        Returns:
            The index entry of the sample's file and the sample's row in it.
        """
        file_idx: int = int(np.searchsorted(self.cumulative_counts, idx, side="right"))
        local_idx: int = int(idx if file_idx == 0 else idx - self.cumulative_counts[file_idx - 1])
        return self.samples_index[file_idx], local_idx

    def read_recorded(self, idx: int) -> np.ndarray:
        """Read one hologram as recorded, in digital numbers of the camera.

        Args:
            idx: Global index of the sample.

        Returns:
            The hologram before normalisation, shape (img_dim, img_dim), float32.
        """
        file_info: dict[str, Any]
        row: int
        file_info, row = self._locate(idx)
        file: pq.ParquetFile
        shape: tuple[int, int]
        file, shape = self._open_image_file(Path(file_info["holo_path"]))
        recorded: np.ndarray = self._read_image(file, row, "intensity", shape)
        return recorded

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
        file_info: dict[str, Any]
        row: int
        file_info, row = self._locate(idx)
        h_np: np.ndarray = self.read_recorded(idx)

        # Ground truth, from the same row of the ground-truth file.
        truth_file: pq.ParquetFile
        shape: tuple[int, int]
        truth_file, shape = self._open_image_file(Path(file_info["gt_path"]))
        r_np: np.ndarray = self._read_image(truth_file, row, "real", shape)
        i_np: np.ndarray = self._read_image(truth_file, row, "imag", shape)

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
