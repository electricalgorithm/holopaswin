"""Small datasets in the hologen file layout, shared by the tests."""

import pathlib

import numpy as np
import pandas as pd

SIZE: int = 64
COUNT: int = 4
GEOMETRY: str = (
    "[geometry]\nresolution = {size}\nsensor_pixel_size = 2.4\nmagnification = 10.0\n"
    "numerical_aperture = 0.45\nwavelength = 0.532\nz_distance = 20.0\nz_offset = {offset}\nz_error = 0.0\n"
)


def write_dataset(folder: pathlib.Path, offset: float = 0.0, size: int = SIZE, count: int = COUNT) -> None:
    """Write a small hologen-style dataset recorded at one distance offset.

    Holograms are noisy flat fields and objects are phase disks, enough for the
    code paths, not for meaningful reconstructions.

    Args:
        folder (pathlib.Path): Folder of the dataset.
        offset (float): Distance offset in um, written to the recipe and metadata.
        size (int): Image size in pixels.
        count (int): Number of holograms, an even number.
    """
    folder.mkdir(parents=True)
    rng: np.random.Generator = np.random.default_rng(int(offset * 10) + 100)
    pixels: int = size * size
    y: np.ndarray
    x: np.ndarray
    y, x = np.mgrid[:size, :size]
    disk: np.ndarray = (((x - size / 2) ** 2 + (y - size / 2) ** 2) < (size / 5) ** 2).astype(float)
    phase: np.ndarray = np.tile(disk.ravel(), count)
    pd.DataFrame({"intensity": 1000.0 + 50.0 * rng.standard_normal(count * pixels)}).to_parquet(
        folder / "hologram_0000.parquet"
    )
    pd.DataFrame({"real": np.cos(phase), "imag": np.sin(phase)}).to_parquet(folder / "ground_truth_0000.parquet")
    pd.DataFrame(
        {
            "object_family": ["red_blood_cells", "adherent_cells"] * (count // 2),
            "config_name": ["camera/red_blood_cells", "camera_laser_speckle/adherent_cells"] * (count // 2),
            "z_true": [20.0 + offset] * count,
            "z_nominal": [20.0] * count,
        }
    ).to_parquet(folder / "metadata_0000.parquet")
    (folder / "recipe.toml").write_text(GEOMETRY.format(size=size, offset=offset))
