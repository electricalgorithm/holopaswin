"""Small datasets in the hologen file layout, shared by the tests."""

import pathlib

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

SIZE: int = 64
COUNT: int = 4
GEOMETRY: str = (
    "[geometry]\nresolution = {size}\nsensor_pixel_size = 2.4\nmagnification = 10.0\n"
    "numerical_aperture = 0.45\nwavelength = 0.532\nz_distance = 20.0\nz_offset = {offset}\nz_error = 0.0\n"
)


def _image_column(images: np.ndarray, value_type: pa.DataType) -> pa.Array:
    """Build a column of 2-D lists, as hologen format 2 stores images.

    Args:
        images (np.ndarray): Images of shape (N, H, W).
        value_type (pa.DataType): Type of the stored values.

    Returns:
        pa.Array: One list of H lists of W values per image.
    """
    count, height, width = images.shape
    values: pa.Array = pa.array(images.reshape(-1), type=value_type)
    rows: pa.Array = pa.ListArray.from_arrays(
        pa.array(np.arange(0, count * height * width + 1, width), pa.int32()), values
    )
    return pa.ListArray.from_arrays(pa.array(np.arange(0, count * height + 1, height), pa.int32()), rows)


def write_images(folder: pathlib.Path, counts: np.ndarray, fields: np.ndarray, batch: int = 0) -> None:
    """Write holograms and ground-truth fields in hologen format 2.

    Args:
        folder (pathlib.Path): Dataset folder.
        counts (np.ndarray): Holograms as whole camera counts, shape (N, H, W).
        fields (np.ndarray): Complex ground-truth fields, shape (N, H, W).
        batch (int): Batch number of the files.
    """
    ids: pa.Array = pa.array(range(len(counts)), type=pa.int64())
    shape: str = f"{counts.shape[1]},{counts.shape[2]}"
    holograms: pa.Table = pa.table(
        {"sample_id": ids, "intensity": _image_column(counts.astype(np.uint16), pa.uint16())}
    )
    truths: pa.Table = pa.table(
        {
            "sample_id": ids,
            "real": _image_column(fields.real.astype(np.float32), pa.float32()),
            "imag": _image_column(fields.imag.astype(np.float32), pa.float32()),
        }
    )
    for name, table in (("hologram", holograms), ("ground_truth", truths)):
        metadata: dict[str, str] = {"hologen.format": "2", "hologen.kind": name, "hologen.shape": shape}
        pq.write_table(
            table.replace_schema_metadata(metadata), folder / f"{name}_{batch:04d}.parquet", row_group_size=1
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
    y: np.ndarray
    x: np.ndarray
    y, x = np.mgrid[:size, :size]
    disk: np.ndarray = (((x - size / 2) ** 2 + (y - size / 2) ** 2) < (size / 5) ** 2).astype(float)
    phase: np.ndarray = np.tile(disk.ravel(), count)
    counts: np.ndarray = np.round(1000.0 + 50.0 * rng.standard_normal((count, size, size)))
    write_images(folder, counts, np.exp(1j * phase.reshape(count, size, size)))
    pd.DataFrame(
        {
            "object_family": ["red_blood_cells", "adherent_cells"] * (count // 2),
            "config_name": ["camera/red_blood_cells", "camera_laser_speckle/adherent_cells"] * (count // 2),
            "z_true": [20.0 + offset] * count,
            "z_nominal": [20.0] * count,
        }
    ).to_parquet(folder / "metadata_0000.parquet")
    (folder / "recipe.toml").write_text(GEOMETRY.format(size=size, offset=offset))
