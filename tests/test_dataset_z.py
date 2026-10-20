"""Tests for reading the recording distance of each hologram from a dataset."""

import pathlib

import numpy as np
import pandas as pd
import pytest
import torch
from holo_fixtures import write_images

from holopaswin.dataset import HoloDataset

SIZE: int = 8
COUNT: int = 3


def write_dataset(folder: pathlib.Path, z_true: list[float] | None, nominal: float = 20.0) -> None:
    """Write a tiny dataset in the hologen file format.

    Args:
        folder (pathlib.Path): Folder for the files.
        z_true (list[float] | None): Recording distance of each hologram in um,
            or None for a dataset made before the distance was recorded.
        nominal (float): Nominal distance in um, written to recipe.toml.
    """
    folder.mkdir(parents=True, exist_ok=True)
    write_images(folder, np.full((COUNT, SIZE, SIZE), 1000.0), np.ones((COUNT, SIZE, SIZE), dtype=complex))
    metadata: dict[str, list] = {"object_family": ["red_blood_cells"] * COUNT}
    if z_true is not None:
        metadata["z_true"] = z_true
        metadata["z_nominal"] = [nominal] * COUNT
    pd.DataFrame(metadata).to_parquet(folder / "metadata_0000.parquet")
    (folder / "recipe.toml").write_text(f"[geometry]\nresolution = {SIZE}  # pixels\nz_distance = {nominal}  # um\n")


def test_items_stay_pairs_by_default(tmp_path: pathlib.Path) -> None:
    """Without return_z, existing callers keep their (hologram, object) pairs."""
    write_dataset(tmp_path, [14.0, 20.0, 26.0])
    item: tuple = HoloDataset(str(tmp_path), target_size=SIZE, img_dim=SIZE)[0]
    assert len(item) == 2


def test_distance_comes_from_the_metadata_in_meters(tmp_path: pathlib.Path) -> None:
    """The z_true column in um becomes a float32 distance in meters."""
    write_dataset(tmp_path, [14.0, 20.5, 26.0])
    dataset: HoloDataset = HoloDataset(str(tmp_path), target_size=SIZE, img_dim=SIZE, return_z=True)
    distances: list[float] = [dataset[i][2].item() for i in range(COUNT)]
    assert distances == pytest.approx([14e-6, 20.5e-6, 26e-6])
    assert dataset[0][2].dtype == torch.float32


def test_older_datasets_fall_back_to_the_nominal_distance(tmp_path: pathlib.Path) -> None:
    """A dataset without z_true uses the z_distance of its recipe.toml."""
    write_dataset(tmp_path, None, nominal=18.0)
    dataset: HoloDataset = HoloDataset(str(tmp_path), target_size=SIZE, img_dim=SIZE, return_z=True)
    assert dataset[1][2].item() == pytest.approx(18e-6)


def test_missing_distance_information_is_a_clear_error(tmp_path: pathlib.Path) -> None:
    """Without z_true and recipe.toml the error names the missing file."""
    write_dataset(tmp_path, None)
    (tmp_path / "recipe.toml").unlink()
    with pytest.raises(FileNotFoundError, match=r"recipe\.toml"):
        HoloDataset(str(tmp_path), target_size=SIZE, img_dim=SIZE, return_z=True)


def test_folder_names_do_not_confuse_the_metadata_lookup(tmp_path: pathlib.Path) -> None:
    """Only the file name is mapped to its metadata file, not the folder path."""
    folder: pathlib.Path = tmp_path / "holograms"
    write_dataset(folder, [15.0, 16.0, 17.0])
    dataset: HoloDataset = HoloDataset(str(folder), target_size=SIZE, img_dim=SIZE, return_z=True)
    assert dataset[2][2].item() == pytest.approx(17e-6)


def test_short_metadata_is_an_error_not_a_shift(tmp_path: pathlib.Path) -> None:
    """A metadata file with fewer rows than holograms would move every later z."""
    write_dataset(tmp_path, [14.0, 20.0, 26.0])
    pd.DataFrame({"object_family": ["a", "b"], "z_true": [14.0, 20.0]}).to_parquet(tmp_path / "metadata_0000.parquet")
    with pytest.raises(ValueError, match="metadata_0000"):
        HoloDataset(str(tmp_path), target_size=SIZE, img_dim=SIZE, return_z=True)


def test_metadata_follows_the_indexed_files(tmp_path: pathlib.Path) -> None:
    """A batch skipped for missing ground truth is skipped in the metadata too."""
    write_dataset(tmp_path, [14.0, 15.0, 16.0])
    for kind in ("hologram", "metadata"):
        (tmp_path / f"{kind}_0000.parquet").rename(tmp_path / f"{kind}_0001.parquet")
    (tmp_path / "ground_truth_0000.parquet").rename(tmp_path / "ground_truth_0001.parquet")
    write_dataset(tmp_path / "first", [21.0, 22.0, 23.0])
    for kind in ("hologram", "metadata"):
        (tmp_path / "first" / f"{kind}_0000.parquet").rename(tmp_path / f"{kind}_0000.parquet")
    dataset: HoloDataset = HoloDataset(str(tmp_path), target_size=SIZE, img_dim=SIZE, return_z=True)
    assert len(dataset) == COUNT
    assert dataset[0][2].item() == pytest.approx(14e-6)
    labels: pd.DataFrame = dataset.read_metadata(["object_family", "z_true"])
    assert labels["z_true"].tolist() == [14.0, 15.0, 16.0]


def test_metadata_columns_missing_from_old_datasets_are_empty(tmp_path: pathlib.Path) -> None:
    """Old datasets have no z_true column; it comes back as missing values."""
    write_dataset(tmp_path, None)
    labels: pd.DataFrame = HoloDataset(str(tmp_path), target_size=SIZE, img_dim=SIZE).read_metadata(
        ["object_family", "z_true"]
    )
    assert labels["object_family"].tolist() == ["red_blood_cells"] * COUNT
    assert labels["z_true"].isna().all()


def test_the_old_per_pixel_layout_is_rejected(tmp_path: pathlib.Path) -> None:
    """Datasets written before hologen format 2 give a message to regenerate them."""
    write_dataset(tmp_path, [20.0] * COUNT)
    pd.DataFrame({"intensity": np.full(COUNT * SIZE * SIZE, 1000.0)}).to_parquet(tmp_path / "hologram_0000.parquet")
    with pytest.raises(ValueError, match="regenerate"):
        HoloDataset(str(tmp_path), target_size=SIZE, img_dim=SIZE)


def test_an_image_size_other_than_the_stored_one_is_rejected(tmp_path: pathlib.Path) -> None:
    """img_dim must match the shape stored in the files."""
    write_dataset(tmp_path, [20.0] * COUNT)
    with pytest.raises(ValueError, match="8 x 8"):
        HoloDataset(str(tmp_path), target_size=4, img_dim=16)
