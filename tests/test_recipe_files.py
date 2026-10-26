"""The paper's hologen recipes give every set its size and seeds and pair the splits with the test set."""

import pathlib
import tomllib
from typing import Any

import pytest

RECIPES: pathlib.Path = pathlib.Path(__file__).parent.parent / "recipes"
PAIRED: list[str] = [
    "test-phase",
    "test-absorption",
    "test-low-light",
    "test-strong-speckle",
]
EXPECTED: dict[str, tuple[int, int]] = {
    "train": (8000, 10000),
    "train-z-augmented": (8000, 10000),
    "test": (1000, 200000),
    "test-dense": (400, 300000),
    "sweep": (200, 900000),
}


def read(name: str) -> dict[str, Any]:
    """Read one recipe file of the recipes folder.

    Args:
        name: File name without the .toml extension.

    Returns:
        dict[str, Any]: The tables the file sets; everything else is hologen's default.
    """
    path: pathlib.Path = RECIPES / f"{name}.toml"
    with path.open("rb") as handle:
        recipe: dict[str, Any] = tomllib.load(handle)
    return recipe


def test_every_set_has_its_size_and_seed() -> None:
    """Every set has its own file with the planned number of holograms and base seed."""
    names: set[str] = {path.stem for path in RECIPES.glob("*.toml")}
    assert names == set(EXPECTED) | set(PAIRED)
    for name, (samples, seed) in EXPECTED.items():
        dataset: dict[str, Any] = read(name)["dataset"]
        assert (dataset["total_samples"], dataset["seed"]) == (samples, seed)


@pytest.mark.parametrize("name", PAIRED)
def test_paired_splits_give_every_seed_the_test_set_group(name: str) -> None:
    """A paired split repeats the test set's dataset table.

    The dataset table holds the total, the seed and the groups (configurations and
    objects), so equal tables give hologram i the same cells as test hologram i.

    Args:
        name: Recipe file name of the split.
    """
    assert read(name)["dataset"] == read("test")["dataset"]


def test_the_two_training_sets_differ_only_in_the_distance_error() -> None:
    """The distance-error training set is the training set plus a 6 um distance error."""
    varied: dict[str, Any] = read("train-z-augmented")
    assert varied.pop("geometry") == {"z_error": 6.0}
    assert varied == read("train")
