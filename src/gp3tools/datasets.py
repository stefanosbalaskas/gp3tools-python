"""Bundled synthetic datasets."""

from __future__ import annotations

from importlib.resources import files

import pandas as pd

_NAMES = {
    "master": "gazepoint_example_master.csv",
    "fixations": "gazepoint_example_fixations.csv",
    "aoi_geometry": "gazepoint_example_aoi_geometry.csv",
    "aoi_windows": "gazepoint_example_aoi_windows.csv",
    "pupil_windows": "gazepoint_example_pupil_windows.csv",
}


def load_example_data(name="master") -> pd.DataFrame:
    key = _NAMES.get(name, name if name.endswith(".csv") else None)
    if key is None:
        raise KeyError(f"Unknown example dataset {name!r}. Available: {sorted(_NAMES)}")
    return pd.read_csv(files("gp3tools").joinpath("data", key))


def load_example_master():
    return load_example_data("master")


def load_example_fixations():
    return load_example_data("fixations")
