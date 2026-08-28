"""Internal utilities shared across gp3tools modules."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

COLUMN_CANDIDATES: dict[str, tuple[str, ...]] = {
    "subject": (
        "subject",
        "participant",
        "Participant",
        "PARTICIPANT",
        "participant_id",
        "SUBJECT",
        "USER",
        "USER_ID",
        "user",
    ),
    "trial": ("trial", "trial_id", "trial_global", "TRIAL", "TRIAL_ID"),
    "time": ("time", "TIME", "timestamp", "TIMESTAMP", "time_ms", "TIME_MS", "CNT"),
    "x": ("x", "X", "FPOGX", "BPOGX", "LPOGX", "RPOGX", "gaze_x"),
    "y": ("y", "Y", "FPOGY", "BPOGY", "LPOGY", "RPOGY", "gaze_y"),
    "pupil": (
        "pupil",
        "PUPIL",
        "BPOGS",
        "LPD",
        "RPD",
        "pupil_mean",
        "pupil_combined",
        "LPMM",
        "RPMM",
    ),
    "left_pupil": (
        "LPD",
        "left_pupil",
        "pupil_left",
        "left_pupil_diameter",
        "LPMM",
    ),
    "right_pupil": (
        "RPD",
        "right_pupil",
        "pupil_right",
        "right_pupil_diameter",
        "RPMM",
    ),
    "aoi": ("aoi_current", "AOI", "aoi", "AOI_NAME", "aoi_label"),
    "condition": ("condition", "CONDITION", "group", "GROUP"),
    "media": ("MEDIA_ID", "media_id", "stimulus", "stimulus_id", "MEDIA_NAME"),
    "validity": ("valid", "VALID", "BPOGV", "FPOGV", "LPOGV", "RPOGV", "trackloss"),
    "fixation_id": ("FPOGID", "fixation_id", "FIXATION_ID"),
    "fixation_duration": ("FPOGD", "fixation_duration", "duration", "duration_ms"),
}


def ensure_dataframe(data: Any, *, copy: bool = True) -> pd.DataFrame:
    """Return *data* as a pandas DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data.copy() if copy else data
    if isinstance(data, pd.Series):
        return data.to_frame().copy() if copy else data.to_frame()
    if isinstance(data, dict):
        return pd.DataFrame(data)
    if isinstance(data, (list, tuple)):
        return pd.DataFrame(data)
    raise TypeError("Expected a pandas DataFrame or DataFrame-compatible object.")


def infer_column(
    data: pd.DataFrame, role: str, explicit: str | None = None, *, required: bool = False
) -> str | None:
    """Infer a column from common Gazepoint/R-package naming conventions."""
    if explicit is not None:
        if explicit not in data.columns:
            if required:
                raise KeyError(f"Column {explicit!r} was not found.")
            return None
        return explicit
    for candidate in COLUMN_CANDIDATES.get(role, (role,)):
        if candidate in data.columns:
            return candidate
    if required:
        raise KeyError(f"Could not infer a {role!r} column from {list(data.columns)!r}.")
    return None


def infer_columns(data: pd.DataFrame, roles: Sequence[str]) -> dict[str, str | None]:
    return {role: infer_column(data, role) for role in roles}


def normalize_group_cols(data: pd.DataFrame, group_cols: str | Sequence[str] | None) -> list[str]:
    if group_cols is None:
        return []
    cols = [group_cols] if isinstance(group_cols, str) else list(group_cols)
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise KeyError(f"Grouping columns not found: {missing}")
    return cols


def finite_numeric(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    return x.where(np.isfinite(x))


def as_bool(series: pd.Series, *, invert_trackloss: bool = False) -> pd.Series:
    """Coerce common logical/validity representations to bool."""
    if pd.api.types.is_bool_dtype(series):
        out = series.fillna(False).astype(bool)
    elif pd.api.types.is_numeric_dtype(series):
        out = pd.to_numeric(series, errors="coerce").fillna(0).ne(0)
    else:
        values = series.astype("string").str.strip().str.lower()
        out = values.isin({"true", "t", "yes", "y", "1", "valid", "ok", "good"})
    return ~out if invert_trackloss else out


def robust_mad(values: Iterable[float]) -> float:
    x = np.asarray(list(values), dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def group_iter(data: pd.DataFrame, group_cols: Sequence[str]):
    if not group_cols:
        yield (), data
        return
    grouped = data.groupby(list(group_cols), dropna=False, sort=False)
    for key, frame in grouped:
        if not isinstance(key, tuple):
            key = (key,)
        yield key, frame


def attach_attrs(data: pd.DataFrame, **attrs: Any) -> pd.DataFrame:
    data.attrs.update(attrs)
    return data


def safe_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def result_table(**values: Any) -> pd.DataFrame:
    return pd.DataFrame([values])


def ordered_unique(values: Iterable[Any]) -> list[Any]:
    seen: set[Any] = set()
    out: list[Any] = []
    for value in values:
        key = (
            value
            if isinstance(value, (str, int, float, tuple, frozenset, type(None)))
            else repr(value)
        )
        if key not in seen:
            seen.add(key)
            out.append(value)
    return out


def collapse_consecutive(values: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    sentinel = object()
    previous: Any = sentinel
    for value in values:
        if previous is sentinel or value != previous:
            out.append(value)
        previous = value
    return out


def time_to_seconds(series: pd.Series) -> pd.Series:
    """Convert a numeric Gazepoint time column to seconds using conservative heuristics."""
    x = pd.to_numeric(series, errors="coerce")
    finite = x[np.isfinite(x)]
    if finite.empty:
        return x
    diffs = np.diff(np.sort(finite.unique()))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size:
        median_diff = float(np.median(diffs))
        # Typical TIME is seconds (~0.0167), TIME_MS is milliseconds (~16.7), TIMETICK is much larger.
        if median_diff > 2:
            return x / 1000.0
    return x


def require_optional(module_name: str, purpose: str):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise ImportError(
            f"{purpose} requires optional dependency {module_name!r}. Install the appropriate gp3tools extra."
        ) from exc
