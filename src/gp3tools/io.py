"""Gazepoint import and column-normalisation utilities."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._compat import r_aliases
from ._utils import attach_attrs, safe_path

_AUTO_COLUMN_RE = re.compile(r"^(?:\.\.\d+|Unnamed.*)$", re.IGNORECASE)


def _standardise_one_name(name: Any) -> str:
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return "EMPTY_TRAILING"
    value = str(name).strip()
    if value.startswith("TIME("):
        return "TIME"
    if value.startswith("TIMETICK("):
        return "TIMETICK"
    return value or "EMPTY_TRAILING"


def _column_is_empty(series: pd.Series) -> bool:
    non_missing = series.dropna()
    if non_missing.empty:
        return True
    if pd.api.types.is_string_dtype(series.dtype) or series.dtype == object:
        return bool(non_missing.astype(str).str.strip().eq("").all())
    return False


def _drop_empty_columns(data: pd.DataFrame) -> pd.DataFrame:
    keep: list[bool] = []
    for i, name in enumerate(data.columns):
        name_s = "" if name is None else str(name)
        auto = name_s in {"", "EMPTY_TRAILING"} or bool(_AUTO_COLUMN_RE.match(name_s))
        keep.append(not (auto and _column_is_empty(data.iloc[:, i])))
    return data.loc[:, keep].copy()


def standardise_gazepoint_names(x):
    """Standardise Gazepoint column names or a sequence of names."""
    if isinstance(x, pd.DataFrame):
        out = x.copy()
        out.columns = [_standardise_one_name(c) for c in out.columns]
        return _drop_empty_columns(out)
    if isinstance(x, str):
        return _standardise_one_name(x)
    return [_standardise_one_name(c) for c in x]


def classify_gazepoint_export(path: str | Path) -> str:
    """Classify a Gazepoint export by filename and header content."""
    source = safe_path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{source} does not exist.")
    name = source.name
    if re.search(r"Data_Summary_export", name, re.I):
        return "summary"
    if re.search(r"fix", name, re.I):
        return "fixations"
    if re.search(r"all_gaze|user\.csv", name, re.I):
        return "all_gaze"
    first = source.open("r", encoding="utf-8-sig", errors="replace").readline()
    if "Gazepoint Analysis" in first:
        return "summary"
    if any(x in first for x in ("FPOGX", "BPOGX", "LPOGX", "RPOGX")):
        return "gaze_table"
    return "unknown"


def read_gazepoint(
    path: str | Path,
    standardise_names: bool = True,
    drop_empty_cols: bool = True,
    **read_csv_kwargs,
) -> pd.DataFrame:
    """Read a Gazepoint all-gaze or fixation CSV export."""
    source = safe_path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{source} does not exist.")
    kind = classify_gazepoint_export(source)
    if kind == "summary":
        raise ValueError(
            "This appears to be a Gazepoint Analysis summary export. Use read_gazepoint_summary()."
        )
    data = pd.read_csv(source, **read_csv_kwargs)
    if standardise_names:
        data = standardise_gazepoint_names(data)
    if drop_empty_cols:
        data = _drop_empty_columns(data)
    return attach_attrs(data, gp3_file_type=kind, gp3_source_file=source.name)


def read_gazepoint_folder(
    folder: str | Path,
    pattern: str = r"\.csv$",
    source_col: str = "USER_FILE",
    recursive: bool = False,
    **read_kwargs,
) -> pd.DataFrame:
    """Read matching Gazepoint exports in a folder and row-bind them."""
    root = safe_path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"{root} does not exist.")
    rx = re.compile(pattern)
    candidates = root.rglob("*") if recursive else root.iterdir()
    files = sorted(
        (p for p in candidates if p.is_file() and rx.search(p.name)),
        key=lambda p: str(p).casefold(),
    )
    if not files:
        raise FileNotFoundError(f"No files matching pattern {pattern!r} were found in {root}.")
    frames = []
    for source in files:
        if classify_gazepoint_export(source) == "summary":
            continue
        frame = read_gazepoint(source, **read_kwargs)
        frame[source_col] = source.name
        frames.append(frame)
    if not frames:
        raise ValueError("Matching files were found, but none were row-level Gazepoint exports.")
    return pd.concat(frames, ignore_index=True, sort=False)


def read_gazepoint_summary(path: str | Path) -> dict[str, Any]:
    """Parse an official Gazepoint Analysis summary export conservatively.

    The exact vendor layout can vary by Gazepoint Analysis version, therefore
    the raw rows are always returned together with best-effort metadata and
    tabular sections.
    """
    source = safe_path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{source} does not exist.")
    lines = source.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    metadata: dict[str, str] = {}
    for line in lines[:30]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] and len(parts[0]) < 80:
            metadata.setdefault(parts[0], parts[1])
    try:
        raw = pd.read_csv(source, header=None, dtype=str, on_bad_lines="skip")
    except Exception:
        raw = pd.DataFrame({"raw_line": lines})
    tables: list[pd.DataFrame] = []
    # Discover likely header rows and create independent tables.
    for idx, line in enumerate(lines):
        if any(token in line.upper() for token in ("AOI", "USER", "FIXATION")) and "," in line:
            header = [x.strip() for x in line.split(",")]
            body = []
            for later in lines[idx + 1 :]:
                if not later.strip():
                    break
                values = [x.strip() for x in later.split(",")]
                if len(values) != len(header):
                    break
                body.append(values)
            if body:
                tables.append(pd.DataFrame(body, columns=header))
    return {"metadata": metadata, "tables": tables, "raw": raw, "source_file": source.name}


def read_gazepoint_face_export(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read an external facial-behaviour CSV/TSV export."""
    source = safe_path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{source} does not exist.")
    if source.suffix.lower() in {".tsv", ".txt"}:
        kwargs.setdefault("sep", "\t")
    out = pd.read_csv(source, **kwargs)
    out.attrs["gp3_source_file"] = source.name
    return out


def inspect_gazepoint_columns(data=None, *, x=None) -> pd.DataFrame:
    """Inspect Gazepoint columns with R 2.3.0 semantics plus legacy aliases."""
    import pandas as pd

    if data is not None and x is not None:
        raise TypeError(
            "inspect_gazepoint_columns received both 'data' and "
            "R-compatible alias 'x'; supply only one."
        )
    target = x if x is not None else data
    if isinstance(target, (str, bytes)):
        target = read_gazepoint(target)
    else:
        target = standardise_gazepoint_names(target)
    if not isinstance(target, pd.DataFrame):
        raise ValueError("x must be a data frame or path to a Gazepoint CSV export.")

    groups = {
        "identification": {"MEDIA_ID", "MEDIA_NAME", "CNT"},
        "time": {"TIME", "TIMETICK"},
        "fixation_gaze": {"FPOGX", "FPOGY", "FPOGS", "FPOGD", "FPOGID", "FPOGV"},
        "best_gaze": {"BPOGX", "BPOGY", "BPOGV"},
        "cursor_keyboard_user": {"CX", "CY", "CS", "KB", "KBS", "USER"},
        "left_eye_pupil": {"LPCX", "LPCY", "LPD", "LPS", "LPV", "LPMM", "LPMMV"},
        "right_eye_pupil": {"RPCX", "RPCY", "RPD", "RPS", "RPV", "RPMM", "RPMMV"},
        "blink": {"BKID", "BKDUR", "BKPMIN"},
        "biometrics": {
            "DIAL",
            "DIALV",
            "GSR",
            "GSR_US",
            "GSR_US_TONIC",
            "GSR_US_PHASIC",
            "GSRV",
            "HR",
            "HRV",
            "HRP",
            "IBI",
        },
        "ttl": {"TTL0", "TTL1", "TTL2", "TTL3", "TTL4", "TTL5", "TTL6", "TTLV"},
        "derived": {"PIXS", "PIXV", "AOI", "SACCADE_MAG", "SACCADE_DIR", "VID_FRAME"},
    }

    def semantic_group(column):
        for name, members in groups.items():
            if column in members:
                return name
        return "other"

    def r_class(series):
        dtype = series.dtype
        if pd.api.types.is_bool_dtype(dtype):
            return "logical"
        if pd.api.types.is_integer_dtype(dtype):
            return "integer"
        if pd.api.types.is_float_dtype(dtype):
            return "numeric"
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "POSIXct/POSIXt"
        if isinstance(dtype, pd.CategoricalDtype):
            return "factor"
        return "character"

    out = pd.DataFrame(
        {
            "column": [str(c) for c in target.columns],
            "semantic_group": [semantic_group(str(c)) for c in target.columns],
            "dtype": [r_class(target[c]) for c in target.columns],
            "n_missing": [int(target[c].isna().sum()) for c in target.columns],
            "pct_missing": [float(target[c].isna().mean() * 100) for c in target.columns],
        }
    )
    out["missing_prop"] = out["pct_missing"] / 100.0
    out["n_unique"] = [int(target[c].nunique(dropna=True)) for c in target.columns]
    return out


# BEGIN R V2.3.0 CALL-SURFACE ALIASES
inspect_gazepoint_columns = r_aliases(inspect_gazepoint_columns, x="data")
# END R V2.3.0 CALL-SURFACE ALIASES
