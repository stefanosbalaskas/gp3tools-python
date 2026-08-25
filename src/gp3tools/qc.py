"""Master-table, sampling, tracking, missingness, and QC helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._compat import r_aliases
from ._utils import (
    as_bool,
    attach_attrs,
    ensure_dataframe,
    finite_numeric,
    infer_column,
    normalize_group_cols,
    result_table,
    time_to_seconds,
)


def as_gazepoint_master(data, copy: bool = True) -> pd.DataFrame:
    """Coerce sample-level data to a standard Gazepoint master table."""
    df = ensure_dataframe(data, copy=copy)
    mapping = {}
    roles = {
        "subject": "subject",
        "trial": "trial_global",
        "time": "time",
        "x": "x",
        "y": "y",
        "pupil": "pupil",
        "aoi": "aoi_current",
        "condition": "condition",
        "media": "MEDIA_ID",
    }
    for role, canonical in roles.items():
        col = infer_column(df, role)
        if col is not None and canonical not in df.columns:
            mapping[col] = canonical
    df = df.rename(columns=mapping)
    return attach_attrs(df, gp3_class="gazepoint_master")


def create_gazepoint_master(data, **kwargs) -> pd.DataFrame:
    """Create a standardized sample-level master table."""
    df = as_gazepoint_master(data)
    if "sample_index" not in df.columns:
        df["sample_index"] = np.arange(len(df), dtype=int)
    if "subject" not in df.columns:
        source = df.get("USER_FILE")
        if source is not None:
            df["subject"] = (
                source.astype(str).str.extract(r"(\d+)", expand=False).fillna(source.astype(str))
            )
    if "trial_global" not in df.columns:
        media = df.get("MEDIA_ID")
        subject = df.get("subject")
        if media is not None and subject is not None:
            df["trial_global"] = subject.astype(str) + "::" + media.astype(str)
        else:
            df["trial_global"] = 1
    return attach_attrs(df, gp3_class="gazepoint_master")


def validate_gazepoint_master(
    data, required: tuple[str, ...] = ("subject", "time")
) -> dict[str, Any]:
    """Validate a master table and return an explicit pass/fail gate."""
    df = ensure_dataframe(data, copy=False)
    checks = []
    for col in required:
        checks.append(
            {
                "check": f"required:{col}",
                "passed": col in df.columns,
                "detail": "present" if col in df.columns else "missing",
            }
        )
    checks.extend(
        [
            {"check": "nonempty", "passed": len(df) > 0, "detail": f"n={len(df)}"},
            {
                "check": "unique_columns",
                "passed": not df.columns.duplicated().any(),
                "detail": "no duplicate names"
                if not df.columns.duplicated().any()
                else "duplicates present",
            },
        ]
    )
    table = pd.DataFrame(checks)
    return {
        "valid": bool(table["passed"].all()),
        "summary": result_table(
            n_rows=len(df),
            n_columns=df.shape[1],
            n_checks=len(table),
            n_failed=int((~table["passed"]).sum()),
        ),
        "checks": table,
    }


def audit_gazepoint_master(data) -> dict[str, pd.DataFrame]:
    """Create structural and signal-availability summaries for a master table."""
    df = ensure_dataframe(data, copy=False)
    overview = result_table(
        n_rows=len(df),
        n_columns=df.shape[1],
        n_subjects=df["subject"].nunique() if "subject" in df else np.nan,
        n_trials=df["trial_global"].nunique() if "trial_global" in df else np.nan,
    )
    missing = pd.DataFrame(
        {
            "column": df.columns,
            "n_missing": [int(df[c].isna().sum()) for c in df],
            "missing_prop": [float(df[c].isna().mean()) for c in df],
        }
    )
    out: dict[str, pd.DataFrame] = {"overview": overview, "missingness": missing}
    for label, col in (
        ("subject_summary", "subject"),
        ("media_summary", "MEDIA_ID"),
        ("aoi_summary", "aoi_current"),
    ):
        if col in df:
            out[label] = df.groupby(col, dropna=False).size().rename("n_samples").reset_index()
    pupil = infer_column(df, "pupil")
    if pupil:
        x = finite_numeric(df[pupil])
        out["pupil_summary"] = result_table(
            n=int(x.notna().sum()),
            mean=float(x.mean()),
            sd=float(x.std()),
            minimum=float(x.min()),
            maximum=float(x.max()),
        )
    xcol, ycol = infer_column(df, "x"), infer_column(df, "y")
    if xcol and ycol:
        out["coordinate_summary"] = result_table(
            x_min=float(pd.to_numeric(df[xcol], errors="coerce").min()),
            x_max=float(pd.to_numeric(df[xcol], errors="coerce").max()),
            y_min=float(pd.to_numeric(df[ycol], errors="coerce").min()),
            y_max=float(pd.to_numeric(df[ycol], errors="coerce").max()),
        )
    return out


def check_sampling_rate(
    data,
    time_col: str | None = None,
    group_cols=None,
    expected_hz: float = 60.0,
    tolerance_hz: float = 5.0,
) -> pd.DataFrame:
    """Estimate effective sampling rate from timestamp differences."""
    df = ensure_dataframe(data, copy=False)
    time_col = infer_column(df, "time", time_col, required=True)
    groups = normalize_group_cols(df, group_cols)
    rows = []
    iterator = [((), df)] if not groups else df.groupby(groups, dropna=False, sort=False)
    for key, frame in iterator:
        if groups and not isinstance(key, tuple):
            key = (key,)
        t = time_to_seconds(frame[time_col]).dropna().sort_values().to_numpy(float)
        diffs = np.diff(t)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        hz = float(1.0 / np.median(diffs)) if diffs.size else np.nan
        row = {c: v for c, v in zip(groups, key, strict=False)} if groups else {}
        row.update(
            n_samples=len(frame),
            sampling_hz=hz,
            expected_hz=float(expected_hz),
            deviation_hz=hz - expected_hz if np.isfinite(hz) else np.nan,
            within_tolerance=bool(np.isfinite(hz) and abs(hz - expected_hz) <= tolerance_hz),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_tracking_quality(
    data,
    validity_col: str | None = None,
    group_cols=None,
    x_col: str | None = None,
    y_col: str | None = None,
) -> pd.DataFrame:
    """Summarise usable gaze samples and tracking loss."""
    df = ensure_dataframe(data, copy=False)
    groups = normalize_group_cols(df, group_cols)
    validity_col = infer_column(df, "validity", validity_col)
    x_col = infer_column(df, "x", x_col)
    y_col = infer_column(df, "y", y_col)
    usable = pd.Series(True, index=df.index)
    if validity_col:
        invert = validity_col.lower() == "trackloss"
        usable &= as_bool(df[validity_col], invert_trackloss=invert)
    if x_col:
        usable &= pd.to_numeric(df[x_col], errors="coerce").notna()
    if y_col:
        usable &= pd.to_numeric(df[y_col], errors="coerce").notna()
    work = df.assign(_gp3_usable=usable)
    if groups:
        out = (
            work.groupby(groups, dropna=False)
            .agg(n_samples=("_gp3_usable", "size"), n_usable=("_gp3_usable", "sum"))
            .reset_index()
        )
    else:
        out = pd.DataFrame({"n_samples": [len(work)], "n_usable": [int(usable.sum())]})
    out["usable_prop"] = out["n_usable"] / out["n_samples"].replace(0, np.nan)
    out["trackloss_prop"] = 1 - out["usable_prop"]
    return out


def flag_tracking_quality(data, min_usable_prop: float = 0.8, **kwargs) -> pd.DataFrame:
    out = summarise_tracking_quality(data, **kwargs)
    out["quality_flag"] = np.where(out["usable_prop"] >= min_usable_prop, "pass", "flag")
    return out


def clean_gazepoint_by_trackloss(
    data,
    group_cols=None,
    tracking_col=None,
    x_col=None,
    y_col=None,
    max_trackloss=0.25,
    action=None,
    treat_zero_zero_as_loss=True,
    rate_col=".gp3_trackloss_rate",
    exclude_col=".gp3_trackloss_exclude",
    *,
    validity_col=None,
    drop=None,
) -> pd.DataFrame:
    """Flag/filter groups by trackloss, with the legacy row-validity API retained."""
    if validity_col is not None or drop is not None:
        df = ensure_dataframe(data)
        validity_col = infer_column(df, "validity", validity_col)
        if validity_col is None:
            return df
        invert = validity_col.lower() == "trackloss"
        valid = as_bool(df[validity_col], invert_trackloss=invert)
        drop = True if drop is None else bool(drop)
        return df.loc[valid].copy() if drop else df.assign(gp3_track_valid=valid)

    df = ensure_dataframe(data)
    if not np.isfinite(max_trackloss) or not 0 <= float(max_trackloss) <= 1:
        raise ValueError("max_trackloss must be between 0 and 1")

    action = "flag" if action is None else action
    if action not in {"flag", "filter"}:
        raise ValueError("action must be 'flag' or 'filter'")

    groups = (
        []
        if group_cols is None
        else ([group_cols] if isinstance(group_cols, str) else list(group_cols))
    )
    missing_groups = [column for column in groups if column not in df.columns]
    if missing_groups:
        raise ValueError("data is missing required column(s): " + ", ".join(missing_groups))

    if tracking_col is not None:
        if tracking_col not in df.columns:
            raise ValueError(f"data is missing required column(s): {tracking_col}")
        raw = df[tracking_col]
        if pd.api.types.is_bool_dtype(raw):
            trackloss = raw.isna() | ~raw.fillna(False)
        elif pd.api.types.is_numeric_dtype(raw):
            numeric = pd.to_numeric(raw, errors="coerce")
            trackloss = numeric.isna() | numeric.le(0)
        else:
            text = raw.astype("string").str.strip().str.lower()
            trackloss = raw.isna() | text.isin(
                ["", "0", "false", "f", "invalid", "lost", "missing", "na", "nan"]
            )
    else:
        if x_col is None or y_col is None:
            raise ValueError("Supply either tracking_col or both x_col and y_col")
        missing = [column for column in (x_col, y_col) if column not in df.columns]
        if missing:
            raise ValueError("data is missing required column(s): " + ", ".join(missing))
        x = pd.to_numeric(df[x_col], errors="coerce")
        y = pd.to_numeric(df[y_col], errors="coerce")
        trackloss = ~(np.isfinite(x) & np.isfinite(y))
        if bool(treat_zero_zero_as_loss):
            trackloss = trackloss | (x.eq(0) & y.eq(0))

    work = df.copy()
    work["_gp3_trackloss_internal"] = pd.Series(trackloss, index=work.index).astype(bool)

    if groups:
        group_keys = work[groups].astype("string")
        valid_group = ~group_keys.isna().any(axis=1)
        key = group_keys.astype(str).agg(".".join, axis=1).where(valid_group)
    else:
        key = pd.Series(".gp3_all_rows", index=work.index, dtype="object")

    rates = (
        work.loc[key.notna()].groupby(key[key.notna()], sort=True)["_gp3_trackloss_internal"].mean()
    )
    counts = (
        work.loc[key.notna()].groupby(key[key.notna()], sort=True)["_gp3_trackloss_internal"].size()
    )
    lost = (
        work.loc[key.notna()].groupby(key[key.notna()], sort=True)["_gp3_trackloss_internal"].sum()
    )

    row_rate = key.map(rates)
    row_exclude = row_rate.gt(float(max_trackloss))

    out = df.copy()
    out[rate_col] = row_rate.to_numpy()
    out[exclude_col] = row_exclude.to_numpy()

    summary = pd.DataFrame(
        {
            "group_id": rates.index.astype(str),
            "n_rows": counts.reindex(rates.index).astype(int).to_numpy(),
            "n_trackloss_rows": lost.reindex(rates.index).astype(int).to_numpy(),
            "trackloss_rate": rates.to_numpy(float),
            "exclude": rates.gt(float(max_trackloss)).to_numpy(bool),
        }
    )
    out.attrs["gp3_trackloss_summary"] = summary

    if action == "filter":
        out = out.loc[~out[exclude_col].fillna(False)].reset_index(drop=True)
        out.attrs["gp3_trackloss_summary"] = summary

    return out


def summarise_gazepoint_missingness(
    data,
    group_cols=None,
    columns=None,
    *,
    cols=None,
    include_group_cols=False,
) -> pd.DataFrame:
    """Summarise missingness with R v2.3.0 fields plus legacy aliases."""
    df = ensure_dataframe(data, copy=False)
    groups = (
        []
        if group_cols is None
        else ([group_cols] if isinstance(group_cols, str) else list(group_cols))
    )
    missing_groups = [column for column in groups if column not in df.columns]
    if missing_groups:
        raise ValueError("data is missing required column(s): " + ", ".join(missing_groups))
    if cols is not None and columns is not None:
        raise TypeError("supply either cols or columns, not both")
    selected = cols if cols is not None else columns
    if selected is None:
        selected = list(df.columns)
        if not include_group_cols and groups:
            selected = [column for column in selected if column not in groups]
    elif isinstance(selected, str):
        selected = [selected]
    else:
        selected = list(selected)
    if not selected:
        raise ValueError("cols must identify at least one column")
    missing_cols = [column for column in selected if column not in df.columns]
    if missing_cols:
        raise ValueError("data is missing required column(s): " + ", ".join(missing_cols))

    if groups:
        group_id = df[groups].astype("string").fillna("<NA>").agg(".".join, axis=1)
    else:
        group_id = pd.Series("all", index=df.index)

    rows = []
    for gid in sorted(group_id.unique()):
        mask = group_id.eq(gid)
        block = df.loc[mask]
        for column in selected:
            missing = block[column].isna()
            row = {
                "group_id": gid,
                "variable": column,
                "n_rows": len(block),
                "n_missing": int(missing.sum()),
                "n_observed": int((~missing).sum()),
                "missing_rate": float(missing.mean()),
                "observed_rate": float((~missing).mean()),
                # Legacy aliases.
                "column": column,
                "n": len(block),
                "missing_prop": float(missing.mean()),
            }
            if groups:
                first = block.iloc[0]
                for group in groups:
                    row[group] = first[group]
            rows.append(row)
    result = pd.DataFrame(rows)
    result.attrs["gp3_missingness_settings"] = {
        "cols": selected,
        "group_cols": groups or None,
        "include_group_cols": bool(include_group_cols),
    }
    return result


summarize_gazepoint_missingness = summarise_gazepoint_missingness


def audit_gazepoint_gaze_signal_quality(data, **kwargs) -> dict[str, pd.DataFrame]:
    return {
        "tracking": summarise_tracking_quality(data, **kwargs),
        "missingness": summarise_gazepoint_missingness(data),
    }


def audit_gazepoint_screen_bounds(
    data,
    x_col=None,
    y_col=None,
    width: float = 1.0,
    height: float = 1.0,
    normalized: bool = True,
    group_cols=None,
    margin=0,
    treat_zero_zero_as_out_of_bounds=True,
):
    """Audit screen bounds with legacy summary or R v2.3.0 detailed output."""
    df = ensure_dataframe(data, copy=False)
    x_col = infer_column(df, "x", x_col, required=True)
    y_col = infer_column(df, "y", y_col, required=True)
    r_mode = group_cols is not None or margin != 0 or treat_zero_zero_as_out_of_bounds is not True

    if not r_mode:
        x = finite_numeric(df[x_col])
        y = finite_numeric(df[y_col])
        xmax, ymax = (1.0, 1.0) if normalized else (float(width), float(height))
        inside = x.between(0, xmax) & y.between(0, ymax)
        return result_table(
            n=len(df),
            n_finite=int((x.notna() & y.notna()).sum()),
            n_inside=int(inside.sum()),
            n_outside=int((x.notna() & y.notna() & ~inside).sum()),
            inside_prop=float(inside.mean()),
        )

    screen_width = float(width)
    screen_height = float(height)
    if not np.isfinite(screen_width) or screen_width <= 0:
        raise ValueError("screen_width must be positive")
    if not np.isfinite(screen_height) or screen_height <= 0:
        raise ValueError("screen_height must be positive")
    if not isinstance(margin, (int, float, np.integer, np.floating)) or margin < 0:
        raise ValueError("margin must be a single non-negative numeric value")

    groups = (
        []
        if group_cols is None
        else ([group_cols] if isinstance(group_cols, str) else list(group_cols))
    )
    missing = [column for column in [x_col, y_col, *groups] if column not in df.columns]
    if missing:
        raise ValueError("data is missing required column(s): " + ", ".join(missing))

    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(float)
    missing_coordinate = ~(np.isfinite(x) & np.isfinite(y))
    zero_zero = ~missing_coordinate & (x == 0) & (y == 0)
    outside_x = ~missing_coordinate & ((x < -margin) | (x > screen_width + margin))
    outside_y = ~missing_coordinate & ((y < -margin) | (y > screen_height + margin))
    outside_bounds = outside_x | outside_y
    invalid_coordinate = missing_coordinate | outside_bounds
    if treat_zero_zero_as_out_of_bounds:
        invalid_coordinate = invalid_coordinate | zero_zero

    row_flags = pd.DataFrame(
        {
            "row_id": np.arange(1, len(df) + 1),
            "x": x,
            "y": y,
            "missing_coordinate": missing_coordinate,
            "zero_zero": zero_zero,
            "outside_x": outside_x,
            "outside_y": outside_y,
            "outside_bounds": outside_bounds,
            "invalid_coordinate": invalid_coordinate,
        }
    )
    if groups:
        group_labels = df[groups].astype("string").fillna("<NA>").agg(".".join, axis=1)
    else:
        group_labels = pd.Series("all", index=df.index)
    row_flags[".gp3_group_id"] = group_labels.to_numpy()

    summary_rows = []
    for group_id, block in row_flags.groupby(".gp3_group_id", sort=True):
        summary_rows.append(
            {
                "group_id": group_id,
                "n_rows": len(block),
                "n_missing_coordinate": int(block["missing_coordinate"].sum()),
                "n_zero_zero": int(block["zero_zero"].sum()),
                "n_outside_bounds": int(block["outside_bounds"].sum()),
                "n_invalid_coordinate": int(block["invalid_coordinate"].sum()),
                "missing_coordinate_rate": float(block["missing_coordinate"].mean()),
                "zero_zero_rate": float(block["zero_zero"].mean()),
                "outside_bounds_rate": float(block["outside_bounds"].mean()),
                "invalid_coordinate_rate": float(block["invalid_coordinate"].mean()),
            }
        )
    group_summary = pd.DataFrame(summary_rows)
    overall_summary = pd.DataFrame(
        [
            {
                "n_rows": len(row_flags),
                "n_missing_coordinate": int(missing_coordinate.sum()),
                "n_zero_zero": int(zero_zero.sum()),
                "n_outside_bounds": int(outside_bounds.sum()),
                "n_invalid_coordinate": int(invalid_coordinate.sum()),
                "missing_coordinate_rate": float(missing_coordinate.mean()),
                "zero_zero_rate": float(zero_zero.mean()),
                "outside_bounds_rate": float(outside_bounds.mean()),
                "invalid_coordinate_rate": float(invalid_coordinate.mean()),
            }
        ]
    )
    return {
        "row_flags": row_flags,
        "group_summary": group_summary,
        "overall_summary": overall_summary,
        "settings": {
            "x_col": x_col,
            "y_col": y_col,
            "screen_width": screen_width,
            "screen_height": screen_height,
            "group_cols": groups or None,
            "margin": margin,
            "treat_zero_zero_as_out_of_bounds": bool(treat_zero_zero_as_out_of_bounds),
        },
    }


def harmonize_gazepoint_screen_coordinates(
    data,
    x_col=None,
    y_col=None,
    width: float | None = None,
    height: float | None = None,
    output_x: str = "x_norm",
    output_y: str = "y_norm",
    *,
    from_width: float | None = None,
    from_height: float | None = None,
    to_width: float | None = None,
    to_height: float | None = None,
    output_x_col: str | None = None,
    output_y_col: str | None = None,
    keep_original: bool = True,
) -> pd.DataFrame:
    """Harmonize screen coordinates.

    The original Python normalization interface is retained. Supplying any
    ``from_*``/``to_*`` argument activates the R v2.3.0 scaling interface.
    """
    df = ensure_dataframe(data)
    x_col, y_col = (
        infer_column(df, "x", x_col, required=True),
        infer_column(df, "y", y_col, required=True),
    )

    r_values = (
        from_width,
        from_height,
        to_width,
        to_height,
    )
    r_mode = any(value is not None for value in r_values)

    if r_mode:
        if any(value is None for value in r_values):
            raise ValueError(
                "from_width, from_height, to_width, and to_height "
                "must all be supplied for R-compatible harmonization"
            )

        values = {
            "from_width": from_width,
            "from_height": from_height,
            "to_width": to_width,
            "to_height": to_height,
        }

        for name, value in values.items():
            value = float(value)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")

        output_x_col = output_x_col or "gaze_x_harmonized"
        output_y_col = output_y_col or "gaze_y_harmonized"

        x = finite_numeric(df[x_col])
        y = finite_numeric(df[y_col])

        x_scale = float(to_width) / float(from_width)
        y_scale = float(to_height) / float(from_height)

        df[output_x_col] = x * x_scale
        df[output_y_col] = y * y_scale

        if not keep_original:
            remove = {
                x_col,
                y_col,
            } - {
                output_x_col,
                output_y_col,
            }

            if remove:
                df = df.drop(columns=list(remove))

        df.attrs["gp3_screen_harmonization"] = {
            "x_col": x_col,
            "y_col": y_col,
            "from_width": float(from_width),
            "from_height": float(from_height),
            "to_width": float(to_width),
            "to_height": float(to_height),
            "output_x_col": output_x_col,
            "output_y_col": output_y_col,
            "x_scale": x_scale,
            "y_scale": y_scale,
        }

        return df

    x = finite_numeric(df[x_col])
    y = finite_numeric(df[y_col])

    if width is None:
        width = 1.0 if x.max(skipna=True) <= 1.5 else float(x.max(skipna=True))

    if height is None:
        height = 1.0 if y.max(skipna=True) <= 1.5 else float(y.max(skipna=True))

    df[output_x] = x / width
    df[output_y] = y / height
    return df


def summarise_gazepoint_coordinate_coverage(
    data,
    x_col=None,
    y_col=None,
    group_cols=None,
    *,
    screen_width: float | None = None,
    screen_height: float | None = None,
    grid_n_x: int = 10,
    grid_n_y: int = 10,
    include_out_of_bounds: bool = False,
) -> pd.DataFrame:
    """Summarise coordinate coverage.

    Screen dimensions activate the R v2.3.0 coverage/grid calculation.
    Without them the original Python range summary is returned.
    """
    df = ensure_dataframe(
        data,
        copy=False,
    )

    x_col, y_col = (
        infer_column(
            df,
            "x",
            x_col,
            required=True,
        ),
        infer_column(
            df,
            "y",
            y_col,
            required=True,
        ),
    )

    if screen_width is None and screen_height is None:
        groups = normalize_group_cols(
            df,
            group_cols,
        )

        work = df.assign(
            _x=finite_numeric(df[x_col]),
            _y=finite_numeric(df[y_col]),
        )

        if groups:
            return (
                work.groupby(
                    groups,
                    dropna=False,
                )
                .agg(
                    n_samples=("_x", "size"),
                    n_xy=(
                        "_x",
                        lambda s: int(
                            (
                                s.notna()
                                & work.loc[
                                    s.index,
                                    "_y",
                                ].notna()
                            ).sum()
                        ),
                    ),
                    x_min=("_x", "min"),
                    x_max=("_x", "max"),
                    y_min=("_y", "min"),
                    y_max=("_y", "max"),
                )
                .reset_index()
            )

        return result_table(
            n_samples=len(work),
            n_xy=int((work._x.notna() & work._y.notna()).sum()),
            x_min=float(work._x.min()),
            x_max=float(work._x.max()),
            y_min=float(work._y.min()),
            y_max=float(work._y.max()),
        )

    if screen_width is None or screen_height is None:
        raise ValueError("screen_width and screen_height must be supplied together")

    screen_width = float(screen_width)
    screen_height = float(screen_height)

    if (
        not np.isfinite(screen_width)
        or screen_width <= 0
        or not np.isfinite(screen_height)
        or screen_height <= 0
    ):
        raise ValueError("screen dimensions must be finite and positive")

    if (
        not isinstance(grid_n_x, int)
        or grid_n_x <= 0
        or not isinstance(grid_n_y, int)
        or grid_n_y <= 0
    ):
        raise ValueError("grid_n_x and grid_n_y must be positive integers")

    groups = normalize_group_cols(
        df,
        group_cols,
    )

    x = finite_numeric(df[x_col]).to_numpy(float)
    y = finite_numeric(df[y_col]).to_numpy(float)

    finite = np.isfinite(x) & np.isfinite(y)

    inside = finite & (x >= 0) & (x <= screen_width) & (y >= 0) & (y <= screen_height)

    if groups:
        group_labels = df[groups].astype("string").fillna("<NA>").agg(".".join, axis=1)
    else:
        group_labels = pd.Series(
            "all",
            index=df.index,
            dtype="string",
        )

    rows = []

    for group_id in pd.unique(group_labels):
        idx = np.flatnonzero(group_labels.to_numpy() == group_id)

        range_mask = finite[idx] if include_out_of_bounds else inside[idx]

        range_idx = idx[range_mask]
        grid_idx = idx[inside[idx]]

        if len(grid_idx):
            gx = np.floor(x[grid_idx] / screen_width * grid_n_x).astype(int)

            gy = np.floor(y[grid_idx] / screen_height * grid_n_y).astype(int)

            gx = np.clip(
                gx,
                0,
                grid_n_x - 1,
            )

            gy = np.clip(
                gy,
                0,
                grid_n_y - 1,
            )

            occupied = len(set(zip(gx, gy, strict=False)))
        else:
            occupied = 0

        def safe_stat(values, function):
            if not len(values):
                return np.nan
            return float(function(values))

        rows.append(
            {
                "group_id": str(group_id),
                "n_rows": int(len(idx)),
                "n_finite_coordinates": int(finite[idx].sum()),
                "n_inside_screen": int(inside[idx].sum()),
                "finite_coordinate_rate": float(finite[idx].mean()),
                "inside_screen_rate": float(inside[idx].mean()),
                "x_min": safe_stat(
                    x[range_idx],
                    np.min,
                ),
                "x_max": safe_stat(
                    x[range_idx],
                    np.max,
                ),
                "y_min": safe_stat(
                    y[range_idx],
                    np.min,
                ),
                "y_max": safe_stat(
                    y[range_idx],
                    np.max,
                ),
                "x_mean": safe_stat(
                    x[range_idx],
                    np.mean,
                ),
                "y_mean": safe_stat(
                    y[range_idx],
                    np.mean,
                ),
                "occupied_grid_cells": int(occupied),
                "total_grid_cells": int(grid_n_x * grid_n_y),
                "occupied_grid_rate": float(occupied / (grid_n_x * grid_n_y)),
            }
        )

    return pd.DataFrame(rows)


summarize_gazepoint_coordinate_coverage = summarise_gazepoint_coordinate_coverage


def audit_gazepoint_design_balance(
    data,
    subject_col="subject",
    condition_col="condition",
    unit_cols=("media_id", "trial_global"),
    expected_conditions=None,
    min_units_per_condition=1,
    max_condition_ratio=2,
    require_all_conditions_per_subject=True,
    *,
    group_cols=None,
):
    """Audit experimental balance; group_cols retains the historical count table."""
    frame = ensure_dataframe(data, copy=False)
    if group_cols is not None:
        columns = [column for column in group_cols if column in frame.columns]
        if not columns:
            return result_table(n_rows=len(frame))
        return frame.groupby(columns, dropna=False).size().rename("n").reset_index()

    if frame.empty:
        raise ValueError("data must contain at least one row")
    work = frame.copy()
    if "MEDIA_ID" in work.columns and "media_id" not in work.columns:
        work["media_id"] = work["MEDIA_ID"]
    if "USER_FILE" in work.columns and "subject" not in work.columns:
        work["subject"] = work["USER_FILE"]
    if subject_col == "USER_FILE" and "subject" in work.columns:
        subject_col = "subject"
    if condition_col not in work.columns or subject_col not in work.columns:
        missing = [column for column in (subject_col, condition_col) if column not in work.columns]
        raise ValueError("data is missing required column(s): " + ", ".join(missing))

    unit_cols = (
        []
        if unit_cols is None
        else ([unit_cols] if isinstance(unit_cols, str) else list(unit_cols))
    )
    unit_cols = [
        "media_id" if column == "MEDIA_ID" else "subject" if column == "USER_FILE" else column
        for column in unit_cols
    ]
    unit_cols = [column for column in unit_cols if column in work.columns]
    if min_units_per_condition <= 0 or not np.isfinite(min_units_per_condition):
        raise ValueError("min_units_per_condition must be positive")
    if max_condition_ratio <= 0 or not np.isfinite(max_condition_ratio):
        raise ValueError("max_condition_ratio must be positive")
    if not isinstance(require_all_conditions_per_subject, (bool, np.bool_)):
        raise ValueError("require_all_conditions_per_subject must be TRUE or FALSE")

    observed = sorted(
        value
        for value in work[condition_col].astype("string").dropna().unique().tolist()
        if str(value)
    )
    if not observed:
        raise ValueError("condition_col must contain at least one non-missing condition")
    conditions = observed if expected_conditions is None else list(expected_conditions)
    if not conditions or any(not isinstance(value, str) or not value for value in conditions):
        raise ValueError("expected_conditions must be a non-empty character vector")

    keep = list(dict.fromkeys([subject_col, condition_col, *unit_cols]))
    units = work[keep].copy()
    units[subject_col] = units[subject_col].astype("string")
    units[condition_col] = units[condition_col].astype("string")
    units = units.loc[
        units[subject_col].notna()
        & units[subject_col].ne("")
        & units[condition_col].notna()
        & units[condition_col].ne("")
    ].drop_duplicates()
    if units.empty:
        raise ValueError("subject_col and condition_col must define at least one usable row")

    subjects = sorted(units[subject_col].astype(str).unique())
    grid = pd.MultiIndex.from_product(
        [subjects, conditions], names=[subject_col, condition_col]
    ).to_frame(index=False)
    counts = (
        units.groupby([subject_col, condition_col], sort=True)
        .size()
        .rename("n_units")
        .reset_index()
    )
    cells = grid.merge(counts, on=[subject_col, condition_col], how="left", sort=False)
    cells["n_units"] = cells["n_units"].fillna(0).astype(int)
    cells["design_cell_status"] = np.select(
        [
            cells["n_units"].eq(0),
            cells["n_units"].lt(min_units_per_condition),
        ],
        ["missing_condition", "too_few_units"],
        default="ok",
    )

    subject_rows = []
    for subject, block in cells.groupby(subject_col, sort=True):
        counts_array = block["n_units"].to_numpy(int)
        nonzero = counts_array[counts_array > 0]
        n_missing = int(np.sum(counts_array == 0))
        n_low = int(block["design_cell_status"].eq("too_few_units").sum())
        ratio = float(nonzero.max() / nonzero.min()) if len(nonzero) > 1 else np.nan
        if require_all_conditions_per_subject and n_missing > 0:
            status = "missing_condition"
        elif n_low > 0:
            status = "too_few_units"
        elif np.isfinite(ratio) and ratio > max_condition_ratio:
            status = "condition_count_imbalance"
        else:
            status = "ok"
        subject_rows.append(
            {
                subject_col: str(subject),
                "n_conditions_expected": len(conditions),
                "n_conditions_observed": int(np.sum(counts_array > 0)),
                "min_units_per_condition_observed": int(nonzero.min()) if len(nonzero) else np.nan,
                "max_units_per_condition_observed": int(nonzero.max()) if len(nonzero) else np.nan,
                "condition_count_ratio": ratio,
                "n_missing_conditions": n_missing,
                "n_low_count_conditions": n_low,
                "design_balance_status": status,
            }
        )
    subject_summary = pd.DataFrame(subject_rows)

    condition_rows = []
    for condition, block in cells.groupby(condition_col, sort=True):
        nonzero = block.loc[block["n_units"].gt(0), "n_units"]
        condition_rows.append(
            {
                condition_col: str(condition),
                "n_subject_cells": len(block),
                "n_subjects_with_condition": int(block["n_units"].gt(0).sum()),
                "n_subjects_missing_condition": int(block["n_units"].eq(0).sum()),
                "total_units": int(block["n_units"].sum()),
                "min_units_per_subject": int(nonzero.min()) if len(nonzero) else np.nan,
                "max_units_per_subject": int(nonzero.max()) if len(nonzero) else np.nan,
                "mean_units_per_subject": float(block["n_units"].mean()),
                "condition_summary_status": (
                    "ok" if block["n_units"].eq(0).sum() == 0 else "missing_for_some_subjects"
                ),
            }
        )
    condition_summary = pd.DataFrame(condition_rows)
    imbalance_summary = (
        subject_summary["design_balance_status"]
        .value_counts(sort=False)
        .rename_axis("design_balance_status")
        .rename("n_subjects")
        .reset_index()
        .sort_values("design_balance_status", kind="stable")
        .reset_index(drop=True)
    )
    flagged_cells = cells.loc[~cells["design_cell_status"].eq("ok")].copy()
    n_flagged_subjects = int(subject_summary["design_balance_status"].ne("ok").sum())
    overview = pd.DataFrame(
        [
            {
                "n_rows": len(work),
                "n_units": len(units),
                "n_subjects": units[subject_col].nunique(),
                "n_conditions": len(conditions),
                "n_flagged_subjects": n_flagged_subjects,
                "n_flagged_cells": len(flagged_cells),
                "design_balance_status": (
                    "ok" if n_flagged_subjects == 0 and len(flagged_cells) == 0 else "review"
                ),
            }
        ]
    )
    settings = pd.DataFrame(
        {
            "setting": [
                "subject_col",
                "condition_col",
                "unit_cols",
                "expected_conditions",
                "min_units_per_condition",
                "max_condition_ratio",
                "require_all_conditions_per_subject",
            ],
            "value": [
                subject_col,
                condition_col,
                ", ".join(unit_cols),
                pd.NA if expected_conditions is None else ", ".join(conditions),
                str(min_units_per_condition),
                str(max_condition_ratio),
                "TRUE" if require_all_conditions_per_subject else "FALSE",
            ],
        }
    )
    return {
        "overview": overview,
        "subject_summary": subject_summary,
        "condition_summary": condition_summary,
        "cell_summary": cells,
        "imbalance_summary": imbalance_summary,
        "flagged_cells": flagged_cells,
        "settings": settings,
        "_gp3_class": "gp3_design_balance_audit",
    }


def audit_gazepoint_condition_quality_imbalance(data, condition_col=None, **kwargs) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    condition_col = infer_column(df, "condition", condition_col, required=True)
    return summarise_tracking_quality(df, group_cols=[condition_col], **kwargs)


def audit_gazepoint_post_exclusion_balance(
    data, excluded_col: str = "excluded", group_cols=("condition",)
) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    cols = [c for c in group_cols if c in df]
    if excluded_col in df:
        retained = df.loc[~as_bool(df[excluded_col])]
    else:
        retained = df
    return (
        retained.groupby(cols, dropna=False).size().rename("n_retained").reset_index()
        if cols
        else result_table(n_retained=len(retained))
    )


def audit_gazepoint_exclusion_flow(data, stages: list[str] | None = None) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    if stages is None:
        stages = [c for c in df.columns if c.lower().startswith(("exclude", "flag_", "excluded"))]
    rows = [{"stage": "input", "n": len(df)}]
    current = pd.Series(True, index=df.index)
    for stage in stages:
        current &= ~as_bool(df[stage])
        rows.append({"stage": stage, "n": int(current.sum())})
    return pd.DataFrame(rows)


def check_gazepoint_file_pairs(
    folder,
    all_gaze_pattern=r"_all_gaze\.csv$",
    fixation_pattern=r"_fixations\.csv$",
    recursive=False,
) -> pd.DataFrame:
    """Check paired all-gaze/fixation exports with R v2.3.0 diagnostics."""
    root = Path(folder)
    if not root.is_dir():
        raise ValueError(f"folder does not exist: {folder}")
    import re as _re

    files = list(root.rglob("*") if recursive else root.glob("*"))
    files = [path for path in files if path.is_file()]
    all_re = _re.compile(all_gaze_pattern)
    fix_re = _re.compile(fixation_pattern)
    all_files = [path for path in files if all_re.search(path.name)]
    fix_files = [path for path in files if fix_re.search(path.name)]
    if not all_files and not fix_files:
        raise ValueError(
            f"No files matching {all_gaze_pattern!r} or {fixation_pattern!r} were found in {folder}"
        )

    def ids_for(paths, pattern):
        groups = {}
        for path in paths:
            participant = pattern.sub("", path.name)
            groups.setdefault(participant, []).append(path.name)
        return groups

    all_groups = ids_for(all_files, all_re)
    fix_groups = ids_for(fix_files, fix_re)
    participants = sorted(set(all_groups) | set(fix_groups))
    rows = []
    for participant in participants:
        all_names = sorted(set(all_groups.get(participant, [])))
        fix_names = sorted(set(fix_groups.get(participant, [])))
        n_all = len(all_groups.get(participant, []))
        n_fix = len(fix_groups.get(participant, []))
        duplicate_all = n_all > 1
        duplicate_fix = n_fix > 1
        if n_all == 0:
            status = "missing_all_gaze"
        elif n_fix == 0:
            status = "missing_fixation"
        elif duplicate_all or duplicate_fix:
            status = "duplicate_files"
        else:
            status = "complete"
        user_match = _re.search(r"(\d+)", participant)
        user = user_match.group(1) if user_match else participant
        rows.append(
            {
                "participant": participant,
                "all_gaze_file": "; ".join(all_names),
                "fixation_file": "; ".join(fix_names),
                "n_all_gaze": n_all,
                "n_fixation": n_fix,
                "has_all_gaze": n_all > 0,
                "has_fixation": n_fix > 0,
                "duplicate_all_gaze": duplicate_all,
                "duplicate_fixation": duplicate_fix,
                "status": status,
                # Legacy additive fields.
                "user": user,
                "paired": status == "complete",
            }
        )
    return pd.DataFrame(rows)


def segment_gazepoint_task_phases(
    data,
    time_col=None,
    boundaries=None,
    labels=None,
    output_col: str = "phase",
    *,
    phase_windows=None,
    phase_col: str | None = None,
    window_phase_col: str = "phase",
    window_start_col: str = "start",
    window_end_col: str = "end",
    outside_label="outside",
    include_lower: bool = True,
    include_upper: bool = False,
    keep_window_metadata: bool = False,
) -> pd.DataFrame:
    """Segment samples into task phases.

    ``phase_windows`` activates the R v2.3.0 window-based interface.
    Otherwise the original Python boundary-based behaviour is retained.
    """
    df = ensure_dataframe(data)
    time_col = infer_column(df, "time", time_col, required=True)

    if phase_windows is not None:
        windows = ensure_dataframe(
            phase_windows,
            copy=False,
        ).copy()

        required = {
            window_phase_col,
            window_start_col,
            window_end_col,
        }

        missing = required - set(windows.columns)
        if missing:
            raise ValueError(
                "phase_windows is missing required columns: " + ", ".join(sorted(missing))
            )

        phase_col = phase_col or "task_phase"

        windows = windows[
            [
                window_phase_col,
                window_start_col,
                window_end_col,
            ]
        ].copy()

        windows.columns = [
            "phase",
            "start",
            "end",
        ]

        windows["start"] = pd.to_numeric(
            windows["start"],
            errors="coerce",
        )

        windows["end"] = pd.to_numeric(
            windows["end"],
            errors="coerce",
        )

        if windows["start"].isna().any() or windows["end"].isna().any():
            raise ValueError("phase-window start/end values must be numeric")

        if (windows["end"] < windows["start"]).any():
            raise ValueError("phase-window end values must not precede start values")

        t = finite_numeric(df[time_col]).to_numpy(float)

        assigned = np.full(
            len(df),
            None,
            dtype=object,
        )

        assigned_start = np.full(
            len(df),
            np.nan,
            dtype=float,
        )

        assigned_end = np.full(
            len(df),
            np.nan,
            dtype=float,
        )

        already = np.zeros(
            len(df),
            dtype=bool,
        )

        for row in windows.itertuples(index=False):
            lower = t >= row.start if include_lower else t > row.start

            upper = t <= row.end if include_upper else t < row.end

            mask = np.isfinite(t) & lower & upper & ~already

            assigned[mask] = row.phase
            assigned_start[mask] = row.start
            assigned_end[mask] = row.end
            already[mask] = True

        if outside_label is not None:
            assigned[~already] = outside_label

        df[phase_col] = pd.Series(
            assigned,
            index=df.index,
            dtype="object",
        )

        df[".gp3_phase_assigned"] = already

        if keep_window_metadata:
            df[".gp3_phase_window_start"] = assigned_start
            df[".gp3_phase_window_end"] = assigned_end

        df.attrs["gp3_phase_windows"] = windows
        df.attrs["gp3_phase_segmentation"] = {
            "time_col": time_col,
            "phase_col": phase_col,
            "outside_label": outside_label,
            "include_lower": include_lower,
            "include_upper": include_upper,
        }

        return df

    t = finite_numeric(df[time_col])

    if boundaries is None:
        q = t.quantile([0, 1 / 3, 2 / 3, 1]).to_numpy()
        boundaries = np.unique(q)

    boundaries = np.asarray(
        boundaries,
        dtype=float,
    )

    if labels is None:
        labels = [f"phase_{i + 1}" for i in range(len(boundaries) - 1)]

    df[output_col] = pd.cut(
        t,
        boundaries,
        labels=labels,
        include_lowest=True,
    )

    return df


def summarise_gazepoint_phase_coverage(
    data,
    phase_col="phase",
    group_cols=None,
    time_col=None,
    value_cols=None,
) -> pd.DataFrame:
    """Summarise task-phase coverage with legacy counts or R v2.3.0 diagnostics."""
    df = ensure_dataframe(data, copy=False)
    r_mode = time_col is not None or value_cols is not None or phase_col == "task_phase"
    if r_mode and phase_col == "phase" and "phase" not in df.columns and "task_phase" in df.columns:
        phase_col = "task_phase"

    if not r_mode:
        groups = normalize_group_cols(df, group_cols) + [phase_col]
        return df.groupby(groups, dropna=False).size().rename("n_samples").reset_index()

    groups = (
        []
        if group_cols is None
        else ([group_cols] if isinstance(group_cols, str) else list(group_cols))
    )
    values = (
        []
        if value_cols is None
        else ([value_cols] if isinstance(value_cols, str) else list(value_cols))
    )
    required = [phase_col, *groups, *values] + ([time_col] if time_col is not None else [])
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("data is missing required column(s): " + ", ".join(missing))

    if groups:
        group_id = df[groups].astype("string").fillna("<NA>").agg(".".join, axis=1)
    else:
        group_id = pd.Series("all", index=df.index)
    phase_value = df[phase_col].astype("string")
    rows = []
    keys = pd.DataFrame({"group_id": group_id, "phase": phase_value})
    for (gid, phase), indices in keys.groupby(
        ["group_id", "phase"], dropna=False, sort=True
    ).groups.items():
        block = df.loc[list(indices)]
        row = {"group_id": gid, "phase": str(phase), "n_rows": len(block)}
        if time_col is None:
            row.update(
                {
                    "n_finite_time": np.nan,
                    "min_time": np.nan,
                    "max_time": np.nan,
                    "time_span": np.nan,
                }
            )
        else:
            time_values = pd.to_numeric(block[time_col], errors="coerce")
            finite = time_values[np.isfinite(time_values)]
            if len(finite):
                minimum = float(finite.min())
                maximum = float(finite.max())
                row.update(
                    {
                        "n_finite_time": int(len(finite)),
                        "min_time": minimum,
                        "max_time": maximum,
                        "time_span": maximum - minimum,
                    }
                )
            else:
                row.update(
                    {
                        "n_finite_time": 0,
                        "min_time": np.nan,
                        "max_time": np.nan,
                        "time_span": np.nan,
                    }
                )
        if values:
            complete = block[values].notna().all(axis=1)
            row.update(
                {
                    "n_complete_value_rows": int(complete.sum()),
                    "complete_value_rate": float(complete.mean()),
                    "n_any_value_missing": int((~complete).sum()),
                    "any_value_missing_rate": float((~complete).mean()),
                }
            )
        else:
            row.update(
                {
                    "n_complete_value_rows": np.nan,
                    "complete_value_rate": np.nan,
                    "n_any_value_missing": np.nan,
                    "any_value_missing_rate": np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


summarize_gazepoint_phase_coverage = summarise_gazepoint_phase_coverage


def _gp3_qc_has_overview(obj) -> bool:
    return (
        isinstance(obj, dict)
        and "overview" in obj
        and isinstance(
            obj["overview"],
            pd.DataFrame,
        )
    )


def _gp3_qc_normalise_objects(objects):
    if objects is None:
        raise ValueError("objects must contain at least one object")

    if isinstance(objects, pd.DataFrame):
        return [("", objects)]

    if _gp3_qc_has_overview(objects):
        return [("", objects)]

    if isinstance(objects, dict):
        if not objects:
            raise ValueError("objects must contain at least one object")

        return [(str(name), obj) for name, obj in objects.items()]

    if isinstance(objects, (list, tuple)):
        if not objects:
            raise ValueError("objects must contain at least one object")

        return [("", obj) for obj in objects]

    return [("", objects)]


def _gp3_qc_extract_overview(obj):
    if isinstance(obj, pd.DataFrame):
        return obj

    if _gp3_qc_has_overview(obj):
        return obj["overview"]

    return None


def _gp3_qc_status_columns(overview):
    status_pattern = re.compile(
        r"status|decision|ready|valid|passed|complete|"
        r"review|flag|warn|fail|error",
        re.IGNORECASE,
    )

    message_pattern = re.compile(
        r"message|reason|recommendation|caution|note|evidence",
        re.IGNORECASE,
    )

    candidates = [str(col) for col in overview.columns if status_pattern.search(str(col))]

    return [col for col in candidates if not message_pattern.search(col)]


def _gp3_qc_message_columns(overview):
    pattern = re.compile(
        r"message|reason|recommendation|caution|note|evidence",
        re.IGNORECASE,
    )

    return [str(col) for col in overview.columns if pattern.search(str(col))]


def _gp3_qc_worse_status(current, candidate):
    severity = {
        "pass": 0,
        "info": 1,
        "unknown": 1,
        "warn": 2,
        "fail": 3,
    }

    if severity[candidate] > severity[current]:
        return candidate

    return current


def _gp3_qc_status_from_overview(
    overview,
    status_cols,
):
    if not status_cols or not len(overview):
        return "unknown"

    worst = "pass"

    for col in status_cols:
        values = overview[col]
        col_lower = col.lower()

        non_missing = values.dropna()

        logical_column = pd.api.types.is_bool_dtype(values.dtype) or (
            len(non_missing)
            and non_missing.map(
                lambda value: isinstance(
                    value,
                    (bool, np.bool_),
                )
            ).all()
        )

        if logical_column:
            bool_values = non_missing.astype(bool)

            if bool_values.eq(True).any() and re.search(
                r"review|flag|warn|fail|error|exclude|problem",
                col_lower,
            ):
                worst = _gp3_qc_worse_status(
                    worst,
                    "warn",
                )

            if bool_values.eq(False).any() and re.search(
                r"ready|valid|passed|complete",
                col_lower,
            ):
                worst = _gp3_qc_worse_status(
                    worst,
                    "fail",
                )

            continue

        char_values = values.astype("string").dropna().str.strip().str.lower()

        char_values = char_values[char_values.ne("")]

        if not len(char_values):
            continue

        if char_values.str.contains(
            r"fail|failed|error|invalid|not_ready|not ready|blocked",
            regex=True,
        ).any():
            candidate = "fail"

        elif char_values.str.contains(
            r"warn|warning|review|caution|partial|"
            r"incomplete|singular|conditional",
            regex=True,
        ).any():
            candidate = "warn"

        elif char_values.str.contains(
            r"info|unknown|not_run|not run|missing",
            regex=True,
        ).any():
            candidate = "info"

        elif char_values.str.contains(
            r"pass|passed|ok|ready|valid|complete|completed|"
            r"clean|true|yes",
            regex=True,
        ).any():
            candidate = "pass"

        else:
            candidate = "info"

        worst = _gp3_qc_worse_status(
            worst,
            candidate,
        )

    return worst


def _gp3_qc_message_from_overview(
    overview,
    message_cols,
    qc_status,
):
    fallback = f"QC status interpreted as '{qc_status}'."

    if not message_cols or not len(overview):
        return fallback

    values = []

    for col in message_cols:
        for value in overview[col]:
            if pd.isna(value):
                continue

            text = str(value)

            if text:
                values.append(text)

    if not values:
        return fallback

    unique = list(dict.fromkeys(values))

    return " | ".join(unique[:3])


def _gp3_qc_collapse(values):
    if not values:
        return np.nan

    return ", ".join(dict.fromkeys(str(value) for value in values))


def _gp3_qc_prepare_overview_rows(
    overview,
    object_name,
    index,
):
    out = overview.copy()

    out.insert(
        0,
        ".gp3_qc_row",
        np.arange(
            1,
            len(out) + 1,
            dtype=int,
        ),
    )

    out.insert(
        0,
        ".gp3_qc_object_index",
        int(index),
    )

    out.insert(
        0,
        ".gp3_qc_object_name",
        object_name,
    )

    return out


def _gp3_qc_bind_overview_rows(rows):
    frames = [
        frame
        for frame in rows
        if isinstance(
            frame,
            pd.DataFrame,
        )
        and len(frame)
    ]

    if not frames:
        return pd.DataFrame()

    columns = []

    for frame in frames:
        for col in frame.columns:
            if col not in columns:
                columns.append(col)

    normalized = []

    for frame in frames:
        current = frame.copy()

        for col in columns:
            if col not in current.columns:
                current[col] = np.nan

        normalized.append(current[columns])

    return pd.concat(
        normalized,
        ignore_index=True,
    )


def _gp3_qc_object_class(obj):
    if isinstance(obj, pd.DataFrame):
        return "data.frame"

    if isinstance(obj, dict):
        return "list"

    return type(obj).__name__


def _gp3_qc_collect_one(
    obj,
    object_name,
    index,
):
    overview = _gp3_qc_extract_overview(obj)

    object_class = _gp3_qc_object_class(obj)

    if overview is None:
        object_summary = pd.DataFrame(
            {
                "object_name": [object_name],
                "object_index": [int(index)],
                "object_class": [object_class],
                "overview_available": [False],
                "n_overview_rows": [0],
                "status_columns": [np.nan],
                "message_columns": [np.nan],
                "qc_status": ["unknown"],
                "qc_message": ["Object had no interpretable overview data frame."],
            }
        )

        return {
            "object_summary": object_summary,
            "overview_rows": pd.DataFrame(),
        }

    status_cols = _gp3_qc_status_columns(overview)

    message_cols = _gp3_qc_message_columns(overview)

    qc_status = _gp3_qc_status_from_overview(
        overview,
        status_cols,
    )

    qc_message = _gp3_qc_message_from_overview(
        overview,
        message_cols,
        qc_status,
    )

    object_summary = pd.DataFrame(
        {
            "object_name": [object_name],
            "object_index": [int(index)],
            "object_class": [object_class],
            "overview_available": [True],
            "n_overview_rows": [int(len(overview))],
            "status_columns": [_gp3_qc_collapse(status_cols)],
            "message_columns": [_gp3_qc_collapse(message_cols)],
            "qc_status": [qc_status],
            "qc_message": [qc_message],
        }
    )

    return {
        "object_summary": object_summary,
        "overview_rows": _gp3_qc_prepare_overview_rows(
            overview,
            object_name,
            index,
        ),
    }


def _gp3_qc_status_counts(status):
    values = pd.Series(status).astype("string")

    levels = [
        "pass",
        "warn",
        "fail",
        "info",
        "unknown",
    ]

    return pd.DataFrame(
        {
            "qc_status": levels,
            "n_objects": [int(values.eq(level).sum()) for level in levels],
        }
    )


def _gp3_qc_overall_status(status):
    values = set(pd.Series(status).dropna().astype(str))

    if "fail" in values:
        return "fail"

    if "warn" in values:
        return "warn"

    if "info" in values or "unknown" in values:
        return "info"

    return "pass"


def collect_gazepoint_qc_summaries(
    data=None,
    *,
    objects=None,
    object_names=None,
    name="gazepoint_qc_summary_bundle",
    include_overview_rows=True,
) -> dict[str, Any]:
    """Collect Gazepoint QC summaries.

    Explicit ``objects=`` activates the R gp3tools v2.3.0 QC-summary
    bundle interface. Passing a raw DataFrame positionally retains the
    historical Python convenience workflow.
    """
    if objects is None:
        if data is None:
            raise TypeError("data or objects must be supplied")

        df = ensure_dataframe(
            data,
            copy=False,
        )

        return {
            "master": audit_gazepoint_master(df),
            "tracking": summarise_tracking_quality(df),
            "missingness": summarise_gazepoint_missingness(df),
            "screen": audit_gazepoint_screen_bounds(df)
            if (infer_column(df, "x") and infer_column(df, "y"))
            else None,
        }

    if data is not None:
        raise TypeError("supply either data or objects, not both")

    if not isinstance(name, str) or not name:
        raise ValueError("name must be a single non-empty string")

    if not isinstance(
        include_overview_rows,
        (bool, np.bool_),
    ):
        raise ValueError("include_overview_rows must be True or False")

    normalized = _gp3_qc_normalise_objects(objects)

    if object_names is not None:
        if (
            isinstance(
                object_names,
                str,
            )
            or not isinstance(
                object_names,
                (list, tuple),
            )
            or len(object_names) != len(normalized)
        ):
            raise ValueError("object_names must contain one name per object")

        normalized = [
            (
                str(object_names[index]),
                obj,
            )
            for index, (_, obj) in enumerate(normalized)
        ]

    named_objects = []

    for index, (
        object_name,
        obj,
    ) in enumerate(
        normalized,
        start=1,
    ):
        if not object_name:
            object_name = f"object_{index}"

        named_objects.append(
            (
                object_name,
                obj,
            )
        )

    object_summaries = []
    overview_frames = []

    for index, (
        object_name,
        obj,
    ) in enumerate(
        named_objects,
        start=1,
    ):
        collected = _gp3_qc_collect_one(
            obj,
            object_name,
            index,
        )

        object_summaries.append(collected["object_summary"])

        if include_overview_rows:
            overview_frames.append(collected["overview_rows"])

    object_summary = pd.concat(
        object_summaries,
        ignore_index=True,
    )

    if include_overview_rows:
        overview_rows = _gp3_qc_bind_overview_rows(overview_frames)
    else:
        overview_rows = pd.DataFrame()

    status_counts = _gp3_qc_status_counts(object_summary["qc_status"])

    overall_status = _gp3_qc_overall_status(object_summary["qc_status"])

    def count_status(value):
        return int(object_summary["qc_status"].eq(value).sum())

    overview = pd.DataFrame(
        {
            "object_name": [name],
            "n_objects": [int(len(object_summary))],
            "n_overview_rows": [
                int(
                    pd.to_numeric(
                        object_summary["n_overview_rows"],
                        errors="coerce",
                    )
                    .fillna(0)
                    .sum()
                )
            ],
            "n_pass": [count_status("pass")],
            "n_warn": [count_status("warn")],
            "n_fail": [count_status("fail")],
            "n_info": [count_status("info")],
            "n_unknown": [count_status("unknown")],
            "qc_bundle_status": [overall_status],
        }
    )

    settings = pd.DataFrame(
        {
            "setting": [
                "name",
                "include_overview_rows",
            ],
            "value": [
                name,
                ("TRUE" if include_overview_rows else "FALSE"),
            ],
        }
    )

    return {
        "overview": overview,
        "object_summary": object_summary,
        "status_counts": status_counts,
        "overview_rows": overview_rows,
        "settings": settings,
    }


def summarise_gazepoint_qc_status(qc) -> pd.DataFrame:
    rows = []
    if isinstance(qc, dict):
        for name, value in qc.items():
            status = "available" if value is not None else "not_available"
            rows.append({"component": name, "status": status})
    return pd.DataFrame(rows)


summarize_gazepoint_qc_status = summarise_gazepoint_qc_status


def check_gazepoint_real_data_readiness(data) -> dict[str, Any]:
    validation = validate_gazepoint_master(as_gazepoint_master(data), required=("subject", "time"))
    df = ensure_dataframe(data, copy=False)
    has_gaze = infer_column(df, "x") is not None and infer_column(df, "y") is not None
    has_pupil = infer_column(df, "pupil") is not None or (
        infer_column(df, "left_pupil") and infer_column(df, "right_pupil")
    )
    checks = validation["checks"].copy()
    checks = pd.concat(
        [
            checks,
            pd.DataFrame(
                [
                    {
                        "check": "gaze_coordinates",
                        "passed": has_gaze,
                        "detail": "available" if has_gaze else "missing",
                    },
                    {
                        "check": "pupil_signal",
                        "passed": bool(has_pupil),
                        "detail": "available" if has_pupil else "missing",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    return {"ready": bool(checks["passed"].all()), "checks": checks}


def recommend_gazepoint_exclusions(
    data,
    participant_col=None,
    trial_col=None,
    validity_col=None,
    x_col=None,
    y_col=None,
    pupil_col=None,
    artifact_col=None,
    min_trial_samples: int = 20,
    max_trial_missing_prop: float = 0.5,
    max_trial_artifact_prop: float = 0.5,
    min_participant_trials: int = 1,
    min_participant_valid_trials: int = 1,
    max_participant_missing_prop: float = 0.5,
    max_participant_artifact_prop: float = 0.5,
    require_both_gaze_coordinates: bool = True,
    name: str = "gazepoint_exclusions",
    **kwargs,
) -> dict[str, Any]:
    """Recommend exclusions without removing rows."""
    df = ensure_dataframe(data, copy=False)
    participant_col = infer_column(df, "subject", participant_col, required=True)
    trial_col = infer_column(df, "trial", trial_col) or "__gp3_trial"
    work = df.copy()
    if trial_col == "__gp3_trial":
        work[trial_col] = 1
    validity_col = infer_column(work, "validity", validity_col)
    x_col, y_col = infer_column(work, "x", x_col), infer_column(work, "y", y_col)
    pupil_col = infer_column(work, "pupil", pupil_col)
    usable = pd.Series(True, index=work.index)
    if validity_col:
        usable &= as_bool(work[validity_col], invert_trackloss=validity_col.lower() == "trackloss")
    coords = pd.Series(True, index=work.index)
    if x_col:
        coords &= finite_numeric(work[x_col]).notna()
    if y_col:
        coords &= finite_numeric(work[y_col]).notna()
    if require_both_gaze_coordinates:
        usable &= coords
    if pupil_col:
        usable &= finite_numeric(work[pupil_col]).notna()
    artifact = (
        as_bool(work[artifact_col])
        if artifact_col and artifact_col in work
        else pd.Series(False, index=work.index)
    )
    work = work.assign(_usable=usable, _artifact=artifact)
    trial = (
        work.groupby([participant_col, trial_col], dropna=False)
        .agg(
            n_samples=("_usable", "size"),
            usable_prop=("_usable", "mean"),
            artifact_prop=("_artifact", "mean"),
        )
        .reset_index()
    )
    trial["exclude"] = (
        (trial.n_samples < min_trial_samples)
        | ((1 - trial.usable_prop) > max_trial_missing_prop)
        | (trial.artifact_prop > max_trial_artifact_prop)
    )
    part = (
        trial.groupby(participant_col, dropna=False)
        .agg(n_trials=(trial_col, "size"), n_valid_trials=("exclude", lambda s: int((~s).sum())))
        .reset_index()
    )
    sample = (
        work.groupby(participant_col, dropna=False)
        .agg(
            missing_prop=("_usable", lambda s: float(1 - s.mean())),
            artifact_prop=("_artifact", "mean"),
        )
        .reset_index()
    )
    part = part.merge(sample, on=participant_col, how="left")
    part["exclude"] = (
        (part.n_trials < min_participant_trials)
        | (part.n_valid_trials < min_participant_valid_trials)
        | (part.missing_prop > max_participant_missing_prop)
        | (part.artifact_prop > max_participant_artifact_prop)
    )
    exclusion_table = pd.concat(
        [
            trial.loc[trial.exclude, [participant_col, trial_col]].assign(level="trial"),
            part.loc[part.exclude, [participant_col]].assign(
                **{trial_col: pd.NA}, level="participant"
            ),
        ],
        ignore_index=True,
    )
    return {
        "name": name,
        "overview": result_table(
            n_trial_exclusions=int(trial.exclude.sum()),
            n_participant_exclusions=int(part.exclude.sum()),
        ),
        "trial_recommendations": trial,
        "participant_recommendations": part,
        "exclusions": exclusion_table,
        "settings": kwargs
        | {
            "min_trial_samples": min_trial_samples,
            "max_trial_missing_prop": max_trial_missing_prop,
        },
    }


def audit_gazepoint_naming_consistency(exports=None) -> dict[str, Any]:
    """Audit British/American summary-helper naming pairs."""
    if exports is None:
        from ._exports import R_EXPORTS

        exports = list(R_EXPORTS)
    values = []
    for value in exports:
        if value is None:
            continue
        text = str(value)
        if text and text not in values:
            values.append(text)

    british = [value for value in values if value.startswith("summarise_")]
    american = [value for value in values if value.startswith("summarize_")]
    stems = sorted(
        set(value.removeprefix("summarise_") for value in british)
        | set(value.removeprefix("summarize_") for value in american)
    )
    rows = []
    for stem in stems:
        british_name = f"summarise_{stem}"
        american_name = f"summarize_{stem}"
        british_exported = british_name in values
        american_exported = american_name in values
        status = (
            "paired"
            if british_exported and american_exported
            else "canonical_only"
            if british_exported
            else "missing_british_alias"
        )
        rows.append(
            {
                "stem": stem,
                "british_name": british_name,
                "american_name": american_name,
                "british_exported": british_exported,
                "american_exported": american_exported,
                "canonical_name": british_name,
                "status": status,
            }
        )
    pairs = pd.DataFrame(
        rows,
        columns=[
            "stem",
            "british_name",
            "american_name",
            "british_exported",
            "american_exported",
            "canonical_name",
            "status",
        ],
    )
    summary = pd.DataFrame(
        [
            {
                "status": "needs_review"
                if len(pairs) and pairs["status"].eq("missing_british_alias").any()
                else "pass",
                "n_summary_stems": len(pairs),
                "n_paired": int(pairs["status"].eq("paired").sum()) if len(pairs) else 0,
                "n_canonical_only": int(pairs["status"].eq("canonical_only").sum())
                if len(pairs)
                else 0,
                "n_missing_british_alias": int(pairs["status"].eq("missing_british_alias").sum())
                if len(pairs)
                else 0,
                # Legacy Python diagnostics retained as additive fields.
                "n_names": len(values),
                "n_alias_pairs": int(pairs["status"].eq("paired").sum()) if len(pairs) else 0,
                "n_issues": int(pairs["status"].eq("missing_british_alias").sum())
                if len(pairs)
                else 0,
            }
        ]
    )
    return {
        "summary": summary,
        "pairs": pairs,
        "policy": gp3tools_naming_policy(),
    }


def gp3tools_naming_policy() -> dict[str, Any]:
    return {
        "canonical": "British English summarise_*",
        "compatibility": "Existing summarize_* aliases are retained",
        "rules": pd.DataFrame(
            [
                {"rule": "summary helpers", "canonical": "summarise_*"},
                {"rule": "legacy aliases", "canonical": "retain summarize_*"},
            ]
        ),
    }


def write_gazepoint_naming_audit(
    path=None,
    names=None,
    *,
    x=None,
    output_file=None,
) -> Path:
    """Write a naming audit.

    ``x`` plus ``output_file`` implements the R v2.3.0 audit-object
    interface. ``path`` plus ``names`` retains the original Python API.
    """
    if x is not None or output_file is not None:
        if x is None or output_file is None:
            raise ValueError("x and output_file must be supplied together")

        if path is not None:
            raise TypeError("path cannot be combined with the R-compatible x/output_file interface")

        if isinstance(x, dict):
            pairs = x.get("pairs")
        else:
            pairs = getattr(
                x,
                "pairs",
                None,
            )

        if pairs is None:
            raise TypeError("x must contain a 'pairs' table")

        out = Path(output_file)

        if not str(out).strip():
            raise ValueError("output_file must be a non-empty path")

        out.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        ensure_dataframe(
            pairs,
            copy=False,
        ).to_csv(
            out,
            index=False,
        )

        return out.resolve()

    if path is None:
        raise TypeError("path is required for the Python interface")

    out = Path(path)
    out.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit = audit_gazepoint_naming_consistency(names)

    audit["summary"].to_csv(
        out,
        index=False,
    )

    return out


# BEGIN R V2.3.0 CALL-SURFACE ALIASES
audit_gazepoint_master = r_aliases(audit_gazepoint_master, master="data")
audit_gazepoint_screen_bounds = r_aliases(
    audit_gazepoint_screen_bounds, screen_width="width", screen_height="height"
)
summarise_gazepoint_qc_status = r_aliases(summarise_gazepoint_qc_status, qc_bundle="qc")
validate_gazepoint_master = r_aliases(validate_gazepoint_master, master="data")
# END R V2.3.0 CALL-SURFACE ALIASES

# R v2.3.0 alias rebinding after compatibility wrappers
summarize_gazepoint_qc_status = summarise_gazepoint_qc_status
