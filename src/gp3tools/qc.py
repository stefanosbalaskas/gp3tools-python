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

_GP3_QC_R_UNSET = object()


def _gp3_qc_r_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _gp3_qc_r_as_bool_series(values, index):
    series = pd.Series(values, index=index)
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.astype("boolean")
    if pd.api.types.is_numeric_dtype(series.dtype):
        numeric = pd.to_numeric(series, errors="coerce")
        out = pd.Series(pd.NA, index=index, dtype="boolean")
        out.loc[numeric.notna()] = numeric.loc[numeric.notna()].ne(0).to_numpy()
        return out
    text = series.astype("string").str.strip().str.lower()
    true_values = {"true", "t", "yes", "y", "1", "valid", "ok"}
    false_values = {"false", "f", "no", "n", "0", "invalid", "bad"}
    out = pd.Series(pd.NA, index=index, dtype="boolean")
    out.loc[text.isin(true_values)] = True
    out.loc[text.isin(false_values)] = False
    return out


def as_gazepoint_master(
    data,
    copy: bool = True,
    *,
    screen_width_px=_GP3_QC_R_UNSET,
    screen_height_px=_GP3_QC_R_UNSET,
    source_col=_GP3_QC_R_UNSET,
    media_col=_GP3_QC_R_UNSET,
    media_name_col=_GP3_QC_R_UNSET,
    time_col=_GP3_QC_R_UNSET,
    coordinate_unit=_GP3_QC_R_UNSET,
    event_latency_offset_ms=_GP3_QC_R_UNSET,
) -> pd.DataFrame:
    """Coerce data to a legacy Python or R v2.3.0 Gazepoint master table."""
    r_mode = any(
        value is not _GP3_QC_R_UNSET
        for value in (
            screen_width_px,
            screen_height_px,
            source_col,
            media_col,
            media_name_col,
            time_col,
            coordinate_unit,
            event_latency_offset_ms,
        )
    )

    if not r_mode:
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

    from .io import standardise_gazepoint_names

    screen_width_px = None if screen_width_px is _GP3_QC_R_UNSET else screen_width_px
    screen_height_px = None if screen_height_px is _GP3_QC_R_UNSET else screen_height_px
    source_col = "USER_FILE" if source_col is _GP3_QC_R_UNSET else source_col
    media_col = "MEDIA_ID" if media_col is _GP3_QC_R_UNSET else media_col
    media_name_col = "MEDIA_NAME" if media_name_col is _GP3_QC_R_UNSET else media_name_col
    time_col = "TIME" if time_col is _GP3_QC_R_UNSET else time_col
    coordinate_unit = "auto" if coordinate_unit is _GP3_QC_R_UNSET else coordinate_unit
    event_latency_offset_ms = (
        0.0 if event_latency_offset_ms is _GP3_QC_R_UNSET else event_latency_offset_ms
    )

    if coordinate_unit not in {"auto", "normalised", "pixels"}:
        raise ValueError("coordinate_unit must be 'auto', 'normalised', or 'pixels'")
    for value, label in (
        (screen_width_px, "screen_width_px"),
        (screen_height_px, "screen_height_px"),
    ):
        if value is not None and (not np.isscalar(value) or not np.isfinite(float(value))):
            raise ValueError(f"{label} must be None or a single finite numeric value")
    if not np.isscalar(event_latency_offset_ms) or not np.isfinite(float(event_latency_offset_ms)):
        raise ValueError("event_latency_offset_ms must be a single finite numeric value")

    raw = ensure_dataframe(data, copy=True)
    df = standardise_gazepoint_names(raw).reset_index(drop=True)
    if time_col not in df.columns:
        raise KeyError(f"Missing required column: {time_col}")
    n = len(df)

    def numeric(candidates, default=np.nan):
        candidates = [candidates] if isinstance(candidates, str) else list(candidates)
        hit = next((col for col in candidates if col in df.columns), None)
        if hit is None:
            return pd.Series(np.full(n, default), dtype=float)
        return pd.to_numeric(df[hit], errors="coerce").reset_index(drop=True)

    def character(candidates):
        candidates = [candidates] if isinstance(candidates, str) else list(candidates)
        hit = next((col for col in candidates if col in df.columns), None)
        if hit is None:
            return pd.Series(pd.array([pd.NA] * n, dtype="string"))
        return df[hit].astype("string").reset_index(drop=True)

    def first_existing(candidates):
        return next((col for col in candidates if col in df.columns), None)

    source_file = character(source_col)
    media_id = character(media_col)
    media_name = character(media_name_col)
    subject = source_file.str.replace(r"_all_gaze\.csv$", "", regex=True)
    subject = subject.str.replace(r"\.csv$", "", regex=True)
    subject = subject.mask(source_file.isna() | source_file.eq(""))
    trial_global = pd.Series(pd.array([pd.NA] * n, dtype="string"))
    trial_mask = subject.notna() & media_id.notna()
    trial_global.loc[trial_mask] = (
        subject.loc[trial_mask].astype("string")
        + "_MEDIA_"
        + media_id.loc[trial_mask].astype("string")
    )

    raw_time_sec = numeric(time_col)
    time_ms = raw_time_sec * 1000.0 + float(event_latency_offset_ms)

    best_x_raw = numeric(["BPOGX", "FPOGX"])
    best_y_raw = numeric(["BPOGY", "FPOGY"])
    left_x_raw = numeric(["LPOGX"])
    left_y_raw = numeric(["LPOGY"])
    right_x_raw = numeric(["RPOGX"])
    right_y_raw = numeric(["RPOGY"])

    coordinate_values = pd.concat(
        [best_x_raw, best_y_raw, left_x_raw, left_y_raw, right_x_raw, right_y_raw],
        ignore_index=True,
    ).to_numpy(float)
    coordinate_values = coordinate_values[np.isfinite(coordinate_values)]
    detected_coordinate_unit = coordinate_unit
    if coordinate_unit == "auto":
        prop_central = (
            float(np.mean((coordinate_values >= -0.25) & (coordinate_values <= 1.25)))
            if coordinate_values.size
            else 0.0
        )
        detected_coordinate_unit = (
            "normalised" if coordinate_values.size and prop_central >= 0.80 else "pixels"
        )

    def scale_x(values):
        if detected_coordinate_unit == "normalised" and screen_width_px is not None:
            return values * float(screen_width_px)
        return values.copy()

    def scale_y(values):
        if detected_coordinate_unit == "normalised" and screen_height_px is not None:
            return values * float(screen_height_px)
        return values.copy()

    x = scale_x(best_x_raw)
    y = scale_y(best_y_raw)
    left_x = scale_x(left_x_raw)
    left_y = scale_y(left_y_raw)
    right_x = scale_x(right_x_raw)
    right_y = scale_y(right_y_raw)

    best_valid = numeric(["BPOGV", "FPOGV"], default=1.0)
    left_gaze_valid_raw = numeric(["LPOGV"])
    right_gaze_valid_raw = numeric(["RPOGV"])
    left_pupil_source = first_existing(["LPMM", "LPUPILD", "LPD"])
    right_pupil_source = first_existing(["RPMM", "RPUPILD", "RPD"])
    left_pupil = numeric(["LPMM", "LPUPILD", "LPD"])
    right_pupil = numeric(["RPMM", "RPUPILD", "RPD"])
    left_pupil_valid_raw = numeric(["LPMMV", "LPUPILV", "LPV"])
    right_pupil_valid_raw = numeric(["RPMMV", "RPUPILV", "RPV"])
    left_pupil = left_pupil.mask(left_pupil_valid_raw.notna() & left_pupil_valid_raw.eq(0))
    right_pupil = right_pupil.mask(right_pupil_valid_raw.notna() & right_pupil_valid_raw.eq(0))
    mean_pupil = pd.concat([left_pupil, right_pupil], axis=1).mean(axis=1, skipna=True)

    if left_pupil_source == "LPMM" or right_pupil_source == "RPMM":
        pupil_unit = "diameter_mm"
    elif left_pupil_source == "LPUPILD" or right_pupil_source == "RPUPILD":
        pupil_unit = "diameter_meters"
    elif left_pupil_source == "LPD" or right_pupil_source == "RPD":
        pupil_unit = "tracker_units"
    else:
        pupil_unit = pd.NA

    gaze_unit = (
        "pixels"
        if detected_coordinate_unit == "normalised"
        and screen_width_px is not None
        and screen_height_px is not None
        else detected_coordinate_unit
    )
    valid_sample = best_valid.notna() & best_valid.eq(1)
    missing_gaze = (~valid_sample) | x.isna() | y.isna()
    missing_pupil = left_pupil.isna() & right_pupil.isna()
    gaze_offscreen = pd.Series(pd.array([pd.NA] * n, dtype="boolean"))
    if screen_width_px is not None and screen_height_px is not None:
        off = (
            (~missing_gaze)
            & x.notna()
            & y.notna()
            & (x.lt(0) | x.gt(float(screen_width_px)) | y.lt(0) | y.gt(float(screen_height_px)))
        )
        gaze_offscreen = off.astype("boolean")

    blink_id = numeric(["BKID"])
    blink_duration = numeric(["BKDUR"])
    has_blink_columns = blink_id.notna().any() or blink_duration.notna().any()
    blink = (
        ((blink_id.notna() & blink_id.gt(0)) | (blink_duration.notna() & blink_duration.gt(0)))
        if has_blink_columns
        else (missing_gaze & missing_pupil)
    )
    trackloss = missing_gaze.copy()
    message = character(["USER", "USER_DATA", "MESSAGE"])
    event_type = pd.Series(pd.array([pd.NA] * n, dtype="string"))
    has_message = message.notna() & message.ne("")
    for token, label in (
        ("TRIAL_START", "trial_start"),
        ("STIMULUS_ONSET", "stimulus_onset"),
        ("TARGET_ONSET", "target_onset"),
        ("TRIAL_END", "trial_end"),
    ):
        event_type.loc[has_message & message.str.contains(token, regex=False, na=False)] = label

    aoi = character(["AOI"])
    aoi_current = aoi.mask(aoi.isna() | aoi.eq(""))
    aoi_current = aoi_current.mask(gaze_offscreen.fillna(False), "offscreen")
    aoi_current = aoi_current.mask(missing_gaze, "missing")
    artifact_flag = missing_gaze | missing_pupil
    artifact_reason = pd.Series(pd.array([pd.NA] * n, dtype="string"))
    artifact_reason.loc[missing_gaze & missing_pupil] = "missing_gaze_and_pupil"
    artifact_reason.loc[missing_gaze & ~missing_pupil] = "missing_gaze"
    artifact_reason.loc[~missing_gaze & missing_pupil] = "missing_pupil"

    fixation_x_raw = numeric(["FPOGX"])
    fixation_y_raw = numeric(["FPOGY"])
    out = pd.DataFrame(
        {
            "source_file": source_file,
            "subject": subject,
            "pID": subject,
            "media_id": media_id,
            "media_name": media_name,
            "trial_global": trial_global,
            "time": time_ms,
            "time_ms": time_ms,
            "time_orig_sec": raw_time_sec,
            "time_orig_ms": raw_time_sec * 1000.0,
            "sample_index": numeric(["CNT"]),
            "time_bin_25ms": np.floor(time_ms / 25.0) * 25.0,
            "time_bin_50ms": np.floor(time_ms / 50.0) * 50.0,
            "time_bin_100ms": np.floor(time_ms / 100.0) * 100.0,
            "x": x,
            "y": y,
            "raw_x": best_x_raw,
            "raw_y": best_y_raw,
            "left_x": left_x,
            "left_y": left_y,
            "right_x": right_x,
            "right_y": right_y,
            "left_pupil": left_pupil,
            "right_pupil": right_pupil,
            "mean_pupil": mean_pupil,
            "pupil": mean_pupil,
            "pupil_unit": pd.Series([pupil_unit] * n, dtype="string"),
            "pupil_source_left": pd.Series([left_pupil_source] * n, dtype="string"),
            "pupil_source_right": pd.Series([right_pupil_source] * n, dtype="string"),
            "gaze_unit": pd.Series([gaze_unit] * n, dtype="string"),
            "coordinate_unit_detected": pd.Series([detected_coordinate_unit] * n, dtype="string"),
            "screen_width_px": np.nan if screen_width_px is None else float(screen_width_px),
            "screen_height_px": np.nan if screen_height_px is None else float(screen_height_px),
            "valid_sample": valid_sample.to_numpy(bool),
            "missing_gaze": missing_gaze.to_numpy(bool),
            "missing_pupil": missing_pupil.to_numpy(bool),
            "gaze_offscreen": gaze_offscreen,
            "trackloss": trackloss.to_numpy(bool),
            "Trackloss": trackloss.to_numpy(bool),
            "blink": blink.to_numpy(bool),
            "left_gaze_valid": _gp3_qc_r_as_bool_series(
                left_gaze_valid_raw.eq(1).where(left_gaze_valid_raw.notna()), df.index
            ),
            "right_gaze_valid": _gp3_qc_r_as_bool_series(
                right_gaze_valid_raw.eq(1).where(right_gaze_valid_raw.notna()), df.index
            ),
            "left_pupil_valid": _gp3_qc_r_as_bool_series(
                left_pupil_valid_raw.eq(1).where(left_pupil_valid_raw.notna()), df.index
            ),
            "right_pupil_valid": _gp3_qc_r_as_bool_series(
                right_pupil_valid_raw.eq(1).where(right_pupil_valid_raw.notna()), df.index
            ),
            "aoi": aoi,
            "AOI": aoi,
            "aoi_current": aoi_current,
            "aoi_count": (~aoi_current.isna() & ~aoi_current.isin(["missing", "offscreen"])).astype(
                int
            ),
            "message": message,
            "event_type": event_type,
            "event_label": message,
            "event_latency_offset_ms": float(event_latency_offset_ms),
            "fixation_x": scale_x(fixation_x_raw),
            "fixation_y": scale_y(fixation_y_raw),
            "fixation_start_sec": numeric(["FPOGS"]),
            "fixation_duration_sec": numeric(["FPOGD"]),
            "fixation_id": numeric(["FPOGID"]),
            "fixation_event": numeric(["FPOGV"]).eq(1).to_numpy(bool),
            "artifact_flag": artifact_flag.to_numpy(bool),
            "artifact_reason": artifact_reason,
            "tracker_model": "Gazepoint",
            "tracker_sampling_rate": np.nan,
        }
    )
    return attach_attrs(
        out,
        gp3_class="gazepoint_master",
        r_class="gazepoint_master",
        coordinate_unit_detected=detected_coordinate_unit,
    )


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
    data,
    required=_GP3_QC_R_UNSET,
    *,
    min_valid_sample_pct=_GP3_QC_R_UNSET,
    max_missing_gaze_pct=_GP3_QC_R_UNSET,
    max_missing_pupil_pct=_GP3_QC_R_UNSET,
    max_offscreen_gaze_pct=_GP3_QC_R_UNSET,
    require_pupil=_GP3_QC_R_UNSET,
    require_aoi=_GP3_QC_R_UNSET,
    fail_on_error=_GP3_QC_R_UNSET,
) -> dict[str, Any]:
    """Validate a Gazepoint master table using legacy or R v2.3.0 checks."""
    r_mode = any(
        value is not _GP3_QC_R_UNSET
        for value in (
            min_valid_sample_pct,
            max_missing_gaze_pct,
            max_missing_pupil_pct,
            max_offscreen_gaze_pct,
            require_pupil,
            require_aoi,
            fail_on_error,
        )
    )

    if not r_mode:
        df = ensure_dataframe(data, copy=False)
        required_cols = ("subject", "time") if required is _GP3_QC_R_UNSET else tuple(required)
        checks = []
        for col in required_cols:
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
                    "detail": (
                        "no duplicate names"
                        if not df.columns.duplicated().any()
                        else "duplicates present"
                    ),
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

    df = ensure_dataframe(data, copy=False)
    min_valid_sample_pct = (
        75.0 if min_valid_sample_pct is _GP3_QC_R_UNSET else float(min_valid_sample_pct)
    )
    max_missing_gaze_pct = (
        25.0 if max_missing_gaze_pct is _GP3_QC_R_UNSET else float(max_missing_gaze_pct)
    )
    max_missing_pupil_pct = (
        50.0 if max_missing_pupil_pct is _GP3_QC_R_UNSET else float(max_missing_pupil_pct)
    )
    max_offscreen_gaze_pct = (
        25.0 if max_offscreen_gaze_pct is _GP3_QC_R_UNSET else float(max_offscreen_gaze_pct)
    )
    require_pupil = False if require_pupil is _GP3_QC_R_UNSET else bool(require_pupil)
    require_aoi = False if require_aoi is _GP3_QC_R_UNSET else bool(require_aoi)
    fail_on_error = False if fail_on_error is _GP3_QC_R_UNSET else bool(fail_on_error)

    def detect(candidates):
        return next((col for col in candidates if col in df.columns), None)

    roles = {
        "subject": detect(["subject", "pID", "participant"]),
        "media_id": detect(["media_id", "MEDIA_ID"]),
        "time": detect(["time_ms", "time", "time_orig"]),
        "x": detect(["x", "gaze_x"]),
        "y": detect(["y", "gaze_y"]),
        "valid_sample": detect(["valid_sample"]),
        "missing_gaze": detect(["missing_gaze"]),
        "missing_pupil": detect(["missing_pupil"]),
        "gaze_offscreen": detect(["gaze_offscreen"]),
        "pupil": detect(["mean_pupil", "pupil", "pupil_raw"]),
        "aoi_current": detect(["aoi_current", "AOI"]),
        "aoi_count": detect(["aoi_count"]),
        "screen_width_px": detect(["screen_width_px"]),
        "screen_height_px": detect(["screen_height_px"]),
    }
    column_map = pd.DataFrame({"role": list(roles), "column": [roles[r] for r in roles]})

    def get(role):
        col = roles[role]
        return None if col is None else df[col]

    def prop_true_pct(values):
        if values is None or len(values) == 0:
            return np.nan
        logical = _gp3_qc_r_as_bool_series(values, df.index)
        if logical.isna().all():
            return np.nan
        return float(logical.mean(skipna=True) * 100.0)

    def safe_minmax(values, which):
        if values is None:
            return np.nan
        arr = pd.to_numeric(values, errors="coerce").to_numpy(float)
        arr = arr[np.isfinite(arr)]
        if not arr.size:
            return np.nan
        return float(np.min(arr) if which == "min" else np.max(arr))

    def n_distinct(values):
        if values is None:
            return np.nan
        return int(pd.Series(values).dropna().nunique())

    def real_aoi(values):
        if values is None:
            return pd.Series(False, index=df.index)
        text = values.astype("string")
        return (
            text.notna()
            & text.ne("")
            & ~text.isin(["missing", "offscreen", "non_aoi", "unclassified"])
        )

    subject = get("subject")
    media = get("media_id")
    time_values = get("time")
    x = get("x")
    y = get("y")
    valid_sample = get("valid_sample")
    missing_gaze = get("missing_gaze")
    missing_pupil = get("missing_pupil")
    gaze_offscreen = get("gaze_offscreen")
    pupil = get("pupil")
    aoi_current = get("aoi_current")
    aoi_count = get("aoi_count")
    screen_width = get("screen_width_px")
    screen_height = get("screen_height_px")

    n_rows = len(df)
    n_subjects = n_distinct(subject)
    n_media = n_distinct(media)
    valid_sample_pct = prop_true_pct(valid_sample)
    missing_gaze_pct = prop_true_pct(missing_gaze)
    missing_pupil_pct = prop_true_pct(missing_pupil)
    offscreen_gaze_pct = prop_true_pct(gaze_offscreen)
    has_pupil = bool(pupil is not None and pd.to_numeric(pupil, errors="coerce").notna().any())
    real_aoi_flag = real_aoi(aoi_current)
    has_aoi = bool(real_aoi_flag.any())
    time_min = safe_minmax(time_values, "min")
    time_max = safe_minmax(time_values, "max")

    required_roles = [
        "subject",
        "media_id",
        "time",
        "x",
        "y",
        "valid_sample",
        "missing_gaze",
        "missing_pupil",
        "gaze_offscreen",
        "aoi_current",
        "aoi_count",
    ]
    missing_required_roles = [role for role in required_roles if roles[role] is None]

    checks = []

    def add(check_id, check_name, passed, severity, value, threshold, message, warning=False):
        status = "pass" if bool(passed) else ("warning" if warning else "fail")
        checks.append(
            {
                "check_id": check_id,
                "check_name": check_name,
                "status": status,
                "severity": severity,
                "value": str(value),
                "threshold": str(threshold),
                "message": message,
            }
        )

    add(
        "C001",
        "Non-empty data frame",
        n_rows > 0,
        "error",
        n_rows,
        "> 0",
        "The master table must contain at least one row.",
    )
    add(
        "C002",
        "Required columns detected",
        not missing_required_roles,
        "error",
        "none missing" if not missing_required_roles else ", ".join(missing_required_roles),
        "all required roles present",
        "The master table must contain the required identifier, time, gaze, quality, and AOI columns.",
    )
    add(
        "C003",
        "Subject identifiers available",
        np.isfinite(n_subjects) and n_subjects > 0,
        "error",
        n_subjects,
        "> 0",
        "At least one non-missing subject identifier must be available.",
    )
    add(
        "C004",
        "Media identifiers available",
        np.isfinite(n_media) and n_media > 0,
        "error",
        n_media,
        "> 0",
        "At least one non-missing media/stimulus identifier must be available.",
    )
    add(
        "C005",
        "Time column is numeric",
        time_values is not None and pd.api.types.is_numeric_dtype(time_values),
        "error",
        "missing" if time_values is None else str(time_values.dtype),
        "numeric",
        "The detected time column must be numeric.",
    )
    add(
        "C006",
        "Time span is positive",
        np.isfinite(time_min) and np.isfinite(time_max) and time_max > time_min,
        "error",
        f"{time_min} to {time_max}",
        "max > min",
        "The master table must contain a positive time span.",
    )
    add(
        "C007",
        "Gaze coordinates are numeric",
        x is not None
        and y is not None
        and pd.api.types.is_numeric_dtype(x)
        and pd.api.types.is_numeric_dtype(y),
        "error",
        f"x={'missing' if x is None else x.dtype}; y={'missing' if y is None else y.dtype}",
        "numeric x and y",
        "The gaze coordinate columns must be numeric.",
    )
    add(
        "C008",
        "Valid-sample percentage acceptable",
        np.isfinite(valid_sample_pct) and valid_sample_pct >= min_valid_sample_pct,
        "error",
        round(valid_sample_pct, 3) if np.isfinite(valid_sample_pct) else np.nan,
        f">= {min_valid_sample_pct}",
        "The percentage of valid samples should be above the minimum threshold.",
    )
    add(
        "C009",
        "Missing-gaze percentage acceptable",
        np.isfinite(missing_gaze_pct) and missing_gaze_pct <= max_missing_gaze_pct,
        "error",
        round(missing_gaze_pct, 3) if np.isfinite(missing_gaze_pct) else np.nan,
        f"<= {max_missing_gaze_pct}",
        "The percentage of missing gaze samples should be below the maximum threshold.",
    )
    add(
        "C010",
        "Pupil data available",
        has_pupil,
        "error" if require_pupil else "warning",
        has_pupil,
        "required" if require_pupil else "recommended",
        "Pupil data are required only when pupil preprocessing or pupil modelling will be performed.",
        warning=(not require_pupil and not has_pupil),
    )
    add(
        "C011",
        "Missing-pupil percentage acceptable",
        (not np.isfinite(missing_pupil_pct)) or missing_pupil_pct <= max_missing_pupil_pct,
        "warning",
        round(missing_pupil_pct, 3) if np.isfinite(missing_pupil_pct) else np.nan,
        f"<= {max_missing_pupil_pct}",
        "High missing-pupil percentages may affect pupil preprocessing and pupil-based modelling.",
        warning=np.isfinite(missing_pupil_pct) and missing_pupil_pct > max_missing_pupil_pct,
    )
    add(
        "C012",
        "Off-screen gaze percentage acceptable",
        (not np.isfinite(offscreen_gaze_pct)) or offscreen_gaze_pct <= max_offscreen_gaze_pct,
        "warning",
        round(offscreen_gaze_pct, 3) if np.isfinite(offscreen_gaze_pct) else np.nan,
        f"<= {max_offscreen_gaze_pct}",
        "High off-screen gaze percentages may indicate calibration, stimulus, or participant-quality problems.",
        warning=np.isfinite(offscreen_gaze_pct) and offscreen_gaze_pct > max_offscreen_gaze_pct,
    )
    add(
        "C013",
        "AOI samples available",
        has_aoi,
        "error" if require_aoi else "warning",
        has_aoi,
        "required" if require_aoi else "recommended",
        "Real AOI samples are required only for AOI-based analyses.",
        warning=(not require_aoi and not has_aoi),
    )

    if aoi_count is not None and aoi_current is not None:
        count_numeric = pd.to_numeric(aoi_count, errors="coerce")
        aoi_mismatch = int(
            (count_numeric.notna() & count_numeric.ne(real_aoi_flag.astype(int))).sum()
        )
    else:
        aoi_mismatch = np.nan
    add(
        "C014",
        "AOI count matches AOI state",
        np.isfinite(aoi_mismatch) and aoi_mismatch == 0,
        "error",
        aoi_mismatch,
        "0 mismatches",
        "`aoi_count` should equal 1 only for real AOI samples and 0 otherwise.",
    )

    offscreen_mismatch = np.nan
    if all(
        v is not None for v in (x, y, missing_gaze, gaze_offscreen, screen_width, screen_height)
    ):
        sw = pd.to_numeric(screen_width, errors="coerce").dropna().unique()
        sh = pd.to_numeric(screen_height, errors="coerce").dropna().unique()
        if len(sw) == 1 and len(sh) == 1:
            mg = _gp3_qc_r_as_bool_series(missing_gaze, df.index).fillna(False)
            go = _gp3_qc_r_as_bool_series(gaze_offscreen, df.index)
            xn = pd.to_numeric(x, errors="coerce")
            yn = pd.to_numeric(y, errors="coerce")
            expected = (
                (~mg)
                & xn.notna()
                & yn.notna()
                & (xn.lt(0) | xn.gt(sw[0]) | yn.lt(0) | yn.gt(sh[0]))
            )
            comparable = go.notna()
            offscreen_mismatch = int(
                (go.loc[comparable].astype(bool) != expected.loc[comparable]).sum()
            )
    add(
        "C015",
        "Off-screen flag matches screen bounds",
        np.isfinite(offscreen_mismatch) and offscreen_mismatch == 0,
        "warning",
        "not checked" if not np.isfinite(offscreen_mismatch) else offscreen_mismatch,
        "0 mismatches",
        "When screen dimensions are available, `gaze_offscreen` should match the x/y screen bounds.",
        warning=not np.isfinite(offscreen_mismatch),
    )

    checks_df = pd.DataFrame(checks)
    failed = checks_df.loc[checks_df["status"].eq("fail")].reset_index(drop=True)
    warnings = checks_df.loc[checks_df["status"].eq("warning")].reset_index(drop=True)
    summary = pd.DataFrame(
        [
            {
                "validation_passed": len(failed) == 0,
                "n_checks": len(checks_df),
                "n_passed": int(checks_df["status"].eq("pass").sum()),
                "n_failed": len(failed),
                "n_warnings": len(warnings),
                "n_rows": n_rows,
                "n_subjects": n_subjects,
                "n_media": n_media,
                "time_min": time_min,
                "time_max": time_max,
                "time_span": time_max - time_min
                if np.isfinite(time_min) and np.isfinite(time_max)
                else np.nan,
                "valid_sample_pct": valid_sample_pct,
                "missing_gaze_pct": missing_gaze_pct,
                "missing_pupil_pct": missing_pupil_pct,
                "offscreen_gaze_pct": offscreen_gaze_pct,
                "has_pupil": has_pupil,
                "has_aoi": has_aoi,
            }
        ]
    )
    result = {
        "summary": summary,
        "checks": checks_df,
        "failed_checks": failed,
        "warning_checks": warnings,
        "column_map": column_map,
        "valid": bool(len(failed) == 0),
    }
    if fail_on_error and len(failed):
        raise ValueError(f"master failed validation with {len(failed)} failing check(s)")
    return result


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
    data,
    excluded_col: str = "excluded",
    group_cols=("condition",),
    *,
    subject_col=_GP3_QC_R_UNSET,
    condition_col=_GP3_QC_R_UNSET,
    unit_cols=_GP3_QC_R_UNSET,
    retained_col=_GP3_QC_R_UNSET,
    include_col=_GP3_QC_R_UNSET,
    exclude_col=_GP3_QC_R_UNSET,
    status_col=_GP3_QC_R_UNSET,
    expected_conditions=_GP3_QC_R_UNSET,
    included_values=_GP3_QC_R_UNSET,
    excluded_values=_GP3_QC_R_UNSET,
    min_retained_units_per_condition=_GP3_QC_R_UNSET,
    min_retained_units_per_subject_condition=_GP3_QC_R_UNSET,
    max_condition_count_ratio=_GP3_QC_R_UNSET,
    max_subject_condition_ratio=_GP3_QC_R_UNSET,
    require_all_conditions_per_subject=_GP3_QC_R_UNSET,
):
    r_mode = any(
        v is not _GP3_QC_R_UNSET
        for v in (
            subject_col,
            condition_col,
            unit_cols,
            retained_col,
            include_col,
            exclude_col,
            status_col,
            expected_conditions,
            included_values,
            excluded_values,
            min_retained_units_per_condition,
            min_retained_units_per_subject_condition,
            max_condition_count_ratio,
            max_subject_condition_ratio,
            require_all_conditions_per_subject,
        )
    )
    if not r_mode:
        df = ensure_dataframe(data, copy=False)
        cols = [c for c in group_cols if c in df]
        retained = df.loc[~as_bool(df[excluded_col])] if excluded_col in df else df
        return (
            retained.groupby(cols, dropna=False).size().rename("n_retained").reset_index()
            if cols
            else result_table(n_retained=len(retained))
        )
    df = _gp3_exclusion_r_aliases(ensure_dataframe(data))
    if df.empty:
        raise ValueError("data must contain at least one row")
    subject_col = _gp3_exclusion_r_col(
        df, "subject" if subject_col is _GP3_QC_R_UNSET else subject_col, "subject_col"
    )
    condition_col = _gp3_exclusion_r_col(
        df, "condition" if condition_col is _GP3_QC_R_UNSET else condition_col, "condition_col"
    )
    raw_units = (
        ("media_id", "trial_global")
        if unit_cols is _GP3_QC_R_UNSET
        else (
            () if unit_cols is None else ([unit_cols] if isinstance(unit_cols, str) else unit_cols)
        )
    )
    unit_cols = [
        {"MEDIA_ID": "media_id", "USER_FILE": "subject"}.get(c, c)
        for c in raw_units
        if {"MEDIA_ID": "media_id", "USER_FILE": "subject"}.get(c, c) in df
    ]
    retained_col = _gp3_exclusion_r_col(
        df, None if retained_col is _GP3_QC_R_UNSET else retained_col, "retained_col", optional=True
    )
    include_col = _gp3_exclusion_r_col(
        df, None if include_col is _GP3_QC_R_UNSET else include_col, "include_col", optional=True
    )
    exclude_col_r = _gp3_exclusion_r_col(
        df, None if exclude_col is _GP3_QC_R_UNSET else exclude_col, "exclude_col", optional=True
    )
    status_col = _gp3_exclusion_r_col(
        df, None if status_col is _GP3_QC_R_UNSET else status_col, "status_col", optional=True
    )
    included_values = (
        ["included", "include", "kept", "keep", "retained", "ok", "ready", "complete", "completed"]
        if included_values is _GP3_QC_R_UNSET
        else list(included_values)
    )
    excluded_values = (
        [
            "excluded",
            "exclude",
            "drop",
            "dropped",
            "removed",
            "fail",
            "failed",
            "not_ready",
            "review",
            "invalid",
        ]
        if excluded_values is _GP3_QC_R_UNSET
        else list(excluded_values)
    )
    min_cond = (
        1
        if min_retained_units_per_condition is _GP3_QC_R_UNSET
        else int(min_retained_units_per_condition)
    )
    min_cell = (
        1
        if min_retained_units_per_subject_condition is _GP3_QC_R_UNSET
        else int(min_retained_units_per_subject_condition)
    )
    max_cond = (
        2.0 if max_condition_count_ratio is _GP3_QC_R_UNSET else float(max_condition_count_ratio)
    )
    max_subj = (
        2.0
        if max_subject_condition_ratio is _GP3_QC_R_UNSET
        else float(max_subject_condition_ratio)
    )
    require_all = (
        True
        if require_all_conditions_per_subject is _GP3_QC_R_UNSET
        else bool(require_all_conditions_per_subject)
    )
    if min_cond <= 0 or min_cell <= 0 or max_cond <= 0 or max_subj <= 0:
        raise ValueError("post-exclusion thresholds must be positive")
    df = df.loc[
        df[subject_col].notna()
        & df[condition_col].notna()
        & df[subject_col].astype(str).ne("")
        & df[condition_col].astype(str).ne("")
    ].copy()
    if df.empty:
        raise ValueError("subject_col and condition_col must define at least one usable row")
    if all(x is None for x in (retained_col, include_col, exclude_col_r, status_col)):
        flags = pd.Series(True, index=df.index, dtype="boolean")
    else:
        flags = pd.Series(pd.NA, index=df.index, dtype="boolean")
    if retained_col is not None:
        flags = _gp3_exclusion_r_bool(df[retained_col], "retained_col")
        flags.index = df.index
    if include_col is not None:
        flags = _gp3_exclusion_r_bool(df[include_col], "include_col")
        flags.index = df.index
    if exclude_col_r is not None:
        ex = _gp3_exclusion_r_bool(df[exclude_col_r], "exclude_col")
        ex.index = df.index
        flags.loc[ex.notna()] = (~ex.loc[ex.notna()]).to_numpy()
    if status_col is not None:
        text = df[status_col].astype("string").str.lower()
        st = pd.Series(pd.NA, index=df.index, dtype="boolean")
        st.loc[text.isin({str(x).lower() for x in included_values})] = True
        st.loc[text.isin({str(x).lower() for x in excluded_values})] = False
        flags.loc[st.notna()] = st.loc[st.notna()].to_numpy()
    units = _gp3_exclusion_r_units(
        df, flags, subject_col, condition_col, unit_cols, status_name="post_exclusion_unit_status"
    )
    observed = sorted(df[condition_col].astype(str).unique())
    conditions = (
        observed
        if expected_conditions is _GP3_QC_R_UNSET or expected_conditions is None
        else list(expected_conditions)
    )
    subjects = sorted(units[subject_col].astype(str).unique())
    rows = []
    for subj in subjects:
        for cond in conditions:
            g = units.loc[
                units[subject_col].astype(str).eq(str(subj))
                & units[condition_col].astype(str).eq(str(cond))
            ]
            total = len(g)
            nr = int(g["retained"].sum()) if total else 0
            status = (
                "missing_retained_condition"
                if nr == 0
                else ("too_few_retained_units" if nr < min_cell else "ok")
            )
            rows.append(
                {
                    subject_col: subj,
                    condition_col: cond,
                    "n_total_units": total,
                    "n_retained_units": nr,
                    "retained_prop": nr / total if total else np.nan,
                    "post_exclusion_cell_status": status,
                }
            )
    cells = pd.DataFrame(rows)
    cond_rows = []
    for cond, g in cells.groupby(condition_col, sort=True):
        total = int(g["n_total_units"].sum())
        nr = int(g["n_retained_units"].sum())
        cond_rows.append(
            {
                condition_col: cond,
                "n_subject_cells": len(g),
                "n_subjects_with_retained": int((g["n_retained_units"] > 0).sum()),
                "n_subjects_missing_retained": int((g["n_retained_units"] == 0).sum()),
                "total_units": total,
                "total_retained_units": nr,
                "retained_prop": nr / total if total else np.nan,
                "min_retained_units_per_subject": int(g["n_retained_units"].min()),
                "max_retained_units_per_subject": int(g["n_retained_units"].max()),
                "mean_retained_units_per_subject": float(g["n_retained_units"].mean()),
                "post_exclusion_condition_status": "too_few_retained_units"
                if nr < min_cond
                else "ok",
            }
        )
    condition_summary = pd.DataFrame(cond_rows)
    subj_rows = []
    for subj, g in cells.groupby(subject_col, sort=True):
        counts = g["n_retained_units"].to_numpy(int)
        nonzero = counts[counts > 0]
        missing = int((counts == 0).sum())
        low = int((g["post_exclusion_cell_status"] == "too_few_retained_units").sum())
        ratio = np.nan if len(nonzero) <= 1 else float(nonzero.max() / nonzero.min())
        status = (
            "missing_retained_condition"
            if require_all and missing > 0
            else (
                "too_few_retained_units"
                if low > 0
                else (
                    "retained_condition_imbalance"
                    if np.isfinite(ratio) and ratio > max_subj
                    else "ok"
                )
            )
        )
        subj_rows.append(
            {
                subject_col: subj,
                "n_conditions_expected": len(conditions),
                "n_conditions_with_retained": int((counts > 0).sum()),
                "total_retained_units": int(counts.sum()),
                "min_retained_units_per_condition": int(nonzero.min()) if len(nonzero) else np.nan,
                "max_retained_units_per_condition": int(nonzero.max()) if len(nonzero) else np.nan,
                "retained_condition_ratio": ratio,
                "n_missing_retained_conditions": missing,
                "n_low_retained_conditions": low,
                "post_exclusion_subject_status": status,
            }
        )
    subject_summary = pd.DataFrame(subj_rows)
    flagged_cells = cells.loc[cells["post_exclusion_cell_status"].ne("ok")].copy()
    flagged_subjects = subject_summary.loc[
        subject_summary["post_exclusion_subject_status"].ne("ok")
    ].copy()
    count_ratio = _gp3_exclusion_r_ratio(
        condition_summary["total_retained_units"], zero_returns_one=False
    )
    ratio_status = (
        "condition_count_imbalance"
        if np.isinf(count_ratio) or (np.isfinite(count_ratio) and count_ratio > max_cond)
        else "ok"
    )
    problem = int(
        units["post_exclusion_unit_status"].isin(["conflicting_flags", "unclear_status"]).sum()
    )
    retained_n = int(units["retained"].sum())
    flagged_cond = int(condition_summary["post_exclusion_condition_status"].ne("ok").sum())
    review = (
        problem > 0
        or len(flagged_cells) > 0
        or len(flagged_subjects) > 0
        or flagged_cond > 0
        or ratio_status != "ok"
    )
    overview = pd.DataFrame(
        [
            {
                "n_rows": len(df),
                "n_units": len(units),
                "n_retained_units": retained_n,
                "n_excluded_units": len(units) - retained_n,
                "retained_prop": retained_n / len(units),
                "n_subjects": len(subjects),
                "n_conditions": len(conditions),
                "n_problem_units": problem,
                "n_flagged_cells": len(flagged_cells),
                "n_flagged_subjects": len(flagged_subjects),
                "n_flagged_conditions": flagged_cond,
                "condition_count_ratio": count_ratio,
                "condition_ratio_status": ratio_status,
                "post_exclusion_balance_status": "review" if review else "ok",
            }
        ]
    )
    settings = pd.DataFrame(
        {
            "setting": [
                "subject_col",
                "condition_col",
                "unit_cols",
                "retained_col",
                "include_col",
                "exclude_col",
                "status_col",
                "expected_conditions",
                "included_values",
                "excluded_values",
                "min_retained_units_per_condition",
                "min_retained_units_per_subject_condition",
                "max_condition_count_ratio",
                "max_subject_condition_ratio",
                "require_all_conditions_per_subject",
            ],
            "value": [
                subject_col,
                condition_col,
                ", ".join(unit_cols),
                retained_col,
                include_col,
                exclude_col_r,
                status_col,
                None if expected_conditions is _GP3_QC_R_UNSET else ", ".join(conditions),
                ", ".join(included_values),
                ", ".join(excluded_values),
                str(min_cond),
                str(min_cell),
                str(max_cond),
                str(max_subj),
                str(require_all),
            ],
        }
    )
    return {
        "overview": overview,
        "unit_flow": units,
        "cell_summary": cells,
        "condition_summary": condition_summary,
        "subject_summary": subject_summary,
        "flagged_cells": flagged_cells,
        "flagged_subjects": flagged_subjects,
        "settings": settings,
    }


def _gp3_exclusion_r_bool(values, arg):
    ser = pd.Series(values)
    out = pd.Series(pd.NA, index=ser.index, dtype="boolean")
    if pd.api.types.is_bool_dtype(ser.dtype):
        return ser.astype("boolean")
    if pd.api.types.is_numeric_dtype(ser.dtype):
        num = pd.to_numeric(ser, errors="coerce")
        bad = num.notna() & ~num.isin([0, 1])
        if bad.any():
            raise ValueError(f"{arg} numeric values must be 0, 1, or missing")
        out.loc[num.eq(1)] = True
        out.loc[num.eq(0)] = False
        return out
    text = ser.astype("string").str.strip().str.lower()
    true_values = {
        "true",
        "t",
        "yes",
        "y",
        "1",
        "included",
        "include",
        "keep",
        "kept",
        "retained",
        "ok",
        "ready",
    }
    false_values = {
        "false",
        "f",
        "no",
        "n",
        "0",
        "excluded",
        "exclude",
        "drop",
        "dropped",
        "removed",
        "fail",
        "failed",
        "not_ready",
        "review",
    }
    out.loc[text.isin(true_values)] = True
    out.loc[text.isin(false_values)] = False
    bad = text.notna() & text.ne("") & out.isna()
    if bad.any():
        raise ValueError(
            f"{arg} character values must be interpretable as inclusion/exclusion flags"
        )
    return out


def _gp3_exclusion_r_aliases(df):
    out = df.copy()
    if "MEDIA_ID" in out and "media_id" not in out:
        out["media_id"] = out["MEDIA_ID"]
    if "USER_FILE" in out and "subject" not in out:
        out["subject"] = out["USER_FILE"]
    return out


def _gp3_exclusion_r_col(df, col, arg, optional=False):
    if col is None and optional:
        return None
    if not isinstance(col, str) or not col:
        raise ValueError(f"{arg} must be a non-empty string")
    alias = {"MEDIA_ID": "media_id", "USER_FILE": "subject"}.get(col, col)
    if alias not in df:
        raise KeyError(f"{arg} must be present in data")
    return alias


def _gp3_exclusion_r_units(
    df,
    flags,
    subject_col,
    condition_col,
    unit_cols,
    reason=None,
    status_name="exclusion_flow_status",
):
    work = df.copy()
    work["__flag"] = flags.to_numpy()
    if reason is not None:
        work["__reason"] = reason.to_numpy()
    ids = []
    for col in [subject_col, condition_col, *unit_cols]:
        if col is not None and col in work and col not in ids:
            ids.append(col)
    if not ids:
        work["__unit_id"] = np.arange(len(work))
        ids = ["__unit_id"]
    rows = []
    for _, part in work.groupby(ids, dropna=False, sort=True):
        vals = part["__flag"].astype("boolean")
        any_true = bool(vals.fillna(False).any())
        any_false = bool((~vals.fillna(True)).any())
        all_unknown = bool(vals.isna().all())
        if all_unknown:
            status = "unclear_status"
        elif any_true and any_false:
            status = "conflicting_flags"
        elif any_false:
            status = "excluded"
        elif any_true:
            status = "retained"
        else:
            status = "unclear_status"
        row = {c: part.iloc[0][c] for c in ids if c != "__unit_id"}
        row.update(
            {"n_source_rows": len(part), "retained": status == "retained", status_name: status}
        )
        if reason is not None:
            reasons = sorted({str(x) for x in part["__reason"].dropna() if str(x)})
            if not reasons:
                reasons = [
                    {
                        "retained": "retained",
                        "conflicting_flags": "conflicting_flags",
                        "unclear_status": "unclear_status",
                    }.get(status, "excluded_unspecified")
                ]
            row["exclusion_reason"] = "; ".join(reasons)
        rows.append(row)
    return pd.DataFrame(rows)


def _gp3_exclusion_r_ratio(values, zero_returns_one=False):
    vals = pd.to_numeric(pd.Series(values), errors="coerce").dropna().to_numpy(float)
    if len(vals) <= 1:
        return np.nan
    if np.all(vals == 0):
        return 1.0 if zero_returns_one else np.nan
    positive = vals[vals > 0]
    if len(positive) == 0:
        return 1.0 if zero_returns_one else np.nan
    if np.any(vals == 0):
        return np.inf
    return float(np.max(positive) / np.min(positive))


def audit_gazepoint_exclusion_flow(
    data,
    stages: list[str] | None = None,
    *,
    subject_col=_GP3_QC_R_UNSET,
    condition_col=_GP3_QC_R_UNSET,
    unit_cols=_GP3_QC_R_UNSET,
    include_col=_GP3_QC_R_UNSET,
    exclude_col=_GP3_QC_R_UNSET,
    status_col=_GP3_QC_R_UNSET,
    reason_col=_GP3_QC_R_UNSET,
    included_values=_GP3_QC_R_UNSET,
    excluded_values=_GP3_QC_R_UNSET,
    min_retained_prop=_GP3_QC_R_UNSET,
    max_condition_exclusion_ratio=_GP3_QC_R_UNSET,
):
    r_mode = any(
        v is not _GP3_QC_R_UNSET
        for v in (
            subject_col,
            condition_col,
            unit_cols,
            include_col,
            exclude_col,
            status_col,
            reason_col,
            included_values,
            excluded_values,
            min_retained_prop,
            max_condition_exclusion_ratio,
        )
    )
    if not r_mode:
        df = ensure_dataframe(data, copy=False)
        if stages is None:
            stages = [
                c for c in df.columns if c.lower().startswith(("exclude", "flag_", "excluded"))
            ]
        rows = [{"stage": "input", "n": len(df)}]
        current = pd.Series(True, index=df.index)
        for stage in stages:
            current &= ~as_bool(df[stage])
            rows.append({"stage": stage, "n": int(current.sum())})
        return pd.DataFrame(rows)
    df = _gp3_exclusion_r_aliases(ensure_dataframe(data))
    if df.empty:
        raise ValueError("data must contain at least one row")
    subject_col = _gp3_exclusion_r_col(
        df, "subject" if subject_col is _GP3_QC_R_UNSET else subject_col, "subject_col"
    )
    condition_col = _gp3_exclusion_r_col(
        df,
        None if condition_col is _GP3_QC_R_UNSET else condition_col,
        "condition_col",
        optional=True,
    )
    raw_units = (
        ("media_id", "trial_global")
        if unit_cols is _GP3_QC_R_UNSET
        else (
            () if unit_cols is None else ([unit_cols] if isinstance(unit_cols, str) else unit_cols)
        )
    )
    unit_cols = [
        {"MEDIA_ID": "media_id", "USER_FILE": "subject"}.get(c, c)
        for c in raw_units
        if {"MEDIA_ID": "media_id", "USER_FILE": "subject"}.get(c, c) in df
    ]
    include_col = _gp3_exclusion_r_col(
        df, None if include_col is _GP3_QC_R_UNSET else include_col, "include_col", optional=True
    )
    exclude_col = _gp3_exclusion_r_col(
        df, None if exclude_col is _GP3_QC_R_UNSET else exclude_col, "exclude_col", optional=True
    )
    status_col = _gp3_exclusion_r_col(
        df, None if status_col is _GP3_QC_R_UNSET else status_col, "status_col", optional=True
    )
    reason_col = _gp3_exclusion_r_col(
        df, None if reason_col is _GP3_QC_R_UNSET else reason_col, "reason_col", optional=True
    )
    if include_col is None and exclude_col is None and status_col is None:
        raise ValueError("One of include_col, exclude_col, or status_col must be supplied")
    included_values = (
        ["included", "include", "kept", "keep", "retained", "ok", "ready", "complete", "completed"]
        if included_values is _GP3_QC_R_UNSET
        else list(included_values)
    )
    excluded_values = (
        [
            "excluded",
            "exclude",
            "drop",
            "dropped",
            "removed",
            "fail",
            "failed",
            "not_ready",
            "review",
            "invalid",
        ]
        if excluded_values is _GP3_QC_R_UNSET
        else list(excluded_values)
    )
    min_retained_prop = 0.70 if min_retained_prop is _GP3_QC_R_UNSET else float(min_retained_prop)
    max_condition_exclusion_ratio = (
        2.0
        if max_condition_exclusion_ratio is _GP3_QC_R_UNSET
        else float(max_condition_exclusion_ratio)
    )
    if (
        not 0 < min_retained_prop <= 1
        or not np.isfinite(max_condition_exclusion_ratio)
        or max_condition_exclusion_ratio <= 0
    ):
        raise ValueError("retention thresholds must be positive and min_retained_prop <= 1")
    flags = pd.Series(pd.NA, index=df.index, dtype="boolean")
    if include_col is not None:
        flags = _gp3_exclusion_r_bool(df[include_col], "include_col")
    if exclude_col is not None:
        ex = _gp3_exclusion_r_bool(df[exclude_col], "exclude_col")
        flags.loc[ex.notna()] = (~ex.loc[ex.notna()]).to_numpy()
    if status_col is not None:
        text = df[status_col].astype("string").str.lower()
        st = pd.Series(pd.NA, index=df.index, dtype="boolean")
        st.loc[text.isin({str(x).lower() for x in included_values})] = True
        st.loc[text.isin({str(x).lower() for x in excluded_values})] = False
        flags.loc[st.notna()] = st.loc[st.notna()].to_numpy()
    reason = (
        pd.Series(pd.NA, index=df.index, dtype="string")
        if reason_col is None
        else df[reason_col].astype("string").replace("", pd.NA)
    )
    units = _gp3_exclusion_r_units(df, flags, subject_col, condition_col, unit_cols, reason=reason)
    excluded_units = units.loc[units["exclusion_flow_status"].ne("retained")].copy()
    reasons = []
    for value in excluded_units.get("exclusion_reason", pd.Series(dtype=str)).dropna():
        reasons.extend([x for x in str(value).split("; ") if x])
    reason_summary = (
        pd.Series(reasons, dtype="string")
        .value_counts()
        .rename_axis("exclusion_reason")
        .reset_index(name="n_units")
        if reasons
        else pd.DataFrame(columns=["exclusion_reason", "n_units"])
    )
    reason_summary["reason_prop"] = (
        reason_summary["n_units"] / len(excluded_units)
        if len(excluded_units)
        else pd.Series(dtype=float)
    )

    def summary(col, status_col_name):
        if col is None:
            return pd.DataFrame(
                columns=[
                    "condition" if status_col_name.startswith("condition") else "subject",
                    "n_units",
                    "n_retained_units",
                    "n_excluded_units",
                    "retained_prop",
                    "excluded_prop",
                    status_col_name,
                ]
            )
        rows = []
        for key, g in units.groupby(col, dropna=False, sort=True):
            nr = int(g["retained"].sum())
            prop = nr / len(g)
            rows.append(
                {
                    col: key,
                    "n_units": len(g),
                    "n_retained_units": nr,
                    "n_excluded_units": len(g) - nr,
                    "retained_prop": prop,
                    "excluded_prop": 1 - prop,
                    status_col_name: "low_retention" if prop < min_retained_prop else "ok",
                }
            )
        return pd.DataFrame(rows)

    condition_summary = summary(condition_col, "condition_exclusion_status")
    subject_summary = summary(subject_col, "subject_exclusion_status")
    ratio = _gp3_exclusion_r_ratio(
        condition_summary.get("excluded_prop", []), zero_returns_one=True
    )
    retained_n = int(units["retained"].sum())
    retained_prop = retained_n / len(units)
    review = (
        units["exclusion_flow_status"].isin(["conflicting_flags", "unclear_status"]).any()
        or retained_prop < min_retained_prop
        or (np.isfinite(ratio) and ratio > max_condition_exclusion_ratio)
        or np.isinf(ratio)
    )
    overview = pd.DataFrame(
        [
            {
                "n_rows": len(df),
                "n_units": len(units),
                "n_subjects": units[subject_col].nunique(dropna=False),
                "n_retained_units": retained_n,
                "n_excluded_units": len(units) - retained_n,
                "retained_prop": retained_prop,
                "excluded_prop": 1 - retained_prop,
                "n_flagged_units": len(excluded_units),
                "n_exclusion_reasons": len(reason_summary),
                "condition_exclusion_ratio": ratio,
                "exclusion_flow_status": "review" if review else "ok",
            }
        ]
    )
    settings = pd.DataFrame(
        {
            "setting": [
                "subject_col",
                "condition_col",
                "unit_cols",
                "include_col",
                "exclude_col",
                "status_col",
                "reason_col",
                "included_values",
                "excluded_values",
                "min_retained_prop",
                "max_condition_exclusion_ratio",
            ],
            "value": [
                subject_col,
                condition_col,
                ", ".join(unit_cols),
                include_col,
                exclude_col,
                status_col,
                reason_col,
                ", ".join(included_values),
                ", ".join(excluded_values),
                str(min_retained_prop),
                str(max_condition_exclusion_ratio),
            ],
        }
    )
    return {
        "overview": overview,
        "unit_flow": units,
        "reason_summary": reason_summary,
        "condition_summary": condition_summary,
        "subject_summary": subject_summary,
        "flagged_units": excluded_units,
        "settings": settings,
    }


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


def check_gazepoint_real_data_readiness(
    data,
    *,
    analysis_type=_GP3_QC_R_UNSET,
    participant_col=_GP3_QC_R_UNSET,
    trial_col=_GP3_QC_R_UNSET,
    time_col=_GP3_QC_R_UNSET,
    condition_col=_GP3_QC_R_UNSET,
    stimulus_col=_GP3_QC_R_UNSET,
    aoi_col=_GP3_QC_R_UNSET,
    pupil_col=_GP3_QC_R_UNSET,
    gaze_x_col=_GP3_QC_R_UNSET,
    gaze_y_col=_GP3_QC_R_UNSET,
    tracking_valid_col=_GP3_QC_R_UNSET,
    required_cols=_GP3_QC_R_UNSET,
    audit_objects=_GP3_QC_R_UNSET,
    min_rows=_GP3_QC_R_UNSET,
    min_participants=_GP3_QC_R_UNSET,
    min_trials=_GP3_QC_R_UNSET,
    max_missing_pupil_prop=_GP3_QC_R_UNSET,
    max_missing_gaze_prop=_GP3_QC_R_UNSET,
    max_condition_imbalance_ratio=_GP3_QC_R_UNSET,
    name=_GP3_QC_R_UNSET,
) -> dict[str, Any]:
    """Check real-data readiness using legacy or R v2.3.0 gate semantics."""
    r_mode = any(
        value is not _GP3_QC_R_UNSET
        for value in (
            analysis_type,
            participant_col,
            trial_col,
            time_col,
            condition_col,
            stimulus_col,
            aoi_col,
            pupil_col,
            gaze_x_col,
            gaze_y_col,
            tracking_valid_col,
            required_cols,
            audit_objects,
            min_rows,
            min_participants,
            min_trials,
            max_missing_pupil_prop,
            max_missing_gaze_prop,
            max_condition_imbalance_ratio,
            name,
        )
    )
    if not r_mode:
        validation = validate_gazepoint_master(
            as_gazepoint_master(data), required=("subject", "time")
        )
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

    df = ensure_dataframe(data, copy=False)
    analysis_type = "general" if analysis_type is _GP3_QC_R_UNSET else analysis_type
    if analysis_type not in {"general", "pupil", "aoi", "combined"}:
        raise ValueError("analysis_type must be general, pupil, aoi, or combined")
    defaults = {
        "participant_col": None,
        "trial_col": None,
        "time_col": None,
        "condition_col": None,
        "stimulus_col": None,
        "aoi_col": None,
        "pupil_col": None,
        "gaze_x_col": None,
        "gaze_y_col": None,
        "tracking_valid_col": None,
        "required_cols": [],
        "audit_objects": None,
        "min_rows": 1,
        "min_participants": 1,
        "min_trials": 1,
        "max_missing_pupil_prop": 0.40,
        "max_missing_gaze_prop": 0.40,
        "max_condition_imbalance_ratio": 3.0,
        "name": "gazepoint_real_data_readiness_gate",
    }
    participant_col = (
        defaults["participant_col"] if participant_col is _GP3_QC_R_UNSET else participant_col
    )
    trial_col = defaults["trial_col"] if trial_col is _GP3_QC_R_UNSET else trial_col
    time_col = defaults["time_col"] if time_col is _GP3_QC_R_UNSET else time_col
    condition_col = defaults["condition_col"] if condition_col is _GP3_QC_R_UNSET else condition_col
    stimulus_col = defaults["stimulus_col"] if stimulus_col is _GP3_QC_R_UNSET else stimulus_col
    aoi_col = defaults["aoi_col"] if aoi_col is _GP3_QC_R_UNSET else aoi_col
    pupil_col = defaults["pupil_col"] if pupil_col is _GP3_QC_R_UNSET else pupil_col
    gaze_x_col = defaults["gaze_x_col"] if gaze_x_col is _GP3_QC_R_UNSET else gaze_x_col
    gaze_y_col = defaults["gaze_y_col"] if gaze_y_col is _GP3_QC_R_UNSET else gaze_y_col
    tracking_valid_col = (
        defaults["tracking_valid_col"]
        if tracking_valid_col is _GP3_QC_R_UNSET
        else tracking_valid_col
    )
    required_cols = (
        defaults["required_cols"]
        if required_cols is _GP3_QC_R_UNSET
        else _gp3_qc_r_list(required_cols)
    )
    audit_objects = defaults["audit_objects"] if audit_objects is _GP3_QC_R_UNSET else audit_objects
    min_rows = defaults["min_rows"] if min_rows is _GP3_QC_R_UNSET else int(min_rows)
    min_participants = (
        defaults["min_participants"]
        if min_participants is _GP3_QC_R_UNSET
        else int(min_participants)
    )
    min_trials = defaults["min_trials"] if min_trials is _GP3_QC_R_UNSET else int(min_trials)
    max_missing_pupil_prop = (
        defaults["max_missing_pupil_prop"]
        if max_missing_pupil_prop is _GP3_QC_R_UNSET
        else float(max_missing_pupil_prop)
    )
    max_missing_gaze_prop = (
        defaults["max_missing_gaze_prop"]
        if max_missing_gaze_prop is _GP3_QC_R_UNSET
        else float(max_missing_gaze_prop)
    )
    max_condition_imbalance_ratio = (
        defaults["max_condition_imbalance_ratio"]
        if max_condition_imbalance_ratio is _GP3_QC_R_UNSET
        else float(max_condition_imbalance_ratio)
    )
    name = defaults["name"] if name is _GP3_QC_R_UNSET else name

    names_data = set(df.columns)

    def resolve(current, candidates):
        if current is not None:
            if current not in names_data:
                raise KeyError(f"Missing required column: {current}")
            return current
        return next((c for c in candidates if c in names_data), None)

    participant_col = resolve(
        participant_col,
        ["subject", "participant", "participant_id", "pID", "USER_FILE", "user", "recording_id"],
    )
    trial_col = resolve(
        trial_col,
        ["trial_global", "trial", "trial_id", "TRIAL_INDEX", "item_id", "media_id", "MEDIA_ID"],
    )
    time_col = resolve(
        time_col, ["time", "time_ms", "timestamp", "TIMESTAMP", "TIME", "sample_index", "CNT"]
    )
    condition_col = resolve(
        condition_col, ["condition", "CONDITION", "group", "GROUP", "trial_type"]
    )
    stimulus_col = resolve(
        stimulus_col,
        [
            "stimulus",
            "stimulus_id",
            "stimulus_file",
            "image_file",
            "image",
            "media",
            "media_id",
            "MEDIA_ID",
        ],
    )
    aoi_col = resolve(
        aoi_col,
        [
            "aoi",
            "AOI",
            "aoi_label",
            "AOI_LABEL",
            "aoi_name",
            "AOI_NAME",
            "CURRENT_FIX_INTEREST_AREA_LABEL",
        ],
    )
    pupil_col = resolve(
        pupil_col,
        [
            "pupil_bc_processed",
            "pupil_smoothed",
            "pupil_interpolated",
            "pupil_clean",
            "pupil_for_preprocessing",
            "pupil_raw",
            "mean_pupil",
            "pupil",
            "LPD",
            "RPD",
        ],
    )
    gaze_x_col = resolve(gaze_x_col, ["gaze_x", "x", "X", "FPOGX", "LPOGX", "RPOGX", "POGX"])
    gaze_y_col = resolve(gaze_y_col, ["gaze_y", "y", "Y", "FPOGY", "LPOGY", "RPOGY", "POGY"])
    tracking_valid_col = resolve(
        tracking_valid_col,
        ["tracking_valid", "valid_gaze", "is_valid", "valid", "FPOGV", "LPOGV", "RPOGV", "POGV"],
    )

    detected = {
        "participant_col": participant_col,
        "trial_col": trial_col,
        "time_col": time_col,
        "condition_col": condition_col,
        "stimulus_col": stimulus_col,
        "aoi_col": aoi_col,
        "pupil_col": pupil_col,
        "gaze_x_col": gaze_x_col,
        "gaze_y_col": gaze_y_col,
        "tracking_valid_col": tracking_valid_col,
    }
    detected_columns = pd.DataFrame(
        [
            {"role": role, "column": col if col is not None else pd.NA, "detected": col is not None}
            for role, col in detected.items()
        ]
    )

    checks = []

    def add(check_id, area, status, severity, message, observed=np.nan, threshold=np.nan):
        checks.append(
            {
                "check_id": check_id,
                "check_area": area,
                "status": status,
                "severity": severity,
                "message": message,
                "observed": observed,
                "threshold": threshold,
            }
        )

    add(
        "data_non_empty",
        "structure",
        "pass" if len(df) else "fail",
        "blocking",
        "Data contain at least one row." if len(df) else "Data contain no rows.",
        len(df),
        1,
    )
    add(
        "minimum_rows",
        "structure",
        "pass" if len(df) >= min_rows else "fail",
        "blocking",
        f"Rows available: {len(df)}. Minimum required: {min_rows}.",
        len(df),
        min_rows,
    )
    required_roles = {
        "general": {
            "participant_col": True,
            "trial_col": True,
            "time_col": False,
            "aoi_col": False,
            "pupil_col": False,
        },
        "pupil": {
            "participant_col": True,
            "trial_col": True,
            "time_col": True,
            "aoi_col": False,
            "pupil_col": True,
        },
        "aoi": {
            "participant_col": True,
            "trial_col": True,
            "time_col": False,
            "aoi_col": True,
            "pupil_col": False,
        },
        "combined": {
            "participant_col": True,
            "trial_col": True,
            "time_col": True,
            "aoi_col": True,
            "pupil_col": True,
        },
    }[analysis_type]
    for role, required_role in required_roles.items():
        col = detected[role]
        add(
            f"required_{role}",
            "required_columns",
            "pass" if col is not None else ("fail" if required_role else "info"),
            "blocking" if required_role else "informational",
            f"Detected required role `{role}` as column `{col}`."
            if col is not None
            else (
                f"Required role `{role}` was not detected."
                if required_role
                else f"Optional role `{role}` was not detected."
            ),
            1 if col is not None else 0,
            1 if required_role else 0,
        )
    missing_user = [c for c in required_cols if c not in df.columns]
    add(
        "user_required_columns",
        "required_columns",
        "pass" if not missing_user else "fail",
        "blocking",
        "All user-specified required columns are present."
        if not missing_user
        else "Missing user-specified required columns: " + ", ".join(missing_user),
        len(required_cols) - len(missing_user),
        len(required_cols),
    )

    n_participants = int(df[participant_col].dropna().nunique()) if participant_col else np.nan
    if participant_col and trial_col:
        trial_keys = df[[participant_col, trial_col]].astype("string").agg("||".join, axis=1)
        n_trials = int(trial_keys.dropna().nunique())
    else:
        n_trials = np.nan
    n_conditions = int(df[condition_col].dropna().nunique()) if condition_col else np.nan
    n_stimuli = int(df[stimulus_col].dropna().nunique()) if stimulus_col else np.nan
    add(
        "minimum_participants",
        "sample_structure",
        "pass" if np.isfinite(n_participants) and n_participants >= min_participants else "fail",
        "blocking",
        f"Participants available: {n_participants}. Minimum required: {min_participants}.",
        n_participants,
        min_participants,
    )
    add(
        "minimum_trials",
        "sample_structure",
        "pass" if np.isfinite(n_trials) and n_trials >= min_trials else "fail",
        "blocking",
        f"Participant-trial units available: {n_trials}. Minimum required: {min_trials}.",
        n_trials,
        min_trials,
    )

    if time_col:
        times = pd.to_numeric(df[time_col], errors="coerce")
        bad_time = float((times.isna() | ~np.isfinite(times)).mean())
        add(
            "finite_time_values",
            "time_structure",
            "pass" if bad_time == 0 else "fail",
            "blocking",
            f"Proportion of missing/non-finite time values: {bad_time:.4f}.",
            bad_time,
            0,
        )
        if participant_col and trial_col:
            key = df[[participant_col, trial_col, time_col]].astype("string")
            duplicate_prop = float(key.duplicated().mean())
            add(
                "duplicate_participant_trial_time",
                "time_structure",
                "pass" if duplicate_prop == 0 else "warn",
                "warning",
                f"Proportion of duplicated participant-trial-time keys: {duplicate_prop:.4f}.",
                duplicate_prop,
                0,
            )

    if pupil_col:
        pupil_values = pd.to_numeric(df[pupil_col], errors="coerce")
        prop_missing_pupil = float((pupil_values.isna() | ~np.isfinite(pupil_values)).mean())
        pupil_required = analysis_type in {"pupil", "combined"}
        status = (
            "pass"
            if prop_missing_pupil <= max_missing_pupil_prop
            else ("fail" if pupil_required else "warn")
        )
        add(
            "pupil_missingness",
            "signal_quality",
            status,
            "blocking" if pupil_required else "warning",
            f"Pupil missingness/non-finite proportion: {prop_missing_pupil:.4f}. Threshold: {max_missing_pupil_prop}.",
            prop_missing_pupil,
            max_missing_pupil_prop,
        )

    if gaze_x_col and gaze_y_col:
        gx = pd.to_numeric(df[gaze_x_col], errors="coerce")
        gy = pd.to_numeric(df[gaze_y_col], errors="coerce")
        missing_gaze = float((gx.isna() | gy.isna() | ~np.isfinite(gx) | ~np.isfinite(gy)).mean())
        add(
            "gaze_coordinate_missingness",
            "signal_quality",
            "pass" if missing_gaze <= max_missing_gaze_prop else "warn",
            "warning",
            f"Gaze-coordinate missingness/non-finite proportion: {missing_gaze:.4f}. Threshold: {max_missing_gaze_prop}.",
            missing_gaze,
            max_missing_gaze_prop,
        )
    elif bool(gaze_x_col) ^ bool(gaze_y_col):
        add(
            "paired_gaze_coordinates",
            "signal_quality",
            "warn",
            "warning",
            "Only one gaze-coordinate column was detected; both x and y are preferable for gaze-quality checks.",
            1,
            2,
        )

    if tracking_valid_col:
        valid = _gp3_qc_r_as_bool_series(df[tracking_valid_col], df.index)
        prop_invalid = float((~valid.fillna(False)).mean())
        add(
            "tracking_validity",
            "signal_quality",
            "pass" if prop_invalid <= max_missing_gaze_prop else "warn",
            "warning",
            f"Tracking-invalid proportion: {prop_invalid:.4f}. Threshold: {max_missing_gaze_prop}.",
            prop_invalid,
            max_missing_gaze_prop,
        )

    if condition_col:
        condition_summary = (
            df[condition_col]
            .astype("string")
            .dropna()
            .value_counts()
            .rename_axis("condition")
            .reset_index(name="n_rows")
        )
        condition_summary["proportion"] = (
            condition_summary["n_rows"] / condition_summary["n_rows"].sum()
        )
        if len(condition_summary) <= 1:
            add(
                "condition_count",
                "design_balance",
                "warn",
                "warning",
                "Only one condition/group was detected.",
                len(condition_summary),
                2,
            )
        else:
            imbalance = float(condition_summary["n_rows"].max() / condition_summary["n_rows"].min())
            add(
                "condition_imbalance",
                "design_balance",
                "pass" if imbalance <= max_condition_imbalance_ratio else "warn",
                "warning",
                f"Condition row-count imbalance ratio: {imbalance:.4f}. Threshold: {max_condition_imbalance_ratio}.",
                imbalance,
                max_condition_imbalance_ratio,
            )
    else:
        condition_summary = pd.DataFrame(columns=["condition", "n_rows", "proportion"])
        add(
            "condition_detected",
            "design_balance",
            "info",
            "informational",
            "No condition/group column was detected.",
            0,
            0,
        )

    if audit_objects is not None:
        objs = audit_objects if isinstance(audit_objects, (list, tuple)) else [audit_objects]
        for i, obj in enumerate(objs, 1):
            overview = (
                obj.get("overview")
                if isinstance(obj, dict)
                else (obj if isinstance(obj, pd.DataFrame) else None)
            )
            status = "info"
            if isinstance(overview, pd.DataFrame) and len(overview):
                values = " ".join(str(v).lower() for v in overview.iloc[0].tolist())
                if any(x in values for x in ["fail", "not ready", "false"]):
                    status = "fail"
                elif any(x in values for x in ["warn", "review"]):
                    status = "warn"
                elif any(x in values for x in ["pass", "ready", "ok", "true"]):
                    status = "pass"
            add(
                f"audit_object_{i}",
                "upstream_audits",
                status,
                "blocking"
                if status == "fail"
                else ("warning" if status == "warn" else "informational"),
                f"Upstream audit object {i} interpreted as status `{status}`.",
            )

    checks_df = pd.DataFrame(checks)
    order = pd.Categorical(checks_df["status"], ["fail", "warn", "pass", "info"], ordered=True)
    checks_df = (
        checks_df.assign(_order=order)
        .sort_values(["_order", "check_area", "check_id"])
        .drop(columns="_order")
        .reset_index(drop=True)
    )
    counts = checks_df["status"].value_counts()
    n_fail = int(counts.get("fail", 0))
    n_warn = int(counts.get("warn", 0))
    n_pass = int(counts.get("pass", 0))
    n_info = int(counts.get("info", 0))
    readiness_status = "fail" if n_fail else ("warn" if n_warn else "pass")
    decision_message = (
        f"Not ready for real-data analysis: {n_fail} blocking issue(s) must be resolved."
        if readiness_status == "fail"
        else f"Conditionally ready: no blocking issues, but {n_warn} warning-level issue(s) should be reviewed."
        if readiness_status == "warn"
        else "Ready for real-data analysis: no blocking or warning-level issues detected."
    )
    gate = pd.DataFrame(
        [
            {
                "object_name": name,
                "analysis_type": analysis_type,
                "readiness_status": readiness_status,
                "ready_for_real_data_analysis": n_fail == 0,
                "n_fail": n_fail,
                "n_warn": n_warn,
                "n_pass": n_pass,
                "n_info": n_info,
                "decision_message": decision_message,
            }
        ]
    )
    data_summary = pd.DataFrame(
        [
            {
                "object_name": name,
                "n_rows": len(df),
                "n_columns": df.shape[1],
                "n_participants": n_participants,
                "n_trial_units": n_trials,
                "n_conditions": n_conditions,
                "n_stimuli": n_stimuli,
                "analysis_type": analysis_type,
            }
        ]
    )
    overview = pd.DataFrame(
        [
            {
                "object_name": name,
                "analysis_type": analysis_type,
                "readiness_status": readiness_status,
                "ready_for_real_data_analysis": n_fail == 0,
                "n_rows": len(df),
                "n_participants": n_participants,
                "n_trial_units": n_trials,
                "n_fail": n_fail,
                "n_warn": n_warn,
                "n_pass": n_pass,
                "n_info": n_info,
            }
        ]
    )
    settings_map = {
        "analysis_type": analysis_type,
        "participant_col": participant_col,
        "trial_col": trial_col,
        "time_col": time_col,
        "condition_col": condition_col,
        "stimulus_col": stimulus_col,
        "aoi_col": aoi_col,
        "pupil_col": pupil_col,
        "gaze_x_col": gaze_x_col,
        "gaze_y_col": gaze_y_col,
        "tracking_valid_col": tracking_valid_col,
        "required_cols": ", ".join(required_cols),
        "min_rows": min_rows,
        "min_participants": min_participants,
        "min_trials": min_trials,
        "max_missing_pupil_prop": max_missing_pupil_prop,
        "max_missing_gaze_prop": max_missing_gaze_prop,
        "max_condition_imbalance_ratio": max_condition_imbalance_ratio,
        "name": name,
    }
    settings = pd.DataFrame(
        {
            "setting": list(settings_map),
            "value": [str(v) if v is not None else pd.NA for v in settings_map.values()],
        }
    )
    return {
        "overview": overview,
        "gate_decision": gate,
        "checks": checks_df,
        "detected_columns": detected_columns,
        "data_summary": data_summary,
        "condition_summary": condition_summary,
        "settings": settings,
        "ready": bool(n_fail == 0),
        "gp3_class": "gp3_real_data_readiness_gate",
    }


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
