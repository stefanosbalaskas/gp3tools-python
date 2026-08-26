"""Pupil preprocessing, blink handling, and binocular reconstruction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import interpolate, ndimage, signal, stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from ._compat import r_aliases
from ._utils import (
    ensure_dataframe,
    finite_numeric,
    infer_column,
    normalize_group_cols,
    robust_mad,
    time_to_seconds,
)


def mean_gazepoint_pupil(
    data,
    left_col=None,
    right_col=None,
    output_col="pupil_mean",
    require_both=False,
    min_eyes=1,
) -> pd.DataFrame:
    """Average binocular pupil values with explicit minimum-eye policy."""
    df = ensure_dataframe(data)
    if min_eyes not in {1, 2}:
        raise ValueError("min_eyes must be 1 or 2")
    left_col = infer_column(df, "left_pupil", left_col, required=True)
    right_col = infer_column(df, "right_pupil", right_col, required=True)
    left = finite_numeric(df[left_col])
    right = finite_numeric(df[right_col])
    required_eyes = 2 if require_both else int(min_eyes)
    available = left.notna().astype(int) + right.notna().astype(int)
    out = pd.concat([left, right], axis=1).mean(axis=1, skipna=True)
    out = out.where(available >= required_eyes)
    df[output_col] = out
    df.attrs["gazepoint_mean_pupil"] = {
        "lp_col": left_col,
        "rp_col": right_col,
        "output_col": output_col,
        "min_eyes": required_eyes,
    }
    return df


def combine_gazepoint_eyes(
    data,
    left_col=None,
    right_col=None,
    output_col="pupil_combined",
    policy="available_eye",
    method=None,
    valid_min=None,
    valid_max=None,
) -> pd.DataFrame:
    """Combine left/right eye values with legacy or R v2.3.0 policies."""
    df = ensure_dataframe(data)
    left_col = infer_column(df, "left_pupil", left_col, required=True)
    right_col = infer_column(df, "right_pupil", right_col, required=True)
    left = finite_numeric(df[left_col]).astype(float)
    right = finite_numeric(df[right_col]).astype(float)

    def bounded(values):
        result = values.copy()
        if valid_min is not None:
            result = result.where(result >= float(valid_min))
        if valid_max is not None:
            result = result.where(result <= float(valid_max))
        return result

    left = bounded(left)
    right = bounded(right)

    if method is not None:
        if method not in {"mean", "left", "right", "prefer_left", "prefer_right", "best"}:
            raise ValueError("Unknown eye-combination method")
        if method == "mean":
            combined = pd.concat([left, right], axis=1).mean(axis=1, skipna=True)
        elif method == "left":
            combined = left
        elif method == "right":
            combined = right
        elif method == "prefer_left":
            combined = left.where(left.notna(), right)
        elif method == "prefer_right":
            combined = right.where(right.notna(), left)
        else:
            if float(left.isna().mean()) <= float(right.isna().mean()):
                combined = left.where(left.notna(), right)
            else:
                combined = right.where(right.notna(), left)
        df[output_col] = combined
        return df

    if policy in {"mean", "available_eye"}:
        df[output_col] = pd.concat([left, right], axis=1).mean(axis=1, skipna=True)
    elif policy == "bilateral_mean":
        df[output_col] = pd.concat([left, right], axis=1).mean(axis=1, skipna=False)
    elif policy == "left_only":
        df[output_col] = left
    elif policy == "right_only":
        df[output_col] = right
    elif policy == "complete_case":
        df[output_col] = ((left + right) / 2).where(left.notna() & right.notna())
    else:
        raise ValueError(f"Unknown eye-combination policy: {policy}")
    df["pupil_eye_source"] = np.select(
        [left.notna() & right.notna(), left.notna(), right.notna()],
        ["both", "left", "right"],
        default="missing",
    )
    return df


def construct_gazepoint_combined_pupil(
    data,
    left_col,
    right_col,
    policy="available_eye",
    prefix="gp3_binocular",
    output_col="pupil_combined",
    status_col="pupil_binocular_status",
    valid_min=None,
    valid_max=None,
    overwrite=False,
):
    """Construct a provenance-aware combined pupil series."""
    if policy == "bilateral_mean":
        policy = "complete_case"
    if policy not in {
        "complete_case",
        "available_eye",
        "reconstructed_mean",
        "left_only",
        "right_only",
    }:
        raise ValueError("Unknown binocular pupil policy")
    frame = ensure_dataframe(data)
    if not overwrite:
        existing = [column for column in (output_col, status_col) if column in frame.columns]
        if existing:
            raise ValueError("output column(s) already exist: " + ", ".join(existing))

    def observed(column):
        values = finite_numeric(frame[column]).astype(float)
        if valid_min is not None:
            values = values.where(values >= float(valid_min))
        if valid_max is not None:
            values = values.where(values <= float(valid_max))
        return values

    if policy == "reconstructed_mean":
        lf = f"{prefix}_left_final"
        rf = f"{prefix}_right_final"
        lr = f"{prefix}_left_reconstructed"
        rr = f"{prefix}_right_reconstructed"
        missing = [column for column in (lf, rf, lr, rr) if column not in frame.columns]
        if missing:
            raise ValueError("reconstruction columns missing: " + ", ".join(missing))
        left = finite_numeric(frame[lf])
        right = finite_numeric(frame[rf])
        left_rec = frame[lr].fillna(False).astype(bool)
        right_rec = frame[rr].fillna(False).astype(bool)
    else:
        left = observed(left_col)
        right = observed(right_col)
        left_rec = pd.Series(False, index=frame.index)
        right_rec = pd.Series(False, index=frame.index)

    left_ok = left.notna()
    right_ok = right.notna()
    value = pd.Series(np.nan, index=frame.index, dtype=float)
    source = pd.Series("unavailable", index=frame.index, dtype=object)

    if policy == "complete_case":
        both = left_ok & right_ok
        value.loc[both] = (left.loc[both] + right.loc[both]) / 2
        source.loc[both] = "bilateral_observed"
    elif policy in {"available_eye", "reconstructed_mean"}:
        both = left_ok & right_ok
        only_left = left_ok & ~right_ok
        only_right = ~left_ok & right_ok
        value.loc[both] = (left.loc[both] + right.loc[both]) / 2
        value.loc[only_left] = left.loc[only_left]
        value.loc[only_right] = right.loc[only_right]
        source.loc[both] = "bilateral_observed"
        source.loc[only_left] = "left_only_observed"
        source.loc[only_right] = "right_only_observed"
        if policy == "reconstructed_mean":
            source.loc[both & (left_rec | right_rec)] = "bilateral_with_reconstruction"
            source.loc[only_left & left_rec] = "left_reconstructed_only"
            source.loc[only_right & right_rec] = "right_reconstructed_only"
    elif policy == "left_only":
        value.loc[left_ok] = left.loc[left_ok]
        source.loc[left_ok] = "left_observed"
    else:
        value.loc[right_ok] = right.loc[right_ok]
        source.loc[right_ok] = "right_observed"

    frame[output_col] = value
    frame[status_col] = source
    frame.attrs["gp3_binocular_combination"] = {
        "policy": policy,
        "left_col": left_col,
        "right_col": right_col,
        "prefix": prefix,
        "output_col": output_col,
        "status_col": status_col,
    }
    return frame


def flag_gazepoint_pupil(
    data,
    pupil_col=None,
    physiological_min: float = 1.0,
    physiological_max: float = 9.0,
    output_col="pupil_flag",
    *,
    time_col=None,
    missing_pupil_col=None,
    group_cols=None,
    outlier_k: float = 1.5,
    flag_iqr_outliers: bool = True,
) -> pd.DataFrame:
    """Flag pupil samples with legacy or R v2.3.0 quality columns."""
    df = ensure_dataframe(data)
    pupil_col = infer_column(df, "pupil", pupil_col, required=True)
    r_mode = (
        any(v is not None for v in (time_col, missing_pupil_col, group_cols))
        or outlier_k != 1.5
        or flag_iqr_outliers is not True
    )
    x = finite_numeric(df[pupil_col])
    if not r_mode:
        flag = pd.Series("ok", index=df.index, dtype="string")
        flag[x.isna()] = "missing"
        flag[x < physiological_min] = "below_min"
        flag[x > physiological_max] = "above_max"
        df[output_col] = flag
        df["pupil_valid"] = flag.eq("ok")
        return df

    if physiological_max <= physiological_min:
        raise ValueError("max_pupil must be greater than min_pupil")
    if not (np.isfinite(outlier_k) and outlier_k >= 0):
        raise ValueError("outlier_k must be a finite non-negative number")
    if not isinstance(flag_iqr_outliers, (bool, np.bool_)):
        raise ValueError("flag_iqr_outliers must be TRUE or FALSE")
    time_col = infer_column(df, "time", time_col, required=True)
    subject_col = infer_column(df, "subject", None, required=True)
    media_col = infer_column(df, "media", None, required=True)
    if group_cols is None:
        groups = ["subject", "media_id"]
    elif isinstance(group_cols, str):
        groups = [group_cols]
    else:
        groups = list(group_cols)
    if set(groups) - {"subject", "media_id"}:
        raise ValueError("group_cols can only contain subject and media_id")

    raw = pd.to_numeric(df[pupil_col], errors="coerce").to_numpy(float)
    if missing_pupil_col is None:
        missing = np.isnan(raw)
    else:
        if missing_pupil_col not in df:
            raise KeyError(f"Missing required column: {missing_pupil_col}")
        missing = df[missing_pupil_col].fillna(False).astype(bool).to_numpy() | np.isnan(raw)
    nonfinite = ~missing & ~np.isfinite(raw)
    low = ~missing & ~nonfinite & (raw < physiological_min)
    high = ~missing & ~nonfinite & (raw > physiological_max)
    implausible = low | high
    candidate = raw.copy()
    candidate[missing | nonfinite | implausible] = np.nan
    iqr_flags = np.zeros(len(df), dtype=bool)
    if flag_iqr_outliers:
        keys = pd.DataFrame(
            {"subject": df[subject_col].astype(str), "media_id": df[media_col].astype(str)}
        )
        iterator = (
            [(None, np.arange(len(df)))]
            if not groups
            else keys.groupby(groups, dropna=False, sort=False).indices.items()
        )
        for _, pos in iterator:
            pos = np.asarray(pos, dtype=int)
            vals = candidate[pos]
            finite = vals[np.isfinite(vals)]
            if finite.size < 4:
                continue
            q1, q3 = np.quantile(finite, [0.25, 0.75])
            iqr = q3 - q1
            if not np.isfinite(iqr) or iqr == 0:
                continue
            lo = q1 - outlier_k * iqr
            hi = q3 + outlier_k * iqr
            local = np.isfinite(vals) & ((vals < lo) | (vals > hi))
            iqr_flags[pos[local]] = True
    invalid = missing | nonfinite | implausible | iqr_flags
    reason = np.full(len(df), "valid", dtype=object)
    reason[iqr_flags] = "iqr_outlier"
    reason[high] = "implausible_high"
    reason[low] = "implausible_low"
    reason[nonfinite] = "nonfinite"
    reason[missing] = "missing"
    df["pupil_raw_value"] = raw
    df["pupil_flag_missing"] = missing
    df["pupil_flag_nonfinite"] = nonfinite
    df["pupil_flag_implausible_low"] = low
    df["pupil_flag_implausible_high"] = high
    df["pupil_flag_implausible"] = implausible
    df["pupil_flag_iqr_outlier"] = iqr_flags
    df["pupil_flag_invalid"] = invalid
    df["pupil_flag_reason"] = reason
    df["pupil_for_preprocessing"] = np.where(invalid, np.nan, raw)
    df["pupil_flag_pupil_column"] = pupil_col
    df["pupil_flag_time_column"] = time_col
    df["pupil_flag_min_plausible"] = physiological_min
    df["pupil_flag_max_plausible"] = physiological_max
    df["pupil_flag_outlier_k"] = outlier_k
    legacy_reason = pd.Series(reason, index=df.index, dtype="string").replace(
        {"valid": "ok", "implausible_low": "below_min", "implausible_high": "above_max"}
    )
    df[output_col] = legacy_reason
    df["pupil_valid"] = ~invalid
    return df


def flag_gazepoint_pupil_hampel(
    data,
    pupil_col=None,
    window=None,
    n_sigma=None,
    output_col=None,
    *,
    time_col=None,
    grouping_cols=None,
    window_size_samples: int = 7,
    k: float = 3.0,
    min_valid_samples: int = 3,
    scale_mad: float = 1.4826,
    flag_col="pupil_hampel_outlier",
    median_col="pupil_hampel_median",
    mad_col="pupil_hampel_mad",
    threshold_col="pupil_hampel_threshold",
    corrected_col=None,
    status_col="pupil_hampel_status",
    overwrite: bool = False,
    name="gazepoint_pupil_hampel",
) -> pd.DataFrame:
    """Apply a Hampel filter with legacy or R v2.3.0 outputs."""
    df = ensure_dataframe(data)
    pupil_col = infer_column(df, "pupil", pupil_col, required=True)
    legacy = window is not None or n_sigma is not None or output_col is not None
    if legacy:
        window = 7 if window is None else int(window)
        n_sigma = 3.0 if n_sigma is None else float(n_sigma)
        output_col = output_col or "pupil_hampel_flag"
        x = finite_numeric(df[pupil_col])
        med = x.rolling(window, center=True, min_periods=max(1, window // 2)).median()
        dev = (x - med).abs()
        mad = dev.rolling(window, center=True, min_periods=max(1, window // 2)).median()
        threshold = 1.4826 * n_sigma * mad
        df[output_col] = dev > threshold
        return df
    if len(df) == 0:
        raise ValueError("data must contain at least one row")
    if window_size_samples < 1 or window_size_samples % 2 == 0:
        raise ValueError("window_size_samples must be an odd positive integer")
    if min_valid_samples < 1 or min_valid_samples > window_size_samples:
        raise ValueError("min_valid_samples must be between 1 and window_size_samples")
    if not (np.isfinite(k) and k > 0 and np.isfinite(scale_mad) and scale_mad > 0):
        raise ValueError("k and scale_mad must be positive finite numbers")
    if grouping_cols is None:
        groups = []
    elif isinstance(grouping_cols, str):
        groups = [grouping_cols]
    else:
        groups = list(grouping_cols)
    missing = [c for c in groups if c not in df]
    if missing:
        raise KeyError(f"Grouping columns not found: {missing}")
    if time_col is not None and time_col not in df:
        raise KeyError(f"Missing required column: {time_col}")
    outputs = [flag_col, median_col, mad_col, threshold_col, status_col]
    if corrected_col is not None:
        outputs.append(corrected_col)
    if len(outputs) != len(set(outputs)):
        raise ValueError("Output column names must be unique")
    if not overwrite:
        existing = [c for c in outputs if c in df]
        if existing:
            raise ValueError(f"Output column(s) already exist in data: {', '.join(existing)}")
    work = df.copy()
    work["_row"] = np.arange(len(work))
    work["_p"] = pd.to_numeric(work[pupil_col], errors="coerce")
    work["_time"] = (
        np.arange(len(work), dtype=float)
        if time_col is None
        else pd.to_numeric(work[time_col], errors="coerce")
    )
    if not np.isfinite(work["_time"].to_numpy(float)).all():
        raise ValueError("time_col must contain finite numeric values")
    med_all = np.full(len(df), np.nan)
    mad_all = np.full(len(df), np.nan)
    threshold_all = np.full(len(df), np.nan)
    flags = np.zeros(len(df), dtype=bool)
    statuses = np.full(len(df), "complete", dtype=object)
    iterator = [(None, work)] if not groups else work.groupby(groups, dropna=False, sort=False)
    n_groups = 0
    half = window_size_samples // 2
    for _, part in iterator:
        n_groups += 1
        part = part.sort_values(["_time", "_row"], kind="stable")
        vals = part["_p"].to_numpy(float)
        rows = part["_row"].to_numpy(int)
        for j, row in enumerate(rows):
            if not np.isfinite(vals[j]):
                statuses[row] = "missing_or_nonfinite_pupil"
                continue
            lo = max(0, j - half)
            hi = min(len(vals), j + half + 1)
            valid = vals[lo:hi][np.isfinite(vals[lo:hi])]
            if len(valid) < min_valid_samples:
                statuses[row] = "insufficient_valid_window"
                continue
            med = float(np.median(valid))
            mad = float(np.median(np.abs(valid - med)) * scale_mad)
            threshold = float(k * mad)
            med_all[row] = med
            mad_all[row] = mad
            threshold_all[row] = threshold
            flags[row] = abs(vals[j] - med) > threshold
            statuses[row] = "complete_zero_mad" if threshold == 0 else "complete"
    df[median_col] = med_all
    df[mad_col] = mad_all
    df[threshold_col] = threshold_all
    df[flag_col] = flags
    df[status_col] = statuses
    if corrected_col is not None:
        corrected = pd.to_numeric(df[pupil_col], errors="coerce").to_numpy(dtype=float, copy=True)
        mask = flags & np.isfinite(med_all)
        corrected[mask] = med_all[mask]
        df[corrected_col] = corrected
    df.attrs["gp3_hampel_overview"] = pd.DataFrame(
        [
            {
                "object_name": name,
                "filter": "hampel",
                "pupil_col": pupil_col,
                "time_col": time_col,
                "grouping_cols": ", ".join(groups) if groups else None,
                "n_input_rows": len(df),
                "n_groups": n_groups,
                "window_size_samples": window_size_samples,
                "k": k,
                "min_valid_samples": min_valid_samples,
                "scale_mad": scale_mad,
                "n_flagged": int(flags.sum()),
                "flagged_proportion": float(flags.mean()),
                "n_complete": int(np.isin(statuses, ["complete", "complete_zero_mad"]).sum()),
                "n_problem_rows": int(
                    (~np.isin(statuses, ["complete", "complete_zero_mad"])).sum()
                ),
            }
        ]
    )
    df.attrs["gp3_hampel_status_summary"] = (
        pd.Series(statuses).value_counts().sort_index().rename_axis("status").reset_index(name="n")
    )
    return df


def detect_gazepoint_blinks(
    data,
    pupil_col=None,
    time_col=None,
    min_duration_ms: float = 50.0,
    max_duration_ms: float = 800.0,
    output_col="blink",
    *,
    id_col=None,
    group_cols=None,
    z_thresh=None,
    zero_threshold=None,
    merge_gap_ms=None,
    time_unit=None,
    include_rapid_changes=None,
    return_mode=None,
    **kwargs,
):
    """Detect blink periods with legacy or R v2.3.0 event-table semantics."""
    r_return = kwargs.pop("return", None)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")
    if r_return is not None:
        if return_mode is not None:
            raise TypeError("Specify only one of return_mode or the R-compatible 'return' argument")
        return_mode = r_return

    r_mode = any(
        value is not None
        for value in (
            id_col,
            group_cols,
            z_thresh,
            zero_threshold,
            merge_gap_ms,
            time_unit,
            include_rapid_changes,
            return_mode,
        )
    )
    if not r_mode:
        df = ensure_dataframe(data)
        pupil_col = infer_column(df, "pupil", pupil_col, required=True)
        time_col = infer_column(df, "time", time_col)
        missing = finite_numeric(df[pupil_col]).isna().to_numpy()
        labels, n = ndimage.label(missing.astype(int))
        blink = np.zeros(len(df), dtype=bool)
        t = time_to_seconds(df[time_col]).to_numpy(float) if time_col else np.arange(len(df)) / 60.0
        for lab in range(1, n + 1):
            idx = np.flatnonzero(labels == lab)
            if not idx.size:
                continue
            duration_ms = (
                (t[idx[-1]] - t[idx[0]] + np.nanmedian(np.diff(t[np.isfinite(t)]))) * 1000
                if idx.size > 1
                else 1000 / 60
            )
            if min_duration_ms <= duration_ms <= max_duration_ms:
                blink[idx] = True
        df[output_col] = blink
        return df

    df = ensure_dataframe(data)
    id_col = "USER_ID" if id_col is None else id_col
    time_col = "TIME" if time_col is None else time_col
    z_thresh = 4.0 if z_thresh is None else float(z_thresh)
    zero_threshold = 0.0 if zero_threshold is None else float(zero_threshold)
    merge_gap_ms = 20.0 if merge_gap_ms is None else float(merge_gap_ms)
    time_unit = "auto" if time_unit is None else str(time_unit)
    include_rapid_changes = True if include_rapid_changes is None else include_rapid_changes
    return_mode = "events" if return_mode is None else str(return_mode)
    if time_unit not in {"auto", "seconds", "milliseconds"}:
        raise ValueError("time_unit must be 'auto', 'seconds', or 'milliseconds'")
    if return_mode not in {"events", "samples", "both"}:
        raise ValueError("return must be 'events', 'samples', or 'both'")
    for value, name in (
        (min_duration_ms, "min_duration"),
        (z_thresh, "z_thresh"),
        (merge_gap_ms, "merge_gap_ms"),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be one finite non-negative number")
    if not np.isfinite(zero_threshold):
        raise ValueError("zero_threshold must be finite")
    if not isinstance(include_rapid_changes, (bool, np.bool_)):
        raise ValueError("include_rapid_changes must be TRUE or FALSE")

    if pupil_col is None:
        candidates = [
            "mean_pupil",
            "pupil_regressed",
            "pupil_smoothed",
            "pupil_interpolated",
            "pupil_clean",
            "pupil",
            "LPupil",
            "RPupil",
            "LPD",
            "RPD",
            "LPMM",
            "RPMM",
        ]
        pupil_cols = [column for column in candidates if column in df.columns]
        if not pupil_cols:
            raise ValueError("No pupil column was supplied or detected")
    elif isinstance(pupil_col, str):
        pupil_cols = [pupil_col]
    else:
        pupil_cols = list(dict.fromkeys(pupil_col))
    extra_groups = (
        []
        if group_cols is None
        else ([group_cols] if isinstance(group_cols, str) else list(group_cols))
    )
    groups = list(dict.fromkeys([id_col, *extra_groups]))
    required = list(dict.fromkeys([*groups, *pupil_cols, time_col]))
    missing_cols = [column for column in required if column not in df.columns]
    if missing_cols:
        raise ValueError("all_gaze is missing required column(s): " + ", ".join(missing_cols))

    def seconds(values):
        raw = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
        if time_unit == "seconds":
            return raw
        if time_unit == "milliseconds":
            return raw * 0.001
        finite = raw[np.isfinite(raw)]
        delta = np.diff(np.unique(np.sort(finite))) if finite.size else np.array([])
        delta = delta[np.isfinite(delta) & (delta > 0)]
        typical = float(np.median(delta)) if delta.size else np.nan
        return raw * 0.001 if np.isfinite(typical) and typical >= 1 else raw

    labelled = df.copy()
    blink_detected = np.zeros(len(df), dtype=bool)
    blink_id = np.full(len(df), np.nan)
    blink_reason = np.full(len(df), None, dtype=object)
    events_rows = []

    grouped = df.groupby(groups, sort=False, dropna=False).indices
    for _, positions in grouped.items():
        positions = np.asarray(positions, dtype=int)
        raw_time = pd.to_numeric(df.iloc[positions][time_col], errors="coerce").to_numpy(
            dtype=float
        )
        order = np.argsort(np.where(np.isfinite(raw_time), raw_time, np.inf), kind="stable")
        pos = positions[order]
        time_sec = seconds(df.iloc[pos][time_col])
        pupil_matrix = np.column_stack(
            [
                pd.to_numeric(df.iloc[pos][column], errors="coerce").to_numpy(dtype=float)
                for column in pupil_cols
            ]
        )
        available = np.isfinite(pupil_matrix).sum(axis=1)
        totals = np.nansum(pupil_matrix, axis=1)
        pupil = np.divide(
            totals,
            available,
            out=np.full(len(available), np.nan, dtype=float),
            where=available > 0,
        )
        missing_flag = ~np.isfinite(pupil)
        zero_flag = np.isfinite(pupil) & (pupil <= zero_threshold)
        finite_pupil = pupil[np.isfinite(pupil) & (pupil > zero_threshold)]
        pupil_med = float(np.median(finite_pupil)) if finite_pupil.size else np.nan
        if finite_pupil.size >= 3:
            mad_raw = float(np.median(np.abs(finite_pupil - np.median(finite_pupil))))
            pupil_mad = 1.4826 * mad_raw
        else:
            pupil_mad = np.nan
        low_flag = np.zeros(len(pupil), dtype=bool)
        if np.isfinite(pupil_mad) and pupil_mad > 0:
            low_flag = np.isfinite(pupil) & (pupil < pupil_med - z_thresh * pupil_mad)
        drop_flag = np.zeros(len(pupil), dtype=bool)
        recovery_flag = np.zeros(len(pupil), dtype=bool)
        if include_rapid_changes and len(pupil) >= 3:
            delta = np.r_[np.nan, np.diff(pupil)]
            finite_delta = delta[np.isfinite(delta)]
            if finite_delta.size >= 3:
                center = float(np.median(finite_delta))
                delta_mad = 1.4826 * float(np.median(np.abs(finite_delta - center)))
            else:
                delta_mad = np.nan
            if np.isfinite(delta_mad) and delta_mad > 0:
                raw_drop = np.isfinite(delta) & (delta < -z_thresh * delta_mad)
                raw_recovery = np.isfinite(delta) & (delta > z_thresh * delta_mad)
                drop_flag = raw_drop | np.r_[raw_drop[1:], False]
                recovery_flag = raw_recovery | np.r_[False, raw_recovery[:-1]]
        candidate = missing_flag | zero_flag | low_flag | drop_flag | recovery_flag
        if candidate.any():
            starts = np.flatnonzero(candidate & np.r_[True, ~candidate[:-1]])
            ends = np.flatnonzero(candidate & np.r_[~candidate[1:], True])
            runs = [[int(a), int(b)] for a, b in zip(starts, ends, strict=True)]
        else:
            runs = []
        if merge_gap_ms > 0 and len(runs) > 1:
            merged = [runs[0]]
            for start, end in runs[1:]:
                prev = merged[-1]
                gap = time_sec[start] - time_sec[prev[1]]
                if np.isfinite(gap) and gap <= merge_gap_ms / 1000.0:
                    prev[1] = end
                else:
                    merged.append([start, end])
            runs = merged
        positive_dt = np.diff(time_sec)
        positive_dt = positive_dt[np.isfinite(positive_dt) & (positive_dt > 0)]
        sample_interval = float(np.median(positive_dt)) if positive_dt.size else 0.0
        local_id = 0
        for start, end in runs:
            run = np.arange(start, end + 1)
            duration_ms = (
                max(0.0, float(time_sec[end] - time_sec[start] + sample_interval)) * 1000.0
            )
            if duration_ms + np.sqrt(np.finfo(float).eps) < min_duration_ms:
                continue
            reasons = []
            for flag, reason in (
                (missing_flag, "missing"),
                (zero_flag, "zero"),
                (low_flag, "low_outlier"),
                (drop_flag, "rapid_drop"),
                (recovery_flag, "rapid_recovery"),
            ):
                if flag[run].any():
                    reasons.append(reason)
            reason = ";".join(dict.fromkeys(reasons))
            local_id += 1
            run_pos = pos[run]
            blink_detected[run_pos] = True
            blink_id[run_pos] = local_id
            blink_reason[run_pos] = reason
            row = {column: df.iloc[run_pos[0]][column] for column in groups}
            row.update(
                {
                    "blink_id": local_id,
                    "start_time": df.iloc[run_pos[0]][time_col],
                    "end_time": df.iloc[run_pos[-1]][time_col],
                    "duration": duration_ms,
                    "duration_ms": duration_ms,
                    "n_samples": len(run),
                    "reason": reason,
                    "pupil_columns": ";".join(pupil_cols),
                }
            )
            events_rows.append(row)

    labelled["blink_detected"] = blink_detected
    labelled["blink_id"] = pd.array(blink_id, dtype="Int64")
    labelled["blink_reason"] = pd.Series(blink_reason, index=labelled.index, dtype="string")
    labelled.attrs["_gp3_class"] = "gp3_blink_samples"
    columns = [
        *groups,
        "blink_id",
        "start_time",
        "end_time",
        "duration",
        "duration_ms",
        "n_samples",
        "reason",
        "pupil_columns",
    ]
    events = pd.DataFrame(events_rows, columns=columns)
    events.attrs["_gp3_class"] = "gp3_blink_events"
    if return_mode == "events":
        return events
    if return_mode == "samples":
        return labelled
    return {"events": events, "samples": labelled, "_gp3_class": "gp3_blink_detection_result"}


def flag_gazepoint_pupil_artifacts(
    data,
    pupil_col=None,
    time_col=None,
    physiological_min=1.0,
    physiological_max=9.0,
    pupil_speed_mad_k: float = 6.0,
    blink_padding_pre_ms: float = 100.0,
    blink_padding_post_ms: float = 100.0,
    output_col="pupil_artifact",
) -> pd.DataFrame:
    df = flag_gazepoint_pupil(
        data,
        pupil_col=pupil_col,
        physiological_min=physiological_min,
        physiological_max=physiological_max,
    )
    pupil_col = infer_column(df, "pupil", pupil_col, required=True)
    time_col = infer_column(df, "time", time_col)
    x = finite_numeric(df[pupil_col])
    t = (
        time_to_seconds(df[time_col])
        if time_col
        else pd.Series(np.arange(len(df)) / 60.0, index=df.index)
    )
    dt = t.diff().replace(0, np.nan)
    speed = x.diff().abs() / dt
    med, mad = np.nanmedian(speed), robust_mad(speed.dropna())
    speed_flag = (
        speed > (med + pupil_speed_mad_k * 1.4826 * mad)
        if np.isfinite(mad)
        else pd.Series(False, index=df.index)
    )
    artifact = ~df["pupil_valid"] | speed_flag.fillna(False)
    # Pad artifacts in sample units estimated from timing.
    diffs = np.diff(t.dropna().to_numpy(float))
    hz = 1 / np.median(diffs[diffs > 0]) if np.any(diffs > 0) else 60.0
    pre = int(round(blink_padding_pre_ms / 1000 * hz))
    post = int(round(blink_padding_post_ms / 1000 * hz))
    arr = artifact.to_numpy(bool)
    padded = arr.copy()
    bad = np.flatnonzero(arr)
    for i in bad:
        padded[max(0, i - pre) : min(len(arr), i + post + 1)] = True
    df[output_col] = padded
    df["pupil_speed"] = speed
    return df


def _interpolate_series(x: pd.Series, method: str, limit: int | None = None) -> pd.Series:
    if method == "linear":
        return x.interpolate(method="linear", limit=limit, limit_area="inside")
    if method in {"pchip", "cubic"}:
        idx = np.arange(len(x), dtype=float)
        ok = x.notna().to_numpy()
        if ok.sum() < 2:
            return x.copy()
        if method == "pchip":
            fn = interpolate.PchipInterpolator(idx[ok], x.to_numpy(float)[ok], extrapolate=False)
        else:
            kind = "cubic" if ok.sum() >= 4 else "linear"
            fn = interpolate.interp1d(idx[ok], x.to_numpy(float)[ok], kind=kind, bounds_error=False)
        out = x.copy()
        miss = ~ok
        values = fn(idx[miss])
        out.iloc[np.flatnonzero(miss)] = values
        if limit is not None:
            runs, n = ndimage.label(miss.astype(int))
            for lab in range(1, n + 1):
                ii = np.flatnonzero(runs == lab)
                if len(ii) > limit:
                    out.iloc[ii] = np.nan
        return out
    raise ValueError(f"Unsupported interpolation method {method!r}")


def interpolate_gazepoint_pupil(
    data,
    pupil_col=None,
    time_col=None,
    group_cols=None,
    max_gap_ms: float = 150.0,
    max_gap_samples=np.inf,
    min_valid_points: int = 2,
    *,
    output_col=None,
    method=None,
) -> pd.DataFrame:
    """Interpolate pupil gaps with R v2.3.0 semantics and legacy compatibility."""
    if output_col is not None or method is not None:
        df = ensure_dataframe(data)
        pupil_col = infer_column(df, "pupil", pupil_col, required=True)
        output_col = output_col or pupil_col
        time_col = infer_column(df, "time", time_col)
        groups = normalize_group_cols(df, group_cols)
        out = pd.Series(np.nan, index=df.index, dtype=float)
        iterator = [((), df)] if not groups else df.groupby(groups, dropna=False, sort=False)
        for _, frame in iterator:
            x = finite_numeric(frame[pupil_col])
            limit = None
            if max_gap_ms is not None:
                if time_col:
                    t = time_to_seconds(frame[time_col]).dropna().to_numpy(float)
                    diffs = np.diff(t)
                    hz = 1 / np.median(diffs[diffs > 0]) if np.any(diffs > 0) else 60.0
                else:
                    hz = 60.0
                limit = max(1, int(round(float(max_gap_ms) / 1000 * hz)))
            selected_method = "linear" if method is None else method
            out.loc[frame.index] = _interpolate_series(x, selected_method, limit)
        df[output_col] = out
        df[f"{output_col}_interpolated"] = finite_numeric(df[pupil_col]).isna() & out.notna()
        return df

    df = ensure_dataframe(data, copy=False)
    if group_cols is None:
        group_cols = ["subject", "media_id"]
    if pupil_col is not None and (not isinstance(pupil_col, str) or not pupil_col):
        raise ValueError("pupil_col must be None or a single character string")
    if time_col is not None and (not isinstance(time_col, str) or not time_col):
        raise ValueError("time_col must be None or a single character string")
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    else:
        group_cols = list(group_cols)
    if not all(isinstance(col, str) and col for col in group_cols):
        raise ValueError("group_cols must be a character vector")
    if (
        isinstance(max_gap_ms, (bool, np.bool_))
        or not isinstance(max_gap_ms, (int, float, np.integer, np.floating))
        or np.isnan(max_gap_ms)
        or max_gap_ms < 0
    ):
        raise ValueError("max_gap_ms must be greater than or equal to 0")
    if (
        isinstance(max_gap_samples, (bool, np.bool_))
        or not isinstance(max_gap_samples, (int, float, np.integer, np.floating))
        or np.isnan(max_gap_samples)
        or max_gap_samples < 0
    ):
        raise ValueError("max_gap_samples must be greater than or equal to 0")
    if (
        isinstance(min_valid_points, (bool, np.bool_))
        or not isinstance(min_valid_points, (int, float, np.integer, np.floating))
        or not np.isfinite(min_valid_points)
        or min_valid_points < 2
    ):
        raise ValueError("min_valid_points must be greater than or equal to 2")
    min_valid_points = int(min_valid_points)

    def detect(candidates):
        return next((candidate for candidate in candidates if candidate in df.columns), None)

    role_sources = {
        "subject": detect(("subject", "pID", "participant")),
        "media_id": detect(("media_id", "MEDIA_ID")),
        "trial": detect(("trial",)),
        "trial_global": detect(("trial_global",)),
    }
    pupil_source = pupil_col or detect(
        (
            "pupil_clean",
            "pupil_for_preprocessing",
            "mean_pupil",
            "pupil",
            "pupil_raw",
            "left_pupil",
            "right_pupil",
        )
    )
    time_source = time_col or detect(("time_ms", "time", "time_orig", "time_orig_ms"))
    if pupil_source is None or pupil_source not in df.columns:
        raise ValueError("No pupil column was found")
    if time_source is None or time_source not in df.columns:
        raise ValueError("No time column was found")
    allowed = list(role_sources)
    invalid = [col for col in group_cols if col not in allowed]
    if invalid:
        raise ValueError("group_cols can only contain: " + ", ".join(allowed))
    missing = [col for col in group_cols if role_sources[col] is None]
    if missing:
        raise ValueError(
            "The following grouping column role(s) were requested but not found: "
            + ", ".join(missing)
        )

    work = pd.DataFrame({"row_id": np.arange(1, len(df) + 1, dtype=int)})
    for role, source_col in role_sources.items():
        work[role] = (
            df[source_col].astype("string").to_numpy()
            if source_col is not None
            else pd.array([pd.NA] * len(df), dtype="string")
        )
    work["time_ms"] = pd.to_numeric(df[time_source], errors="coerce").to_numpy(float)
    work["pupil_input_value"] = pd.to_numeric(df[pupil_source], errors="coerce").to_numpy(float)

    def interpolate_group(frame):
        frame = frame.sort_values(["time_ms", "row_id"], kind="stable").copy()
        values = frame["pupil_input_value"].to_numpy(float)
        times = frame["time_ms"].to_numpy(float)
        n = len(frame)
        observed = np.isfinite(values)
        time_valid = np.isfinite(times)
        endpoint_valid = observed & time_valid
        needs = ~observed
        pupil_after = values.copy()
        was = np.zeros(n, dtype=bool)
        status = np.full(n, "observed", dtype=object)
        status[needs] = "missing_unfilled"
        status[needs & ~time_valid] = "missing_no_time"
        gap_id = np.full(n, np.nan)
        gap_n = np.full(n, np.nan)
        gap_duration = np.full(n, np.nan)

        if endpoint_valid.sum() < min_valid_points:
            status[needs & time_valid] = "missing_insufficient_valid"
        elif needs.any():
            valid_times = times[endpoint_valid]
            valid_values = values[endpoint_valid]
            unique_times = np.unique(valid_times)
            unique_values = np.array(
                [valid_values[valid_times == value].mean() for value in unique_times]
            )
            starts = np.flatnonzero(needs & np.r_[True, ~needs[:-1]])
            ends = np.flatnonzero(needs & np.r_[~needs[1:], True])
            for counter, (start, end) in enumerate(zip(starts, ends, strict=True), start=1):
                idx = np.arange(start, end + 1)
                gap_id[idx] = counter
                gap_n[idx] = len(idx)
                previous = np.flatnonzero(endpoint_valid & (np.arange(n) < start))
                following = np.flatnonzero(endpoint_valid & (np.arange(n) > end))
                if len(previous) == 0 or len(following) == 0:
                    status[idx[time_valid[idx]]] = "missing_edge_gap"
                    continue
                left = previous.max()
                right = following.min()
                duration = times[right] - times[left]
                gap_duration[idx] = duration
                if (
                    (not np.isfinite(duration))
                    or duration > float(max_gap_ms)
                    or len(idx) > float(max_gap_samples)
                ):
                    status[idx[time_valid[idx]]] = "missing_long_gap"
                    continue
                interpolated = np.interp(
                    times[idx], unique_times, unique_values, left=np.nan, right=np.nan
                )
                fillable = time_valid[idx] & np.isfinite(interpolated)
                if fillable.any():
                    fill_idx = idx[fillable]
                    pupil_after[fill_idx] = interpolated[fillable]
                    was[fill_idx] = True
                    status[fill_idx] = "interpolated"
                if (~fillable).any():
                    unfilled_idx = idx[(~fillable) & time_valid[idx]]
                    status[unfilled_idx] = "missing_unfilled"

        return pd.DataFrame(
            {
                "row_id": frame["row_id"].to_numpy(int),
                "pupil_interpolated": pupil_after,
                "pupil_was_interpolated": was,
                "pupil_interpolation_status": status,
                "pupil_gap_id": pd.array(gap_id, dtype="Int64"),
                "pupil_gap_n_samples": pd.array(gap_n, dtype="Int64"),
                "pupil_gap_duration_ms": gap_duration,
            }
        )

    if not group_cols:
        interpolated = interpolate_group(work)
    else:
        parts = []
        for _, frame in work.groupby(group_cols, dropna=False, sort=True):
            parts.append(interpolate_group(frame))
        interpolated = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    interpolated = interpolated.sort_values("row_id", kind="stable").reset_index(drop=True)
    interpolated["pupil_interp_pupil_column"] = pupil_source
    interpolated["pupil_interp_time_column"] = time_source
    interpolated["pupil_interp_max_gap_ms"] = max_gap_ms
    interpolated["pupil_interp_max_gap_samples"] = max_gap_samples
    interpolated["pupil_interp_min_valid_points"] = min_valid_points
    output_cols = [
        "pupil_interpolated",
        "pupil_was_interpolated",
        "pupil_interpolation_status",
        "pupil_gap_id",
        "pupil_gap_n_samples",
        "pupil_gap_duration_ms",
        "pupil_interp_pupil_column",
        "pupil_interp_time_column",
        "pupil_interp_max_gap_ms",
        "pupil_interp_max_gap_samples",
        "pupil_interp_min_valid_points",
    ]
    original = df.drop(columns=[col for col in output_cols if col in df.columns]).reset_index(
        drop=True
    )
    return pd.concat([original, interpolated[output_cols]], axis=1)


def interpolate_gazepoint_pupil_pchip(data, **kwargs) -> pd.DataFrame:
    kwargs["method"] = "pchip"
    return interpolate_gazepoint_pupil(data, **kwargs)


def interpolate_gazepoint_blinks(data, pupil_col=None, blink_col="blink", **kwargs) -> pd.DataFrame:
    df = ensure_dataframe(data)
    pupil_col = infer_column(df, "pupil", pupil_col, required=True)
    if blink_col not in df:
        df = detect_gazepoint_blinks(df, pupil_col=pupil_col, output_col=blink_col)
    working = df.copy()
    working.loc[working[blink_col].astype(bool), pupil_col] = np.nan
    return interpolate_gazepoint_pupil(working, pupil_col=pupil_col, **kwargs)


def smooth_gazepoint_pupil(
    data,
    pupil_col=None,
    time_col=None,
    group_cols=None,
    window_samples: int = 5,
    method="mean",
    align="center",
    min_points: int = 1,
    preserve_missing: bool = True,
    *,
    output_col=None,
    window=None,
) -> pd.DataFrame:
    """Smooth pupil data with R v2.3.0 semantics and legacy compatibility."""
    legacy_mode = (
        output_col is not None or window is not None or method in {"moving_average", "savgol"}
    )
    if legacy_mode:
        df = ensure_dataframe(data)
        pupil_col = infer_column(df, "pupil", pupil_col, required=True)
        selected_window = int(window_samples if window is None else window)
        output_col = output_col or f"{pupil_col}_smoothed"
        x = finite_numeric(df[pupil_col])
        if method in {"moving_average", "mean"}:
            y = x.rolling(selected_window, center=True, min_periods=1).mean()
        elif method == "median":
            y = x.rolling(selected_window, center=True, min_periods=1).median()
        elif method == "savgol":
            win = max(3, selected_window + (1 - selected_window % 2))
            vals = x.interpolate(limit_direction="both").to_numpy(float)
            y = pd.Series(signal.savgol_filter(vals, win, min(2, win - 1)), index=df.index)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
        df[output_col] = y
        return df

    df = ensure_dataframe(data, copy=False)
    if group_cols is None:
        group_cols = ["subject", "media_id"]
    if method not in {"mean", "median"}:
        raise ValueError(f"Unknown smoothing method: {method}")
    if align not in {"center", "right", "left"}:
        raise ValueError("align must be one of: center, right, left")
    if not isinstance(preserve_missing, (bool, np.bool_)):
        raise ValueError("preserve_missing must be True or False")
    if (
        isinstance(window_samples, (bool, np.bool_))
        or not isinstance(window_samples, (int, float, np.integer, np.floating))
        or not np.isfinite(window_samples)
        or window_samples < 1
    ):
        raise ValueError("window_samples must be greater than or equal to 1")
    if (
        isinstance(min_points, (bool, np.bool_))
        or not isinstance(min_points, (int, float, np.integer, np.floating))
        or not np.isfinite(min_points)
        or min_points < 1
    ):
        raise ValueError("min_points must be greater than or equal to 1")
    window_samples = int(window_samples)
    min_points = int(min_points)
    if min_points > window_samples:
        raise ValueError("min_points must be less than or equal to window_samples")
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    else:
        group_cols = list(group_cols)

    def detect(candidates):
        return next((candidate for candidate in candidates if candidate in df.columns), None)

    role_sources = {
        "subject": detect(("subject", "pID", "participant")),
        "media_id": detect(("media_id", "MEDIA_ID")),
        "trial": detect(("trial",)),
        "trial_global": detect(("trial_global",)),
    }
    pupil_source = pupil_col or detect(
        (
            "pupil_baseline_corrected",
            "pupil_baseline_percent_change",
            "pupil_interpolated",
            "pupil_for_preprocessing",
            "mean_pupil",
            "pupil",
            "pupil_raw",
            "left_pupil",
            "right_pupil",
        )
    )
    time_source = time_col or detect(("time_ms", "time", "time_orig", "time_orig_ms"))
    if pupil_source is None or pupil_source not in df.columns:
        raise ValueError("No pupil column was found")
    if time_source is None or time_source not in df.columns:
        raise ValueError("No time column was found")
    standard_roles = set(role_sources)
    missing_roles = [
        col for col in group_cols if col in standard_roles and role_sources[col] is None
    ]
    if missing_roles:
        raise ValueError(
            "The following grouping column role(s) were requested but not found: "
            + ", ".join(missing_roles)
        )
    custom = [col for col in group_cols if col not in standard_roles]
    missing_custom = [col for col in custom if col not in df.columns]
    if missing_custom:
        raise ValueError(
            "The following grouping column(s) were requested but not found: "
            + ", ".join(missing_custom)
        )

    work = pd.DataFrame({"row_id": np.arange(1, len(df) + 1, dtype=int)})
    for role, source_col in role_sources.items():
        work[role] = (
            df[source_col].astype("string").to_numpy()
            if source_col is not None
            else pd.array([pd.NA] * len(df), dtype="string")
        )
    work["time_ms"] = pd.to_numeric(df[time_source], errors="coerce").to_numpy(float)
    work["pupil_input_value"] = pd.to_numeric(df[pupil_source], errors="coerce").to_numpy(float)
    for col in custom:
        work[col] = df[col].to_numpy()

    def smooth_vector(values):
        x = np.asarray(values, dtype=float)
        n = len(x)
        smoothed = np.full(n, np.nan)
        window_n = np.zeros(n, dtype=int)
        status = np.full(n, "insufficient_window", dtype=object)
        finite = np.isfinite(x)
        status[~finite] = "missing_input"
        for i in range(n):
            if align == "center":
                before = (window_samples - 1) // 2
                after = window_samples - 1 - before
                start = max(0, i - before)
                end = min(n, i + after + 1)
            elif align == "right":
                start = max(0, i - window_samples + 1)
                end = i + 1
            else:
                start = i
                end = min(n, i + window_samples)
            vals = x[start:end]
            vals = vals[np.isfinite(vals)]
            window_n[i] = len(vals)
            if preserve_missing and not finite[i]:
                continue
            if len(vals) < min_points:
                continue
            smoothed[i] = float(np.median(vals) if method == "median" else np.mean(vals))
            status[i] = "smoothed"
        return smoothed, status, window_n

    parts = []
    iterator = (
        [(None, work)] if not group_cols else work.groupby(group_cols, dropna=False, sort=True)
    )
    for _, frame in iterator:
        smoothed, status, window_n = smooth_vector(frame["pupil_input_value"].to_numpy(float))
        parts.append(
            pd.DataFrame(
                {
                    "row_id": frame["row_id"].to_numpy(int),
                    "pupil_smoothed": smoothed,
                    "pupil_smoothing_status": status,
                    "pupil_smoothing_window_n": window_n,
                }
            )
        )
    result = (
        pd.concat(parts, ignore_index=True)
        .sort_values("row_id", kind="stable")
        .reset_index(drop=True)
    )
    result["pupil_smoothing_input_column"] = pupil_source
    result["pupil_smoothing_time_column"] = time_source
    result["pupil_smoothing_method"] = method
    result["pupil_smoothing_align"] = align
    result["pupil_smoothing_window_samples"] = window_samples
    result["pupil_smoothing_min_points"] = min_points
    result["pupil_smoothing_preserve_missing"] = bool(preserve_missing)
    output_cols = [
        "pupil_smoothed",
        "pupil_smoothing_status",
        "pupil_smoothing_window_n",
        "pupil_smoothing_input_column",
        "pupil_smoothing_time_column",
        "pupil_smoothing_method",
        "pupil_smoothing_align",
        "pupil_smoothing_window_samples",
        "pupil_smoothing_min_points",
        "pupil_smoothing_preserve_missing",
    ]
    original = df.drop(columns=[col for col in output_cols if col in df.columns]).reset_index(
        drop=True
    )
    return pd.concat([original, result[output_cols]], axis=1)


def _gp3_pupil_r_roll(
    values,
    *,
    window,
    method,
    min_valid,
):
    values = np.asarray(
        values,
        dtype=float,
    )

    output = np.full(
        len(values),
        np.nan,
    )

    left_width = (window - 1) // 2

    right_width = window - left_width - 1

    for index in range(len(values)):
        lower = max(
            0,
            index - left_width,
        )

        upper = min(
            len(values),
            index + right_width + 1,
        )

        local = values[lower:upper]

        finite = np.isfinite(local)

        if finite.sum() < min_valid:
            continue

        if method == "median":
            output[index] = float(np.median(local[finite]))
        else:
            output[index] = float(np.mean(local[finite]))

    return output


def smooth_gazepoint_coordinate(
    data=None,
    column=None,
    output_col=None,
    window: int = 5,
    method="moving_average",
    *,
    all_gaze=None,
    x_col=None,
    y_col=None,
    id_col="USER_ID",
    group_cols=None,
    suffix="_smooth",
    min_valid=None,
    preserve_missing=True,
) -> pd.DataFrame:
    """Smooth gaze coordinates using Python or R v2.3.0 semantics."""
    r_mode = (
        all_gaze is not None
        or x_col is not None
        or y_col is not None
        or group_cols is not None
        or id_col != "USER_ID"
        or suffix != "_smooth"
        or min_valid is not None
        or preserve_missing is not True
        or method
        in {
            "mean",
            "median",
        }
    )

    if not r_mode:
        df = ensure_dataframe(data)

        if column is None:
            column = infer_column(
                df,
                "x",
                required=True,
            )

        output_col = output_col or f"{column}_smoothed"

        values = finite_numeric(df[column])

        if method == "moving_average":
            smoothed = values.rolling(
                window,
                center=True,
                min_periods=1,
            ).mean()
        else:
            smoothed = values.rolling(
                window,
                center=True,
                min_periods=1,
            ).median()

        df[output_col] = smoothed

        return df

    if all_gaze is not None and data is not None:
        raise TypeError("supply either data or all_gaze, not both")

    frame = ensure_dataframe(
        all_gaze if all_gaze is not None else data,
        copy=False,
    )

    x_col = x_col or "FPOGX"

    y_col = y_col or "FPOGY"

    r_method = "median" if (all_gaze is not None and method == "moving_average") else method

    if r_method not in {
        "median",
        "mean",
    }:
        raise ValueError("method must be 'median' or 'mean'")

    if (
        isinstance(window, bool)
        or not isinstance(
            window,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(window)
        or window < 1
        or float(window) != int(window)
    ):
        raise ValueError("window must be one positive integer")

    window = int(window)

    min_valid = 1 if min_valid is None else min_valid

    if (
        isinstance(min_valid, bool)
        or not isinstance(
            min_valid,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(min_valid)
        or min_valid < 1
        or float(min_valid) != int(min_valid)
    ):
        raise ValueError("min_valid must be one positive integer")

    min_valid = int(min_valid)

    groups = _gp3_pupil_r_group_cols(
        id_col,
        group_cols,
    )

    required = list(
        dict.fromkeys(
            groups
            + [
                x_col,
                y_col,
            ]
        )
    )

    missing = [column_name for column_name in required if column_name not in frame.columns]

    if missing:
        raise ValueError("all_gaze is missing required column(s): " + ", ".join(missing))

    output = frame.copy()

    x_out = f"{x_col}{suffix}"

    y_out = f"{y_col}{suffix}"

    output[x_out] = np.nan
    output[y_out] = np.nan

    x_position = output.columns.get_loc(x_out)

    y_position = output.columns.get_loc(y_out)

    for indices in _gp3_pupil_r_group_positions(
        frame,
        groups,
    ):
        x = _gp3_pupil_r_numeric(frame.iloc[indices][x_col])

        y = _gp3_pupil_r_numeric(frame.iloc[indices][y_col])

        smoothed_x = _gp3_pupil_r_roll(
            x,
            window=window,
            method=r_method,
            min_valid=min_valid,
        )

        smoothed_y = _gp3_pupil_r_roll(
            y,
            window=window,
            method=r_method,
            min_valid=min_valid,
        )

        if preserve_missing is True:
            smoothed_x[~np.isfinite(x)] = np.nan

            smoothed_y[~np.isfinite(y)] = np.nan

        output.iloc[
            indices,
            x_position,
        ] = smoothed_x

        output.iloc[
            indices,
            y_position,
        ] = smoothed_y

    output.attrs["gazepoint_coordinate_smoothing"] = {
        "method": r_method,
        "window": window,
        "x_col": x_col,
        "y_col": y_col,
        "group_cols": groups,
    }

    return output


def baseline_correct_gazepoint_pupil(
    data,
    pupil_col=None,
    time_col=None,
    baseline_time_col=None,
    baseline_window=(-200.0, 0.0),
    baseline_flag_col=None,
    group_cols=None,
    baseline_method="mean",
    min_baseline_samples: int = 1,
    *,
    baseline=None,
    output_col=None,
    mode=None,
) -> pd.DataFrame:
    """Baseline-correct pupil data with R v2.3.0 semantics and legacy compatibility."""
    legacy_mode = baseline is not None or output_col is not None or mode is not None
    if legacy_mode:
        df = ensure_dataframe(data)
        pupil_col = infer_column(df, "pupil", pupil_col, required=True)
        time_col = infer_column(df, "time", time_col, required=True)
        groups = normalize_group_cols(df, group_cols)
        selected_baseline = baseline_window if baseline is None else baseline
        selected_mode = "subtract" if mode is None else mode
        output_col = output_col or f"{pupil_col}_baseline_corrected"
        out = pd.Series(np.nan, index=df.index, dtype=float)
        bases = pd.Series(np.nan, index=df.index, dtype=float)
        iterator = [((), df)] if not groups else df.groupby(groups, dropna=False, sort=False)
        for _, frame in iterator:
            t = finite_numeric(frame[time_col])
            x = finite_numeric(frame[pupil_col])
            mask = t.between(float(selected_baseline[0]), float(selected_baseline[1]))
            b = float(x.loc[mask].mean()) if mask.any() else np.nan
            bases.loc[frame.index] = b
            if selected_mode == "subtract":
                y = x - b
            elif selected_mode == "divide":
                y = x / b
            elif selected_mode in {"percent", "percent_change"}:
                y = (x - b) / b * 100
            else:
                raise ValueError(f"Unknown baseline mode {selected_mode!r}")
            out.loc[frame.index] = y
        df[output_col] = out
        df["pupil_baseline"] = bases
        return df

    df = ensure_dataframe(data, copy=False)
    if group_cols is None:
        group_cols = ["subject", "media_id"]
    if baseline_method not in {"mean", "median"}:
        raise ValueError("baseline_method must be one of: mean, median")
    if (
        isinstance(min_baseline_samples, (bool, np.bool_))
        or not isinstance(min_baseline_samples, (int, float, np.integer, np.floating))
        or not np.isfinite(min_baseline_samples)
        or min_baseline_samples < 1
    ):
        raise ValueError("min_baseline_samples must be greater than or equal to 1")
    min_baseline_samples = int(min_baseline_samples)
    if baseline_flag_col is None:
        if baseline_window is None or len(baseline_window) != 2:
            raise ValueError("baseline_window must be a numeric vector of length 2")
        start_window, end_window = map(float, baseline_window)
        if not np.isfinite([start_window, end_window]).all() or end_window < start_window:
            raise ValueError(
                "baseline_window[2] must be greater than or equal to baseline_window[1]"
            )
    else:
        if baseline_flag_col not in df.columns:
            raise ValueError("baseline_flag_col was not found in data")
        if baseline_window is None:
            start_window = end_window = np.nan
        else:
            if len(baseline_window) != 2:
                raise ValueError("baseline_window must be None or a numeric vector of length 2")
            start_window, end_window = map(float, baseline_window)
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    else:
        group_cols = list(group_cols)

    def detect(candidates):
        return next((candidate for candidate in candidates if candidate in df.columns), None)

    role_sources = {
        "subject": detect(("subject", "pID", "participant")),
        "media_id": detect(("media_id", "MEDIA_ID")),
        "trial": detect(("trial",)),
        "trial_global": detect(("trial_global",)),
    }
    pupil_source = pupil_col or detect(
        (
            "pupil_interpolated",
            "pupil_for_preprocessing",
            "mean_pupil",
            "pupil",
            "pupil_raw",
            "left_pupil",
            "right_pupil",
        )
    )
    time_source = time_col or detect(("time_ms", "time", "time_orig", "time_orig_ms"))
    baseline_time_source = baseline_time_col or detect(
        (
            "time_relative_ms",
            "relative_time_ms",
            "event_time_ms",
            "time_ms",
            "time",
            "time_orig",
            "time_orig_ms",
        )
    )
    if pupil_source is None or pupil_source not in df.columns:
        raise ValueError("No pupil column was found")
    if time_source is None or time_source not in df.columns:
        raise ValueError("No time column was found")
    if baseline_time_source is None or baseline_time_source not in df.columns:
        raise ValueError("No baseline-time column was found")
    standard_roles = set(role_sources)
    missing_roles = [
        col for col in group_cols if col in standard_roles and role_sources[col] is None
    ]
    if missing_roles:
        raise ValueError(
            "The following grouping column role(s) were requested but not found: "
            + ", ".join(missing_roles)
        )
    custom = [col for col in group_cols if col not in standard_roles]
    missing_custom = [col for col in custom if col not in df.columns]
    if missing_custom:
        raise ValueError(
            "The following grouping column(s) were requested but not found: "
            + ", ".join(missing_custom)
        )

    work = pd.DataFrame({"row_id": np.arange(1, len(df) + 1, dtype=int)})
    for role, source_col in role_sources.items():
        work[role] = (
            df[source_col].astype("string").to_numpy()
            if source_col is not None
            else pd.array([pd.NA] * len(df), dtype="string")
        )
    work["time_ms"] = pd.to_numeric(df[time_source], errors="coerce").to_numpy(float)
    work["baseline_time_ms"] = pd.to_numeric(df[baseline_time_source], errors="coerce").to_numpy(
        float
    )
    work["pupil_input_value"] = pd.to_numeric(df[pupil_source], errors="coerce").to_numpy(float)
    for col in custom:
        work[col] = df[col].to_numpy()
    if baseline_flag_col is not None:
        flag = df[baseline_flag_col].fillna(False).astype(bool).to_numpy()
        work["baseline_candidate"] = flag & np.isfinite(work["pupil_input_value"].to_numpy(float))
    else:
        bt = work["baseline_time_ms"].to_numpy(float)
        pv = work["pupil_input_value"].to_numpy(float)
        work["baseline_candidate"] = (
            np.isfinite(pv) & np.isfinite(bt) & (bt >= start_window) & (bt <= end_window)
        )

    def summarize(frame):
        mask = frame["baseline_candidate"].to_numpy(bool)
        values = frame.loc[mask, "pupil_input_value"].to_numpy(float)
        times = frame.loc[mask, "baseline_time_ms"].to_numpy(float)
        values = values[np.isfinite(values)]
        times = times[np.isfinite(times)]
        baseline_value = (
            float(np.median(values) if baseline_method == "median" else np.mean(values))
            if len(values)
            else np.nan
        )
        baseline_sd = float(np.std(values, ddof=1)) if len(values) >= 2 else np.nan
        return {
            "pupil_baseline_n": int(mask.sum()),
            "pupil_baseline_value": baseline_value,
            "pupil_baseline_sd": baseline_sd,
            "pupil_baseline_time_min": float(times.min()) if len(times) else np.nan,
            "pupil_baseline_time_max": float(times.max()) if len(times) else np.nan,
        }

    summaries = pd.DataFrame(
        index=work.index,
        columns=[
            "pupil_baseline_n",
            "pupil_baseline_value",
            "pupil_baseline_sd",
            "pupil_baseline_time_min",
            "pupil_baseline_time_max",
        ],
    )
    iterator = (
        [(None, work)] if not group_cols else work.groupby(group_cols, dropna=False, sort=True)
    )
    for _, frame in iterator:
        summary = summarize(frame)
        for key, value in summary.items():
            summaries.loc[frame.index, key] = value
    for col in summaries.columns:
        work[col] = pd.to_numeric(summaries[col], errors="coerce")
    work["pupil_baseline_n"] = work["pupil_baseline_n"].astype("Int64")
    available = (work["pupil_baseline_n"].astype(float) >= min_baseline_samples) & np.isfinite(
        work["pupil_baseline_value"].to_numpy(float)
    )
    pv = work["pupil_input_value"].to_numpy(float)
    bv = work["pupil_baseline_value"].to_numpy(float)
    sd = work["pupil_baseline_sd"].to_numpy(float)
    finite_pupil = np.isfinite(pv)
    work["pupil_baseline_available"] = available
    work["pupil_baseline_corrected"] = np.where(available & finite_pupil, pv - bv, np.nan)
    nonzero = available & finite_pupil & (bv != 0)
    work["pupil_baseline_percent_change"] = np.where(nonzero, ((pv - bv) / bv) * 100, np.nan)
    work["pupil_baseline_ratio"] = np.where(nonzero, pv / bv, np.nan)
    z_ok = available & finite_pupil & np.isfinite(sd) & (sd > 0)
    work["pupil_baseline_z"] = np.where(z_ok, (pv - bv) / sd, np.nan)
    work["pupil_baseline_status"] = np.where(
        ~finite_pupil, "missing_pupil", np.where(~available, "no_baseline", "corrected")
    )
    work["pupil_baseline_used"] = work["baseline_candidate"].astype(bool)
    work["pupil_baseline_pupil_column"] = pupil_source
    work["pupil_baseline_time_column"] = baseline_time_source
    work["pupil_baseline_flag_column"] = pd.NA if baseline_flag_col is None else baseline_flag_col
    work["pupil_baseline_window_start"] = start_window
    work["pupil_baseline_window_end"] = end_window
    work["pupil_baseline_method"] = baseline_method
    work["pupil_baseline_min_samples"] = min_baseline_samples
    work = work.sort_values("row_id", kind="stable").reset_index(drop=True)
    output_cols = [
        "pupil_baseline_value",
        "pupil_baseline_sd",
        "pupil_baseline_n",
        "pupil_baseline_available",
        "pupil_baseline_time_min",
        "pupil_baseline_time_max",
        "pupil_baseline_used",
        "pupil_baseline_corrected",
        "pupil_baseline_percent_change",
        "pupil_baseline_ratio",
        "pupil_baseline_z",
        "pupil_baseline_status",
        "pupil_baseline_pupil_column",
        "pupil_baseline_time_column",
        "pupil_baseline_flag_column",
        "pupil_baseline_window_start",
        "pupil_baseline_window_end",
        "pupil_baseline_method",
        "pupil_baseline_min_samples",
    ]
    original = df.drop(columns=[col for col in output_cols if col in df.columns]).reset_index(
        drop=True
    )
    return pd.concat([original, work[output_cols]], axis=1)


def audit_gazepoint_pupil_gaps(
    data,
    group_cols=None,
    status_col="pupil_interpolation_status",
    gap_id_col="pupil_gap_id",
    gap_n_samples_col="pupil_gap_n_samples",
    gap_duration_col="pupil_gap_duration_ms",
    interpolated_col="pupil_was_interpolated",
    pupil_col="pupil_interpolated",
    *,
    time_col=None,
) -> pd.DataFrame:
    """Audit pupil gaps with R v2.3.0 semantics and legacy raw-gap compatibility."""
    df = ensure_dataframe(data, copy=False)
    if time_col is not None or status_col not in df.columns:
        resolved_pupil = infer_column(
            df, "pupil", pupil_col if pupil_col in df.columns else None, required=True
        )
        groups = normalize_group_cols(df, group_cols)
        rows = []
        iterator = [((), df)] if not groups else df.groupby(groups, dropna=False, sort=False)
        for key, frame in iterator:
            if groups and not isinstance(key, tuple):
                key = (key,)
            miss = finite_numeric(frame[resolved_pupil]).isna().to_numpy()
            labels, n = ndimage.label(miss.astype(int))
            lengths = [int((labels == lab).sum()) for lab in range(1, n + 1)]
            row = {c: v for c, v in zip(groups, key, strict=False)} if groups else {}
            row.update(
                n_samples=len(frame),
                n_missing=int(miss.sum()),
                missing_prop=float(miss.mean()),
                n_gaps=n,
                max_gap_samples=max(lengths, default=0),
                mean_gap_samples=float(np.mean(lengths)) if lengths else 0.0,
            )
            rows.append(row)
        return pd.DataFrame(rows)

    if group_cols is None:
        group_cols = ["subject", "media_id"]
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    else:
        group_cols = list(group_cols)
    if len(set(group_cols)) != len(group_cols) or any(
        not isinstance(col, str) or not col for col in group_cols
    ):
        raise ValueError("group_cols must be a character vector of unique column names")
    column_args = [
        status_col,
        gap_id_col,
        gap_n_samples_col,
        gap_duration_col,
        interpolated_col,
        pupil_col,
    ]
    if any(not isinstance(col, str) or not col for col in column_args):
        raise ValueError("Column-name arguments must be non-missing character scalars")
    missing = [col for col in dict.fromkeys(group_cols + column_args) if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    def logical_flag(series):
        if pd.api.types.is_bool_dtype(series):
            return series.fillna(False).to_numpy(bool)
        if pd.api.types.is_numeric_dtype(series):
            values = pd.to_numeric(series, errors="coerce").to_numpy(float)
            return np.isfinite(values) & (values != 0)
        return (
            series.astype("string")
            .str.strip()
            .str.lower()
            .isin(["true", "t", "1", "yes", "y"])
            .to_numpy(bool)
        )

    work = df.copy()
    work[".status"] = work[status_col].astype("string")
    work[".was"] = logical_flag(work[interpolated_col]) | work[".status"].eq("interpolated").fillna(
        False
    ).to_numpy(bool)
    work[".gap_id"] = work[gap_id_col]
    work[".gap_n"] = pd.to_numeric(work[gap_n_samples_col], errors="coerce")
    work[".gap_duration"] = pd.to_numeric(work[gap_duration_col], errors="coerce")
    work[".pupil"] = pd.to_numeric(work[pupil_col], errors="coerce")

    no_time = {"missing_no_time", "missing_no_time_gap"}
    insufficient = {"missing_insufficient_valid_samples", "missing_insufficient_valid"}
    unfilled = {"missing_unfilled", "unfilled"}
    rows = []
    iterator = [((), work)] if not group_cols else work.groupby(group_cols, dropna=False, sort=True)
    for key, frame in iterator:
        if group_cols and not isinstance(key, tuple):
            key = (key,)
        base = {col: value for col, value in zip(group_cols, key, strict=False)}
        status = frame[".status"].astype("string")
        pupil = pd.to_numeric(frame[".pupil"], errors="coerce")
        n_rows = len(frame)
        row = {
            **base,
            "n_rows": n_rows,
            "n_observed_samples": int(status.eq("observed").sum()),
            "n_interpolated_samples": int(frame[".was"].sum()),
            "n_missing_edge_gap_samples": int(status.eq("missing_edge_gap").sum()),
            "n_missing_long_gap_samples": int(status.eq("missing_long_gap").sum()),
            "n_missing_no_time_samples": int(status.isin(no_time).sum()),
            "n_missing_insufficient_valid_samples": int(status.isin(insufficient).sum()),
            "n_missing_unfilled_samples": int(status.isin(unfilled).sum()),
            "n_remaining_missing_samples": int(pupil.isna().sum()),
            "n_total_missing_or_gap_samples": int(
                ((status.notna() & ~status.eq("observed")) | pupil.isna()).sum()
            ),
        }
        row["pct_observed_samples"] = 100 * row["n_observed_samples"] / n_rows if n_rows else np.nan
        row["pct_interpolated_samples"] = (
            100 * row["n_interpolated_samples"] / n_rows if n_rows else np.nan
        )
        row["pct_remaining_missing_samples"] = (
            100 * row["n_remaining_missing_samples"] / n_rows if n_rows else np.nan
        )
        gaps = frame.loc[frame[".gap_id"].notna()].groupby(".gap_id", dropna=False, sort=True)
        gap_records = []
        for _, gap in gaps:
            gap_status = gap[".status"].astype("string")
            nvals = pd.to_numeric(gap[".gap_n"], errors="coerce").to_numpy(float)
            dvals = pd.to_numeric(gap[".gap_duration"], errors="coerce").to_numpy(float)
            nfinite = nvals[np.isfinite(nvals)]
            dfinite = dvals[np.isfinite(dvals)]
            gap_records.append(
                {
                    "interpolated": bool(gap[".was"].any()),
                    "edge": bool(gap_status.eq("missing_edge_gap").any()),
                    "long": bool(gap_status.eq("missing_long_gap").any()),
                    "n": float(nfinite.max()) if len(nfinite) else np.nan,
                    "duration": float(dfinite.max()) if len(dfinite) else np.nan,
                }
            )
        row["n_gaps_total"] = len(gap_records)
        row["n_gaps_interpolated"] = sum(record["interpolated"] for record in gap_records)
        row["n_gaps_edge"] = sum(record["edge"] for record in gap_records)
        row["n_gaps_long"] = sum(record["long"] for record in gap_records)
        durations = np.array([record["duration"] for record in gap_records], dtype=float)
        ns = np.array([record["n"] for record in gap_records], dtype=float)
        durations = durations[np.isfinite(durations)]
        ns = ns[np.isfinite(ns)]
        row["mean_gap_duration_ms"] = float(durations.mean()) if len(durations) else np.nan
        row["max_gap_duration_ms"] = float(durations.max()) if len(durations) else np.nan
        row["mean_gap_n_samples"] = float(ns.mean()) if len(ns) else np.nan
        row["max_gap_n_samples"] = float(ns.max()) if len(ns) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _gp3_r_first_present(data, candidates):
    for candidate in candidates:
        if candidate in data.columns:
            return candidate
    return None


def _gp3_r_bool(values):
    series = pd.Series(values, copy=False)
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series.dtype):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.notna() & numeric.ne(0)
    text = series.astype("string").str.strip().str.lower()
    return text.isin(["true", "t", "1", "yes", "y"])


def _gp3_r_group_parts(data, group_cols):
    if not group_cols:
        return [((), data)]
    grouper = group_cols[0] if len(group_cols) == 1 else group_cols
    return data.groupby(grouper, dropna=False, sort=False)


def _gp3_r_group_row(group_cols, key):
    if not group_cols:
        return {}
    if len(group_cols) == 1:
        key = (key,)
    return dict(zip(group_cols, key, strict=True))


def audit_gazepoint_pupil_baseline(
    data,
    pupil_col=None,
    time_col=None,
    baseline=(-200, 0),
    group_cols=None,
    *,
    baseline_n_col=None,
    baseline_status_col=None,
    baseline_available_col=None,
    baseline_used_col=None,
    baseline_window_start_col=None,
    baseline_window_end_col=None,
    baseline_flag_col=None,
    interpolated_col=None,
    artifact_col=None,
    artifact_reason_col=None,
    min_baseline_samples=None,
    max_missing_pct=None,
    max_interpolated_pct=None,
    max_artifact_pct=None,
) -> pd.DataFrame:
    """Audit pupil baselines using legacy or R v2.3.0 semantics."""
    r_mode = any(
        value is not None
        for value in (
            baseline_n_col,
            baseline_status_col,
            baseline_available_col,
            baseline_used_col,
            baseline_window_start_col,
            baseline_window_end_col,
            baseline_flag_col,
            interpolated_col,
            artifact_col,
            artifact_reason_col,
            min_baseline_samples,
            max_missing_pct,
            max_interpolated_pct,
            max_artifact_pct,
        )
    )

    if not r_mode:
        df = ensure_dataframe(data, copy=False)
        pupil_col = infer_column(df, "pupil", pupil_col, required=True)
        time_col = infer_column(df, "time", time_col, required=True)
        groups = normalize_group_cols(df, group_cols)
        work = df.loc[finite_numeric(df[time_col]).between(*baseline)].copy()
        x = finite_numeric(work[pupil_col])
        work["_p"] = x
        if groups:
            return (
                work.groupby(groups, dropna=False)
                .agg(
                    n_baseline=("_p", "size"),
                    n_valid=("_p", "count"),
                    baseline_mean=("_p", "mean"),
                    baseline_sd=("_p", "std"),
                )
                .reset_index()
                .assign(missing_prop=lambda z: 1 - z.n_valid / z.n_baseline)
            )
        return pd.DataFrame(
            [
                {
                    "n_baseline": len(work),
                    "n_valid": int(x.notna().sum()),
                    "baseline_mean": float(x.mean()),
                    "baseline_sd": float(x.std()),
                    "missing_prop": float(x.isna().mean()),
                }
            ]
        )

    df = ensure_dataframe(data, copy=False)
    groups = (
        ["subject", "media_id"]
        if group_cols is None
        else ([group_cols] if isinstance(group_cols, str) else list(group_cols))
    )
    if len(groups) != len(set(groups)) or any(not isinstance(c, str) or not c for c in groups):
        raise ValueError("group_cols must contain unique non-empty column names")

    time_col = "time" if time_col is None else time_col
    pupil_col = "pupil_interpolated" if pupil_col is None else pupil_col
    baseline_n_col = "pupil_baseline_n" if baseline_n_col is None else baseline_n_col
    baseline_status_col = (
        "pupil_baseline_status" if baseline_status_col is None else baseline_status_col
    )
    baseline_available_col = (
        "pupil_baseline_available" if baseline_available_col is None else baseline_available_col
    )
    baseline_used_col = "pupil_baseline_used" if baseline_used_col is None else baseline_used_col
    baseline_window_start_col = (
        "pupil_baseline_window_start"
        if baseline_window_start_col is None
        else baseline_window_start_col
    )
    baseline_window_end_col = (
        "pupil_baseline_window_end" if baseline_window_end_col is None else baseline_window_end_col
    )
    interpolated_col = "pupil_was_interpolated" if interpolated_col is None else interpolated_col
    min_baseline_samples = 1 if min_baseline_samples is None else float(min_baseline_samples)
    max_missing_pct = 50 if max_missing_pct is None else float(max_missing_pct)
    max_interpolated_pct = 50 if max_interpolated_pct is None else float(max_interpolated_pct)
    max_artifact_pct = 50 if max_artifact_pct is None else float(max_artifact_pct)

    if artifact_col is None:
        artifact_col = _gp3_r_first_present(
            df, ["pupil_artifact_flag", "pupil_flag_invalid", "artifact_flag"]
        )
    if artifact_reason_col is None:
        artifact_reason_col = _gp3_r_first_present(
            df, ["pupil_artifact_reason", "pupil_flag_reason", "artifact_reason"]
        )

    required = groups + [
        time_col,
        pupil_col,
        baseline_n_col,
        baseline_status_col,
        baseline_available_col,
        baseline_used_col,
        baseline_window_start_col,
        baseline_window_end_col,
        interpolated_col,
    ]
    for optional in (baseline_flag_col, artifact_col, artifact_reason_col):
        if optional is not None:
            required.append(optional)
    missing = [c for c in dict.fromkeys(required) if c not in df.columns]
    if missing:
        raise KeyError("Missing required columns: " + ", ".join(missing))

    work = df.copy()
    time = pd.to_numeric(work[time_col], errors="coerce")
    start = pd.to_numeric(work[baseline_window_start_col], errors="coerce")
    end = pd.to_numeric(work[baseline_window_end_col], errors="coerce")
    if baseline_flag_col is None:
        is_baseline = time.notna() & start.notna() & end.notna() & time.ge(start) & time.le(end)
    else:
        is_baseline = _gp3_r_bool(work[baseline_flag_col])
    work["_gp3_is_baseline"] = is_baseline
    work["_gp3_pupil"] = pd.to_numeric(work[pupil_col], errors="coerce")
    work["_gp3_n"] = pd.to_numeric(work[baseline_n_col], errors="coerce")
    work["_gp3_status"] = work[baseline_status_col].astype("string")
    work["_gp3_available"] = _gp3_r_bool(work[baseline_available_col])
    work["_gp3_used"] = _gp3_r_bool(work[baseline_used_col])
    work["_gp3_interp"] = _gp3_r_bool(work[interpolated_col])
    if artifact_col is not None:
        work["_gp3_artifact"] = _gp3_r_bool(work[artifact_col])
    elif artifact_reason_col is not None:
        reason = work[artifact_reason_col].astype("string")
        work["_gp3_artifact"] = reason.notna() & reason.ne("") & reason.ne("valid")
    else:
        work["_gp3_artifact"] = pd.Series(pd.NA, index=work.index, dtype="boolean")

    def first_nonmissing(series):
        values = series.dropna()
        return values.iloc[0] if len(values) else np.nan

    rows = []
    for key, part in _gp3_r_group_parts(work, groups):
        row = _gp3_r_group_row(groups, key)
        baseline_mask = part["_gp3_is_baseline"].fillna(False).astype(bool)
        baseline_part = part.loc[baseline_mask]
        n_rows = len(part)
        n_baseline_rows = len(baseline_part)
        n_valid = int(baseline_part["_gp3_pupil"].notna().sum())
        n_missing = int(baseline_part["_gp3_pupil"].isna().sum())
        n_interp = int(baseline_part["_gp3_interp"].fillna(False).sum())
        artifact_values = baseline_part["_gp3_artifact"]
        n_artifact = int(artifact_values.fillna(False).sum()) if len(artifact_values) else 0
        n_values = part["_gp3_n"].dropna()
        status = first_nonmissing(part["_gp3_status"])
        available = first_nonmissing(part["_gp3_available"])
        used = first_nonmissing(part["_gp3_used"])
        baseline_n_min = float(n_values.min()) if len(n_values) else np.nan
        baseline_n_mean = float(n_values.mean()) if len(n_values) else np.nan
        baseline_n_max = float(n_values.max()) if len(n_values) else np.nan
        missing_pct = 100 * n_missing / n_baseline_rows if n_baseline_rows else np.nan
        interp_pct = 100 * n_interp / n_baseline_rows if n_baseline_rows else np.nan
        artifact_pct = 100 * n_artifact / n_baseline_rows if n_baseline_rows else np.nan
        available_bool = bool(available) if not pd.isna(available) else False
        no_baseline = (
            status == "no_baseline"
            or not available_bool
            or (np.isfinite(baseline_n_max) and baseline_n_max < min_baseline_samples)
        )
        low_quality = (
            no_baseline
            or not np.isfinite(baseline_n_max)
            or baseline_n_max < min_baseline_samples
            or (np.isfinite(missing_pct) and missing_pct > max_missing_pct)
            or (np.isfinite(interp_pct) and interp_pct > max_interpolated_pct)
            or (np.isfinite(artifact_pct) and artifact_pct > max_artifact_pct)
        )
        if no_baseline:
            reason = "no_baseline"
        elif not np.isfinite(baseline_n_max):
            reason = "missing_baseline_n"
        elif baseline_n_max < min_baseline_samples:
            reason = "too_few_baseline_samples"
        elif np.isfinite(missing_pct) and missing_pct > max_missing_pct:
            reason = "high_baseline_missing_pct"
        elif np.isfinite(interp_pct) and interp_pct > max_interpolated_pct:
            reason = "high_baseline_interpolated_pct"
        elif np.isfinite(artifact_pct) and artifact_pct > max_artifact_pct:
            reason = "high_baseline_artifact_pct"
        else:
            reason = "ok"
        row.update(
            n_rows=n_rows,
            n_baseline_rows=n_baseline_rows,
            n_baseline_valid_samples=n_valid,
            n_baseline_missing_samples=n_missing,
            n_baseline_interpolated_samples=n_interp,
            n_baseline_artifact_samples=n_artifact,
            baseline_missing_pct=missing_pct,
            baseline_interpolated_pct=interp_pct,
            baseline_artifact_pct=artifact_pct,
            baseline_n_min=baseline_n_min,
            baseline_n_mean=baseline_n_mean,
            baseline_n_max=baseline_n_max,
            baseline_status=status,
            baseline_available=available,
            baseline_used=used,
            n_no_baseline_rows=int(part["_gp3_status"].eq("no_baseline").sum()),
            n_missing_pupil_baseline_rows=int(part["_gp3_status"].eq("missing_pupil").sum()),
            no_baseline_case=bool(no_baseline),
            low_quality_baseline_flag=bool(low_quality),
            baseline_quality_reason=reason,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def audit_gazepoint_pupil_drift(
    data,
    pupil_col=None,
    time_col=None,
    group_cols=None,
    *,
    order_col=None,
    condition_col=None,
    exclude_col=None,
    include_excluded: bool = False,
    min_valid_samples: int = 3,
    max_abs_slope_per_min: float = 1.0,
    max_condition_time_mean_diff_ms: float = 1000.0,
    max_condition_order_mean_diff: float = 1.0,
):
    """Audit pupil drift using legacy or R v2.3.0 semantics."""
    df = ensure_dataframe(data, copy=False)
    r_mode = (
        order_col is not None
        or condition_col is not None
        or exclude_col is not None
        or include_excluded
        or min_valid_samples != 3
        or max_abs_slope_per_min != 1.0
        or max_condition_time_mean_diff_ms != 1000.0
        or max_condition_order_mean_diff != 1.0
    )
    if not r_mode:
        pupil_col = infer_column(df, "pupil", pupil_col, required=True)
        time_col = infer_column(df, "time", time_col, required=True)
        groups = normalize_group_cols(df, group_cols)
        rows = []
        iterator = [((), df)] if not groups else df.groupby(groups, dropna=False, sort=False)
        for key, frame in iterator:
            if groups and not isinstance(key, tuple):
                key = (key,)
            t = finite_numeric(frame[time_col]).to_numpy(float)
            y = finite_numeric(frame[pupil_col]).to_numpy(float)
            ok = np.isfinite(t) & np.isfinite(y)
            slope, intercept, r, p, se = (
                (np.nan,) * 5 if ok.sum() < 3 else stats.linregress(t[ok], y[ok])
            )
            row = {c: v for c, v in zip(groups, key, strict=False)} if groups else {}
            row.update(n=int(ok.sum()), slope=slope, r=r, p_value=p)
            rows.append(row)
        return pd.DataFrame(rows)

    if group_cols is None:
        groups = ["subject"]
    elif isinstance(group_cols, str):
        groups = [group_cols]
    else:
        groups = list(group_cols)
    if (
        not groups
        or any(not isinstance(c, str) or not c for c in groups)
        or len(set(groups)) != len(groups)
    ):
        raise ValueError("group_cols must contain unique non-empty column names")
    if not isinstance(min_valid_samples, (int, np.integer)) or min_valid_samples < 1:
        raise ValueError("min_valid_samples must be a positive integer")
    for value, label in [
        (max_abs_slope_per_min, "max_abs_slope_per_min"),
        (max_condition_time_mean_diff_ms, "max_condition_time_mean_diff_ms"),
        (max_condition_order_mean_diff, "max_condition_order_mean_diff"),
    ]:
        if not np.isfinite(value):
            raise ValueError(f"{label} must be a finite numeric scalar")

    if pupil_col is None:
        pupil_col = next(
            (
                c
                for c in [
                    "pupil_smoothed",
                    "pupil_baseline_corrected",
                    "pupil_interpolated",
                    "pupil_clean",
                    "pupil",
                ]
                if c in df
            ),
            None,
        )
    if pupil_col is None:
        raise KeyError("Could not automatically detect a pupil column")
    time_col = "time" if time_col is None else time_col
    order_col = "trial" if order_col is None else order_col
    condition_col = "condition" if condition_col is None else condition_col
    exclude_col = "excluded_trial" if exclude_col is None else exclude_col

    required = list(
        dict.fromkeys(
            groups
            + [pupil_col, time_col]
            + ([order_col] if order_col else [])
            + ([condition_col] if condition_col else [])
        )
    )
    missing = [c for c in required if c not in df]
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")

    work = df.copy()
    work[".gp3_drift_pupil"] = pd.to_numeric(work[pupil_col], errors="coerce")
    work[".gp3_drift_time"] = pd.to_numeric(work[time_col], errors="coerce")
    work[".gp3_drift_order"] = (
        pd.to_numeric(work[order_col], errors="coerce") if order_col else np.nan
    )
    work[".gp3_drift_condition"] = (
        work[condition_col].astype("string")
        if condition_col
        else pd.Series(pd.NA, index=work.index, dtype="string")
    )

    if exclude_col and exclude_col in work and not include_excluded:
        raw = work[exclude_col]
        if pd.api.types.is_bool_dtype(raw):
            excluded = raw.fillna(False).astype(bool)
        elif pd.api.types.is_numeric_dtype(raw):
            excluded = pd.to_numeric(raw, errors="coerce").fillna(0).ne(0)
        else:
            excluded = (
                raw.astype("string").str.strip().str.lower().isin(["true", "t", "1", "yes", "y"])
            )
        work = work.loc[~excluded].copy()

    def _num(series):
        return pd.to_numeric(series, errors="coerce").astype(float)

    def _summary(frame, cols):
        rows = []
        iterator = [((), frame)] if not cols else frame.groupby(cols, dropna=False, sort=False)
        for key, part in iterator:
            if cols and not isinstance(key, tuple):
                key = (key,)
            y = _num(part[".gp3_drift_pupil"]).to_numpy(float)
            t = _num(part[".gp3_drift_time"]).to_numpy(float)
            o = _num(part[".gp3_drift_order"]).to_numpy(float)
            pair = np.isfinite(y) & np.isfinite(t)
            yv = y[np.isfinite(y)]
            tv = t[np.isfinite(t)]
            ov = o[np.isfinite(o)]
            slope = np.nan
            corr = np.nan
            if pair.sum() >= min_valid_samples and np.unique(t[pair]).size >= 2:
                slope = float(np.polyfit(t[pair], y[pair], 1)[0])
                if np.std(y[pair], ddof=1) > 0 and np.std(t[pair], ddof=1) > 0:
                    corr = float(np.corrcoef(y[pair], t[pair])[0, 1])
            row = {c: v for c, v in zip(cols, key, strict=False)} if cols else {}
            row.update(
                {
                    "n_rows": int(len(part)),
                    "n_valid_pupil": int(pair.sum()),
                    "valid_pupil_pct": 100.0 * pair.sum() / len(part) if len(part) else np.nan,
                    "pupil_mean": float(np.mean(yv)) if len(yv) else np.nan,
                    "pupil_sd": float(np.std(yv, ddof=1)) if len(yv) >= 2 else np.nan,
                    "time_min": float(np.min(tv)) if len(tv) else np.nan,
                    "time_mean": float(np.mean(tv)) if len(tv) else np.nan,
                    "time_max": float(np.max(tv)) if len(tv) else np.nan,
                    "time_range": float(np.ptp(tv)) if len(tv) else np.nan,
                    "order_min": float(np.min(ov)) if len(ov) else np.nan,
                    "order_mean": float(np.mean(ov)) if len(ov) else np.nan,
                    "order_max": float(np.max(ov)) if len(ov) else np.nan,
                    "order_range": float(np.ptp(ov)) if len(ov) else np.nan,
                    "pupil_time_slope_per_ms": slope,
                    "pupil_time_r": corr,
                }
            )
            per_min = slope * 60000 if np.isfinite(slope) else np.nan
            abs_per_min = abs(per_min) if np.isfinite(per_min) else np.nan
            warning = bool(np.isfinite(abs_per_min) and abs_per_min > max_abs_slope_per_min)
            row.update(
                {
                    "pupil_time_slope_per_sec": slope * 1000 if np.isfinite(slope) else np.nan,
                    "pupil_time_slope_per_min": per_min,
                    "abs_pupil_time_slope_per_min": abs_per_min,
                    "drift_direction": "not_estimated"
                    if not np.isfinite(per_min)
                    else ("increasing" if per_min > 0 else "decreasing" if per_min < 0 else "flat"),
                    "drift_warning": warning,
                    "drift_status": "insufficient_valid_samples"
                    if pair.sum() < min_valid_samples
                    else "not_estimated"
                    if not np.isfinite(per_min)
                    else "possible_drift"
                    if warning
                    else "ok",
                }
            )
            rows.append(row)
        return pd.DataFrame(rows)

    by_group = _summary(work, groups)
    by_subject = _summary(work, ["subject"]) if "subject" in work else pd.DataFrame()
    has_condition = bool(condition_col and condition_col in work)
    nonmissing_condition = has_condition and work[condition_col].notna().any()
    by_condition = (
        _summary(work.loc[work[condition_col].notna()], [condition_col])
        if nonmissing_condition
        else pd.DataFrame()
    )

    def _range(series):
        vals = pd.to_numeric(series, errors="coerce").dropna().to_numpy(float)
        return float(np.ptp(vals)) if len(vals) else np.nan

    if nonmissing_condition:
        time_range = _range(by_condition["time_mean"])
        order_range = _range(by_condition["order_mean"])
        time_warn = bool(np.isfinite(time_range) and time_range > max_condition_time_mean_diff_ms)
        order_warn = bool(np.isfinite(order_range) and order_range > max_condition_order_mean_diff)
        reason = (
            "time_mean_diff;order_mean_diff"
            if time_warn and order_warn
            else "time_mean_diff"
            if time_warn
            else "order_mean_diff"
            if order_warn
            else "ok"
        )
        condition_balance = pd.DataFrame(
            [
                {
                    "n_conditions": int(len(by_condition)),
                    "condition_time_mean_range": time_range,
                    "condition_order_mean_range": order_range,
                    "condition_time_imbalance_warning": time_warn,
                    "condition_order_imbalance_warning": order_warn,
                    "condition_balance_warning": time_warn or order_warn,
                    "condition_balance_reason": reason,
                }
            ]
        )
    elif has_condition:
        condition_balance = pd.DataFrame(
            [
                {
                    "n_conditions": 0,
                    "condition_time_mean_range": np.nan,
                    "condition_order_mean_range": np.nan,
                    "condition_time_imbalance_warning": False,
                    "condition_order_imbalance_warning": False,
                    "condition_balance_warning": False,
                    "condition_balance_reason": "no_non_missing_conditions",
                }
            ]
        )
    else:
        condition_balance = pd.DataFrame(
            [
                {
                    "n_conditions": np.nan,
                    "condition_time_mean_range": np.nan,
                    "condition_order_mean_range": np.nan,
                    "condition_time_imbalance_warning": np.nan,
                    "condition_order_imbalance_warning": np.nan,
                    "condition_balance_warning": np.nan,
                    "condition_balance_reason": "condition_col_not_available",
                }
            ]
        )

    summary = pd.DataFrame(
        [
            {
                "n_rows": int(len(work)),
                "pupil_column": pupil_col,
                "time_column": time_col,
                "order_column": order_col,
                "condition_column": condition_col,
                "n_by_group": int(len(by_group)),
                "n_subjects": int(len(by_subject)),
                "n_conditions": condition_balance.iloc[0]["n_conditions"],
                "n_group_drift_warnings": int(by_group["drift_warning"].sum())
                if len(by_group)
                else 0,
                "n_subject_drift_warnings": int(by_subject["drift_warning"].sum())
                if len(by_subject)
                else np.nan,
                "condition_balance_warning": condition_balance.iloc[0]["condition_balance_warning"],
            }
        ]
    )
    return {
        "by_group": by_group,
        "by_subject": by_subject,
        "by_condition": by_condition,
        "condition_balance": condition_balance,
        "summary": summary,
        "_gp3_class": "gp3_pupil_drift_audit",
    }


def audit_gazepoint_pupil_reliability(
    data,
    pupil_col=None,
    subject_col=None,
    split_col=None,
    *,
    outcome_cols=None,
    participant_col=None,
    trial_col=None,
    by_cols=None,
    split_method="odd_even",
    aggregate_function="mean",
    correlation_method="pearson",
    min_trials_per_split: int = 2,
    name="gazepoint_pupil_reliability",
):
    """Audit split-half pupil reliability with legacy or R v2.3.0 outputs."""
    df = ensure_dataframe(data, copy=False)
    r_mode = (
        outcome_cols is not None
        or participant_col is not None
        or trial_col is not None
        or by_cols is not None
        or split_method != "odd_even"
        or aggregate_function != "mean"
        or correlation_method != "pearson"
        or min_trials_per_split != 2
        or name != "gazepoint_pupil_reliability"
    )
    if not r_mode:
        pupil_col = infer_column(df, "pupil", pupil_col, required=True)
        subject_col = infer_column(df, "subject", subject_col)
        if subject_col is None:
            x = finite_numeric(df[pupil_col])
            half = np.arange(len(df)) % 2
            means = pd.DataFrame({"half": half, "x": x}).groupby("half").x.mean()
            return pd.DataFrame(
                [
                    {
                        "split_half_correlation": np.nan,
                        "mean_even": means.get(0, np.nan),
                        "mean_odd": means.get(1, np.nan),
                    }
                ]
            )
        work = df[[subject_col, pupil_col]].copy()
        work["_half"] = work.groupby(subject_col).cumcount() % 2
        work["_p"] = finite_numeric(work[pupil_col])
        piv = work.groupby([subject_col, "_half"])._p.mean().unstack()
        r = float(piv.corr().iloc[0, 1]) if piv.shape[1] >= 2 else np.nan
        sb = 2 * r / (1 + r) if np.isfinite(r) and r != -1 else np.nan
        return pd.DataFrame(
            [{"n_subjects": len(piv), "split_half_correlation": r, "spearman_brown": sb}]
        )

    if len(df) == 0:
        raise ValueError("data must contain at least one row")
    if split_method not in {"odd_even", "first_second"}:
        raise ValueError("split_method must be odd_even or first_second")
    if aggregate_function not in {"mean", "median"}:
        raise ValueError("aggregate_function must be mean or median")
    if correlation_method not in {"pearson", "spearman"}:
        raise ValueError("correlation_method must be pearson or spearman")
    if not isinstance(min_trials_per_split, (int, np.integer)) or min_trials_per_split < 1:
        raise ValueError("min_trials_per_split must be a positive integer")
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")

    names = set(df.columns)
    if participant_col is None:
        participant_col = next(
            (
                c
                for c in [
                    "subject",
                    "participant",
                    "participant_id",
                    "pID",
                    "USER_FILE",
                    "user",
                    "user_id",
                    "recording_id",
                ]
                if c in names
            ),
            None,
        )
    if participant_col is None or participant_col not in df:
        raise KeyError("participant_col could not be detected")
    if trial_col is None:
        trial_col = next(
            (
                c
                for c in [
                    "trial_global",
                    "trial",
                    "trial_id",
                    "TRIAL_INDEX",
                    "trial_number",
                    "item_trial",
                    "sample_index",
                ]
                if c in names
            ),
            None,
        )
    elif trial_col not in df:
        raise KeyError("trial_col must be present in data")
    if split_col is not None and split_col not in df:
        raise KeyError("split_col must be present in data")
    by = [] if by_cols is None else ([by_cols] if isinstance(by_cols, str) else list(by_cols))
    missing_by = [c for c in by if c not in df]
    if missing_by:
        raise KeyError(f"Missing by_cols: {missing_by}")

    if outcome_cols is None:
        preferred = [
            "auc_pupil_0_2000",
            "mean_pupil_0_2000",
            "peak_pupil_0_2000",
            "latency_to_peak_ms",
            "mean_pupil_window",
            "pupil_window_mean",
            "pupil_mean",
            "pupil_peak",
            "pupil_auc",
            "mean_pupil",
            "peak_pupil",
            "auc_pupil",
            "pupil",
        ]
        outcomes = [c for c in preferred if c in df and pd.api.types.is_numeric_dtype(df[c])]
        if not outcomes:
            excluded = {participant_col, trial_col, split_col, *by}
            bad_tokens = (
                "count",
                "prop",
                "percent",
                "missing",
                "valid",
                "sample",
                "trial",
                "time_bin",
                "order",
            )
            outcomes = [
                c
                for c in df.columns
                if c not in excluded
                and pd.api.types.is_numeric_dtype(df[c])
                and not c.startswith("n_")
                and not c.endswith("_n")
                and not any(token in c.lower() for token in bad_tokens)
            ]
    else:
        outcomes = [outcome_cols] if isinstance(outcome_cols, str) else list(outcome_cols)
        missing = [c for c in outcomes if c not in df]
        if missing:
            raise KeyError(f"Missing outcome_cols: {missing}")
        outcomes = [c for c in outcomes if pd.api.types.is_numeric_dtype(df[c])]
    if not outcomes:
        raise ValueError("outcome_cols must include at least one numeric column")
    by = [c for c in by if c not in {participant_col, trial_col, split_col, *outcomes}]

    if trial_col is not None:
        trial_raw = df[trial_col]
        trial_text = trial_raw.astype("string")
        if pd.api.types.is_numeric_dtype(trial_raw):
            trial_order = pd.to_numeric(trial_raw, errors="coerce").to_numpy(float)
        else:
            trial_order = np.array(
                [
                    float(m[-1]) if (m := __import__("re").findall(r"[0-9]+", str(v))) else np.nan
                    for v in trial_raw
                ]
            )
    else:
        trial_text = pd.Series(np.arange(1, len(df) + 1), index=df.index).astype("string")
        trial_order = np.arange(1, len(df) + 1, dtype=float)

    split_data = pd.DataFrame(
        {
            ".row_id": np.arange(1, len(df) + 1),
            "participant": df[participant_col].astype("string"),
            "trial": trial_text,
            "trial_order": trial_order,
        }
    )
    for c in by + outcomes:
        split_data[c] = df[c].to_numpy(copy=False)

    if split_col is not None:
        raw = df[split_col].astype("string").replace("", pd.NA)
        levels = sorted(raw.dropna().unique().tolist())
        if len(levels) != 2:
            raise ValueError("split_col must contain exactly two non-missing split levels")
        split_data["split"] = raw
        split_data = split_data.loc[split_data["split"].notna()].copy()
        split_levels = levels
        used_split_method = "predefined_split_col"
    else:
        split_parts = []
        key_cols = ["participant"] + by
        for _, part in split_data.groupby(key_cols, dropna=False, sort=False):
            part = part.sort_values(
                ["trial_order", ".row_id"], kind="stable", na_position="last"
            ).copy()
            part[".within_group_order"] = np.arange(1, len(part) + 1)
            part[".n_in_group"] = len(part)
            split_number = part["trial_order"].to_numpy(float)
            fallback = part[".within_group_order"].to_numpy(float)
            split_number = np.where(np.isfinite(split_number), split_number, fallback)
            part[".split_number"] = split_number
            if split_method == "odd_even":
                part["split"] = np.where(split_number % 2 == 0, "even", "odd")
            else:
                part["split"] = np.where(
                    part[".within_group_order"] <= np.ceil(len(part) / 2), "first", "second"
                )
            split_parts.append(part)
        split_data = pd.concat(split_parts, ignore_index=True)
        split_levels = ["odd", "even"] if split_method == "odd_even" else ["first", "second"]
        used_split_method = split_method

    long = split_data.melt(
        id_vars=[".row_id", "participant", "trial", "trial_order"]
        + by
        + ["split"]
        + [c for c in [".within_group_order", ".n_in_group", ".split_number"] if c in split_data],
        value_vars=outcomes,
        var_name="outcome",
        value_name="value",
    )
    split_rows = []
    group_keys = ["participant"] + by + ["outcome", "split"]
    for key, part in long.groupby(group_keys, dropna=False, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        vals = pd.to_numeric(part["value"], errors="coerce")
        finite = vals[np.isfinite(vals.to_numpy(float))]
        split_value = (
            float(finite.median() if aggregate_function == "median" else finite.mean())
            if len(finite)
            else np.nan
        )
        row = {c: v for c, v in zip(group_keys, key, strict=False)}
        row.update(n_trials=int(len(part)), n_valid=int(len(finite)), split_value=split_value)
        split_rows.append(row)
    split_summary = pd.DataFrame(split_rows)

    key_cols = ["participant"] + by + ["outcome"]

    def _side(label, prefix):
        side = split_summary.loc[
            split_summary["split"].astype(str) == str(label),
            key_cols + ["n_trials", "n_valid", "split_value"],
        ].copy()
        return side.rename(
            columns={
                "n_trials": f"{prefix}_n_trials",
                "n_valid": f"{prefix}_n_valid",
                "split_value": f"{prefix}_value",
            }
        )

    left = _side(split_levels[0], "split1")
    right = _side(split_levels[1], "split2")
    pairs = left.merge(right, on=key_cols, how="outer", sort=False)
    pairs["split1_label"] = split_levels[0]
    pairs["split2_label"] = split_levels[1]
    if len(pairs):
        pairs["complete_pair"] = (
            np.isfinite(pd.to_numeric(pairs["split1_value"], errors="coerce"))
            & np.isfinite(pd.to_numeric(pairs["split2_value"], errors="coerce"))
            & (pairs["split1_n_valid"].fillna(0) >= min_trials_per_split)
            & (pairs["split2_n_valid"].fillna(0) >= min_trials_per_split)
        )
    else:
        pairs["complete_pair"] = pd.Series(dtype=bool)

    rel_rows = []
    rel_keys = by + ["outcome"]
    for key, part in pairs.groupby(rel_keys, dropna=False, sort=False) if len(pairs) else []:
        if not isinstance(key, tuple):
            key = (key,)
        ok = part["complete_pair"].fillna(False).to_numpy(bool)
        x = pd.to_numeric(part.loc[ok, "split1_value"], errors="coerce").to_numpy(float)
        y = pd.to_numeric(part.loc[ok, "split2_value"], errors="coerce").to_numpy(float)
        n_complete = len(x)
        sd1 = float(np.std(x, ddof=1)) if n_complete >= 2 else np.nan
        sd2 = float(np.std(y, ddof=1)) if n_complete >= 2 else np.nan
        corr = np.nan
        if n_complete >= 3 and np.isfinite(sd1) and np.isfinite(sd2) and sd1 != 0 and sd2 != 0:
            corr = (
                float(np.corrcoef(x, y)[0, 1])
                if correlation_method == "pearson"
                else float(stats.spearmanr(x, y).statistic)
            )
        sb = 2 * corr / (1 + corr) if np.isfinite(corr) and (1 + corr) != 0 else np.nan
        status = (
            "too_few_complete_pairs"
            if n_complete < 3 or not np.isfinite(sd1) or not np.isfinite(sd2)
            else "constant_split_values"
            if sd1 == 0 or sd2 == 0
            else "correlation_unavailable"
            if not np.isfinite(corr)
            else "ready"
        )
        row = {c: v for c, v in zip(rel_keys, key, strict=False)}
        row.update(
            {
                "n_participants": int(part["participant"].nunique(dropna=True)),
                "n_complete_pairs": int(n_complete),
                "split1_label": split_levels[0],
                "split2_label": split_levels[1],
                "split1_mean": float(np.mean(x)) if n_complete else np.nan,
                "split2_mean": float(np.mean(y)) if n_complete else np.nan,
                "split1_sd": sd1,
                "split2_sd": sd2,
                "split_half_correlation": corr,
                "spearman_brown_reliability": sb,
                "reliability_status": status,
            }
        )
        rel_rows.append(row)
    reliability_summary = pd.DataFrame(rel_rows)

    n_by_groups = 1 if not by else split_data[by].drop_duplicates().shape[0]
    overview = pd.DataFrame(
        [
            {
                "object_name": name,
                "n_input_rows": int(len(df)),
                "n_rows_used": int(len(split_data)),
                "n_participants": int(split_data["participant"].nunique(dropna=True)),
                "n_outcomes": int(len(outcomes)),
                "n_by_groups": int(n_by_groups),
                "n_reliability_rows": int(len(reliability_summary)),
                "n_ready_reliability_rows": int(
                    (
                        reliability_summary.get("reliability_status", pd.Series(dtype=object))
                        == "ready"
                    ).sum()
                ),
                "split_method": used_split_method,
                "aggregate_function": aggregate_function,
                "correlation_method": correlation_method,
                "min_trials_per_split": int(min_trials_per_split),
            }
        ]
    )

    def _collapse(value):
        if value is None:
            return "NULL"
        if isinstance(value, (list, tuple)):
            return ",".join(map(str, value)) if value else "NULL"
        return str(value)

    settings = pd.DataFrame(
        {
            "setting": [
                "outcome_cols",
                "participant_col",
                "trial_col",
                "split_col",
                "by_cols",
                "split_method",
                "aggregate_function",
                "correlation_method",
                "min_trials_per_split",
                "name",
            ],
            "value": [
                _collapse(outcomes),
                participant_col,
                _collapse(trial_col),
                _collapse(split_col),
                _collapse(by),
                split_method,
                aggregate_function,
                correlation_method,
                str(min_trials_per_split),
                name,
            ],
        }
    )
    return {
        "overview": overview,
        "split_data": split_data,
        "split_summary": split_summary,
        "reliability_pairs": pairs,
        "reliability_summary": reliability_summary,
        "settings": settings,
        "_gp3_class": "gp3_pupil_reliability_audit",
    }


def audit_gazepoint_pupil_imbalance(
    data,
    pupil_col=None,
    condition_col=None,
    *,
    group_cols=None,
    interpolated_col="pupil_was_interpolated",
    interpolation_status_col="pupil_interpolation_status",
    artifact_col=None,
    artifact_reason_col=None,
    min_group_n=1,
    max_valid_pct_diff=10,
    max_artifact_pct_diff=10,
    max_missing_pct_diff=10,
    max_interpolated_pct_diff=10,
) -> pd.DataFrame:
    """Audit pupil preprocessing balance using legacy or R v2.3.0 semantics."""
    r_mode = (
        group_cols is not None
        or interpolated_col != "pupil_was_interpolated"
        or interpolation_status_col != "pupil_interpolation_status"
        or artifact_col is not None
        or artifact_reason_col is not None
        or min_group_n != 1
        or max_valid_pct_diff != 10
        or max_artifact_pct_diff != 10
        or max_missing_pct_diff != 10
        or max_interpolated_pct_diff != 10
    )
    if not r_mode:
        df = ensure_dataframe(data, copy=False)
        pupil_col = infer_column(df, "pupil", pupil_col, required=True)
        condition_col = infer_column(df, "condition", condition_col, required=True)
        work = df.copy()
        work["_p"] = finite_numeric(work[pupil_col])
        return (
            work.groupby(condition_col, dropna=False)
            .agg(
                n=("_p", "size"),
                n_valid=("_p", "count"),
                mean_pupil=("_p", "mean"),
                sd_pupil=("_p", "std"),
            )
            .reset_index()
        )

    df = ensure_dataframe(data, copy=False)
    groups = (
        ["condition"]
        if group_cols is None
        else ([group_cols] if isinstance(group_cols, str) else list(group_cols))
    )
    if len(groups) != len(set(groups)) or any(not isinstance(c, str) or not c for c in groups):
        raise ValueError("group_cols must contain unique non-empty column names")
    pupil_col = "pupil_interpolated" if pupil_col is None else pupil_col
    if artifact_col is None:
        artifact_col = _gp3_r_first_present(
            df, ["pupil_artifact_flag", "pupil_flag_invalid", "artifact_flag"]
        )
    if artifact_reason_col is None:
        artifact_reason_col = _gp3_r_first_present(
            df, ["pupil_artifact_reason", "pupil_flag_reason", "artifact_reason"]
        )
    required = groups + [pupil_col, interpolated_col, interpolation_status_col]
    for optional in (artifact_col, artifact_reason_col):
        if optional is not None:
            required.append(optional)
    missing = [c for c in dict.fromkeys(required) if c not in df.columns]
    if missing:
        raise KeyError("Missing required columns: " + ", ".join(missing))

    work = df.copy()
    work["_gp3_pupil"] = pd.to_numeric(work[pupil_col], errors="coerce")
    work["_gp3_interp"] = _gp3_r_bool(work[interpolated_col])
    work["_gp3_status"] = work[interpolation_status_col].astype("string")
    if artifact_col is not None:
        work["_gp3_artifact"] = _gp3_r_bool(work[artifact_col])
    elif artifact_reason_col is not None:
        reason = work[artifact_reason_col].astype("string")
        work["_gp3_artifact"] = reason.notna() & reason.ne("") & reason.ne("valid")
    else:
        work["_gp3_artifact"] = False

    rows = []
    for key, part in _gp3_r_group_parts(work, groups):
        row = _gp3_r_group_row(groups, key)
        n_rows = len(part)
        n_valid = int(part["_gp3_pupil"].notna().sum())
        n_interp = int(part["_gp3_interp"].sum())
        n_artifact = int(part["_gp3_artifact"].sum())
        n_missing = int(part["_gp3_pupil"].isna().sum())
        row.update(
            n_rows=n_rows,
            n_valid_samples=n_valid,
            n_interpolated_samples=n_interp,
            n_artifact_samples=n_artifact,
            n_remaining_missing_samples=n_missing,
            n_observed_samples=int(part["_gp3_status"].eq("observed").sum()),
            n_missing_edge_gap_samples=int(part["_gp3_status"].eq("missing_edge_gap").sum()),
            n_missing_long_gap_samples=int(part["_gp3_status"].eq("missing_long_gap").sum()),
            valid_sample_pct=100 * n_valid / n_rows if n_rows else np.nan,
            interpolated_sample_pct=100 * n_interp / n_rows if n_rows else np.nan,
            artifact_sample_pct=100 * n_artifact / n_rows if n_rows else np.nan,
            remaining_missing_sample_pct=100 * n_missing / n_rows if n_rows else np.nan,
        )
        rows.append(row)
    out = pd.DataFrame(rows)

    def value_range(column):
        values = pd.to_numeric(out[column], errors="coerce").dropna()
        return float(values.max() - values.min()) if len(values) else np.nan

    valid_range = value_range("valid_sample_pct")
    artifact_range = value_range("artifact_sample_pct")
    missing_range = value_range("remaining_missing_sample_pct")
    interp_range = value_range("interpolated_sample_pct")
    reasons = []
    if np.isfinite(valid_range) and valid_range > max_valid_pct_diff:
        reasons.append("valid_pct_diff")
    if np.isfinite(artifact_range) and artifact_range > max_artifact_pct_diff:
        reasons.append("artifact_pct_diff")
    if np.isfinite(missing_range) and missing_range > max_missing_pct_diff:
        reasons.append("missing_pct_diff")
    if np.isfinite(interp_range) and interp_range > max_interpolated_pct_diff:
        reasons.append("interpolated_pct_diff")
    if bool((out["n_rows"] < min_group_n).any()):
        reasons.append("small_group_n")
    warning = bool(reasons)
    out["valid_sample_pct_range"] = valid_range
    out["artifact_sample_pct_range"] = artifact_range
    out["remaining_missing_sample_pct_range"] = missing_range
    out["interpolated_sample_pct_range"] = interp_range
    out["preprocessing_imbalance_warning"] = warning
    out["preprocessing_imbalance_reason"] = ";".join(reasons) if reasons else "ok"
    return out


def audit_gazepoint_pupil_overlap_risk(
    data,
    trial_duration_ms=3000,
    event_gap_ms=1000,
    trial_col=None,
    time_col=None,
    *,
    group_cols=None,
    event_time_cols=None,
    window_start_ms=None,
    window_end_ms=None,
    min_event_gap_ms=None,
    exclude_col=None,
    include_excluded: bool = False,
):
    """Audit event-related pupil-window overlap using legacy or R semantics."""
    df = ensure_dataframe(data, copy=False)
    r_mode = (
        group_cols is not None
        or event_time_cols is not None
        or window_start_ms is not None
        or window_end_ms is not None
        or min_event_gap_ms is not None
        or exclude_col is not None
        or include_excluded
    )
    if not r_mode:
        trial_col = infer_column(df, "trial", trial_col)
        time_col = infer_column(df, "time", time_col)
        if trial_col and time_col:
            work = df.copy()
            work["_t"] = finite_numeric(work[time_col])
            out = (
                work.groupby(trial_col, dropna=False)._t.agg(["min", "max", "count"]).reset_index()
            )
            out["duration"] = out["max"] - out["min"]
            out["overlap_risk"] = out["duration"] < trial_duration_ms
            return out
        return pd.DataFrame(
            [
                {
                    "trial_duration_ms_threshold": trial_duration_ms,
                    "event_gap_ms_threshold": event_gap_ms,
                    "status": "insufficient_columns",
                }
            ]
        )

    groups = (
        ["subject"]
        if group_cols is None
        else ([group_cols] if isinstance(group_cols, str) else list(group_cols))
    )
    if (
        not groups
        or len(set(groups)) != len(groups)
        or any(not isinstance(c, str) or not c for c in groups)
    ):
        raise ValueError("group_cols must contain unique non-empty column names")
    trial_col = "trial_global" if trial_col is None else trial_col
    time_col = "time" if time_col is None else time_col
    event_time_cols = (
        ["stimulus_onset_time", "target_onset_time", "response_time"]
        if event_time_cols is None
        else ([event_time_cols] if isinstance(event_time_cols, str) else list(event_time_cols))
    )
    if not event_time_cols or len(set(event_time_cols)) != len(event_time_cols):
        raise ValueError("event_time_cols must be a non-empty vector of unique column names")
    window_start_ms = 0.0 if window_start_ms is None else float(window_start_ms)
    window_end_ms = 2000.0 if window_end_ms is None else float(window_end_ms)
    min_event_gap_ms = 1000.0 if min_event_gap_ms is None else float(min_event_gap_ms)
    exclude_col = "excluded_trial" if exclude_col is None else exclude_col
    if not all(np.isfinite(v) for v in [window_start_ms, window_end_ms, min_event_gap_ms]):
        raise ValueError("window/gap arguments must be finite")
    if window_end_ms <= window_start_ms:
        raise ValueError("window_end_ms must be greater than window_start_ms")
    required = list(dict.fromkeys(groups + [trial_col, time_col] + event_time_cols))
    missing = [c for c in required if c not in df]
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")

    work = df.copy()
    if exclude_col and exclude_col in work and not include_excluded:
        raw = work[exclude_col]
        if pd.api.types.is_bool_dtype(raw):
            excluded = raw.fillna(False).astype(bool)
        elif pd.api.types.is_numeric_dtype(raw):
            excluded = pd.to_numeric(raw, errors="coerce").fillna(0).ne(0)
        else:
            excluded = (
                raw.astype("string").str.strip().str.lower().isin(["true", "t", "1", "yes", "y"])
            )
        work = work.loc[~excluded].copy()

    keys = list(dict.fromkeys(groups + [trial_col]))
    trial_rows = []
    iterator = work.groupby(keys, dropna=False, sort=False)
    for key, part in iterator:
        if not isinstance(key, tuple):
            key = (key,)
        t = pd.to_numeric(part[time_col], errors="coerce").dropna().to_numpy(float)
        row = {c: v for c, v in zip(keys, key, strict=False)}
        row.update(
            {
                "n_rows": int(len(part)),
                "trial_time_min": float(np.min(t)) if len(t) else np.nan,
                "trial_time_mean": float(np.mean(t)) if len(t) else np.nan,
                "trial_time_max": float(np.max(t)) if len(t) else np.nan,
                "trial_time_range_ms": float(np.ptp(t)) if len(t) else np.nan,
            }
        )
        trial_rows.append(row)
    by_trial_base = pd.DataFrame(trial_rows)

    event_frames = []
    for event_col in event_time_cols:
        e = work[keys].copy()
        e["event_name"] = event_col
        e["event_time_ms"] = pd.to_numeric(work[event_col], errors="coerce")
        e = e.loc[e["event_time_ms"].notna()].drop_duplicates()
        event_frames.append(e)
    events = (
        pd.concat(event_frames, ignore_index=True)
        if event_frames
        else pd.DataFrame(columns=keys + ["event_name", "event_time_ms"])
    )
    if len(events):
        events["response_window_start_ms"] = events["event_time_ms"] + window_start_ms
        events["response_window_end_ms"] = events["event_time_ms"] + window_end_ms
        events["response_window_duration_ms"] = window_end_ms - window_start_ms
        events = events.sort_values(
            keys + ["event_time_ms", "event_name"], kind="stable"
        ).reset_index(drop=True)
    else:
        for c in [
            "response_window_start_ms",
            "response_window_end_ms",
            "response_window_duration_ms",
        ]:
            events[c] = pd.Series(dtype=float)

    event_gaps = events.copy()
    if len(event_gaps):
        parts = []
        for _, part in event_gaps.groupby(keys, dropna=False, sort=False):
            part = part.copy()
            part["previous_event_name"] = part["event_name"].shift(1)
            part["previous_event_time_ms"] = part["event_time_ms"].shift(1)
            part["previous_response_window_end_ms"] = part["response_window_end_ms"].shift(1)
            part["event_gap_ms"] = part["event_time_ms"] - part["previous_event_time_ms"]
            part["short_event_gap"] = part["event_gap_ms"].notna() & (
                part["event_gap_ms"] < min_event_gap_ms
            )
            part["response_window_overlap"] = part["previous_response_window_end_ms"].notna() & (
                part["response_window_start_ms"] < part["previous_response_window_end_ms"]
            )
            part["overlap_amount_ms"] = np.where(
                part["response_window_overlap"],
                part["previous_response_window_end_ms"] - part["response_window_start_ms"],
                0.0,
            )
            part["event_gap_status"] = np.select(
                [
                    part["previous_event_time_ms"].isna(),
                    part["response_window_overlap"] & part["short_event_gap"],
                    part["response_window_overlap"],
                    part["short_event_gap"],
                ],
                [
                    "first_event",
                    "overlap_and_short_gap",
                    "overlapping_response_window",
                    "short_event_gap",
                ],
                default="ok",
            )
            parts.append(part)
        event_gaps = pd.concat(parts, ignore_index=True)
    else:
        for c, dtype in [
            ("previous_event_name", object),
            ("previous_event_time_ms", float),
            ("previous_response_window_end_ms", float),
            ("event_gap_ms", float),
            ("short_event_gap", bool),
            ("response_window_overlap", bool),
            ("overlap_amount_ms", float),
            ("event_gap_status", object),
        ]:
            event_gaps[c] = pd.Series(dtype=dtype)

    summary_rows = []
    if len(event_gaps):
        for key, part in event_gaps.groupby(keys, dropna=False, sort=False):
            if not isinstance(key, tuple):
                key = (key,)
            gaps = pd.to_numeric(part["event_gap_ms"], errors="coerce").dropna().to_numpy(float)
            overlap_amount = (
                pd.to_numeric(part["overlap_amount_ms"], errors="coerce").dropna().to_numpy(float)
            )
            n_short = int(part["short_event_gap"].sum())
            n_overlap = int(part["response_window_overlap"].sum())
            row = {c: v for c, v in zip(keys, key, strict=False)}
            row.update(
                {
                    "n_events": int(len(part)),
                    "n_event_gaps": int(len(gaps)),
                    "min_event_gap_observed_ms": float(np.min(gaps)) if len(gaps) else np.nan,
                    "mean_event_gap_observed_ms": float(np.mean(gaps)) if len(gaps) else np.nan,
                    "max_event_gap_observed_ms": float(np.max(gaps)) if len(gaps) else np.nan,
                    "n_short_event_gaps": n_short,
                    "n_overlapping_response_windows": n_overlap,
                    "max_overlap_amount_ms": float(np.max(overlap_amount))
                    if len(overlap_amount)
                    else np.nan,
                    "overlap_risk_warning": bool(n_short > 0 or n_overlap > 0),
                    "overlap_risk_reason": "short_event_gap;overlapping_response_window"
                    if n_short and n_overlap
                    else "short_event_gap"
                    if n_short
                    else "overlapping_response_window"
                    if n_overlap
                    else "ok",
                }
            )
            summary_rows.append(row)
    event_summary = pd.DataFrame(summary_rows)
    if len(by_trial_base):
        by_trial = (
            by_trial_base.merge(event_summary, on=keys, how="left")
            if len(event_summary)
            else by_trial_base.copy()
        )
        for c in [
            "n_events",
            "n_event_gaps",
            "n_short_event_gaps",
            "n_overlapping_response_windows",
        ]:
            if c not in by_trial:
                by_trial[c] = 0
            by_trial[c] = by_trial[c].fillna(0).astype(int)
        if "overlap_risk_warning" not in by_trial:
            by_trial["overlap_risk_warning"] = False
        by_trial["overlap_risk_warning"] = (
            by_trial["overlap_risk_warning"].fillna(False).astype(bool)
        )
        if "overlap_risk_reason" not in by_trial:
            by_trial["overlap_risk_reason"] = "no_usable_event_times"
        by_trial["overlap_risk_reason"] = by_trial["overlap_risk_reason"].fillna(
            "no_usable_event_times"
        )
    else:
        by_trial = by_trial_base.copy()

    n_trials = len(by_trial)
    n_events = len(events)
    n_with_events = int((by_trial["n_events"] > 0).sum()) if n_trials else 0
    n_short_trials = int((by_trial["n_short_event_gaps"] > 0).sum()) if n_trials else 0
    n_overlap_trials = (
        int((by_trial["n_overlapping_response_windows"] > 0).sum()) if n_trials else 0
    )
    n_risk = int(by_trial["overlap_risk_warning"].sum()) if n_trials else 0
    status = (
        "no_usable_event_times"
        if n_events == 0
        else "possible_overlap_risk"
        if n_risk > 0
        else "ok"
    )
    summary = pd.DataFrame(
        [
            {
                "n_trials": int(n_trials),
                "n_events": int(n_events),
                "n_trials_with_events": n_with_events,
                "n_trials_without_events": int(n_trials - n_with_events),
                "n_trials_with_short_event_gaps": n_short_trials,
                "n_trials_with_overlapping_windows": n_overlap_trials,
                "n_overlap_risk_trials": n_risk,
                "pct_overlap_risk_trials": 100.0 * n_risk / n_trials if n_trials else np.nan,
                "window_start_ms": window_start_ms,
                "window_end_ms": window_end_ms,
                "response_window_duration_ms": window_end_ms - window_start_ms,
                "min_event_gap_ms": min_event_gap_ms,
                "overlap_assessment_status": status,
            }
        ]
    )
    return {
        "events": events,
        "event_gaps": event_gaps,
        "by_trial": by_trial,
        "summary": summary,
        "_gp3_class": "gp3_pupil_overlap_risk_audit",
    }


def audit_gazepoint_stimulus_luminance(
    data, luminance_col="luminance", pupil_col=None, group_cols=None
) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    pupil_col = infer_column(df, "pupil", pupil_col, required=True)
    if luminance_col not in df:
        return pd.DataFrame([{"status": "luminance_column_missing"}])
    x = finite_numeric(df[luminance_col])
    y = finite_numeric(df[pupil_col])
    ok = x.notna() & y.notna()
    r, p = stats.pearsonr(x[ok], y[ok]) if ok.sum() > 2 else (np.nan, np.nan)
    return pd.DataFrame([{"n": int(ok.sum()), "correlation": r, "p_value": p}])


def summarise_gazepoint_pupil(
    data=None,
    pupil_col=None,
    group_cols=None,
    *,
    master=None,
    time_col=None,
    missing_pupil_col=None,
    min_pupil=0,
    max_pupil=np.inf,
    outlier_k=1.5,
) -> pd.DataFrame:
    """Summarise Gazepoint pupil measurements.

    ``master=`` activates the R gp3tools v2.3.0 summary contract.
    Passing a DataFrame through ``data`` retains the historical Python
    summary interface.
    """
    if master is None:
        if data is None:
            raise TypeError("data or master must be supplied")

        df = ensure_dataframe(
            data,
            copy=False,
        )

        pupil_col = infer_column(
            df,
            "pupil",
            pupil_col,
            required=True,
        )

        groups = normalize_group_cols(
            df,
            group_cols,
        )

        work = df.copy()

        work["_p"] = finite_numeric(work[pupil_col])

        if groups:
            return (
                work.groupby(
                    groups,
                    dropna=False,
                )
                .agg(
                    n_samples=(
                        "_p",
                        "size",
                    ),
                    n_valid=(
                        "_p",
                        "count",
                    ),
                    mean_pupil=(
                        "_p",
                        "mean",
                    ),
                    sd_pupil=(
                        "_p",
                        "std",
                    ),
                    median_pupil=(
                        "_p",
                        "median",
                    ),
                    min_pupil=(
                        "_p",
                        "min",
                    ),
                    max_pupil=(
                        "_p",
                        "max",
                    ),
                )
                .reset_index()
            )

        values = work["_p"]

        return pd.DataFrame(
            [
                {
                    "n_samples": len(values),
                    "n_valid": int(values.notna().sum()),
                    "mean_pupil": float(values.mean()),
                    "sd_pupil": float(values.std()),
                    "median_pupil": float(values.median()),
                    "min_pupil": float(values.min()),
                    "max_pupil": float(values.max()),
                }
            ]
        )

    if data is not None:
        raise TypeError("supply either data or master, not both")

    if not isinstance(
        master,
        pd.DataFrame,
    ):
        raise TypeError("master must be a DataFrame")

    if group_cols is None:
        groups = [
            "subject",
            "media_id",
        ]

    elif isinstance(
        group_cols,
        str,
    ):
        groups = [group_cols]

    elif isinstance(
        group_cols,
        (
            list,
            tuple,
            pd.Index,
            np.ndarray,
        ),
    ):
        groups = list(group_cols)

        if not all(
            isinstance(
                col,
                str,
            )
            for col in groups
        ):
            raise ValueError("group_cols must be a character vector")

    else:
        raise ValueError("group_cols must be a character vector")

    for argument, value in {
        "pupil_col": pupil_col,
        "time_col": time_col,
        "missing_pupil_col": missing_pupil_col,
    }.items():
        if value is not None and (
            not isinstance(
                value,
                str,
            )
            or not value
        ):
            raise ValueError(f"{argument} must be None or a single string")

    def numeric_scalar(
        value,
        argument,
    ):
        if isinstance(
            value,
            (bool, np.bool_),
        ):
            raise ValueError(f"{argument} must be a single numeric value")

        try:
            return float(value)
        except (
            TypeError,
            ValueError,
        ) as exc:
            raise ValueError(f"{argument} must be a single numeric value") from exc

    min_value = numeric_scalar(
        min_pupil,
        "min_pupil",
    )

    max_value = numeric_scalar(
        max_pupil,
        "max_pupil",
    )

    outlier_k = numeric_scalar(
        outlier_k,
        "outlier_k",
    )

    if max_value <= min_value:
        raise ValueError("max_pupil must be greater than min_pupil")

    invalid_groups = [
        col
        for col in groups
        if col
        not in {
            "subject",
            "media_id",
        }
    ]

    if invalid_groups:
        raise ValueError("group_cols can only contain: subject, media_id")

    def detect_col(candidates):
        for candidate in candidates:
            if candidate in master.columns:
                return candidate

        return None

    subject_source = detect_col(
        [
            "subject",
            "pID",
            "participant",
        ]
    )

    media_source = detect_col(
        [
            "media_id",
            "MEDIA_ID",
        ]
    )

    pupil_source = (
        pupil_col
        if pupil_col is not None
        else detect_col(
            [
                "mean_pupil",
                "pupil",
                "pupil_raw",
                "left_pupil",
                "right_pupil",
            ]
        )
    )

    time_source = (
        time_col
        if time_col is not None
        else detect_col(
            [
                "time_ms",
                "time",
                "time_orig",
                "time_orig_ms",
            ]
        )
    )

    missing_source = (
        missing_pupil_col if missing_pupil_col is not None else detect_col(["missing_pupil"])
    )

    if subject_source is None:
        raise ValueError("No subject column was found")

    if media_source is None:
        raise ValueError("No media/stimulus column was found")

    if pupil_source is None or pupil_source not in master.columns:
        raise ValueError("No pupil column was found")

    if time_source is None or time_source not in master.columns:
        raise ValueError("No time column was found")

    if missing_source is not None and missing_source not in master.columns:
        raise ValueError("missing_pupil_col was not found in master")

    pupil = pd.to_numeric(
        master[pupil_source],
        errors="coerce",
    )

    time_values = pd.to_numeric(
        master[time_source],
        errors="coerce",
    )

    def as_r_logical(values):
        series = pd.Series(
            values,
            index=master.index,
        )

        if pd.api.types.is_bool_dtype(series.dtype):
            return series.astype("boolean")

        if pd.api.types.is_numeric_dtype(series.dtype):
            numeric = pd.to_numeric(
                series,
                errors="coerce",
            )

            result = pd.Series(
                pd.NA,
                index=series.index,
                dtype="boolean",
            )

            result.loc[numeric.notna()] = numeric.loc[numeric.notna()] != 0

            return result

        text = series.astype("string").str.strip().str.lower()

        result = pd.Series(
            pd.NA,
            index=series.index,
            dtype="boolean",
        )

        result.loc[
            text.isin(
                [
                    "true",
                    "t",
                    "1",
                ]
            )
        ] = True

        result.loc[
            text.isin(
                [
                    "false",
                    "f",
                    "0",
                ]
            )
        ] = False

        return result

    if missing_source is not None:
        missing = as_r_logical(master[missing_source])

        missing = missing.fillna(pupil.isna())

    else:
        missing = pupil.isna().astype("boolean")

    pupil_valid = ~missing.astype(bool) & pupil.notna()

    work = pd.DataFrame(
        {
            "subject": master[subject_source].astype("string"),
            "media_id": master[media_source].astype("string"),
            "time_ms": time_values,
            "pupil": pupil,
            "missing_pupil": missing.astype(bool),
            "pupil_valid": pupil_valid,
            "pupil_for_summary": pupil.where(
                pupil_valid,
                np.nan,
            ),
        }
    )

    def finite_values(values):
        numeric = pd.to_numeric(
            values,
            errors="coerce",
        ).to_numpy(float)

        return numeric[np.isfinite(numeric)]

    def safe_min(values):
        values = finite_values(values)

        return float(np.min(values)) if len(values) else np.nan

    def safe_max(values):
        values = finite_values(values)

        return float(np.max(values)) if len(values) else np.nan

    def safe_quantile(
        values,
        probability,
    ):
        values = finite_values(values)

        return (
            float(
                np.quantile(
                    values,
                    probability,
                    method="linear",
                )
            )
            if len(values)
            else np.nan
        )

    def safe_mean(values):
        values = pd.to_numeric(
            values,
            errors="coerce",
        ).dropna()

        if not len(values):
            return np.nan

        return float(values.mean())

    def safe_median(values):
        values = pd.to_numeric(
            values,
            errors="coerce",
        ).dropna()

        if not len(values):
            return np.nan

        return float(values.median())

    def safe_sd(values):
        values = pd.to_numeric(
            values,
            errors="coerce",
        ).dropna()

        if len(values) < 2:
            return np.nan

        return float(values.std(ddof=1))

    def percentage(
        numerator,
        denominator,
    ):
        if pd.isna(denominator) or denominator == 0:
            return np.nan

        return float(numerator / denominator * 100)

    def count_iqr_outliers(values):
        values = finite_values(values)

        values = values[(values >= min_value) & (values <= max_value)]

        if len(values) < 4:
            return 0

        q25, q75 = np.quantile(
            values,
            [
                0.25,
                0.75,
            ],
            method="linear",
        )

        iqr_value = q75 - q25

        if not np.isfinite(iqr_value) or iqr_value == 0:
            return 0

        lower = q25 - outlier_k * iqr_value

        upper = q75 + outlier_k * iqr_value

        return int(((values < lower) | (values > upper)).sum())

    def summarise_part(part):
        n_rows = int(len(part))

        n_valid = int(part["pupil_valid"].sum())

        n_missing = int((~part["pupil_valid"]).sum())

        below = int((part["pupil_valid"] & (part["pupil"] < min_value)).sum())

        above = int((part["pupil_valid"] & (part["pupil"] > max_value)).sum())

        iqr_outliers = count_iqr_outliers(part["pupil_for_summary"])

        time_min = safe_min(part["time_ms"])

        time_max = safe_max(part["time_ms"])

        return {
            "n_rows": n_rows,
            "time_min_ms": time_min,
            "time_max_ms": time_max,
            "time_span_ms": (
                time_max - time_min if (np.isfinite(time_min) and np.isfinite(time_max)) else np.nan
            ),
            "n_pupil_samples": n_valid,
            "n_missing_pupil": n_missing,
            "missing_pupil_pct": percentage(
                n_missing,
                n_rows,
            ),
            "valid_pupil_pct": percentage(
                n_valid,
                n_rows,
            ),
            "mean_pupil": safe_mean(part["pupil_for_summary"]),
            "median_pupil": safe_median(part["pupil_for_summary"]),
            "sd_pupil": safe_sd(part["pupil_for_summary"]),
            "min_pupil": safe_min(part["pupil_for_summary"]),
            "max_pupil": safe_max(part["pupil_for_summary"]),
            "q05_pupil": safe_quantile(
                part["pupil_for_summary"],
                0.05,
            ),
            "q25_pupil": safe_quantile(
                part["pupil_for_summary"],
                0.25,
            ),
            "q75_pupil": safe_quantile(
                part["pupil_for_summary"],
                0.75,
            ),
            "q95_pupil": safe_quantile(
                part["pupil_for_summary"],
                0.95,
            ),
            "n_below_plausible": below,
            "n_above_plausible": above,
            "n_implausible": below + above,
            "implausible_pct": percentage(
                below + above,
                n_rows,
            ),
            "n_iqr_outliers": iqr_outliers,
            "iqr_outlier_pct": percentage(
                iqr_outliers,
                n_valid,
            ),
            "pupil_column": pupil_source,
            "time_column": time_source,
            "min_plausible": min_value,
            "max_plausible": max_value,
        }

    if not groups:
        return pd.DataFrame([summarise_part(work)])

    rows = []

    for key, part in work.groupby(
        groups,
        dropna=False,
        sort=False,
    ):
        key_values = (
            key
            if isinstance(
                key,
                tuple,
            )
            else (key,)
        )

        row = dict(
            zip(
                groups,
                key_values,
                strict=True,
            )
        )

        row.update(summarise_part(part))

        rows.append(row)

    return pd.DataFrame(rows)


def summarise_gazepoint_pupil_windows(
    data,
    pupil_col=None,
    time_col=None,
    windows=None,
    group_cols=None,
    *,
    include_window_end=False,
    min_valid_samples=1,
) -> pd.DataFrame:
    """Summarise pupil windows using legacy or R v2.3.0 semantics."""
    numeric_windows = (
        isinstance(windows, (list, tuple, np.ndarray, pd.Series))
        and len(windows) >= 2
        and all(np.isscalar(v) for v in windows)
    )
    r_mode = (
        isinstance(windows, pd.DataFrame)
        or numeric_windows
        or include_window_end
        or min_valid_samples != 1
    )
    if not r_mode:
        df = ensure_dataframe(data, copy=False)
        pupil_col = infer_column(df, "pupil", pupil_col, required=True)
        time_col = infer_column(df, "time", time_col, required=True)
        if windows is None:
            windows = {
                "window": (
                    float(finite_numeric(df[time_col]).min()),
                    float(finite_numeric(df[time_col]).max()),
                )
            }
        rows = []
        iterator = windows.items() if isinstance(windows, dict) else windows
        for name, (lo, hi) in iterator:
            sub = df.loc[finite_numeric(df[time_col]).between(lo, hi)].copy()
            tmp = summarise_gazepoint_pupil(sub, pupil_col=pupil_col, group_cols=group_cols)
            tmp.insert(len(tmp.columns) if len(tmp.columns) else 0, "window", name)
            tmp["window_start"] = lo
            tmp["window_end"] = hi
            rows.append(tmp)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    df = ensure_dataframe(data, copy=False)
    if min_valid_samples < 1:
        raise ValueError("min_valid_samples must be greater than or equal to 1")
    groups = (
        ["subject", "media_id"]
        if group_cols is None
        else ([group_cols] if isinstance(group_cols, str) else list(group_cols))
    )

    def detect(candidates):
        return _gp3_r_first_present(df, candidates)

    pupil_source = pupil_col or detect(
        [
            "pupil_smoothed",
            "pupil_baseline_corrected",
            "pupil_baseline_percent_change",
            "pupil_interpolated",
            "pupil_for_preprocessing",
            "mean_pupil",
            "pupil",
            "pupil_raw",
            "left_pupil",
            "right_pupil",
        ]
    )
    time_source = time_col or detect(
        [
            "time_relative_ms",
            "relative_time_ms",
            "event_time_ms",
            "time_ms",
            "time",
            "time_orig",
            "time_orig_ms",
        ]
    )
    if pupil_source is None or pupil_source not in df.columns:
        raise KeyError("No pupil column was found")
    if time_source is None or time_source not in df.columns:
        raise KeyError("No time column was found")

    role_sources = {
        "subject": detect(["subject", "pID", "participant"]),
        "media_id": detect(["media_id", "MEDIA_ID"]),
        "trial": detect(["trial"]),
        "trial_global": detect(["trial_global"]),
    }
    missing_roles = [g for g in groups if g in role_sources and role_sources[g] is None]
    missing_other = [g for g in groups if g not in role_sources and g not in df.columns]
    if missing_roles or missing_other:
        raise KeyError("Missing grouping columns: " + ", ".join(missing_roles + missing_other))

    def fmt(value):
        value = float(value)
        return str(int(value)) if value.is_integer() else format(value, "g")

    if numeric_windows:
        breaks = np.asarray(windows, dtype=float)
        if not np.isfinite(breaks).all() or len(breaks) < 2 or not np.all(np.diff(breaks) > 0):
            raise ValueError("Numeric windows must contain strictly increasing finite breakpoints")
        window_tbl = pd.DataFrame(
            {
                "window_label": [
                    f"{fmt(a)}_{fmt(b)}ms" for a, b in zip(breaks[:-1], breaks[1:], strict=True)
                ],
                "window_start_ms": breaks[:-1],
                "window_end_ms": breaks[1:],
            }
        )
    elif isinstance(windows, pd.DataFrame):

        def first_col(candidates):
            for c in candidates:
                if c in windows.columns:
                    return c
            return None

        start_col = first_col(["window_start_ms", "window_start", "start_ms", "start"])
        end_col = first_col(["window_end_ms", "window_end", "end_ms", "end"])
        label_col = first_col(["window_label", "label", "window"])
        if start_col is None or end_col is None:
            raise KeyError("Window data must contain start and end columns")
        starts = pd.to_numeric(windows[start_col], errors="coerce")
        ends = pd.to_numeric(windows[end_col], errors="coerce")
        if starts.isna().any() or ends.isna().any():
            raise ValueError("Window start and end values must be numeric and non-missing")
        labels = (
            windows[label_col].astype("string")
            if label_col
            else pd.Series([f"{fmt(a)}_{fmt(b)}ms" for a, b in zip(starts, ends, strict=True)])
        )
        window_tbl = pd.DataFrame(
            {
                "window_label": labels.astype(str),
                "window_start_ms": starts.astype(float),
                "window_end_ms": ends.astype(float),
            }
        )
    else:
        breaks = np.asarray([0, 500, 1000, 2000], dtype=float)
        window_tbl = pd.DataFrame(
            {
                "window_label": [
                    f"{fmt(a)}_{fmt(b)}ms" for a, b in zip(breaks[:-1], breaks[1:], strict=True)
                ],
                "window_start_ms": breaks[:-1],
                "window_end_ms": breaks[1:],
            }
        )
    if (
        len(window_tbl) == 0
        or bool((window_tbl["window_end_ms"] < window_tbl["window_start_ms"]).any())
        or bool(window_tbl["window_label"].eq("").any())
    ):
        raise ValueError("Invalid window definitions")

    work = pd.DataFrame(index=df.index)
    for role, source_col in role_sources.items():
        work[role] = (
            df[source_col].astype("string")
            if source_col is not None
            else pd.Series(pd.NA, index=df.index, dtype="string")
        )
    for g in groups:
        if g not in role_sources:
            work[g] = df[g]
    work["time_ms"] = pd.to_numeric(df[time_source], errors="coerce")
    work["pupil_value"] = pd.to_numeric(df[pupil_source], errors="coerce")
    work = work.loc[np.isfinite(work["time_ms"].to_numpy(float))].copy()

    rows = []
    grouping = groups + ["window_label", "window_start_ms", "window_end_ms"]
    pieces = []
    for window in window_tbl.itertuples(index=False):
        if include_window_end:
            mask = work["time_ms"].ge(window.window_start_ms) & work["time_ms"].le(
                window.window_end_ms
            )
        else:
            mask = work["time_ms"].ge(window.window_start_ms) & work["time_ms"].lt(
                window.window_end_ms
            )
        part = work.loc[mask].copy()
        if len(part):
            part["window_label"] = window.window_label
            part["window_start_ms"] = float(window.window_start_ms)
            part["window_end_ms"] = float(window.window_end_ms)
            pieces.append(part)
    if not pieces:
        columns = groups + [
            "window_label",
            "window_start_ms",
            "window_end_ms",
            "n_samples",
            "n_valid_pupil",
            "n_missing_pupil",
            "valid_pupil_pct",
            "missing_pupil_pct",
            "mean_pupil",
            "sd_pupil",
            "median_pupil",
            "min_pupil",
            "max_pupil",
            "q25_pupil",
            "q75_pupil",
            "pupil_auc",
            "pupil_time_span_ms",
            "pupil_window_status",
            "pupil_window_pupil_column",
            "pupil_window_time_column",
            "pupil_window_min_valid_samples",
            "pupil_window_include_end",
        ]
        return pd.DataFrame(columns=columns)
    windowed = pd.concat(pieces, ignore_index=True)

    grouper = grouping[0] if len(grouping) == 1 else grouping
    for key, part in windowed.groupby(grouper, dropna=False, sort=False):
        if len(grouping) == 1:
            key = (key,)
        row = dict(zip(grouping, key, strict=True))
        values = pd.to_numeric(part["pupil_value"], errors="coerce").to_numpy(float)
        times = pd.to_numeric(part["time_ms"], errors="coerce").to_numpy(float)
        valid = np.isfinite(values)
        n = len(values)
        nv = int(valid.sum())
        nm = n - nv
        finite_values = values[valid]
        both = np.isfinite(times) & np.isfinite(values)
        if both.sum() >= 2:
            order = np.argsort(times[both], kind="stable")
            tt = times[both][order]
            yy = values[both][order]
            auc = float(np.sum(np.diff(tt) * (yy[:-1] + yy[1:]) / 2))
            span = float(tt.max() - tt.min())
        else:
            auc = np.nan
            span = np.nan
        row.update(
            n_samples=n,
            n_valid_pupil=nv,
            n_missing_pupil=nm,
            valid_pupil_pct=100 * nv / n if n else np.nan,
            missing_pupil_pct=100 * nm / n if n else np.nan,
            mean_pupil=float(np.mean(finite_values)) if nv else np.nan,
            sd_pupil=float(np.std(finite_values, ddof=1)) if nv >= 2 else np.nan,
            median_pupil=float(np.median(finite_values)) if nv else np.nan,
            min_pupil=float(np.min(finite_values)) if nv else np.nan,
            max_pupil=float(np.max(finite_values)) if nv else np.nan,
            q25_pupil=float(np.quantile(finite_values, 0.25)) if nv else np.nan,
            q75_pupil=float(np.quantile(finite_values, 0.75)) if nv else np.nan,
            pupil_auc=auc,
            pupil_time_span_ms=span,
            pupil_window_status="valid"
            if nv >= min_valid_samples
            else ("no_valid_pupil" if nv == 0 else "insufficient_valid_pupil"),
            pupil_window_pupil_column=pupil_source,
            pupil_window_time_column=time_source,
            pupil_window_min_valid_samples=int(min_valid_samples),
            pupil_window_include_end=bool(include_window_end),
        )
        rows.append(row)
    out = pd.DataFrame(rows)
    sort_cols = ["window_start_ms", "window_end_ms"] + groups
    return out.sort_values(sort_cols, kind="stable", ignore_index=True)


def summarise_gazepoint_pupil_trial_features(
    data,
    pupil_col=None,
    trial_col=None,
    group_cols=None,
    *,
    time_col=None,
    interpolated_col=None,
    artifact_col=None,
    artifact_reason_col=None,
    early_window=None,
    middle_window=None,
    late_window=None,
    min_valid_samples=None,
) -> pd.DataFrame:
    """Summarise trial-level pupil features using legacy or R v2.3.0 semantics."""
    r_mode = any(
        value is not None
        for value in (
            time_col,
            interpolated_col,
            artifact_col,
            artifact_reason_col,
            early_window,
            middle_window,
            late_window,
            min_valid_samples,
        )
    )
    if not r_mode:
        df = ensure_dataframe(data, copy=False)
        pupil_col = infer_column(df, "pupil", pupil_col, required=True)
        trial_col = infer_column(df, "trial", trial_col, required=True)
        groups = normalize_group_cols(df, group_cols) + [trial_col]
        work = df.copy()
        work["_p"] = finite_numeric(work[pupil_col])
        return (
            work.groupby(groups, dropna=False)
            .agg(
                mean_pupil=("_p", "mean"),
                peak_pupil=("_p", "max"),
                min_pupil=("_p", "min"),
                sd_pupil=("_p", "std"),
                n_valid=("_p", "count"),
            )
            .reset_index()
        )

    df = ensure_dataframe(data, copy=False)
    groups = (
        ["subject", "trial_global"]
        if group_cols is None
        else ([group_cols] if isinstance(group_cols, str) else list(group_cols))
    )
    if len(groups) != len(set(groups)) or any(not isinstance(c, str) or not c for c in groups):
        raise ValueError("group_cols must contain unique non-empty column names")
    time_col = "time" if time_col is None else time_col
    interpolated_col = "pupil_was_interpolated" if interpolated_col is None else interpolated_col
    early_window = (0, 500) if early_window is None else tuple(early_window)
    middle_window = (500, 1500) if middle_window is None else tuple(middle_window)
    late_window = (1500, 3000) if late_window is None else tuple(late_window)
    min_valid_samples = 1 if min_valid_samples is None else float(min_valid_samples)
    for name, window in (
        ("early_window", early_window),
        ("middle_window", middle_window),
        ("late_window", late_window),
    ):
        if len(window) != 2 or not np.isfinite(window).all() or window[1] <= window[0]:
            raise ValueError(f"{name} must contain two increasing finite values")
    if pupil_col is None:
        pupil_col = _gp3_r_first_present(
            df,
            [
                "pupil_smoothed",
                "pupil_baseline_corrected",
                "pupil_baseline_percent_change",
                "pupil_interpolated",
                "pupil_clean",
                "pupil",
            ],
        )
    if pupil_col is None:
        raise KeyError("Could not automatically detect a pupil column")
    if artifact_col is None:
        artifact_col = _gp3_r_first_present(
            df, ["pupil_artifact_flag", "pupil_flag_invalid", "artifact_flag"]
        )
    if artifact_reason_col is None:
        artifact_reason_col = _gp3_r_first_present(
            df, ["pupil_artifact_reason", "pupil_flag_reason", "artifact_reason"]
        )
    required = groups + [pupil_col, time_col]
    missing = [c for c in dict.fromkeys(required) if c not in df.columns]
    if missing:
        raise KeyError("Missing required columns: " + ", ".join(missing))

    work = df.copy()
    work["_gp3_pupil"] = pd.to_numeric(work[pupil_col], errors="coerce")
    work["_gp3_time"] = pd.to_numeric(work[time_col], errors="coerce")
    if interpolated_col in work.columns:
        work["_gp3_interp"] = _gp3_r_bool(work[interpolated_col]).astype("boolean")
    else:
        work["_gp3_interp"] = pd.Series(pd.NA, index=work.index, dtype="boolean")
    if artifact_col is not None and artifact_col in work.columns:
        work["_gp3_artifact"] = _gp3_r_bool(work[artifact_col]).astype("boolean")
    elif artifact_reason_col is not None and artifact_reason_col in work.columns:
        reason = work[artifact_reason_col].astype("string")
        work["_gp3_artifact"] = (reason.notna() & reason.ne("") & reason.ne("valid")).astype(
            "boolean"
        )
    else:
        work["_gp3_artifact"] = pd.Series(pd.NA, index=work.index, dtype="boolean")

    def mean_window(y, t, window):
        mask = np.isfinite(y) & np.isfinite(t) & (t >= window[0]) & (t < window[1])
        return float(np.mean(y[mask])) if mask.any() else np.nan

    def count_window(y, t, window):
        return int((np.isfinite(y) & np.isfinite(t) & (t >= window[0]) & (t < window[1])).sum())

    rows = []
    for key, part in _gp3_r_group_parts(work, groups):
        row = _gp3_r_group_row(groups, key)
        y = part["_gp3_pupil"].to_numpy(dtype=float)
        t = part["_gp3_time"].to_numpy(dtype=float)
        valid = np.isfinite(y) & np.isfinite(t)
        finite_y = y[np.isfinite(y)]
        finite_t = t[np.isfinite(t)]
        n_samples = len(part)
        n_valid = int(valid.sum())
        n_missing = int((~valid).sum())
        interp = part["_gp3_interp"]
        artifact = part["_gp3_artifact"]
        n_interp = np.nan if interp.isna().all() else int(interp.fillna(False).sum())
        n_artifact = np.nan if artifact.isna().all() else int(artifact.fillna(False).sum())
        if n_valid:
            peak = float(np.max(y[valid]))
            first_peak = int(np.flatnonzero(valid & (y == peak))[0])
            peak_time = float(t[first_peak])
        else:
            peak = np.nan
            peak_time = np.nan
        if n_valid >= 2 and len(np.unique(t[valid])) >= 2:
            order = np.argsort(t[valid], kind="stable")
            tt = t[valid][order]
            yy = y[valid][order]
            auc = float(np.sum(np.diff(tt) * (yy[:-1] + yy[1:]) / 2))
        else:
            auc = np.nan
        time_min = float(np.min(finite_t)) if len(finite_t) else np.nan
        time_max = float(np.max(finite_t)) if len(finite_t) else np.nan
        row.update(
            n_samples=n_samples,
            n_valid_pupil=n_valid,
            n_missing_pupil=n_missing,
            valid_sample_pct=100 * n_valid / n_samples if n_samples else np.nan,
            missing_sample_pct=100 * n_missing / n_samples if n_samples else np.nan,
            n_interpolated_samples=n_interp,
            interpolation_pct=(100 * n_interp / n_samples)
            if n_samples and np.isfinite(n_interp)
            else np.nan,
            n_artifact_samples=n_artifact,
            artifact_pct=(100 * n_artifact / n_samples)
            if n_samples and np.isfinite(n_artifact)
            else np.nan,
            time_min=time_min,
            time_max=time_max,
            time_span_ms=time_max - time_min
            if np.isfinite(time_min) and np.isfinite(time_max)
            else np.nan,
            mean_pupil=float(np.mean(finite_y)) if len(finite_y) else np.nan,
            sd_pupil=float(np.std(finite_y, ddof=1)) if len(finite_y) >= 2 else np.nan,
            peak_pupil=peak,
            peak_time_ms=peak_time,
            time_to_peak_ms=peak_time - time_min
            if np.isfinite(peak_time) and np.isfinite(time_min)
            else np.nan,
            pupil_auc=auc,
            early_mean_pupil=mean_window(y, t, early_window),
            middle_mean_pupil=mean_window(y, t, middle_window),
            late_mean_pupil=mean_window(y, t, late_window),
            n_valid_early=count_window(y, t, early_window),
            n_valid_middle=count_window(y, t, middle_window),
            n_valid_late=count_window(y, t, late_window),
            early_window_start_ms=float(early_window[0]),
            early_window_end_ms=float(early_window[1]),
            middle_window_start_ms=float(middle_window[0]),
            middle_window_end_ms=float(middle_window[1]),
            late_window_start_ms=float(late_window[0]),
            late_window_end_ms=float(late_window[1]),
            pupil_feature_status="insufficient_valid_samples"
            if n_valid < min_valid_samples
            else ("missing_pupil" if not len(finite_y) else "ok"),
            pupil_feature_pupil_column=pupil_col,
            pupil_feature_time_column=time_col,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_gazepoint_pupil_response_features(data, **kwargs):
    return summarise_gazepoint_pupil_trial_features(data, **kwargs)


summarise_gazepoint_pupil_response_features = summarize_gazepoint_pupil_response_features


def _gp3_pupil_r_group_cols(id_col, group_cols):
    groups = []

    if id_col is not None:
        groups.append(str(id_col))

    if group_cols is not None:
        if isinstance(group_cols, str):
            group_cols = [group_cols]

        for column in group_cols:
            column = str(column)

            if column not in groups:
                groups.append(column)

    return groups


def _gp3_pupil_r_group_positions(data, columns):
    if len(data) == 0:
        return []

    if not columns:
        return [np.arange(len(data), dtype=int)]

    groups = data.groupby(
        columns,
        dropna=False,
        sort=False,
    )

    return [np.asarray(indices, dtype=int) for indices in groups.indices.values()]


def _gp3_pupil_r_detect_columns(data, pupil_cols):
    if pupil_cols is not None:
        if isinstance(pupil_cols, str):
            columns = [pupil_cols]
        else:
            columns = [str(column) for column in pupil_cols]

        columns = list(dict.fromkeys(columns))

        missing = [column for column in columns if column not in data.columns]

        if missing:
            raise ValueError("data is missing required column(s): " + ", ".join(missing))

        return columns

    candidates = [
        "mean_pupil",
        "pupil_regressed",
        "pupil_smoothed",
        "pupil_interpolated",
        "pupil_clean",
        "pupil",
        "LPupil",
        "RPupil",
        "LPD",
        "RPD",
        "LPMM",
        "RPMM",
    ]

    detected = [column for column in candidates if column in data.columns]

    if not detected:
        raise ValueError(
            "No pupil column was supplied or detected. Provide pupil_col or pupil_cols explicitly."
        )

    return [detected[0]]


def _gp3_pupil_r_numeric(values):
    return pd.to_numeric(
        values,
        errors="coerce",
    ).to_numpy(dtype=float)


def downsample_gazepoint_pupil(
    data=None,
    time_col=None,
    pupil_col=None,
    target_hz: float = 30.0,
    group_cols=None,
    *,
    master_df=None,
    factor=None,
    pupil_cols=None,
    id_col="USER_ID",
    ts_col="TIME",
    method=None,
    keep_bin=False,
) -> pd.DataFrame:
    """Downsample pupil data using Python or R v2.3.0 semantics."""
    r_mode = (
        master_df is not None
        or factor is not None
        or pupil_cols is not None
        or method is not None
        or keep_bin
        or id_col != "USER_ID"
        or ts_col != "TIME"
    )

    if not r_mode:
        df = ensure_dataframe(
            data,
            copy=False,
        )

        time_col = infer_column(
            df,
            "time",
            time_col,
            required=True,
        )

        pupil_col = infer_column(
            df,
            "pupil",
            pupil_col,
            required=True,
        )

        groups = normalize_group_cols(
            df,
            group_cols,
        )

        bin_width = 1 / target_hz

        work = df.copy()

        ts = time_to_seconds(work[time_col])

        work["_bin"] = (ts / bin_width).round().astype("Int64")

        agg_cols = groups + ["_bin"]

        return (
            work.groupby(
                agg_cols,
                dropna=False,
            )
            .agg(
                **{
                    time_col: (
                        time_col,
                        "mean",
                    ),
                    pupil_col: (
                        pupil_col,
                        "mean",
                    ),
                }
            )
            .reset_index()
            .drop(columns="_bin")
        )

    if master_df is not None and data is not None:
        raise TypeError("supply either data or master_df, not both")

    frame = ensure_dataframe(
        master_df if master_df is not None else data,
        copy=False,
    )

    factor = 2 if factor is None else factor
    method = "mean" if method is None else str(method)

    if (
        isinstance(factor, bool)
        or not isinstance(
            factor,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(factor)
        or factor < 1
        or float(factor) != int(factor)
    ):
        raise ValueError("factor must be one positive integer")

    factor = int(factor)

    if method not in {
        "mean",
        "first",
    }:
        raise ValueError("method must be 'mean' or 'first'")

    detected_pupil_cols = _gp3_pupil_r_detect_columns(
        frame,
        pupil_cols,
    )

    groups = _gp3_pupil_r_group_cols(
        id_col,
        group_cols,
    )

    required = list(
        dict.fromkeys(groups + detected_pupil_cols + ([] if ts_col is None else [ts_col]))
    )

    missing = [column for column in required if column not in frame.columns]

    if missing:
        raise ValueError("master_df is missing required column(s): " + ", ".join(missing))

    output_rows = []

    for indices in _gp3_pupil_r_group_positions(
        frame,
        groups,
    ):
        if ts_col is not None:
            ordered = (
                frame.iloc[indices]
                .sort_values(
                    ts_col,
                    kind="stable",
                    na_position="last",
                )
                .index
            )

            position_map = {index: position for position, index in enumerate(frame.index)}

            indices = np.asarray(
                [position_map[index] for index in ordered],
                dtype=int,
            )

        for start in range(
            0,
            len(indices),
            factor,
        ):
            bin_indices = indices[start : start + factor]

            row = frame.iloc[int(bin_indices[0])].copy().to_dict()

            if method == "mean":
                for column in detected_pupil_cols:
                    values = _gp3_pupil_r_numeric(frame.iloc[bin_indices][column])

                    finite = np.isfinite(values)

                    row[column] = float(np.mean(values[finite])) if finite.any() else np.nan

                if ts_col is not None:
                    values = _gp3_pupil_r_numeric(frame.iloc[bin_indices][ts_col])

                    finite = np.isfinite(values)

                    row[ts_col] = float(np.mean(values[finite])) if finite.any() else np.nan

            row["n_samples_aggregated"] = len(bin_indices)

            row["downsample_factor"] = factor

            row["downsample_bin"] = (start // factor) + 1

            output_rows.append(row)

    if output_rows:
        output = pd.DataFrame(output_rows).reset_index(drop=True)
    else:
        output = frame.iloc[0:0].copy()

    if not keep_bin and "downsample_bin" in output.columns:
        output = output.drop(columns=["downsample_bin"])

    output.attrs["gazepoint_downsampling"] = {
        "factor": factor,
        "method": method,
        "pupil_cols": detected_pupil_cols,
        "group_cols": groups,
    }

    return output


def _gp3_pupil_r_row_mean_two(left, right):
    left = np.asarray(
        left,
        dtype=float,
    )

    right = np.asarray(
        right,
        dtype=float,
    )

    left_ok = np.isfinite(left)
    right_ok = np.isfinite(right)

    available = left_ok.astype(int) + right_ok.astype(int)

    left_zero = np.where(
        left_ok,
        left,
        0.0,
    )

    right_zero = np.where(
        right_ok,
        right,
        0.0,
    )

    result = (left_zero + right_zero) / np.maximum(
        available,
        1,
    )

    result[available == 0] = np.nan

    return result


def _gp3_pupil_r_fit_line(x, y):
    x = np.asarray(
        x,
        dtype=float,
    )

    y = np.asarray(
        y,
        dtype=float,
    )

    design = np.column_stack(
        [
            np.ones(len(x)),
            x,
        ]
    )

    coefficients, _, _, _ = np.linalg.lstsq(
        design,
        y,
        rcond=None,
    )

    if len(coefficients) != 2 or not np.isfinite(coefficients).all():
        return np.array(
            [
                float(np.mean(y)),
                0.0,
            ]
        )

    return coefficients


def regress_gazepoint_pupils(
    data=None,
    left_col=None,
    right_col=None,
    direction="right_from_left",
    *,
    master_df=None,
    lp_col=None,
    rp_col=None,
    id_col="USER_ID",
    group_cols=None,
    output_col="pupil_regressed",
    residual_col="pupil_regression_residual",
    min_complete=None,
):
    """Regress binocular pupil signals using Python or R v2.3.0 semantics."""
    r_directions = {
        "bidirectional",
        "right_on_left",
        "left_on_right",
    }

    r_mode = (
        master_df is not None
        or lp_col is not None
        or rp_col is not None
        or group_cols is not None
        or id_col != "USER_ID"
        or output_col != "pupil_regressed"
        or residual_col != "pupil_regression_residual"
        or min_complete is not None
        or direction in r_directions
    )

    if not r_mode:
        df = ensure_dataframe(
            data,
            copy=False,
        )

        left_col = infer_column(
            df,
            "left_pupil",
            left_col,
            required=True,
        )

        right_col = infer_column(
            df,
            "right_pupil",
            right_col,
            required=True,
        )

        left_values = finite_numeric(df[left_col])

        right_values = finite_numeric(df[right_col])

        ok = left_values.notna() & right_values.notna()

        if direction == "right_from_left":
            x = left_values[ok]
            y = right_values[ok]
        else:
            x = right_values[ok]
            y = left_values[ok]

        model = stats.linregress(x, y) if ok.sum() >= 3 else None

        return {
            "direction": direction,
            "n": int(ok.sum()),
            "slope": model.slope if model else np.nan,
            "intercept": model.intercept if model else np.nan,
            "r2": model.rvalue**2 if model else np.nan,
            "model": model,
        }

    if master_df is not None and data is not None:
        raise TypeError("supply either data or master_df, not both")

    frame = ensure_dataframe(
        master_df if master_df is not None else data,
        copy=False,
    )

    lp_col = lp_col or left_col or "LPupil"

    rp_col = rp_col or right_col or "RPupil"

    if direction not in r_directions:
        direction = "bidirectional"

    min_complete = 10 if min_complete is None else min_complete

    if (
        isinstance(min_complete, bool)
        or not isinstance(
            min_complete,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(min_complete)
        or min_complete < 2
        or float(min_complete) != int(min_complete)
    ):
        raise ValueError("min_complete must be an integer of at least 2")

    min_complete = int(min_complete)

    groups = _gp3_pupil_r_group_cols(
        id_col,
        group_cols,
    )

    required = list(
        dict.fromkeys(
            groups
            + [
                lp_col,
                rp_col,
            ]
        )
    )

    missing = [column for column in required if column not in frame.columns]

    if missing:
        raise ValueError("master_df is missing required column(s): " + ", ".join(missing))

    output = frame.copy()

    output[output_col] = np.nan
    output[residual_col] = np.nan
    output["pupil_regression_n"] = pd.Series(
        [pd.NA] * len(output),
        dtype="Int64",
    )

    output["pupil_regression_method"] = pd.Series(
        [pd.NA] * len(output),
        dtype="object",
    )

    output_col_position = output.columns.get_loc(output_col)

    residual_position = output.columns.get_loc(residual_col)

    n_position = output.columns.get_loc("pupil_regression_n")

    method_position = output.columns.get_loc("pupil_regression_method")

    for indices in _gp3_pupil_r_group_positions(
        frame,
        groups,
    ):
        left = _gp3_pupil_r_numeric(frame.iloc[indices][lp_col])

        right = _gp3_pupil_r_numeric(frame.iloc[indices][rp_col])

        complete = np.isfinite(left) & np.isfinite(right)

        n_complete = int(complete.sum())

        fallback = _gp3_pupil_r_row_mean_two(
            left,
            right,
        )

        fused = fallback.copy()

        residual = np.full(
            len(indices),
            np.nan,
        )

        method_used = "binocular_mean_fallback"

        can_fit = (
            n_complete >= min_complete
            and np.std(
                left[complete],
                ddof=1,
            )
            > 0
            and np.std(
                right[complete],
                ddof=1,
            )
            > 0
        )

        if can_fit:
            right_fit = _gp3_pupil_r_fit_line(
                left[complete],
                right[complete],
            )

            left_fit = _gp3_pupil_r_fit_line(
                right[complete],
                left[complete],
            )

            predicted_right = np.full(
                len(indices),
                np.nan,
            )

            predicted_left = np.full(
                len(indices),
                np.nan,
            )

            finite_left = np.isfinite(left)

            finite_right = np.isfinite(right)

            predicted_right[finite_left] = right_fit[0] + right_fit[1] * left[finite_left]

            predicted_left[finite_right] = left_fit[0] + left_fit[1] * right[finite_right]

            both = finite_left & finite_right

            residual[both] = right[both] - predicted_right[both]

            if direction == "bidirectional":
                fused = _gp3_pupil_r_row_mean_two(
                    predicted_left,
                    predicted_right,
                )

            elif direction == "right_on_left":
                fused = predicted_right

            else:
                fused = predicted_left

            missing_fused = ~np.isfinite(fused)

            fused[missing_fused] = fallback[missing_fused]

            method_used = direction

        output.iloc[
            indices,
            output_col_position,
        ] = fused

        output.iloc[
            indices,
            residual_position,
        ] = residual

        output.iloc[
            indices,
            n_position,
        ] = n_complete

        output.iloc[
            indices,
            method_position,
        ] = method_used

    output.attrs["gazepoint_pupil_regression"] = {
        "lp_col": lp_col,
        "rp_col": rp_col,
        "direction": direction,
        "output_col": output_col,
        "group_cols": groups,
    }

    return output


def fit_gazepoint_binocular_calibration(data, left_col=None, right_col=None) -> dict[str, Any]:
    return {
        "right_from_left": regress_gazepoint_pupils(data, left_col, right_col, "right_from_left"),
        "left_from_right": regress_gazepoint_pupils(data, left_col, right_col, "left_from_right"),
    }


def diagnose_gazepoint_binocular_pupil(
    data, left_col=None, right_col=None, group_cols=None
) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    left_col = infer_column(df, "left_pupil", left_col, required=True)
    right_col = infer_column(df, "right_pupil", right_col, required=True)
    groups = normalize_group_cols(df, group_cols)
    rows = []
    iterator = [((), df)] if not groups else df.groupby(groups, dropna=False, sort=False)
    for key, frame in iterator:
        left_values = finite_numeric(frame[left_col])
        right_values = finite_numeric(frame[right_col])
        both = left_values.notna() & right_values.notna()
        diff = left_values - right_values
        corr = float(left_values[both].corr(right_values[both])) if both.sum() > 1 else np.nan
        rmse = float(np.sqrt(np.nanmean(diff[both] ** 2))) if both.any() else np.nan
        row = {c: v for c, v in zip(groups, key, strict=False)} if groups else {}
        row.update(
            n=len(frame),
            left_missing_prop=float(left_values.isna().mean()),
            right_missing_prop=float(right_values.isna().mean()),
            both_valid_prop=float(both.mean()),
            correlation=corr,
            rmse=rmse,
            mae=float(np.nanmean(abs(diff[both]))) if both.any() else np.nan,
        )
        rows.append(row)
    return pd.DataFrame(rows)


def reconstruct_gazepoint_binocular_pupil(
    data,
    left_col=None,
    right_col=None,
    method="linear_regression",
    output_left="left_pupil_reconstructed",
    output_right="right_pupil_reconstructed",
    combined_col="pupil_combined",
) -> pd.DataFrame:
    df = ensure_dataframe(data)
    left_col = infer_column(df, "left_pupil", left_col, required=True)
    right_col = infer_column(df, "right_pupil", right_col, required=True)
    left_values = finite_numeric(df[left_col])
    right_values = finite_numeric(df[right_col])
    left_reconstructed = left_values.copy()
    right_reconstructed = right_values.copy()
    src = pd.Series("observed", index=df.index, dtype="string")
    if method in {"linear_regression", "regression"}:
        models = fit_gazepoint_binocular_calibration(df, left_col, right_col)
        a = models["right_from_left"]
        b = models["left_from_right"]
        miss_r = right_values.isna() & left_values.notna()
        miss_l = left_values.isna() & right_values.notna()
        right_reconstructed.loc[miss_r] = a["intercept"] + a["slope"] * left_values.loc[miss_r]
        left_reconstructed.loc[miss_l] = b["intercept"] + b["slope"] * right_values.loc[miss_l]
        src.loc[miss_r] = "right_reconstructed"
        src.loc[miss_l] = "left_reconstructed"
    elif method == "available_eye":
        miss_r = right_values.isna() & left_values.notna()
        miss_l = left_values.isna() & right_values.notna()
        right_reconstructed.loc[miss_r] = left_values.loc[miss_r]
        left_reconstructed.loc[miss_l] = right_values.loc[miss_l]
        src.loc[miss_r] = "right_from_left"
        src.loc[miss_l] = "left_from_right"
    elif method == "none":
        pass
    else:
        raise ValueError(f"Unknown reconstruction method {method!r}")
    df[output_left] = left_reconstructed
    df[output_right] = right_reconstructed
    df[combined_col] = pd.concat([left_reconstructed, right_reconstructed], axis=1).mean(
        axis=1, skipna=True
    )
    df["pupil_reconstruction_source"] = src
    return df


def audit_gazepoint_binocular_reconstruction(
    data,
    observed_left=None,
    observed_right=None,
    reconstructed_left="left_pupil_reconstructed",
    reconstructed_right="right_pupil_reconstructed",
) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    observed_left = infer_column(df, "left_pupil", observed_left, required=True)
    observed_right = infer_column(df, "right_pupil", observed_right, required=True)
    rows = []
    for eye, obs, recon in (
        ("left", observed_left, reconstructed_left),
        ("right", observed_right, reconstructed_right),
    ):
        if recon not in df:
            continue
        o = finite_numeric(df[obs])
        r = finite_numeric(df[recon])
        replaced = o.isna() & r.notna()
        rows.append(
            {
                "eye": eye,
                "n": len(df),
                "n_reconstructed": int(replaced.sum()),
                "reconstructed_prop": float(replaced.mean()),
            }
        )
    return pd.DataFrame(rows)


def validate_gazepoint_binocular_reconstruction(
    data,
    left_col=None,
    right_col=None,
    fraction: float = 0.1,
    random_state: int = 123,
    method="linear_regression",
) -> pd.DataFrame:
    df = ensure_dataframe(data)
    left_col = infer_column(df, "left_pupil", left_col, required=True)
    right_col = infer_column(df, "right_pupil", right_col, required=True)
    rng = np.random.default_rng(random_state)
    rows = []
    for direction, target, other in (("left", left_col, right_col), ("right", right_col, left_col)):
        valid = df[target].notna() & df[other].notna()
        ids = np.flatnonzero(valid.to_numpy())
        n = max(1, int(round(len(ids) * fraction))) if len(ids) else 0
        hold = (
            rng.choice(ids, size=min(n, len(ids)), replace=False) if n else np.array([], dtype=int)
        )
        tmp = df.copy()
        truth = finite_numeric(tmp[target]).iloc[hold].to_numpy(float)
        tmp.iloc[hold, tmp.columns.get_loc(target)] = np.nan
        rec = reconstruct_gazepoint_binocular_pupil(tmp, left_col, right_col, method=method)
        pred = (
            finite_numeric(
                rec[
                    "left_pupil_reconstructed"
                    if direction == "left"
                    else "right_pupil_reconstructed"
                ]
            )
            .iloc[hold]
            .to_numpy(float)
        )
        ok = np.isfinite(truth) & np.isfinite(pred)
        err = pred[ok] - truth[ok]
        rows.append(
            {
                "direction": direction,
                "n_holdout": int(ok.sum()),
                "rmse": float(np.sqrt(np.mean(err**2))) if ok.any() else np.nan,
                "mae": float(np.mean(abs(err))) if ok.any() else np.nan,
                "bias": float(np.mean(err)) if ok.any() else np.nan,
                "correlation": float(np.corrcoef(truth[ok], pred[ok])[0, 1])
                if ok.sum() > 1
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def stress_test_gazepoint_binocular_reconstruction(
    data, fractions=(0.05, 0.1, 0.2, 0.3), **kwargs
) -> pd.DataFrame:
    rows = []
    for f in fractions:
        out = validate_gazepoint_binocular_reconstruction(data, fraction=f, **kwargs).copy()
        out["missing_fraction"] = f
        rows.append(out)
    return pd.concat(rows, ignore_index=True)


def analyse_gazepoint_binocular_sensitivity(
    data, methods=("available_eye", "linear_regression"), **kwargs
) -> pd.DataFrame:
    rows = []
    for method in methods:
        val = validate_gazepoint_binocular_reconstruction(data, method=method, **kwargs).copy()
        val["method"] = method
        rows.append(val)
    return pd.concat(rows, ignore_index=True)


def summarise_gazepoint_binocular_reporting(data, **kwargs) -> dict[str, pd.DataFrame]:
    return {
        "diagnostics": diagnose_gazepoint_binocular_pupil(data, **kwargs),
        "validation": validate_gazepoint_binocular_reconstruction(data, **kwargs),
    }


def impute_gazepoint_pupil_gp(
    data,
    pupil_col=None,
    time_col=None,
    output_col=None,
    max_points: int = 2000,
    random_state: int = 123,
    *,
    subject=None,
    trial=None,
    length_scale=None,
    noise: float = 1e-4,
    flag="pupil_was_gp_imputed",
) -> pd.DataFrame:
    """Impute pupil values with legacy sklearn or grouped R-style GP smoothing."""
    df = ensure_dataframe(data)
    pupil_col = infer_column(df, "pupil", pupil_col, required=True)
    time_col = infer_column(df, "time", time_col, required=True)
    r_mode = (
        subject is not None
        or trial is not None
        or length_scale is not None
        or noise != 1e-4
        or flag != "pupil_was_gp_imputed"
    )
    if not r_mode:
        output_col = output_col or f"{pupil_col}_gp"
        t = finite_numeric(df[time_col]).to_numpy(float)
        y = finite_numeric(df[pupil_col]).to_numpy(float)
        ok = np.isfinite(t) & np.isfinite(y)
        miss = np.isfinite(t) & ~np.isfinite(y)
        if ok.sum() < 3:
            df[output_col] = y
            return df
        ids = np.flatnonzero(ok)
        if len(ids) > max_points:
            ids = np.linspace(0, len(ids) - 1, max_points).astype(int)
            ids = np.flatnonzero(ok)[ids]
        kernel = 1.0 * RBF(length_scale=max(np.nanstd(t[ok]) * 0.1, 1e-6)) + WhiteKernel(
            noise_level=max(np.nanvar(y[ok]) * 0.01, 1e-6)
        )
        gp = GaussianProcessRegressor(
            kernel=kernel, normalize_y=True, random_state=random_state, n_restarts_optimizer=0
        )
        gp.fit(t[ids, None], y[ids])
        values = y.copy()
        values[miss] = gp.predict(t[miss, None])
        df[output_col] = values
        return df
    for col in (subject, trial):
        if col is not None and col not in df:
            raise KeyError(f"Missing required column: {col}")
    if max_points < 1:
        raise ValueError("max_train must be positive")
    if not (np.isfinite(noise) and noise >= 0):
        raise ValueError("noise must be a finite non-negative number")
    if length_scale is not None and (not np.isfinite(length_scale) or length_scale <= 0):
        raise ValueError("length_scale must be a positive finite number")
    output_col = output_col or "pupil_gp_imputed"
    values = pd.to_numeric(df[pupil_col], errors="coerce").to_numpy(dtype=float, copy=True)
    was_imputed = np.zeros(len(df), dtype=bool)
    groups = [c for c in (subject, trial) if c is not None]
    iterator = (
        [(None, np.arange(len(df)))]
        if not groups
        else df.groupby(groups, dropna=False, sort=False).indices.items()
    )
    for _, positions in iterator:
        positions = np.asarray(positions, dtype=int)
        local_t = pd.to_numeric(df.iloc[positions][time_col], errors="coerce").to_numpy(float)
        order = np.argsort(local_t, kind="stable")
        positions = positions[order]
        local_t = local_t[order]
        local_y = pd.to_numeric(df.iloc[positions][pupil_col], errors="coerce").to_numpy(float)
        obs = np.isfinite(local_t) & np.isfinite(local_y)
        miss = np.isfinite(local_t) & ~np.isfinite(local_y)
        if obs.sum() < 3 or not miss.any():
            continue
        train_t = local_t[obs]
        train_y = local_y[obs]
        if len(train_t) > max_points:
            pick = np.unique(np.rint(np.linspace(0, len(train_t) - 1, max_points)).astype(int))
            train_t = train_t[pick]
            train_y = train_y[pick]
        ell = length_scale
        if ell is None:
            diffs = np.diff(np.unique(np.sort(train_t)))
            ell = float(np.median(diffs) * 10) if len(diffs) else np.nan
            if not np.isfinite(ell) or ell <= 0:
                ell = float(np.ptp(train_t) / 10)
            if not np.isfinite(ell) or ell <= 0:
                ell = 1.0
        ell = float(ell)
        if not np.isfinite(ell) or ell <= 0:
            raise ValueError("length_scale must be a positive finite number")
        k_tt = np.exp(-0.5 * ((train_t[:, None] - train_t[None, :]) / ell) ** 2)
        k_tt.flat[:: len(train_t) + 1] += noise
        mean_y = float(np.mean(train_y))
        try:
            alpha = np.linalg.solve(k_tt, train_y - mean_y)
        except np.linalg.LinAlgError:
            continue
        k_mt = np.exp(-0.5 * ((local_t[miss, None] - train_t[None, :]) / ell) ** 2)
        target = positions[miss]
        values[target] = k_mt @ alpha + mean_y
        was_imputed[target] = True
    df[output_col] = values
    df[flag] = was_imputed
    return df


def preprocess_gazepoint_signals(
    data,
    pupil_col=None,
    physiological_min=1.0,
    physiological_max=9.0,
    interpolate=True,
    smooth=True,
    baseline=None,
    time_col=None,
) -> pd.DataFrame:
    df = flag_gazepoint_pupil_artifacts(
        data,
        pupil_col=pupil_col,
        time_col=time_col,
        physiological_min=physiological_min,
        physiological_max=physiological_max,
    )
    pupil_col = infer_column(df, "pupil", pupil_col, required=True)
    work_col = pupil_col
    df.loc[df["pupil_artifact"], work_col] = np.nan
    if interpolate:
        df = interpolate_gazepoint_pupil(
            df, pupil_col=work_col, time_col=time_col, output_col=f"{pupil_col}_clean"
        )
        work_col = f"{pupil_col}_clean"
    if smooth:
        df = smooth_gazepoint_pupil(df, pupil_col=work_col, output_col=f"{pupil_col}_processed")
        work_col = f"{pupil_col}_processed"
    if baseline is not None:
        df = baseline_correct_gazepoint_pupil(
            df,
            pupil_col=work_col,
            time_col=time_col,
            baseline=baseline,
            output_col=f"{pupil_col}_baseline_corrected",
        )
    return df


def create_gazepoint_preprocessing_registry(
    blink_padding_pre_ms=100,
    blink_padding_post_ms=100,
    max_interpolation_gap_ms=150,
    smoothing_window_ms=50,
    baseline_start_ms=-200,
    baseline_end_ms=0,
    pupil_physiological_min=1,
    pupil_physiological_max=9,
    pupil_speed_mad_k=6,
    binocular_mad_k=6,
    baseline_missing_prop_threshold=0.3,
    baseline_interpolated_prop_threshold=0.3,
    baseline_artifact_prop_threshold=0.3,
    overlap_trial_duration_ms=3000,
    overlap_event_gap_ms=1000,
) -> pd.DataFrame:
    values = locals()
    units = {
        k: (
            "ms"
            if k.endswith("_ms")
            else "proportion"
            if "prop" in k
            else "pupil"
            if "physiological" in k
            else "MAD"
        )
        for k in values
    }
    cats = {
        k: (
            "artifact"
            if "blink" in k or "speed" in k or "physiological" in k
            else "interpolation"
            if "interpolation" in k
            else "smoothing"
            if "smoothing" in k
            else "baseline"
            if "baseline" in k
            else "binocular"
            if "binocular" in k
            else "overlap"
        )
        for k in values
    }
    return pd.DataFrame(
        [
            {
                "parameter": k,
                "value": v,
                "unit": units[k],
                "category": cats[k],
                "description": k.replace("_", " "),
            }
            for k, v in values.items()
        ]
    )


def create_gazepoint_preprocessing_multiverse(**grids) -> pd.DataFrame:
    import itertools

    if not grids:
        grids = {
            "interpolation_method": ["linear", "pchip"],
            "smoothing_window": [3, 5, 9],
            "baseline_mode": ["subtract", "percent"],
        }
    keys = list(grids)
    return pd.DataFrame(
        [
            dict(zip(keys, vals, strict=False))
            for vals in itertools.product(*[grids[k] for k in keys])
        ]
    )


# BEGIN R V2.3.0 CALL-SURFACE ALIASES
detect_gazepoint_blinks = r_aliases(
    detect_gazepoint_blinks, all_gaze="data", ts_col="time_col", min_duration="min_duration_ms"
)
flag_gazepoint_pupil = r_aliases(
    flag_gazepoint_pupil,
    master="data",
    min_pupil="physiological_min",
    max_pupil="physiological_max",
)
impute_gazepoint_pupil_gp = r_aliases(
    impute_gazepoint_pupil_gp,
    pupil="pupil_col",
    time="time_col",
    output="output_col",
    max_train="max_points",
)
mean_gazepoint_pupil = r_aliases(
    mean_gazepoint_pupil, master_df="data", lp_col="left_col", rp_col="right_col"
)
# END R V2.3.0 CALL-SURFACE ALIASES
