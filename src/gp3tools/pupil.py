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
) -> pd.DataFrame:
    df = ensure_dataframe(data)
    pupil_col = infer_column(df, "pupil", pupil_col, required=True)
    x = finite_numeric(df[pupil_col])
    flag = pd.Series("ok", index=df.index, dtype="string")
    flag[x.isna()] = "missing"
    flag[x < physiological_min] = "below_min"
    flag[x > physiological_max] = "above_max"
    df[output_col] = flag
    df["pupil_valid"] = flag.eq("ok")
    return df


def flag_gazepoint_pupil_hampel(
    data, pupil_col=None, window: int = 7, n_sigma: float = 3.0, output_col="pupil_hampel_flag"
) -> pd.DataFrame:
    df = ensure_dataframe(data)
    pupil_col = infer_column(df, "pupil", pupil_col, required=True)
    x = finite_numeric(df[pupil_col])
    med = x.rolling(window, center=True, min_periods=max(1, window // 2)).median()
    dev = (x - med).abs()
    mad = dev.rolling(window, center=True, min_periods=max(1, window // 2)).median()
    threshold = 1.4826 * n_sigma * mad
    df[output_col] = dev > threshold
    return df


def detect_gazepoint_blinks(
    data,
    pupil_col=None,
    time_col=None,
    min_duration_ms: float = 50.0,
    max_duration_ms: float = 800.0,
    output_col="blink",
) -> pd.DataFrame:
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
    output_col=None,
    method="linear",
    max_gap_ms: float | None = 150.0,
    time_col=None,
    group_cols=None,
) -> pd.DataFrame:
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
            limit = max(1, int(round(max_gap_ms / 1000 * hz)))
        out.loc[frame.index] = _interpolate_series(x, method, limit)
    df[output_col] = out
    df[f"{output_col}_interpolated"] = finite_numeric(df[pupil_col]).isna() & out.notna()
    return df


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
    data, pupil_col=None, output_col=None, window: int = 5, method="moving_average"
) -> pd.DataFrame:
    df = ensure_dataframe(data)
    pupil_col = infer_column(df, "pupil", pupil_col, required=True)
    output_col = output_col or f"{pupil_col}_smoothed"
    x = finite_numeric(df[pupil_col])
    if method in {"moving_average", "mean"}:
        y = x.rolling(window, center=True, min_periods=1).mean()
    elif method == "median":
        y = x.rolling(window, center=True, min_periods=1).median()
    elif method == "savgol":
        win = max(3, window + (1 - window % 2))
        vals = x.interpolate(limit_direction="both").to_numpy(float)
        y = pd.Series(signal.savgol_filter(vals, win, min(2, win - 1)), index=df.index)
    else:
        raise ValueError(f"Unknown smoothing method: {method}")
    df[output_col] = y
    return df


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
    baseline=(-200.0, 0.0),
    group_cols=None,
    output_col=None,
    mode="subtract",
) -> pd.DataFrame:
    df = ensure_dataframe(data)
    pupil_col = infer_column(df, "pupil", pupil_col, required=True)
    time_col = infer_column(df, "time", time_col, required=True)
    groups = normalize_group_cols(df, group_cols)
    output_col = output_col or f"{pupil_col}_baseline_corrected"
    out = pd.Series(np.nan, index=df.index, dtype=float)
    bases = pd.Series(np.nan, index=df.index, dtype=float)
    iterator = [((), df)] if not groups else df.groupby(groups, dropna=False, sort=False)
    for _, frame in iterator:
        t = finite_numeric(frame[time_col])
        x = finite_numeric(frame[pupil_col])
        mask = t.between(float(baseline[0]), float(baseline[1]))
        b = float(x.loc[mask].mean()) if mask.any() else np.nan
        bases.loc[frame.index] = b
        if mode == "subtract":
            y = x - b
        elif mode == "divide":
            y = x / b
        elif mode in {"percent", "percent_change"}:
            y = (x - b) / b * 100
        else:
            raise ValueError(f"Unknown baseline mode {mode!r}")
        out.loc[frame.index] = y
    df[output_col] = out
    df["pupil_baseline"] = bases
    return df


def audit_gazepoint_pupil_gaps(
    data, pupil_col=None, time_col=None, group_cols=None
) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    pupil_col = infer_column(df, "pupil", pupil_col, required=True)
    groups = normalize_group_cols(df, group_cols)
    rows = []
    iterator = [((), df)] if not groups else df.groupby(groups, dropna=False, sort=False)
    for key, frame in iterator:
        if groups and not isinstance(key, tuple):
            key = (key,)
        miss = finite_numeric(frame[pupil_col]).isna().to_numpy()
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


def audit_gazepoint_pupil_baseline(
    data, pupil_col=None, time_col=None, baseline=(-200, 0), group_cols=None
) -> pd.DataFrame:
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


def audit_gazepoint_pupil_drift(
    data, pupil_col=None, time_col=None, group_cols=None
) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    pupil_col = infer_column(df, "pupil", pupil_col, required=True)
    time_col = infer_column(df, "time", time_col, required=True)
    groups = normalize_group_cols(df, group_cols)
    rows = []
    iterator = [((), df)] if not groups else df.groupby(groups, dropna=False, sort=False)
    for key, frame in iterator:
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


def audit_gazepoint_pupil_reliability(
    data, pupil_col=None, subject_col=None, split_col=None
) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
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


def audit_gazepoint_pupil_imbalance(data, pupil_col=None, condition_col=None) -> pd.DataFrame:
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


def audit_gazepoint_pupil_overlap_risk(
    data, trial_duration_ms=3000, event_gap_ms=1000, trial_col=None, time_col=None
) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    trial_col = infer_column(df, "trial", trial_col)
    time_col = infer_column(df, "time", time_col)
    if trial_col and time_col:
        work = df.copy()
        work["_t"] = finite_numeric(work[time_col])
        out = work.groupby(trial_col, dropna=False)._t.agg(["min", "max", "count"]).reset_index()
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
    data, pupil_col=None, time_col=None, windows=None, group_cols=None
) -> pd.DataFrame:
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
    for name, (lo, hi) in windows.items() if isinstance(windows, dict) else windows:
        sub = df.loc[finite_numeric(df[time_col]).between(lo, hi)].copy()
        tmp = summarise_gazepoint_pupil(sub, pupil_col=pupil_col, group_cols=group_cols)
        tmp.insert(len(tmp.columns) if len(tmp.columns) else 0, "window", name)
        tmp["window_start"] = lo
        tmp["window_end"] = hi
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarise_gazepoint_pupil_trial_features(
    data, pupil_col=None, trial_col=None, group_cols=None
) -> pd.DataFrame:
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
) -> pd.DataFrame:
    df = ensure_dataframe(data)
    pupil_col = infer_column(df, "pupil", pupil_col, required=True)
    time_col = infer_column(df, "time", time_col, required=True)
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
    out = y.copy()
    out[miss] = gp.predict(t[miss, None])
    df[output_col] = out
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
