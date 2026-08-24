"""Pupil preprocessing, blink handling, and binocular reconstruction."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import interpolate, ndimage, signal, stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

from ._utils import (
    ensure_dataframe,
    finite_numeric,
    infer_column,
    normalize_group_cols,
    robust_mad,
    time_to_seconds,
)


def mean_gazepoint_pupil(
    data, left_col=None, right_col=None, output_col="pupil_mean", require_both=False
) -> pd.DataFrame:
    df = ensure_dataframe(data)
    left_col = infer_column(df, "left_pupil", left_col, required=True)
    right_col = infer_column(df, "right_pupil", right_col, required=True)
    left, right = finite_numeric(df[left_col]), finite_numeric(df[right_col])
    if require_both:
        out = pd.concat([left, right], axis=1).mean(axis=1).where(left.notna() & right.notna())
    else:
        out = pd.concat([left, right], axis=1).mean(axis=1, skipna=True)
    df[output_col] = out
    return df


def combine_gazepoint_eyes(
    data, left_col=None, right_col=None, output_col="pupil_combined", policy="available_eye"
) -> pd.DataFrame:
    df = ensure_dataframe(data)
    left_col = infer_column(df, "left_pupil", left_col, required=True)
    right_col = infer_column(df, "right_pupil", right_col, required=True)
    left_values, right_values = finite_numeric(df[left_col]), finite_numeric(df[right_col])
    if policy in {"mean", "available_eye", "bilateral_mean"}:
        df[output_col] = pd.concat([left_values, right_values], axis=1).mean(
            axis=1, skipna=(policy != "bilateral_mean")
        )
        if policy == "bilateral_mean":
            df.loc[left_values.isna() | right_values.isna(), output_col] = np.nan
    elif policy == "left_only":
        df[output_col] = left_values
    elif policy == "right_only":
        df[output_col] = right_values
    elif policy == "complete_case":
        df[output_col] = ((left_values + right_values) / 2).where(
            left_values.notna() & right_values.notna()
        )
    else:
        raise ValueError(f"Unknown eye-combination policy: {policy}")
    df["pupil_eye_source"] = np.select(
        [left_values.notna() & right_values.notna(), left_values.notna(), right_values.notna()],
        ["both", "left", "right"],
        default="missing",
    )
    return df


construct_gazepoint_combined_pupil = combine_gazepoint_eyes


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


def smooth_gazepoint_coordinate(
    data, column=None, output_col=None, window: int = 5, method="moving_average"
) -> pd.DataFrame:
    df = ensure_dataframe(data)
    if column is None:
        column = infer_column(df, "x", required=True)
    output_col = output_col or f"{column}_smoothed"
    x = finite_numeric(df[column])
    df[output_col] = (
        x.rolling(window, center=True, min_periods=1).mean()
        if method == "moving_average"
        else x.rolling(window, center=True, min_periods=1).median()
    )
    return df


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


def summarise_gazepoint_pupil(data, pupil_col=None, group_cols=None) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    pupil_col = infer_column(df, "pupil", pupil_col, required=True)
    groups = normalize_group_cols(df, group_cols)
    work = df.copy()
    work["_p"] = finite_numeric(work[pupil_col])
    if groups:
        return (
            work.groupby(groups, dropna=False)
            .agg(
                n_samples=("_p", "size"),
                n_valid=("_p", "count"),
                mean_pupil=("_p", "mean"),
                sd_pupil=("_p", "std"),
                median_pupil=("_p", "median"),
                min_pupil=("_p", "min"),
                max_pupil=("_p", "max"),
            )
            .reset_index()
        )
    x = work._p
    return pd.DataFrame(
        [
            {
                "n_samples": len(x),
                "n_valid": int(x.notna().sum()),
                "mean_pupil": float(x.mean()),
                "sd_pupil": float(x.std()),
                "median_pupil": float(x.median()),
                "min_pupil": float(x.min()),
                "max_pupil": float(x.max()),
            }
        ]
    )


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


def downsample_gazepoint_pupil(
    data, time_col=None, pupil_col=None, target_hz: float = 30.0, group_cols=None
) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    time_col = infer_column(df, "time", time_col, required=True)
    pupil_col = infer_column(df, "pupil", pupil_col, required=True)
    groups = normalize_group_cols(df, group_cols)
    bin_width = 1 / target_hz
    work = df.copy()
    ts = time_to_seconds(work[time_col])
    work["_bin"] = (ts / bin_width).round().astype("Int64")
    agg_cols = groups + ["_bin"]
    out = (
        work.groupby(agg_cols, dropna=False)
        .agg(**{time_col: (time_col, "mean"), pupil_col: (pupil_col, "mean")})
        .reset_index()
        .drop(columns="_bin")
    )
    return out


def regress_gazepoint_pupils(
    data, left_col=None, right_col=None, direction="right_from_left"
) -> dict[str, Any]:
    df = ensure_dataframe(data, copy=False)
    left_col = infer_column(df, "left_pupil", left_col, required=True)
    right_col = infer_column(df, "right_pupil", right_col, required=True)
    left_values = finite_numeric(df[left_col])
    right_values = finite_numeric(df[right_col])
    ok = left_values.notna() & right_values.notna()
    x, y = (
        (left_values[ok], right_values[ok])
        if direction == "right_from_left"
        else (right_values[ok], left_values[ok])
    )
    lr = stats.linregress(x, y) if ok.sum() >= 3 else None
    return {
        "direction": direction,
        "n": int(ok.sum()),
        "slope": lr.slope if lr else np.nan,
        "intercept": lr.intercept if lr else np.nan,
        "r2": lr.rvalue**2 if lr else np.nan,
        "model": lr,
    }


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
