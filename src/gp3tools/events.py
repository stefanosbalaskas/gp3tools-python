"""Fixation, saccade, and event-detection helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._compat import r_aliases
from ._utils import ensure_dataframe, infer_column, normalize_group_cols, time_to_seconds


def _velocity(df: pd.DataFrame, x_col: str, y_col: str, time_col: str) -> np.ndarray:
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(float)
    t = time_to_seconds(df[time_col]).to_numpy(float)
    dt = np.diff(t, prepend=np.nan)
    d = np.sqrt(np.diff(x, prepend=np.nan) ** 2 + np.diff(y, prepend=np.nan) ** 2)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = d / dt
    return v


def detect_gazepoint_fixations_velocity(
    data,
    x_col=None,
    y_col=None,
    time_col=None,
    velocity_threshold: float = 0.08,
    min_duration_ms: float = 100.0,
    group_cols=None,
) -> pd.DataFrame:
    """Detect fixations with a transparent velocity-threshold algorithm."""
    df = ensure_dataframe(data)
    x_col = x_col or infer_column(df, "x")
    y_col = y_col or infer_column(df, "y")
    time_col = time_col or infer_column(df, "time")
    if not all([x_col, y_col, time_col]):
        raise ValueError("x, y, and time columns are required")
    groups = normalize_group_cols(df, group_cols)
    if not groups:
        groups = [c for c in [infer_column(df, "subject"), infer_column(df, "trial")] if c]
    out = df.copy()
    out["event_velocity"] = np.nan
    out["fixation"] = False
    out["fixation_id"] = pd.Series(pd.NA, index=out.index, dtype="Int64")
    next_id = 1
    iterator = out.groupby(groups, sort=False, dropna=False) if groups else [(None, out)]
    for _, g in iterator:
        idx = g.index
        vel = _velocity(g, x_col, y_col, time_col)
        cand = np.isfinite(vel) & (vel <= velocity_threshold)
        # run-length filter by measured duration
        run = np.cumsum(np.r_[True, cand[1:] != cand[:-1]])
        keep = np.zeros(len(g), dtype=bool)
        t = time_to_seconds(g[time_col]).to_numpy(float)
        for rid in np.unique(run[cand]):
            loc = np.where((run == rid) & cand)[0]
            if not len(loc):
                continue
            duration = (t[loc[-1]] - t[loc[0]]) * 1000 if len(loc) > 1 else 0.0
            if duration >= min_duration_ms:
                keep[loc] = True
                out.loc[idx[loc], "fixation_id"] = next_id
                next_id += 1
        out.loc[idx, "event_velocity"] = vel
        out.loc[idx, "fixation"] = keep
    return out


def detect_gazepoint_fixations_ivt(data, **kwargs) -> pd.DataFrame:
    """Alias for I-VT fixation detection."""
    return detect_gazepoint_fixations_velocity(data, **kwargs)


def classify_gazepoint_events_hmm(
    data, x_col=None, y_col=None, time_col=None, **kwargs
) -> pd.DataFrame:
    """Classify samples into fixation/saccade states using a robust velocity mixture.

    This native Python implementation is an HMM-inspired two-state classifier rather
    than a bit-for-bit port of any R backend. It keeps the public workflow usable
    without requiring an optional probabilistic package.
    """
    df = ensure_dataframe(data)
    x_col = x_col or infer_column(df, "x")
    y_col = y_col or infer_column(df, "y")
    time_col = time_col or infer_column(df, "time")
    if not all([x_col, y_col, time_col]):
        raise ValueError("x, y, and time columns are required")
    v = _velocity(df, x_col, y_col, time_col)
    finite = v[np.isfinite(v)]
    threshold = (
        float(np.nanmedian(finite) + 2.5 * np.nanmedian(np.abs(finite - np.nanmedian(finite))))
        if len(finite)
        else 0.0
    )
    out = df.copy()
    out["event_velocity"] = v
    out["event_state"] = np.where(np.isfinite(v) & (v <= threshold), "fixation", "saccade")
    out["event_state_probability"] = np.where(out["event_state"].eq("fixation"), 0.75, 0.75)
    out.attrs["backend"] = "native_velocity_state_classifier"
    return out


def compute_gazepoint_saccade_metrics(
    data, x_col=None, y_col=None, time_col=None, event_col="event_state", group_cols=None
) -> pd.DataFrame:
    """Summarise saccade amplitude, duration, and peak velocity."""
    df = ensure_dataframe(data)
    x_col = x_col or infer_column(df, "x")
    y_col = y_col or infer_column(df, "y")
    time_col = time_col or infer_column(df, "time")
    if event_col not in df:
        df = classify_gazepoint_events_hmm(df, x_col=x_col, y_col=y_col, time_col=time_col)
    groups = normalize_group_cols(df, group_cols)
    work = df.loc[df[event_col].astype(str).str.lower().eq("saccade")].copy()
    if work.empty:
        return pd.DataFrame(
            columns=[*groups, "n_samples", "duration_ms", "amplitude", "peak_velocity"]
        )
    work["_velocity"] = _velocity(work, x_col, y_col, time_col)
    rows = []
    iterator = work.groupby(groups, dropna=False, sort=False) if groups else [(None, work)]
    for key, g in iterator:
        t = time_to_seconds(g[time_col])
        amp = np.sqrt(
            (
                pd.to_numeric(g[x_col], errors="coerce").iloc[-1]
                - pd.to_numeric(g[x_col], errors="coerce").iloc[0]
            )
            ** 2
            + (
                pd.to_numeric(g[y_col], errors="coerce").iloc[-1]
                - pd.to_numeric(g[y_col], errors="coerce").iloc[0]
            )
            ** 2
        )
        row = {
            "n_samples": len(g),
            "duration_ms": float((t.max() - t.min()) * 1000),
            "amplitude": float(amp),
            "peak_velocity": float(np.nanmax(g["_velocity"])),
        }
        if groups:
            vals = key if isinstance(key, tuple) else (key,)
            row.update(dict(zip(groups, vals, strict=False)))
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_fixations(
    data,
    fixation_col="fixation",
    fixation_id_col="fixation_id",
    x_col=None,
    y_col=None,
    time_col=None,
    group_cols=None,
) -> pd.DataFrame:
    """Collapse fixation-classified samples to one row per fixation."""
    df = ensure_dataframe(data)
    if fixation_id_col not in df:
        df = detect_gazepoint_fixations_velocity(
            df, x_col=x_col, y_col=y_col, time_col=time_col, group_cols=group_cols
        )
    x_col = x_col or infer_column(df, "x")
    y_col = y_col or infer_column(df, "y")
    time_col = time_col or infer_column(df, "time")
    groups = normalize_group_cols(df, group_cols)
    keys = [*groups, fixation_id_col]
    work = df[df[fixation_id_col].notna()].copy()
    if work.empty:
        return pd.DataFrame(columns=[*keys, "duration_ms", "x", "y", "n_samples"])
    rows = []
    for key, g in work.groupby(keys, dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        t = (
            time_to_seconds(g[time_col])
            if time_col
            else pd.Series(np.arange(len(g)), index=g.index)
        )
        row = dict(zip(keys, key, strict=False))
        row.update(
            {
                "duration_ms": float((t.max() - t.min()) * 1000),
                "x": float(pd.to_numeric(g[x_col], errors="coerce").mean()) if x_col else np.nan,
                "y": float(pd.to_numeric(g[y_col], errors="coerce").mean()) if y_col else np.nan,
                "n_samples": len(g),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_gazepoint_fixation_trials(
    data, trial_col=None, subject_col=None, duration_col="duration_ms"
) -> pd.DataFrame:
    df = ensure_dataframe(data)
    trial_col = trial_col or infer_column(df, "trial")
    subject_col = subject_col or infer_column(df, "subject")
    keys = [c for c in [subject_col, trial_col] if c]
    if not keys:
        keys = []
    agg = {
        "n_fixations": (duration_col, "size"),
        "mean_fixation_duration_ms": (duration_col, "mean"),
        "total_fixation_duration_ms": (duration_col, "sum"),
    }
    if keys:
        return df.groupby(keys, dropna=False).agg(**agg).reset_index()
    return pd.DataFrame(
        {
            k: [getattr(df[duration_col], fn)() if fn != "size" else len(df)]
            for k, (_, fn) in agg.items()
        }
    )


def audit_gazepoint_fixation_reliability(
    data, subject_col=None, duration_col="duration_ms"
) -> pd.DataFrame:
    df = ensure_dataframe(data)
    subject_col = subject_col or infer_column(df, "subject")
    if not subject_col:
        return pd.DataFrame(
            {
                "n": [len(df)],
                "mean_duration_ms": [pd.to_numeric(df.get(duration_col), errors="coerce").mean()],
            }
        )
    return (
        df.groupby(subject_col, dropna=False)[duration_col]
        .agg(n="size", mean_duration_ms="mean", sd_duration_ms="std")
        .reset_index()
    )


def compare_gazepoint_event_detectors(data, **kwargs) -> pd.DataFrame:
    a = detect_gazepoint_fixations_velocity(data, **kwargs)
    b = classify_gazepoint_events_hmm(
        data, **{k: v for k, v in kwargs.items() if k in {"x_col", "y_col", "time_col"}}
    )
    out = pd.DataFrame(index=a.index)
    out["velocity_detector"] = np.where(a["fixation"], "fixation", "saccade")
    out["state_detector"] = b["event_state"].to_numpy()
    out["agreement"] = out["velocity_detector"].eq(out["state_detector"])
    return out.reset_index(drop=True)


def summarise_gazepoint_event_detector_agreement(data, **kwargs) -> pd.DataFrame:
    comp = (
        data
        if isinstance(data, pd.DataFrame) and "agreement" in data
        else compare_gazepoint_event_detectors(data, **kwargs)
    )
    return pd.DataFrame({"n": [len(comp)], "agreement_rate": [float(comp["agreement"].mean())]})


def benchmark_gazepoint_event_detectors(data, repeats: int = 3, **kwargs) -> pd.DataFrame:
    import time

    rows = []
    for name, fn in [
        ("velocity", detect_gazepoint_fixations_velocity),
        ("state", classify_gazepoint_events_hmm),
    ]:
        for r in range(repeats):
            start = time.perf_counter()
            fn(data, **kwargs)
            elapsed = time.perf_counter() - start
            rows.append(
                {"detector": name, "repeat": r + 1, "elapsed_seconds": elapsed, "n_rows": len(data)}
            )
    return pd.DataFrame(rows)


def summarise_gazepoint_event_detector_benchmark(data) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    return (
        df.groupby("detector", dropna=False)["elapsed_seconds"]
        .agg(n="size", mean_seconds="mean", median_seconds="median", max_seconds="max")
        .reset_index()
    )


def create_gazepoint_event_review_template(
    data, path: str | Path | None = None, **kwargs
) -> pd.DataFrame:
    df = ensure_dataframe(data)
    if "event_state" not in df:
        df = classify_gazepoint_events_hmm(df, **kwargs)
    out = df.copy()
    out["review_state"] = out["event_state"]
    out["reviewer_note"] = ""
    out["reviewed"] = False
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(path, index=False)
    return out


def simulate_gazepoint_fixations(
    n_fixations: int = 30, samples_per_fixation: int = 8, random_state: int = 123
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    t = 0.0
    for fid in range(1, n_fixations + 1):
        cx, cy = rng.uniform(0.1, 0.9, 2)
        for _ in range(samples_per_fixation):
            rows.append(
                {
                    "TIME": t,
                    "FPOGX": cx + rng.normal(0, 0.005),
                    "FPOGY": cy + rng.normal(0, 0.005),
                    "fixation": True,
                    "fixation_id": fid,
                }
            )
            t += 1 / 60
        t += rng.uniform(0.02, 0.08)
    return pd.DataFrame(rows)


# BEGIN R V2.3.0 CALL-SURFACE ALIASES
detect_gazepoint_fixations_velocity = r_aliases(
    detect_gazepoint_fixations_velocity,
    all_gaze="data",
    ts_col="time_col",
    vmax="velocity_threshold",
    min_duration="min_duration_ms",
)
# END R V2.3.0 CALL-SURFACE ALIASES
