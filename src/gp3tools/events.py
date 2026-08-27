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
    *,
    id_col=None,
    time_unit=None,
    x_scale: float = 1.0,
    y_scale: float = 1.0,
    return_mode=None,
    keep_single_sample: bool = False,
    **kwargs,
):
    """Detect I-VT fixations with legacy Python or R v2.3.0 semantics.

    The R argument named ``return`` is accepted through ``**{"return": ...}``
    because ``return`` is a reserved Python keyword. Unknown keyword arguments
    are rejected explicitly.
    """
    r_return = kwargs.pop("return", None)
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")
    if r_return is not None:
        if return_mode is not None:
            raise TypeError("Specify only one of return_mode or the R-compatible 'return' argument")
        return_mode = r_return

    r_mode = (
        any(value is not None for value in (id_col, time_unit, return_mode))
        or x_scale != 1.0
        or y_scale != 1.0
        or keep_single_sample
    )

    if not r_mode:
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

    df = ensure_dataframe(data)
    id_col = "USER_ID" if id_col is None else id_col
    x_col = "FPOGX" if x_col is None else x_col
    y_col = "FPOGY" if y_col is None else y_col
    time_col = "TIME" if time_col is None else time_col
    velocity_threshold = 10.0 if velocity_threshold == 0.08 else float(velocity_threshold)
    min_duration_ms = 50.0 if min_duration_ms == 100.0 else float(min_duration_ms)
    time_unit = "auto" if time_unit is None else str(time_unit)
    return_mode = "events" if return_mode is None else str(return_mode)

    if time_unit not in {"auto", "seconds", "milliseconds"}:
        raise ValueError("time_unit must be 'auto', 'seconds', or 'milliseconds'")
    if return_mode not in {"events", "samples", "both"}:
        raise ValueError("return must be 'events', 'samples', or 'both'")
    if not np.isfinite(velocity_threshold) or velocity_threshold <= 0:
        raise ValueError("vmax must be one finite positive number")
    if not np.isfinite(min_duration_ms) or min_duration_ms < 0:
        raise ValueError("min_duration must be one finite non-negative number")
    if not np.isfinite(x_scale) or x_scale <= 0 or not np.isfinite(y_scale) or y_scale <= 0:
        raise ValueError("x_scale and y_scale must be finite positive numbers")
    if not isinstance(keep_single_sample, (bool, np.bool_)):
        raise ValueError("keep_single_sample must be TRUE or FALSE")

    extra_groups = (
        []
        if group_cols is None
        else ([group_cols] if isinstance(group_cols, str) else list(group_cols))
    )
    groups = list(dict.fromkeys([id_col, *extra_groups]))
    required = list(dict.fromkeys([*groups, x_col, y_col, time_col]))
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("all_gaze is missing required column(s): " + ", ".join(missing))

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
    gaze_velocity = np.full(len(df), np.nan)
    fixation_flag_all = np.zeros(len(df), dtype=bool)
    fixation_id_all = np.full(len(df), np.nan)
    event_rows = []

    grouped = df.groupby(groups, sort=False, dropna=False).indices
    for _, positions in grouped.items():
        positions = np.asarray(positions, dtype=int)
        raw_time = pd.to_numeric(df.iloc[positions][time_col], errors="coerce").to_numpy(
            dtype=float
        )
        order = np.argsort(np.where(np.isfinite(raw_time), raw_time, np.inf), kind="stable")
        pos = positions[order]
        time_sec = seconds(df.iloc[pos][time_col])
        x = pd.to_numeric(df.iloc[pos][x_col], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df.iloc[pos][y_col], errors="coerce").to_numpy(dtype=float)
        dt = np.r_[np.nan, np.diff(time_sec)]
        dx = np.r_[np.nan, np.diff(x) * float(x_scale)]
        dy = np.r_[np.nan, np.diff(y) * float(y_scale)]
        with np.errstate(divide="ignore", invalid="ignore"):
            velocity = np.sqrt(dx**2 + dy**2) / dt
        velocity[~np.isfinite(velocity) | ~np.isfinite(dt) | (dt <= 0)] = np.nan
        if len(velocity) >= 2 and np.isnan(velocity[0]):
            velocity[0] = velocity[1]
        valid = np.isfinite(time_sec) & np.isfinite(x) & np.isfinite(y)
        candidate = valid & np.isfinite(velocity) & (velocity <= velocity_threshold)
        gaze_velocity[pos] = velocity

        if candidate.any():
            starts = np.flatnonzero(candidate & np.r_[True, ~candidate[:-1]])
            ends = np.flatnonzero(candidate & np.r_[~candidate[1:], True])
        else:
            starts = ends = np.array([], dtype=int)
        positive_dt = dt[np.isfinite(dt) & (dt > 0)]
        sample_interval = float(np.median(positive_dt)) if positive_dt.size else 0.0
        local_id = 0
        for start, end in zip(starts, ends, strict=True):
            run = np.arange(start, end + 1)
            duration_sec = max(0.0, float(time_sec[end] - time_sec[start]))
            coverage = (
                duration_sec + sample_interval
                if len(run) > 1 or keep_single_sample
                else duration_sec
            )
            duration_ms = coverage * 1000.0
            if duration_ms + np.sqrt(np.finfo(float).eps) < min_duration_ms:
                continue
            local_id += 1
            run_pos = pos[run]
            fixation_flag_all[run_pos] = True
            fixation_id_all[run_pos] = local_id
            row = {column: df.iloc[run_pos[0]][column] for column in groups}
            vv = velocity[run]
            finite_v = vv[np.isfinite(vv)]
            row.update(
                {
                    "fixation_id": local_id,
                    "start_time": df.iloc[run_pos[0]][time_col],
                    "end_time": df.iloc[run_pos[-1]][time_col],
                    "duration": duration_ms,
                    "duration_ms": duration_ms,
                    "n_samples": len(run),
                    "mean_x": float(np.nanmean(x[run])),
                    "mean_y": float(np.nanmean(y[run])),
                    "median_velocity": float(np.median(finite_v)) if finite_v.size else np.nan,
                    "max_velocity": float(np.max(finite_v)) if finite_v.size else np.nan,
                    "velocity_threshold": float(velocity_threshold),
                    "algorithm": "I-VT",
                }
            )
            event_rows.append(row)

    labelled["gaze_velocity"] = gaze_velocity
    labelled["velocity_fixation"] = fixation_flag_all
    labelled["velocity_fixation_id"] = pd.array(fixation_id_all, dtype="Int64")
    labelled.attrs["_gp3_class"] = "gp3_velocity_fixation_samples"

    columns = [
        *groups,
        "fixation_id",
        "start_time",
        "end_time",
        "duration",
        "duration_ms",
        "n_samples",
        "mean_x",
        "mean_y",
        "median_velocity",
        "max_velocity",
        "velocity_threshold",
        "algorithm",
    ]
    events = pd.DataFrame(event_rows, columns=columns)
    events.attrs["_gp3_class"] = "gp3_velocity_fixations"
    if return_mode == "events":
        return events
    if return_mode == "samples":
        return labelled
    return {"events": events, "samples": labelled, "_gp3_class": "gp3_velocity_fixation_result"}


def detect_gazepoint_fixations_ivt(data, **kwargs) -> pd.DataFrame:
    """Alias for I-VT fixation detection."""

    from ._behavioral_r3a import _dispatch_r3a, _should_use_r3a

    if _should_use_r3a(
        "detect_gazepoint_fixations_ivt",
        locals(),
    ):
        return _dispatch_r3a(
            "detect_gazepoint_fixations_ivt",
            locals(),
        )

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
    data,
    x_col=None,
    y_col=None,
    time_col=None,
    event_col="event_state",
    group_cols=None,
    start_time_col=None,
    end_time_col=None,
    distance_scale=1,
    drop_missing=True,
) -> pd.DataFrame:
    """Compute legacy event summaries or R v2.3.0 fixation-to-fixation saccades."""
    r_mode = (
        start_time_col is not None
        or end_time_col is not None
        or distance_scale != 1
        or drop_missing is not True
    )

    if not r_mode:
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

    frame = ensure_dataframe(data, copy=False)
    if not isinstance(x_col, str) or not x_col or not isinstance(y_col, str) or not y_col:
        raise ValueError("x_col and y_col must be supplied in R-compatible mode")
    groups = (
        []
        if group_cols is None
        else ([group_cols] if isinstance(group_cols, str) else list(group_cols))
    )
    needed = [x_col, y_col, *groups]
    for value in (time_col, start_time_col, end_time_col):
        if value is not None:
            needed.append(value)
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise ValueError("data is missing required column(s): " + ", ".join(missing))
    if not np.isfinite(distance_scale) or float(distance_scale) <= 0:
        raise ValueError("distance_scale must be a positive number")

    blocks = [(None, frame)] if not groups else frame.groupby(groups, dropna=True, sort=True)
    rows = []
    for key, block in blocks:
        if time_col is not None:
            block = block.sort_values(time_col, kind="stable", na_position="last")
        if drop_missing:
            block = block.loc[block[x_col].notna() & block[y_col].notna()]
        if len(block) < 2:
            continue
        x = pd.to_numeric(block[x_col], errors="coerce").to_numpy(float)
        y = pd.to_numeric(block[y_col], errors="coerce").to_numpy(float)
        dx = np.diff(x)
        dy = np.diff(y)
        amp = np.sqrt(dx**2 + dy**2) * float(distance_scale)
        angle_rad = np.arctan2(dy, dx)
        angle_deg = np.degrees(angle_rad)
        time_delta = np.full(len(block) - 1, np.nan)
        time_kind = np.full(len(block) - 1, None, dtype=object)
        if start_time_col is not None and end_time_col is not None:
            start = pd.to_numeric(block[start_time_col], errors="coerce").to_numpy(float)
            end = pd.to_numeric(block[end_time_col], errors="coerce").to_numpy(float)
            time_delta = start[1:] - end[:-1]
            time_kind[:] = "next_start_minus_current_end"
        elif time_col is not None:
            time_values = pd.to_numeric(block[time_col], errors="coerce").to_numpy(float)
            time_delta = np.diff(time_values)
            time_kind[:] = "successive_time_difference"
        speed = np.where(np.isfinite(time_delta) & (time_delta > 0), amp / time_delta, np.nan)
        base = {}
        if groups:
            values = key if isinstance(key, tuple) else (key,)
            base = dict(zip(groups, values, strict=True))
        for i in range(len(block) - 1):
            rows.append(
                {
                    **base,
                    "saccade_index": i + 1,
                    "from_fixation_index": i + 1,
                    "to_fixation_index": i + 2,
                    "from_x": x[i],
                    "from_y": y[i],
                    "to_x": x[i + 1],
                    "to_y": y[i + 1],
                    "dx": dx[i],
                    "dy": dy[i],
                    "saccade_amplitude": amp[i],
                    "saccade_angle_rad": angle_rad[i],
                    "saccade_angle_deg": angle_deg[i],
                    "time_delta": time_delta[i],
                    "time_delta_kind": time_kind[i],
                    "saccade_speed": speed[i],
                    "n_fixations": len(block),
                    "saccade_status": "ok",
                }
            )
    return pd.DataFrame(rows)


def summarise_fixations(
    data,
    fixation_col="fixation",
    fixation_id_col="fixation_id",
    x_col=None,
    y_col=None,
    time_col=None,
    group_cols=None,
    aoi_col=None,
) -> pd.DataFrame:
    """Summarise fixation exports or collapse fixation-classified samples."""
    df = ensure_dataframe(data)
    r_mode = {"FPOGD", "FPOGS"}.issubset(df.columns)

    if r_mode:
        resolved_aoi = "AOI" if aoi_col is None else aoi_col
        groups = (
            ["MEDIA_ID"]
            if group_cols is None
            else ([group_cols] if isinstance(group_cols, str) else list(group_cols))
        )
        needed = [*groups, resolved_aoi, "FPOGD", "FPOGS"]
        missing = [column for column in needed if column not in df.columns]
        if missing:
            raise ValueError("Missing columns: " + ", ".join(missing))
        work = df.loc[df[resolved_aoi].notna() & df[resolved_aoi].astype(str).ne("")].copy()
        work["_FPOGD"] = pd.to_numeric(work["FPOGD"], errors="coerce")
        work["_FPOGS"] = pd.to_numeric(work["FPOGS"], errors="coerce")
        rows = []
        keys = [*groups, resolved_aoi]
        for key, block in work.groupby(keys, dropna=False, sort=True):
            values = key if isinstance(key, tuple) else (key,)
            row = dict(zip(keys, values, strict=True))
            row.update(
                {
                    "fixation_count": int(len(block)),
                    "fixation_duration_sum_sec": float(block["_FPOGD"].sum(skipna=True)),
                    "fixation_duration_mean_ms": float(block["_FPOGD"].mean(skipna=True) * 1000),
                    "fixation_ttff_sec": float(block["_FPOGS"].min(skipna=True)),
                }
            )
            rows.append(row)
        return pd.DataFrame(
            rows,
            columns=[
                *keys,
                "fixation_count",
                "fixation_duration_sum_sec",
                "fixation_duration_mean_ms",
                "fixation_ttff_sec",
            ],
        )

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
    data,
    trial_col=None,
    subject_col=None,
    duration_col=None,
    *,
    group_cols=None,
    fixation_id_col=None,
    start_col=None,
    x_col=None,
    y_col=None,
    valid_col=None,
    aoi_col=None,
    start_time_unit="auto",
    duration_unit="auto",
    valid_only=True,
    include_non_aoi=True,
    target_aoi_values=None,
    distractor_aoi_values=None,
    non_aoi_values=(
        "non_aoi",
        "none",
        "background",
        "outside",
        "outside_aoi",
        "missing",
        "missing_aoi",
    ),
    missing_aoi_label="missing_aoi",
) -> pd.DataFrame:
    """Summarise fixation-level data with legacy or R v2.3.0 trial features."""
    df = ensure_dataframe(data)
    r_mode = (
        any(
            value is not None
            for value in (
                group_cols,
                fixation_id_col,
                start_col,
                x_col,
                y_col,
                valid_col,
                aoi_col,
                target_aoi_values,
                distractor_aoi_values,
            )
        )
        or start_time_unit != "auto"
        or duration_unit != "auto"
        or valid_only is not True
        or include_non_aoi is not True
    )

    if not r_mode and (trial_col is not None or subject_col is not None):
        duration_col = duration_col or "duration_ms"
        trial_col = trial_col or infer_column(df, "trial")
        subject_col = subject_col or infer_column(df, "subject")
        keys = [c for c in [subject_col, trial_col] if c]
        if duration_col not in df.columns:
            raise KeyError(f"Missing required column: {duration_col}")
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

    if start_time_unit not in {"auto", "ms", "s"}:
        raise ValueError("start_time_unit must be 'auto', 'ms', or 's'")
    if duration_unit not in {"auto", "ms", "s"}:
        raise ValueError("duration_unit must be 'auto', 'ms', or 's'")
    if not isinstance(valid_only, (bool, np.bool_)) or not isinstance(
        include_non_aoi, (bool, np.bool_)
    ):
        raise ValueError("valid_only and include_non_aoi must be TRUE or FALSE")

    def first_existing(candidates, informative=False):
        hits = [column for column in candidates if column in df.columns]
        if not informative:
            return hits[0] if hits else None
        for column in hits:
            values = df[column].astype("string").str.strip().dropna()
            values = values.loc[values.ne("")]
            if values.nunique() > 0:
                return column
        return hits[0] if hits else None

    if group_cols is None:
        detected = [
            first_existing(
                [
                    "subject",
                    "USER_ID",
                    "USER_FILE",
                    "USER",
                    "user",
                    "participant",
                    "participant_id",
                ],
                True,
            ),
            first_existing(["MEDIA_ID", "media_id", "MEDIA_NAME", "media_name", "stimulus"], True),
            first_existing(["trial_global", "trial", "trial_id", "TRIAL"], True),
        ]
        groups = list(dict.fromkeys([value for value in detected if value is not None]))
        if not groups:
            if trial_col is not None or subject_col is not None:
                groups = [value for value in (subject_col, trial_col) if value is not None]
            else:
                raise ValueError(
                    "Could not automatically detect grouping columns. Please provide group_cols"
                )
    else:
        groups = [group_cols] if isinstance(group_cols, str) else list(group_cols)
        if len(groups) != len(set(groups)) or not groups:
            raise ValueError("group_cols must be NULL or a vector of unique column names")
    missing = [column for column in groups if column not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))

    fixation_id_col = fixation_id_col or first_existing(
        ["FPOGID", "fixation_id", "fixationID", "fix_id", "id"]
    )
    start_col = start_col or first_existing(
        [
            "FPOGS",
            "fixation_start_time",
            "fixation_start_ms",
            "start_time",
            "start_time_ms",
            "time",
            "TIME",
            "TIMETICK",
        ]
    )
    duration_col = duration_col or first_existing(
        [
            "FPOGD",
            "fixation_duration_ms",
            "fixation_duration",
            "duration_ms",
            "duration",
            "FPOGD_MS",
        ]
    )
    x_col = x_col or first_existing(["FPOGX", "fixation_x", "x", "X", "gaze_x"])
    y_col = y_col or first_existing(["FPOGY", "fixation_y", "y", "Y", "gaze_y"])
    valid_col = valid_col or first_existing(["FPOGV", "fixation_valid", "valid", "VALID"])
    aoi_col = aoi_col or first_existing(["AOI", "aoi_current", "aoi_state", "aoi"])
    if start_col is None or duration_col is None:
        missing_names = [
            name
            for name, value in (("start_col", start_col), ("duration_col", duration_col))
            if value is None
        ]
        raise ValueError(
            "Could not automatically detect required fixation columns: " + ", ".join(missing_names)
        )
    for column in [start_col, duration_col, fixation_id_col, x_col, y_col, valid_col, aoi_col]:
        if column is not None and column not in df.columns:
            raise ValueError(f"Missing required column: {column}")

    def to_ms(series, column, unit, role):
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        if unit == "s":
            return values * 1000.0
        if unit == "ms":
            return values
        name = str(column).lower()
        seconds = name in {"fpogs", "fpogd"}
        finite = values[np.isfinite(values)]
        if not seconds and (name == "time" or role == "duration") and finite.size:
            seconds = float(np.max(finite)) <= 60.0
        return values * 1000.0 if seconds else values

    work = df.copy()
    work[".gp3_start_time_ms"] = to_ms(work[start_col], start_col, start_time_unit, "start")
    work[".gp3_duration_ms"] = to_ms(work[duration_col], duration_col, duration_unit, "duration")
    work[".gp3_x"] = pd.to_numeric(work[x_col], errors="coerce") if x_col else np.nan
    work[".gp3_y"] = pd.to_numeric(work[y_col], errors="coerce") if y_col else np.nan

    if valid_col is None:
        work[".gp3_valid"] = True
    else:
        raw_valid = work[valid_col]
        numeric = pd.to_numeric(raw_valid, errors="coerce")
        text = raw_valid.astype("string").str.strip().str.lower()
        valid = pd.Series(False, index=work.index)
        valid.loc[numeric.notna()] = numeric.loc[numeric.notna()].ne(0)
        valid.loc[text.isin(["true", "valid", "yes", "y"])] = True
        work[".gp3_valid"] = valid

    aoi_available = aoi_col is not None
    if aoi_available:
        state = work[aoi_col].astype("string").str.strip()
        state = state.mask(state.isna() | state.eq(""), missing_aoi_label)
        work[".gp3_aoi_state"] = state.astype(object)
    else:
        work[".gp3_aoi_state"] = None
    background = {str(value).strip().lower() for value in non_aoi_values}
    work[".gp3_is_non_aoi"] = (
        work[".gp3_aoi_state"].astype("string").str.strip().str.lower().isin(background)
        if aoi_available
        else False
    )
    targets = (
        set()
        if target_aoi_values is None
        else {str(value).strip().lower() for value in target_aoi_values}
    )
    distractors = (
        set()
        if distractor_aoi_values is None
        else {str(value).strip().lower() for value in distractor_aoi_values}
    )
    target_defined = bool(targets)
    distractor_defined = bool(distractors)
    state_norm = work[".gp3_aoi_state"].astype("string").str.strip().str.lower()
    state_class = np.full(
        len(work), "no_aoi_column" if not aoi_available else "other_aoi", dtype=object
    )
    if aoi_available:
        state_class[work[".gp3_is_non_aoi"].to_numpy(bool)] = "background"
        if targets:
            state_class[state_norm.isin(targets).to_numpy()] = "target"
        if distractors:
            state_class[state_norm.isin(distractors).to_numpy()] = "distractor"
    work[".gp3_state_class"] = state_class
    work[".gp3_fixation_id"] = (
        work[fixation_id_col].astype("string")
        if fixation_id_col
        else pd.Series(np.arange(1, len(work) + 1), index=work.index).astype("string")
    )
    work = work.loc[np.isfinite(work[".gp3_start_time_ms"])].copy()
    if valid_only:
        work = work.loc[work[".gp3_valid"]].copy()
    if not include_non_aoi and aoi_available:
        work = work.loc[~work[".gp3_is_non_aoi"]].copy()
    if work.empty:
        raise ValueError("No fixation rows remain after filtering")

    work = work.sort_values([*groups, ".gp3_start_time_ms"], kind="stable")
    fix_rows = []
    for key, block in work.groupby([*groups, ".gp3_fixation_id"], dropna=False, sort=False):
        values = key if isinstance(key, tuple) else (key,)
        row = dict(zip([*groups, ".gp3_fixation_id"], values, strict=True))
        first = block.iloc[0]
        start = float(first[".gp3_start_time_ms"])
        duration = (
            float(first[".gp3_duration_ms"]) if np.isfinite(first[".gp3_duration_ms"]) else np.nan
        )
        row.update(
            {
                "fixation_start_time_ms": start,
                "fixation_duration_ms": duration,
                "fixation_end_time_ms": start + duration if np.isfinite(duration) else np.nan,
                "fixation_x": first[".gp3_x"],
                "fixation_y": first[".gp3_y"],
                "fixation_valid": bool(first[".gp3_valid"]),
                "fixation_aoi": first[".gp3_aoi_state"],
                "is_non_aoi": bool(first[".gp3_is_non_aoi"]),
                "state_class": first[".gp3_state_class"],
                "n_rows_per_fixation": len(block),
            }
        )
        fix_rows.append(row)
    fix = pd.DataFrame(fix_rows)
    if fix.empty:
        raise ValueError("No fixation rows remain after fixation-level reduction")

    def ratio(a, b):
        return np.nan if not np.isfinite(b) or b <= 0 else float(a) / float(b)

    rows = []
    for key, block in fix.groupby(groups, dropna=False, sort=False):
        values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(groups, values, strict=True))
        duration = pd.to_numeric(block["fixation_duration_ms"], errors="coerce")
        start = pd.to_numeric(block["fixation_start_time_ms"], errors="coerce")
        end = pd.to_numeric(block["fixation_end_time_ms"], errors="coerce")
        x = pd.to_numeric(block["fixation_x"], errors="coerce")
        y = pd.to_numeric(block["fixation_y"], errors="coerce")
        aoi_mask = (
            ~block["is_non_aoi"].astype(bool)
            if aoi_available
            else pd.Series(False, index=block.index)
        )
        target_mask = block["state_class"].eq("target")
        distractor_mask = block["state_class"].eq("distractor")
        other_mask = block["state_class"].eq("other_aoi")
        trial_start = float(start.min())
        trial_end = float(end.max())
        n_aoi = int(aoi_mask.sum())
        n_fix = len(block)
        target_n = int(target_mask.sum())
        distractor_n = int(distractor_mask.sum())
        target_duration = float(duration.loc[target_mask].sum(skipna=True))
        distractor_duration = float(duration.loc[distractor_mask].sum(skipna=True))
        other_duration = float(duration.loc[other_mask].sum(skipna=True))
        aoi_states = (
            block.loc[aoi_mask]
            .sort_values("fixation_start_time_ms", kind="stable")["fixation_aoi"]
            .astype(str)
        )
        row.update(
            {
                "trial_start_time_ms": trial_start,
                "trial_end_time_ms": trial_end,
                "n_fixations": n_fix,
                "n_valid_fixations": int(block["fixation_valid"].astype(bool).sum()),
                "n_rows_represented": int(block["n_rows_per_fixation"].sum()),
                "total_fixation_duration_ms": float(duration.sum(skipna=True)),
                "mean_fixation_duration_ms": float(duration.mean())
                if duration.notna().any()
                else np.nan,
                "median_fixation_duration_ms": float(duration.median())
                if duration.notna().any()
                else np.nan,
                "min_fixation_duration_ms": float(duration.min())
                if duration.notna().any()
                else np.nan,
                "max_fixation_duration_ms": float(duration.max())
                if duration.notna().any()
                else np.nan,
                "mean_fixation_x": float(x.mean()) if x.notna().any() else np.nan,
                "mean_fixation_y": float(y.mean()) if y.notna().any() else np.nan,
                "sd_fixation_x": float(x.std(ddof=1)) if x.notna().sum() >= 2 else np.nan,
                "sd_fixation_y": float(y.std(ddof=1)) if y.notna().sum() >= 2 else np.nan,
                "min_fixation_x": float(x.min()) if x.notna().any() else np.nan,
                "max_fixation_x": float(x.max()) if x.notna().any() else np.nan,
                "min_fixation_y": float(y.min()) if y.notna().any() else np.nan,
                "max_fixation_y": float(y.max()) if y.notna().any() else np.nan,
                "n_aoi_fixations": n_aoi,
                "n_non_aoi_fixations": int(block["is_non_aoi"].astype(bool).sum())
                if aoi_available
                else 0,
                "n_unique_aoi_fixated": int(
                    block.loc[aoi_mask, "fixation_aoi"].astype(str).nunique()
                )
                if aoi_available
                else 0,
                "first_aoi_fixated": aoi_states.iloc[0] if len(aoi_states) else pd.NA,
                "last_aoi_fixated": aoi_states.iloc[-1] if len(aoi_states) else pd.NA,
                "first_aoi_fixation_time_ms": float(start.loc[aoi_mask].min())
                if aoi_mask.any()
                else np.nan,
                "target_fixation_count": target_n,
                "target_revisits": max(target_n - 1, 0),
                "target_fixation_duration_ms": target_duration,
                "target_ttff_ms": float(start.loc[target_mask].min())
                if target_mask.any()
                else np.nan,
                "mean_target_fixation_duration_ms": float(duration.loc[target_mask].mean())
                if target_mask.any()
                else np.nan,
                "distractor_fixation_count": distractor_n,
                "distractor_revisits": max(distractor_n - 1, 0),
                "distractor_fixation_duration_ms": distractor_duration,
                "distractor_ttff_ms": float(start.loc[distractor_mask].min())
                if distractor_mask.any()
                else np.nan,
                "mean_distractor_fixation_duration_ms": float(duration.loc[distractor_mask].mean())
                if distractor_mask.any()
                else np.nan,
                "other_aoi_fixation_count": int(other_mask.sum()),
                "other_aoi_fixation_duration_ms": other_duration,
            }
        )
        trial_duration = trial_end - trial_start
        row.update(
            {
                "trial_duration_ms": trial_duration,
                "fixation_rate_per_sec": ratio(n_fix, trial_duration / 1000.0),
                "fixation_duration_prop": ratio(row["total_fixation_duration_ms"], trial_duration),
                "aoi_fixation_prop": ratio(n_aoi, n_fix),
                "non_aoi_fixation_prop": ratio(row["n_non_aoi_fixations"], n_fix),
                "target_fixation_prop_of_aoi": ratio(target_n, n_aoi),
                "distractor_fixation_prop_of_aoi": ratio(distractor_n, n_aoi),
                "target_duration_prop_of_aoi": ratio(
                    target_duration, target_duration + distractor_duration + other_duration
                ),
                "distractor_duration_prop_of_aoi": ratio(
                    distractor_duration, target_duration + distractor_duration + other_duration
                ),
                "aoi_available": aoi_available,
                "target_aoi_defined": target_defined,
                "distractor_aoi_defined": distractor_defined,
            }
        )
        if not aoi_available:
            status = "no_aoi_column"
        elif n_aoi == 0:
            status = "no_aoi_fixations"
        elif not target_defined and not distractor_defined:
            status = "no_target_or_distractor_defined"
        elif target_defined and target_n == 0:
            status = "target_not_observed"
        elif distractor_defined and distractor_n == 0:
            status = "distractor_not_observed"
        else:
            status = "ok"
        row["fixation_trial_feature_status"] = status
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["_gp3_class"] = "gp3_fixation_trial_features"
    return out


def audit_gazepoint_fixation_reliability(
    data,
    subject_col=None,
    duration_col=None,
    *,
    trial_col=None,
    metric="fixation_count",
    aoi_col=None,
    target_aoi=None,
    time_col=None,
    group_cols=None,
    min_trials=4,
    split_method="odd_even",
    seed=None,
    correlation_method="pearson",
) -> pd.DataFrame:
    """Audit split-half fixation reliability.

    Supplying ``trial_col`` activates the R gp3tools v2.3.0 split-half
    reliability procedure. If ``trial_col`` is omitted, the historical
    Python per-subject duration summary is retained.
    """
    df = ensure_dataframe(data)

    # ------------------------------------------------------------
    # Historical Python behaviour
    # ------------------------------------------------------------
    if trial_col is None:
        legacy_duration_col = duration_col or "duration_ms"
        legacy_subject_col = subject_col or infer_column(df, "subject")

        if not legacy_subject_col:
            duration = pd.to_numeric(
                df.get(legacy_duration_col),
                errors="coerce",
            )

            return pd.DataFrame(
                {
                    "n": [len(df)],
                    "mean_duration_ms": [duration.mean()],
                }
            )

        return (
            df.groupby(
                legacy_subject_col,
                dropna=False,
            )[legacy_duration_col]
            .agg(
                n="size",
                mean_duration_ms="mean",
                sd_duration_ms="std",
            )
            .reset_index()
        )

    # ------------------------------------------------------------
    # R v2.3.0 behaviour
    # ------------------------------------------------------------
    def check_scalar_string(
        value,
        argument,
        *,
        allow_none=False,
    ):
        if value is None and allow_none:
            return

        if not isinstance(value, str) or not value:
            raise ValueError(f"{argument} must be a non-empty string")

    check_scalar_string(
        subject_col,
        "subject_col",
    )
    check_scalar_string(
        trial_col,
        "trial_col",
    )
    check_scalar_string(
        duration_col,
        "duration_col",
        allow_none=True,
    )
    check_scalar_string(
        aoi_col,
        "aoi_col",
        allow_none=True,
    )
    check_scalar_string(
        target_aoi,
        "target_aoi",
        allow_none=True,
    )
    check_scalar_string(
        time_col,
        "time_col",
        allow_none=True,
    )

    valid_metrics = {
        "fixation_count",
        "mean_fixation_duration",
        "total_fixation_duration",
        "aoi_dwell_prop",
        "transition_count",
        "entropy_score",
    }

    if metric not in valid_metrics:
        raise ValueError("metric must be one of: " + ", ".join(sorted(valid_metrics)))

    if split_method not in {
        "odd_even",
        "random",
    }:
        raise ValueError("split_method must be 'odd_even' or 'random'")

    if correlation_method not in {
        "pearson",
        "spearman",
    }:
        raise ValueError("correlation_method must be 'pearson' or 'spearman'")

    if (
        isinstance(min_trials, bool)
        or not isinstance(
            min_trials,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(min_trials)
        or min_trials < 2
    ):
        raise ValueError("min_trials must be a finite numeric scalar of at least 2")

    min_trials = int(min_trials)

    groups = normalize_group_cols(
        df,
        group_cols,
    )

    required_cols = [
        subject_col,
        trial_col,
        *groups,
    ]

    if metric in {
        "mean_fixation_duration",
        "total_fixation_duration",
    }:
        if duration_col is None:
            raise ValueError("duration_col is required for duration reliability metrics")

        required_cols.append(duration_col)

    if metric in {
        "aoi_dwell_prop",
        "transition_count",
        "entropy_score",
    }:
        if aoi_col is None:
            raise ValueError("aoi_col is required for AOI reliability metrics")

        required_cols.append(aoi_col)

        if time_col is not None:
            required_cols.append(time_col)

    if metric == "aoi_dwell_prop" and target_aoi is None:
        raise ValueError("target_aoi is required when metric='aoi_dwell_prop'")

    if metric == "aoi_dwell_prop" and duration_col is not None:
        required_cols.append(duration_col)

    missing = [col for col in dict.fromkeys(required_cols) if col not in df.columns]

    if missing:
        raise ValueError("data is missing required column(s): " + ", ".join(missing))

    def ordered(part):
        if time_col is None:
            return part

        return part.sort_values(
            time_col,
            kind="stable",
            na_position="last",
        )

    def prepared_aoi(part):
        values = part[aoi_col].astype("string")

        valid = values.notna() & values.str.strip().ne("")

        return values.loc[valid].astype(str).tolist()

    def collapsed(values):
        if len(values) <= 1:
            return values

        out = [values[0]]

        for value in values[1:]:
            if value != out[-1]:
                out.append(value)

        return out

    def entropy_score(values):
        if not values:
            return np.nan

        counts = pd.Series(values).value_counts().to_numpy(float)

        counts = counts[np.isfinite(counts) & (counts > 0)]

        if not len(counts):
            return np.nan

        probabilities = counts / counts.sum()

        entropy = float(-np.sum(probabilities * np.log2(probabilities)))

        n_categories = len(counts)

        if n_categories <= 1:
            return 0.0

        maximum = np.log2(n_categories)

        if not np.isfinite(maximum) or maximum <= 0:
            return np.nan

        return float(entropy / maximum)

    def trial_metric(part):
        if metric == "fixation_count":
            return float(len(part))

        if metric in {
            "mean_fixation_duration",
            "total_fixation_duration",
        }:
            values = pd.to_numeric(
                part[duration_col],
                errors="coerce",
            ).to_numpy(float)

            values = values[np.isfinite(values)]

            if not len(values):
                return np.nan

            if metric == "mean_fixation_duration":
                return float(values.mean())

            return float(values.sum())

        if metric == "aoi_dwell_prop":
            aoi = part[aoi_col].astype("string")

            valid_aoi = aoi.notna() & aoi.str.strip().ne("")

            if not bool(valid_aoi.any()):
                return np.nan

            if duration_col is not None:
                duration = pd.to_numeric(
                    part[duration_col],
                    errors="coerce",
                ).to_numpy(float)

                valid = valid_aoi.to_numpy() & np.isfinite(duration) & (duration >= 0)

                if not valid.any() or duration[valid].sum() == 0:
                    return np.nan

                target = aoi.astype("string").eq(str(target_aoi)).fillna(False).to_numpy()

                return float(duration[valid & target].sum() / duration[valid].sum())

            target = aoi.loc[valid_aoi].astype(str).eq(str(target_aoi))

            return float(target.mean())

        part = ordered(part)
        aoi = prepared_aoi(part)

        if metric == "transition_count":
            aoi = collapsed(aoi)
            return float(max(len(aoi) - 1, 0))

        return entropy_score(aoi)

    trial_group_cols = [
        subject_col,
        trial_col,
        *groups,
    ]

    trial_rows = []

    for _, part in df.groupby(
        trial_group_cols,
        dropna=False,
        sort=False,
    ):
        row = {col: part.iloc[0][col] for col in groups}

        row.update(
            {
                ".gp3_subject": str(part.iloc[0][subject_col]),
                ".gp3_trial": str(part.iloc[0][trial_col]),
                ".gp3_metric_value": trial_metric(part),
            }
        )

        trial_rows.append(row)

    if not trial_rows:
        return pd.DataFrame(
            {
                "metric": [metric],
                "split_method": [split_method],
                "correlation_method": [correlation_method],
                "split_half_r": [np.nan],
                "spearman_brown": [np.nan],
                "n_subjects_total": [0],
                "n_subjects_used": [0],
                "n_trials": [0],
                "min_trials": [min_trials],
                "reliability_status": ["no_trials"],
                "reliability_warning": ["No trial-level metrics could be computed."],
            }
        )

    trial_summary = pd.DataFrame(trial_rows)

    rng = np.random.default_rng(seed)

    split_rows = []

    split_group_cols = [
        *groups,
        ".gp3_subject",
    ]

    for _, part in trial_summary.groupby(
        split_group_cols,
        dropna=False,
        sort=False,
    ):
        part = part.copy()

        if split_method == "random":
            order = rng.permutation(len(part))

            part = part.iloc[order].copy()
        else:
            part = (
                part.assign(_gp3_trial_sort=part[".gp3_trial"].astype("string"))
                .sort_values(
                    "_gp3_trial_sort",
                    kind="stable",
                    na_position="last",
                )
                .drop(columns="_gp3_trial_sort")
            )

        part[".gp3_half"] = ["odd" if index % 2 == 0 else "even" for index in range(len(part))]

        split_rows.append(part)

    trial_summary = pd.concat(
        split_rows,
        ignore_index=True,
    )

    if groups:
        reliability_iterator = trial_summary.groupby(
            groups,
            dropna=False,
            sort=False,
        )
    else:
        reliability_iterator = [
            (
                None,
                trial_summary,
            )
        ]

    reliability_rows = []

    for _, part in reliability_iterator:
        group_values = {col: part.iloc[0][col] for col in groups}

        subject_rows = []

        for subject, subpart in part.groupby(
            ".gp3_subject",
            dropna=False,
            sort=False,
        ):
            values = pd.to_numeric(
                subpart[".gp3_metric_value"],
                errors="coerce",
            )

            odd_values = values.loc[subpart[".gp3_half"].eq("odd")]

            even_values = values.loc[subpart[".gp3_half"].eq("even")]

            odd_values = odd_values[np.isfinite(odd_values)]

            even_values = even_values[np.isfinite(even_values)]

            odd = float(odd_values.mean()) if len(odd_values) else np.nan

            even = float(even_values.mean()) if len(even_values) else np.nan

            subject_rows.append(
                {
                    ".gp3_subject": subject,
                    "odd": odd,
                    "even": even,
                    "n_trials": int(len(subpart)),
                }
            )

        subject_summary = pd.DataFrame(subject_rows)

        eligible = subject_summary.loc[
            (subject_summary["n_trials"] >= min_trials)
            & np.isfinite(subject_summary["odd"])
            & np.isfinite(subject_summary["even"])
        ].copy()

        n_subjects_total = int(len(subject_summary))

        n_subjects_used = int(len(eligible))

        n_trials_used = int(eligible["n_trials"].sum())

        if n_subjects_used < 3:
            split_half_r = np.nan
            spearman_brown = np.nan
            status = "too_few_subjects"
            warning = "Fewer than three subjects had enough complete split-half data."

        else:
            odd_sd = float(eligible["odd"].std(ddof=1))

            even_sd = float(eligible["even"].std(ddof=1))

            if odd_sd == 0 or even_sd == 0:
                split_half_r = np.nan
                spearman_brown = np.nan
                status = "no_variance"
                warning = "At least one split had zero between-subject variance."

            else:
                split_half_r = float(
                    eligible["odd"].corr(
                        eligible["even"],
                        method=correlation_method,
                    )
                )

                if np.isfinite(split_half_r) and split_half_r != -1:
                    spearman_brown = float((2 * split_half_r) / (1 + split_half_r))
                else:
                    spearman_brown = np.nan

                status = "ok"
                warning = ""

        reliability_rows.append(
            {
                **group_values,
                "metric": metric,
                "split_method": split_method,
                "correlation_method": correlation_method,
                "split_half_r": split_half_r,
                "spearman_brown": spearman_brown,
                "n_subjects_total": n_subjects_total,
                "n_subjects_used": n_subjects_used,
                "n_trials": n_trials_used,
                "min_trials": min_trials,
                "reliability_status": status,
                "reliability_warning": warning,
            }
        )

    return pd.DataFrame(reliability_rows)


def compare_gazepoint_event_detectors(
    data,
    id_col="USER_ID",
    trial_col=None,
    group_cols=None,
    x_col="FPOGX",
    y_col="FPOGY",
    time_col="TIME",
    methods=None,
    velocity_thresholds=(5, 10, 20),
    min_duration=50,
    hmm_states=3,
    eyetools_method="vti",
    run_optional_eyetools=False,
    min_overlap=0.5,
    velocity_args=None,
    hmm_args=None,
    eyetools_args=None,
    **kwargs,
):
    if methods is None:
        # Historical Python sample-wise detector agreement.
        a = detect_gazepoint_fixations_velocity(data, **kwargs)
        b = classify_gazepoint_events_hmm(
            data,
            **{k: v for k, v in kwargs.items() if k in {"x_col", "y_col", "time_col"}},
        )
        out = pd.DataFrame(index=a.index)
        out["velocity_detector"] = np.where(a["fixation"], "fixation", "saccade")
        out["state_detector"] = b["event_state"].to_numpy()
        out["agreement"] = out["velocity_detector"].eq(out["state_detector"])
        return out.reset_index(drop=True)

    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unknown}")

    from ._behavioral_r2 import compare_event_detectors

    return compare_event_detectors(
        data,
        id_col=id_col,
        trial_col=trial_col,
        group_cols=group_cols,
        x_col=x_col,
        y_col=y_col,
        time_col=time_col,
        methods=methods,
        velocity_thresholds=velocity_thresholds,
        min_duration=min_duration,
        hmm_states=hmm_states,
        eyetools_method=eyetools_method,
        run_optional_eyetools=run_optional_eyetools,
        min_overlap=min_overlap,
        velocity_args=velocity_args,
        hmm_args=hmm_args,
        eyetools_args=eyetools_args,
    )


def summarise_gazepoint_event_detector_agreement(data, **kwargs) -> pd.DataFrame:

    from ._behavioral_r3a import _dispatch_r3a, _should_use_r3a

    if _should_use_r3a(
        "summarise_gazepoint_event_detector_agreement",
        locals(),
    ):
        return _dispatch_r3a(
            "summarise_gazepoint_event_detector_agreement",
            locals(),
        )

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


def summarise_gazepoint_event_detector_benchmark(
    data=None,
    *,
    x=None,
    level="detector",
    sort=True,
) -> pd.DataFrame:
    """Summarise an event-detector benchmark.

    Dict-like benchmark objects use the R v2.3.0 level-selection semantics.
    Plain DataFrames retain the original Python timing-summary behaviour.
    """
    if x is not None:
        if data is not None:
            raise TypeError("supply either data or x, not both")
        data = x

    if isinstance(data, dict) and any(
        key in data
        for key in (
            "detector_metrics",
            "sequence_metrics",
            "matches",
            "errors",
        )
    ):
        valid_levels = {
            "detector": "detector_metrics",
            "sequence": "sequence_metrics",
            "matches": "matches",
            "errors": "errors",
        }

        if level not in valid_levels:
            raise ValueError("level must be one of: detector, sequence, matches, errors")

        key = valid_levels[level]

        if key not in data:
            raise ValueError(f"benchmark object does not contain {key!r}")

        out = ensure_dataframe(
            data[key],
            copy=False,
        ).copy()

        if level == "detector" and sort and len(out) and "f1" in out.columns:
            order_value = pd.to_numeric(
                out["f1"],
                errors="coerce",
            ).fillna(-np.inf)

            out = (
                out.assign(_gp3_order=order_value)
                .sort_values(
                    ["_gp3_order", "detector"],
                    ascending=[False, True],
                    kind="stable",
                )
                .drop(columns="_gp3_order")
                .reset_index(drop=True)
            )

        return out

    df = ensure_dataframe(
        data,
        copy=False,
    )

    return (
        df.groupby(
            "detector",
            dropna=False,
        )["elapsed_seconds"]
        .agg(
            n="size",
            mean_seconds="mean",
            median_seconds="median",
            max_seconds="max",
        )
        .reset_index()
    )


def create_gazepoint_event_review_template(
    data,
    id_col="USER_ID",
    trial_col=None,
    group_cols=None,
    time_col="TIME",
    rows_per_sequence=1,
    event_type="fixation",
    reviewer=None,
    path=None,
    **kwargs,
):
    """Create a sequence-level manual event-review template."""
    import numpy as np
    import pandas as pd

    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a data frame.")

    # Legacy event-state review path for pre-parity Python callers whose data do
    # not use the R default USER_ID/TIME columns.
    if (id_col == "USER_ID" and id_col not in data.columns) or (
        time_col == "TIME" and time_col not in data.columns
    ):
        legacy = data.copy()
        if "event_state" not in legacy.columns:
            legacy = classify_gazepoint_events_hmm(legacy, **kwargs)
        elif kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")
        legacy["review_state"] = legacy["event_state"]
        legacy["reviewer_note"] = ""
        legacy["reviewed"] = False
        if path is not None:
            target = Path(path)
            target.parent.mkdir(parents=True, exist_ok=True)
            legacy.to_csv(target, index=False)
        return legacy

    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unknown}")
    if len(data) == 0:
        raise ValueError("`data` must contain at least one sample.")

    extra = (
        []
        if group_cols is None
        else ([group_cols] if isinstance(group_cols, str) else list(group_cols))
    )
    sequence_cols = []
    for c in [id_col, trial_col, *extra]:
        if c is not None and not pd.isna(c) and str(c) and c not in sequence_cols:
            sequence_cols.append(c)
    required = list(dict.fromkeys(sequence_cols + [time_col]))
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"`data` is missing required column(s): {', '.join(missing)}.")
    if (
        isinstance(rows_per_sequence, (bool, np.bool_))
        or not isinstance(rows_per_sequence, (int, float, np.integer, np.floating))
        or not np.isfinite(rows_per_sequence)
        or rows_per_sequence < 1
        or int(rows_per_sequence) != rows_per_sequence
    ):
        raise ValueError("`rows_per_sequence` must be one positive integer.")
    rows_per_sequence = int(rows_per_sequence)
    if not isinstance(event_type, str) or not event_type:
        raise ValueError("`event_type` must be one non-empty character value.")

    time_values = pd.to_numeric(data[time_col], errors="coerce")
    if not np.isfinite(time_values.to_numpy(float)).any():
        raise ValueError(f"`{time_col}` must contain at least one finite timestamp.")

    if sequence_cols:
        key_frame = data[sequence_cols].astype("string").fillna("<NA>")
        keys = key_frame.agg("\r".join, axis=1)
    else:
        keys = pd.Series([".all"] * len(data), index=data.index)

    rows = []
    for key in sorted(pd.unique(keys)):
        idx = data.index[keys.eq(key)]
        finite = pd.to_numeric(data.loc[idx, time_col], errors="coerce")
        finite = finite[np.isfinite(finite)]
        if finite.empty:
            continue
        base = {c: data.loc[idx[0], c] for c in sequence_cols}
        for review_event_id in range(1, rows_per_sequence + 1):
            rows.append(
                {
                    **base,
                    "review_event_id": review_event_id,
                    "sequence_start": float(finite.min()),
                    "sequence_end": float(finite.max()),
                    "start_time": np.nan,
                    "end_time": np.nan,
                    "event_type": event_type,
                    "review_status": "pending",
                    "reviewer": pd.NA if reviewer is None or pd.isna(reviewer) else str(reviewer),
                    "notes": pd.NA,
                }
            )
    if not rows:
        raise ValueError("No sequence contained a finite timestamp.")
    out = pd.DataFrame(rows)
    if path is not None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(target, index=False)
    return out


def simulate_gazepoint_fixations(
    n_fixations: int = 30,
    samples_per_fixation: int = 8,
    random_state: int = 123,
    *,
    n_subjects=None,
    n_fix=None,
    sd=None,
    coordinate_system=None,
    screen_width=None,
    screen_height=None,
    duration_mean=None,
    duration_sd=None,
    saccade_gap_mean=None,
    seed=None,
) -> pd.DataFrame:
    """Simulate fixation data using legacy sample-level or R v2.3.0 fixation-level output."""
    r_mode = any(
        value is not None
        for value in (
            n_subjects,
            n_fix,
            sd,
            coordinate_system,
            screen_width,
            screen_height,
            duration_mean,
            duration_sd,
            saccade_gap_mean,
            seed,
        )
    )
    if not r_mode:
        rng = np.random.default_rng(random_state)
        rows = []
        time = 0.0
        for fixation_id in range(1, n_fixations + 1):
            center_x, center_y = rng.uniform(0.1, 0.9, 2)
            for _ in range(samples_per_fixation):
                rows.append(
                    {
                        "TIME": time,
                        "FPOGX": center_x + rng.normal(0, 0.005),
                        "FPOGY": center_y + rng.normal(0, 0.005),
                        "fixation": True,
                        "fixation_id": fixation_id,
                    }
                )
                time += 1 / 60
            time += rng.uniform(0.02, 0.08)
        return pd.DataFrame(rows)

    n_subjects = 10 if n_subjects is None else n_subjects
    n_fix = 50 if n_fix is None else n_fix
    sd = 10 if sd is None else sd
    coordinate_system = "pixels" if coordinate_system is None else coordinate_system
    screen_width = 1920 if screen_width is None else screen_width
    screen_height = 1080 if screen_height is None else screen_height
    duration_mean = 250 if duration_mean is None else duration_mean
    duration_sd = 80 if duration_sd is None else duration_sd
    saccade_gap_mean = 40 if saccade_gap_mean is None else saccade_gap_mean

    if coordinate_system not in {"pixels", "normalized"}:
        raise ValueError("coordinate_system must be 'pixels' or 'normalized'")
    for value, name in ((n_subjects, "n_subjects"), (n_fix, "n_fix")):
        if isinstance(value, (bool, np.bool_)) or not float(value).is_integer() or int(value) < 1:
            raise ValueError(f"{name} must be a positive integer")
    for value, name in (
        (sd, "sd"),
        (screen_width, "screen_width"),
        (screen_height, "screen_height"),
        (duration_mean, "duration_mean"),
        (duration_sd, "duration_sd"),
        (saccade_gap_mean, "saccade_gap_mean"),
    ):
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be non-negative and finite")
    if screen_width <= 0 or screen_height <= 0 or duration_mean <= 0:
        raise ValueError("Screen dimensions and duration_mean must be positive")

    rng = np.random.default_rng(seed)
    rows = []
    for subject_index in range(1, int(n_subjects) + 1):
        duration_ms = np.maximum(
            40.0,
            rng.normal(float(duration_mean), float(duration_sd), int(n_fix)),
        )
        gap_ms = np.maximum(
            0.0,
            rng.exponential(max(float(saccade_gap_mean), 1.0), int(n_fix)),
        )
        if coordinate_system == "pixels":
            x0 = rng.uniform(0.2 * screen_width, 0.8 * screen_width)
            y0 = rng.uniform(0.2 * screen_height, 0.8 * screen_height)
            x = x0 + np.cumsum(rng.normal(0, sd, int(n_fix)))
            y = y0 + np.cumsum(rng.normal(0, sd, int(n_fix)))
            x = np.clip(x, 0, screen_width)
            y = np.clip(y, 0, screen_height)
            fpogx = x / screen_width
            fpogy = y / screen_height
        else:
            x0 = rng.uniform(0.2, 0.8)
            y0 = rng.uniform(0.2, 0.8)
            step_sd = sd / max(screen_width, screen_height) if sd > 1 else sd
            x = np.clip(x0 + np.cumsum(rng.normal(0, step_sd, int(n_fix))), 0, 1)
            y = np.clip(y0 + np.cumsum(rng.normal(0, step_sd, int(n_fix))), 0, 1)
            fpogx, fpogy = x, y

        onset_ms = np.concatenate([[0.0], np.cumsum((duration_ms + gap_ms)[:-1])])
        end_ms = onset_ms + duration_ms
        subject = f"P{subject_index:03d}"
        for index in range(int(n_fix)):
            rows.append(
                {
                    "USER_ID": subject,
                    "MEDIA_ID": "simulated_stimulus",
                    "FPOGID": index + 1,
                    "FPOGS": onset_ms[index] / 1000,
                    "FPOGD": duration_ms[index] / 1000,
                    "FPOGX": fpogx[index],
                    "FPOGY": fpogy[index],
                    "FPOGV": 1,
                    "subject": subject,
                    "fixation_id": index + 1,
                    "start_time": onset_ms[index] / 1000,
                    "end_time": end_ms[index] / 1000,
                    "duration": duration_ms[index],
                    "duration_ms": duration_ms[index],
                    "x": x[index],
                    "y": y[index],
                    "coordinate_system": coordinate_system,
                }
            )
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
