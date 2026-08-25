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
