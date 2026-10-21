"""Research-only timestamp integrity and gaze-dispersion interpretation contracts."""
from __future__ import annotations

import numpy as np
import pandas as pd


def acquisition_integrity_certificate(
    timestamps_s,
    *,
    valid=None,
    requested_rate_hz=None,
    native_rate_hz=None,
    sdk_rate_hz=None,
    analysis_grid_rate_hz=None,
    catchup_long_ratio=2.0,
    catchup_short_ratio=0.5,
) -> dict:
    """Preserve delivered timestamp observations without sorting/interpolation.

    Catch-up denotes a candidate long interval followed by a short interval;
    it is NOT a causal diagnosis of clock, driver or operating-system behavior.
    """
    time = np.asarray(timestamps_s, dtype=float)
    if time.ndim != 1 or time.size < 3 or not np.isfinite(time).all():
        raise ValueError("at least three finite timestamp samples required")
    dt = np.diff(time)
    if np.any(dt < 0) or not np.any(dt > 0):
        raise ValueError("timestamps must not reverse and need positive intervals")
    if catchup_long_ratio <= 1 or not 0 < catchup_short_ratio < 1:
        raise ValueError("invalid catch-up thresholds")
    declared = {"requested_rate_hz": requested_rate_hz,
                "native_rate_hz": native_rate_hz, "sdk_rate_hz": sdk_rate_hz,
                "analysis_grid_rate_hz": analysis_grid_rate_hz}
    for key, value in declared.items():
        if value is not None and (not np.isfinite(value) or value <= 0):
            raise ValueError(key + " must be positive")
    valid = np.ones(len(time), dtype=bool) if valid is None else np.asarray(valid)
    if valid.shape != time.shape or valid.dtype.kind != "b":
        raise ValueError("validity flags must be sample-aligned booleans")
    positive = dt[dt > 0]
    median = float(np.median(positive))
    burst = (dt[:-1] > catchup_long_ratio * median) & (
        dt[1:] < catchup_short_ratio * median)
    p = np.quantile(dt * 1000, [0.005, 0.05, 0.5, 0.95, 0.995])
    observed_median_rate = 1 / median
    effective_span_rate = (len(time) - 1) / (time[-1] - time[0])
    return {
        **declared,
        "observed_median_positive_rate_hz": observed_median_rate,
        "observed_span_rate_hz": float(effective_span_rate),
        "interval_p005_ms": float(p[0]), "interval_p05_ms": float(p[1]),
        "interval_median_ms": float(p[2]), "interval_p95_ms": float(p[3]),
        "interval_p995_ms": float(p[4]), "maximum_gap_ms": float(max(dt) * 1000),
        "duplicate_timestamp_intervals": int(np.count_nonzero(dt == 0)),
        "catchup_pair_candidates": int(burst.sum()),
        "invalid_state_fraction": float((~valid).mean()),
        "recording_duration_s": float(time[-1] - time[0]),
        "analysis_grid_exceeds_declared_native_rate": (
            bool(analysis_grid_rate_hz > native_rate_hz)
            if analysis_grid_rate_hz is not None and native_rate_hz is not None
            else None),
        "analysis_grid_exceeds_observed_rate": (
            bool(analysis_grid_rate_hz > observed_median_rate)
            if analysis_grid_rate_hz is not None else None),
        "claim_boundary": "Recorded timestamp process, not native hardware-rate or event synchronization validation.",
    }


def dispersion_interpretation_audit(
    frame: pd.DataFrame,
    *,
    dispersion_col: str = "bcea68",
    source_col: str = "measurement_context",
    target_size_col: str = "target_visual_angle_deg",
    eccentricity_col: str = "eccentricity_deg",
    task_col: str = "task",
    participant_col: str = "participant_id",
) -> pd.DataFrame:
    """Require geometry provenance before interpreting fixation dispersion.

    Does not correct dispersion, infer tracker precision, or impose a
    stimulus-dependent effect size from a single published experiment.
    """
    required = [dispersion_col, source_col, target_size_col, eccentricity_col,
                task_col, participant_col]
    if not set(required).issubset(frame.columns):
        raise ValueError("missing dispersion/geometry provenance columns")
    d = frame[required].copy()
    if d[[source_col, task_col, participant_col]].isna().any().any():
        raise ValueError("missing interpretation scope")
    if not d[source_col].isin(["calibration_target", "task_fixation"]).all():
        raise ValueError("unknown dispersion measurement context")
    for col in [dispersion_col, target_size_col, eccentricity_col]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
        if not np.isfinite(d[col]).all():
            raise ValueError("geometry and dispersion must be finite")
    if ((d[dispersion_col] < 0).any() or (d[target_size_col] <= 0).any()
            or (d[eccentricity_col] < 0).any()):
        raise ValueError("dispersion/eccentricity must be nonnegative, visual angle positive")
    d["interpretation"] = d[source_col].map({
        "calibration_target": "calibration_session_dispersion_not_accuracy",
        "task_fixation": "stimulus_conditioned_task_dispersion_not_tracker_precision",
    })
    return d
