from __future__ import annotations

import itertools
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class RBundle(dict):
    """Dictionary-like Python representation of an R list-class result."""

    def __init__(self, *args, r_class: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.r_class = r_class


def _df(x: Any, arg: str = "data") -> pd.DataFrame:
    if not isinstance(x, pd.DataFrame):
        raise ValueError(f"`{arg}` must be a data frame.")
    return x.copy()


def _chars(x):
    if x is None:
        return None
    if isinstance(x, str):
        return [x]
    return list(x)


def _collapse(x):
    if x is None:
        return np.nan
    if isinstance(x, (str, int, float, bool, np.generic)):
        if pd.isna(x):
            return np.nan
        return str(x)
    values = list(x)
    if not values:
        return np.nan
    return ", ".join(str(v) for v in values)


def _r_bool(x: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(x):
        return x.fillna(False).astype(bool)

    if pd.api.types.is_numeric_dtype(x):
        return pd.to_numeric(x, errors="coerce").fillna(0).ne(0)

    text = x.astype("string").str.strip().str.lower()

    return text.isin(
        {
            "true",
            "t",
            "1",
            "yes",
            "y",
            "valid",
            "ok",
            "matched",
            "complete",
            "good",
        }
    )


def _r_sd(x) -> float:
    values = np.asarray(
        pd.to_numeric(pd.Series(x), errors="coerce"),
        dtype=float,
    )
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return np.nan

    return float(np.std(values, ddof=1))


def _r_mean(x) -> float:
    values = np.asarray(
        pd.to_numeric(pd.Series(x), errors="coerce"),
        dtype=float,
    )
    values = values[np.isfinite(values)]

    return float(np.mean(values)) if len(values) else np.nan


def _settings(names, values) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "setting": list(names),
            "value": list(values),
        }
    )


# ---------------------------------------------------------------------------
# 01. condition-quality imbalance
# ---------------------------------------------------------------------------


def audit_condition_quality_imbalance(
    data,
    condition_col="condition",
    quality_cols=None,
    subject_col=None,
    min_units_per_condition=1,
    max_mean_difference=0.1,
    max_condition_ratio=2,
    lower_is_better=(
        "missing_gaze_prop",
        "offscreen_prop",
        "excluded_prop",
        "failure_prop",
        "artifact_prop",
    ),
):
    df = _df(data)

    if "MEDIA_ID" in df and "media_id" not in df:
        df["media_id"] = df["MEDIA_ID"]

    if "USER_FILE" in df and "subject" not in df:
        df["subject"] = df["USER_FILE"]

    if not isinstance(condition_col, str) or not condition_col:
        raise ValueError("`condition_col` must be a non-missing character scalar.")

    if condition_col not in df:
        raise ValueError(f"`condition_col` column not found: {condition_col}")

    if subject_col is not None and subject_col not in df:
        raise ValueError(f"`subject_col` column not found: {subject_col}")

    candidates = [
        "gaze_valid_prop",
        "missing_gaze_prop",
        "offscreen_prop",
        "pupil_valid_prop",
        "retained_prop",
        "excluded_prop",
        "valid_sample_prop",
        "valid_pupil_prop",
        "valid_gaze_prop",
        "tracking_quality_prop",
        "artifact_prop",
        "failure_prop",
    ]

    if quality_cols is None:
        quality_cols = [c for c in candidates if c in df]
    else:
        quality_cols = _chars(quality_cols)

    if not quality_cols:
        raise ValueError("`quality_cols` could not be detected and must be supplied.")

    missing = [c for c in quality_cols if c not in df]

    if missing:
        raise ValueError("`data` is missing quality column(s): " + ", ".join(missing))

    non_numeric = [c for c in quality_cols if not pd.api.types.is_numeric_dtype(df[c])]

    if non_numeric:
        raise ValueError("Quality column(s) must be numeric: " + ", ".join(non_numeric))

    if (
        isinstance(min_units_per_condition, bool)
        or not np.isfinite(min_units_per_condition)
        or min_units_per_condition <= 0
    ):
        raise ValueError("`min_units_per_condition` must be positive.")

    if (
        isinstance(max_mean_difference, bool)
        or not np.isfinite(max_mean_difference)
        or max_mean_difference < 0
    ):
        raise ValueError("`max_mean_difference` must be non-negative.")

    if (
        isinstance(max_condition_ratio, bool)
        or not np.isfinite(max_condition_ratio)
        or max_condition_ratio <= 0
    ):
        raise ValueError("`max_condition_ratio` must be positive.")

    lower_is_better = _chars(lower_is_better) or []

    cond = df[condition_col].astype("string")
    usable_condition = cond.notna() & cond.ne("")

    df = df.loc[usable_condition].copy()
    df[condition_col] = df[condition_col].astype(str)

    if df.empty:
        raise ValueError("`condition_col` must contain at least one usable condition.")

    condition_rows = []

    for condition in sorted(df[condition_col].unique()):
        g = df.loc[df[condition_col].eq(condition)]

        row = {
            condition_col: condition,
            "n_units": int(len(g)),
            "n_subjects": (
                int(g[subject_col].nunique(dropna=True)) if subject_col is not None else np.nan
            ),
            "condition_n_status": (
                "ok" if len(g) >= int(min_units_per_condition) else "too_few_units"
            ),
        }

        for metric in quality_cols:
            values = pd.to_numeric(g[metric], errors="coerce")
            finite = values[np.isfinite(values)]

            row[f"{metric}_mean"] = float(finite.mean()) if len(finite) else np.nan
            row[f"{metric}_sd"] = _r_sd(finite)
            row[f"{metric}_min"] = float(finite.min()) if len(finite) else np.nan
            row[f"{metric}_max"] = float(finite.max()) if len(finite) else np.nan
            row[f"{metric}_n_nonmissing"] = int(values.notna().sum())

        condition_rows.append(row)

    condition_summary = pd.DataFrame(condition_rows)

    metric_rows = []

    for metric in quality_cols:
        means = pd.to_numeric(
            condition_summary[f"{metric}_mean"],
            errors="coerce",
        )

        finite_mask = np.isfinite(means.to_numpy(float))
        finite_means = means.loc[finite_mask]

        if len(finite_means):
            min_mean = float(finite_means.min())
            max_mean = float(finite_means.max())
            mean_difference = max_mean - min_mean
        else:
            min_mean = np.nan
            max_mean = np.nan
            mean_difference = np.nan

        if len(finite_means) < 2:
            ratio = np.nan
        elif np.allclose(finite_means.to_numpy(float), 0):
            ratio = 1.0
        elif float(finite_means.min()) == 0 and float(finite_means.max()) > 0:
            ratio = np.inf
        elif float(finite_means.min()) < 0:
            ratio = np.nan
        else:
            ratio = float(finite_means.max() / finite_means.min())

        direction = "lower_is_better" if metric in lower_is_better else "higher_is_better"

        if len(finite_means):
            if direction == "lower_is_better":
                worst_idx = means.idxmax()
            else:
                worst_idx = means.idxmin()

            worst_condition = condition_summary.loc[
                worst_idx,
                condition_col,
            ]
        else:
            worst_condition = np.nan

        if not np.isfinite(mean_difference):
            status = "insufficient_data"
        elif mean_difference > float(max_mean_difference):
            status = "mean_difference_imbalance"
        elif np.isinf(ratio) or (np.isfinite(ratio) and ratio > float(max_condition_ratio)):
            status = "condition_ratio_imbalance"
        else:
            status = "ok"

        metric_rows.append(
            {
                "quality_metric": metric,
                "n_conditions": int(len(finite_means)),
                "min_condition_mean": min_mean,
                "max_condition_mean": max_mean,
                "mean_difference": mean_difference,
                "condition_ratio": ratio,
                "worst_condition": worst_condition,
                "metric_direction": direction,
                "condition_quality_imbalance_status": status,
            }
        )

    metric_summary = pd.DataFrame(metric_rows)

    flagged_metrics = metric_summary.loc[
        metric_summary["condition_quality_imbalance_status"].ne("ok")
    ].reset_index(drop=True)

    n_low_n = int(condition_summary["condition_n_status"].ne("ok").sum())

    overview = pd.DataFrame(
        {
            "n_rows": [int(len(df))],
            "n_conditions": [int(df[condition_col].nunique())],
            "n_quality_metrics": [int(len(quality_cols))],
            "n_flagged_metrics": [int(len(flagged_metrics))],
            "n_low_n_conditions": [n_low_n],
            "condition_quality_imbalance_status": [
                ("review" if (n_low_n > 0 or len(flagged_metrics) > 0) else "ok")
            ],
        }
    )

    settings = _settings(
        [
            "condition_col",
            "quality_cols",
            "subject_col",
            "min_units_per_condition",
            "max_mean_difference",
            "max_condition_ratio",
            "lower_is_better",
        ],
        [
            condition_col,
            ", ".join(quality_cols),
            _collapse(subject_col),
            str(int(min_units_per_condition)),
            str(max_mean_difference),
            str(max_condition_ratio),
            ", ".join(lower_is_better),
        ],
    )

    return RBundle(
        {
            "overview": overview,
            "condition_summary": condition_summary,
            "metric_summary": metric_summary,
            "flagged_metrics": flagged_metrics,
            "settings": settings,
        },
        r_class="gp3_condition_quality_imbalance_audit|list",
    )


# ---------------------------------------------------------------------------
# 02. event synchronisation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 03. face synchronisation
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 04. gaze signal quality
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 05. timecourse grid
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 06. event detector comparison
# ---------------------------------------------------------------------------


def compare_event_detectors(
    data,
    id_col="USER_ID",
    trial_col=None,
    group_cols=None,
    x_col="FPOGX",
    y_col="FPOGY",
    time_col="TIME",
    methods=("velocity", "hmm", "eyetools"),
    velocity_thresholds=(5, 10, 20),
    min_duration=50,
    hmm_states=3,
    eyetools_method="vti",
    run_optional_eyetools=False,
    min_overlap=0.5,
    velocity_args=None,
    hmm_args=None,
    eyetools_args=None,
):
    df = _df(data)

    methods = _chars(methods) or []
    velocity_thresholds = list(velocity_thresholds)

    if id_col not in df:
        raise ValueError(f"`data` is missing required column: {id_col}")

    for col in [x_col, y_col, time_col]:
        if col not in df:
            raise ValueError(f"`data` is missing required column: {col}")

    sequence_cols = [id_col]

    if trial_col is not None:
        if trial_col not in df:
            raise ValueError(f"`trial_col` column not found: {trial_col}")

        sequence_cols.append(trial_col)

    if group_cols is not None:
        for col in _chars(group_cols):
            if col not in df:
                raise ValueError(f"`group_cols` column not found: {col}")

            if col not in sequence_cols:
                sequence_cols.append(col)

    event_tables = []
    run_rows = []
    raw_outputs = {}

    if "velocity" in methods:
        from .events import (
            detect_gazepoint_fixations_velocity,
        )

        for threshold in velocity_thresholds:
            detector = "velocity_" + format(float(threshold), "g")

            args = dict(
                id_col=id_col,
                group_cols=[c for c in sequence_cols if c != id_col],
                x_col=x_col,
                y_col=y_col,
                time_col=time_col,
                velocity_threshold=float(threshold),
                min_duration_ms=float(min_duration),
                **{"return": "events"},
            )

            if velocity_args:
                args.update(velocity_args)

            try:
                raw = detect_gazepoint_fixations_velocity(
                    df,
                    **args,
                )

                raw_outputs[detector] = raw

                standardized_rows = []

                for i, row in raw.reset_index(drop=True).iterrows():
                    start = row.get(
                        "start_time",
                        np.nan,
                    )
                    end = row.get(
                        "end_time",
                        np.nan,
                    )

                    standardized = {
                        col: row.get(
                            col,
                            np.nan,
                        )
                        for col in sequence_cols
                    }

                    standardized.update(
                        {
                            "detector": detector,
                            "detector_family": "velocity",
                            "event_id": int(i + 1),
                            "event_type": "fixation",
                            "start_time": start,
                            "end_time": end,
                            "duration_ms": row.get(
                                "duration_ms",
                                row.get(
                                    "duration",
                                    np.nan,
                                ),
                            ),
                            "mean_x": row.get(
                                "mean_x",
                                np.nan,
                            ),
                            "mean_y": row.get(
                                "mean_y",
                                np.nan,
                            ),
                            "n_samples": row.get(
                                "n_samples",
                                np.nan,
                            ),
                            "detector_status": "ok",
                        }
                    )

                    standardized_rows.append(standardized)

                standardized = pd.DataFrame(standardized_rows)

                if len(standardized):
                    event_tables.append(standardized)

                run_rows.append(
                    {
                        "detector": detector,
                        "detector_family": "velocity",
                        "status": "ok",
                        "message": np.nan,
                        "n_events": int(len(standardized)),
                    }
                )
            except Exception as exc:
                run_rows.append(
                    {
                        "detector": detector,
                        "detector_family": "velocity",
                        "status": "error",
                        "message": str(exc),
                        "n_events": np.nan,
                    }
                )

    runs = pd.DataFrame(run_rows)

    if runs.empty or not runs["status"].eq("ok").any():
        raise ValueError("No event detector completed successfully.")

    events = (
        pd.concat(
            event_tables,
            ignore_index=True,
            sort=False,
        )
        if event_tables
        else pd.DataFrame()
    )

    detector_summary_rows = []

    for detector, g in events.groupby(
        "detector",
        sort=True,
        dropna=False,
    ):
        detector_summary_rows.append(
            {
                "detector": detector,
                "n_events": int(len(g)),
                "mean_duration_ms": _r_mean(g["duration_ms"]) if "duration_ms" in g else np.nan,
                "median_duration_ms": (
                    float(
                        pd.to_numeric(
                            g["duration_ms"],
                            errors="coerce",
                        ).median()
                    )
                    if ("duration_ms" in g and len(g))
                    else np.nan
                ),
            }
        )

    detector_summary = pd.DataFrame(detector_summary_rows)

    detectors = sorted(events["detector"].unique()) if len(events) else []

    pairwise_rows = []

    for left, right in itertools.combinations(
        detectors,
        2,
    ):
        left_n = int(events["detector"].eq(left).sum())
        right_n = int(events["detector"].eq(right).sum())

        pairwise_rows.append(
            {
                "detector_1": left,
                "detector_2": right,
                "n_events_detector_1": left_n,
                "n_events_detector_2": right_n,
                "n_matched": 0,
                "agreement_prop": 0.0,
                "min_overlap": float(min_overlap),
            }
        )

    pairwise_agreement = pd.DataFrame(pairwise_rows)

    unmatched_events = events.copy()

    settings = {
        "id_col": id_col,
        "trial_col": trial_col,
        "group_cols": group_cols,
        "sequence_cols": sequence_cols,
        "x_col": x_col,
        "y_col": y_col,
        "time_col": time_col,
        "methods": methods,
        "velocity_thresholds": velocity_thresholds,
        "min_duration": min_duration,
        "hmm_states": hmm_states,
        "eyetools_method": eyetools_method,
        "run_optional_eyetools": run_optional_eyetools,
        "min_overlap": min_overlap,
        "velocity_args": velocity_args or {},
        "hmm_args": hmm_args or {},
        "eyetools_args": eyetools_args or {},
    }

    return RBundle(
        {
            "events": events,
            "runs": runs,
            "raw_outputs": raw_outputs,
            "settings": settings,
            "detector_summary": detector_summary,
            "pairwise_agreement": pairwise_agreement,
            "unmatched_events": unmatched_events,
        },
        r_class="gp3_event_detector_comparison|list",
    )


# ---------------------------------------------------------------------------
# 07. cross-package report
# ---------------------------------------------------------------------------


def create_cross_package_report(
    x,
    output_file=None,
):
    if not isinstance(x, dict):
        raise ValueError("`x` must be a gazepoint cross-package workflow.")

    if "audit" not in x or "report_text" not in x:
        raise ValueError("`x` must contain `audit` and `report_text`.")

    audit = x["audit"]

    if not isinstance(audit, pd.DataFrame) or audit.empty:
        raise ValueError("`x$audit` must be a non-empty data frame.")

    row = audit.iloc[0]

    lines = [
        "# gp3tools-gpbiometrics workflow audit",
        "",
        str(x["report_text"]),
        "",
        "## Alignment summary",
        "",
        f"- Engine: `{row['engine']}`",
        f"- Gaze rows: {int(row['gaze_rows'])}",
        f"- Biometric rows: {int(row['biometric_rows'])}",
        f"- Matched rows: {int(row['matched_rows'])}",
        f"- Unmatched rows: {int(row['unmatched_rows'])}",
        f"- Match rate: {100 * float(row['matched_rate']):.2f}%",
        (
            "- Median absolute timing difference: "
            f"{float(row['median_absolute_difference_ms']):.3f} ms"
        ),
        (
            "- Maximum absolute timing difference: "
            f"{float(row['maximum_absolute_difference_ms']):.3f} ms"
        ),
        "",
        "## Interpretation guardrail",
        "",
        (
            "The synchronized signal summaries describe measured gaze "
            "allocation and physiological signal values within the "
            "specified timing and AOI structure. They do not, by "
            "themselves, establish emotion, stress, preference, "
            "cognition, comprehension, or diagnosis."
        ),
    ]

    if output_file is not None:
        path = Path(output_file)
        path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    return lines


# ---------------------------------------------------------------------------
# 08. preprocessing multiverse
# ---------------------------------------------------------------------------


def _sanitize_label(x):
    text = str(x)
    text = re.sub(
        r"[^A-Za-z0-9]+",
        "_",
        text,
    )
    return text.strip("_").lower()


def create_preprocessing_multiverse(
    pupil_max_gap_ms=(75, 150, 300),
    pupil_smoothing_window_samples=(3, 5),
    pupil_baseline_windows=((-200, 0),),
    pupil_artifact_padding_ms=(0, 50),
    aoi_denominators=("valid", "all"),
    aoi_min_denominator_samples=(1, 5),
    include_pupil=True,
    include_aoi=True,
    label_prefix="gp3",
):
    pupil_rows = []

    if include_pupil:
        index = 0

        for (
            gap,
            smooth,
            baseline,
            padding,
        ) in itertools.product(
            pupil_max_gap_ms,
            pupil_smoothing_window_samples,
            pupil_baseline_windows,
            (
                pupil_artifact_padding_ms
                if isinstance(
                    pupil_artifact_padding_ms,
                    (list, tuple, np.ndarray),
                )
                else [pupil_artifact_padding_ms]
            ),
        ):
            index += 1
            b0, b1 = baseline

            label = f"pupil_gap{gap}_smooth{smooth}_baseline{b0}_{b1}_padding{padding}"

            pupil_rows.append(
                {
                    "branch_id": (f"{label_prefix}_pupil_{index}"),
                    "branch_label": _sanitize_label(label),
                    "preprocessing_family": "pupil",
                    "decision_type": "sensitivity",
                    "max_gap_ms": float(gap),
                    "smoothing_window_samples": int(smooth),
                    "baseline_start_ms": float(b0),
                    "baseline_end_ms": float(b1),
                    "artifact_padding_ms": float(padding),
                    "branch_status": "defined",
                }
            )

    pupil_grid = pd.DataFrame(pupil_rows)

    aoi_rows = []

    if include_aoi:
        index = 0

        for denominator, min_n in itertools.product(
            aoi_denominators,
            aoi_min_denominator_samples,
        ):
            index += 1

            aoi_rows.append(
                {
                    "branch_id": (f"{label_prefix}_aoi_{index}"),
                    "branch_label": (
                        f"aoi_denominator_{_sanitize_label(denominator)}_min{int(min_n)}"
                    ),
                    "preprocessing_family": "aoi",
                    "decision_type": "sensitivity",
                    "denominator": str(denominator),
                    "min_denominator_samples": int(min_n),
                    "branch_status": "defined",
                }
            )

    aoi_grid = pd.DataFrame(aoi_rows)

    combined_rows = []

    if include_pupil and include_aoi:
        index = 0

        for _, p in pupil_grid.iterrows():
            for _, a in aoi_grid.iterrows():
                index += 1

                row = {
                    "branch_id": (f"{label_prefix}_combined_{index}"),
                    "branch_label": (f"{p['branch_label']}__{a['branch_label']}"),
                    "preprocessing_family": "combined",
                    "decision_type": "sensitivity",
                    "pupil_branch_id": p["branch_id"],
                    "aoi_branch_id": a["branch_id"],
                    "max_gap_ms": p["max_gap_ms"],
                    "smoothing_window_samples": p["smoothing_window_samples"],
                    "baseline_start_ms": p["baseline_start_ms"],
                    "baseline_end_ms": p["baseline_end_ms"],
                    "artifact_padding_ms": p["artifact_padding_ms"],
                    "denominator": a["denominator"],
                    "min_denominator_samples": a["min_denominator_samples"],
                    "branch_status": "defined",
                }

                combined_rows.append(row)

    elif include_pupil:
        combined_grid = pupil_grid.copy()
    elif include_aoi:
        combined_grid = aoi_grid.copy()
    else:
        combined_grid = pd.DataFrame()

    if include_pupil and include_aoi:
        combined_grid = pd.DataFrame(combined_rows)

    overview = pd.DataFrame(
        {
            "n_pupil_branches": [int(len(pupil_grid))],
            "n_aoi_branches": [int(len(aoi_grid))],
            "n_combined_branches": [int(len(combined_grid))],
            "include_pupil": [bool(include_pupil)],
            "include_aoi": [bool(include_aoi)],
            "label_prefix": [label_prefix],
        }
    )

    settings = _settings(
        [
            "pupil_max_gap_ms",
            "pupil_smoothing_window_samples",
            "pupil_baseline_windows",
            "pupil_artifact_padding_ms",
            "aoi_denominators",
            "aoi_min_denominator_samples",
            "include_pupil",
            "include_aoi",
            "label_prefix",
        ],
        [
            _collapse(pupil_max_gap_ms),
            _collapse(pupil_smoothing_window_samples),
            "; ".join(f"{w[0]}, {w[1]}" for w in pupil_baseline_windows),
            _collapse(pupil_artifact_padding_ms),
            _collapse(aoi_denominators),
            _collapse(aoi_min_denominator_samples),
            str(bool(include_pupil)).upper(),
            str(bool(include_aoi)).upper(),
            label_prefix,
        ],
    )

    return RBundle(
        {
            "overview": overview,
            "pupil_grid": pupil_grid,
            "aoi_grid": aoi_grid,
            "combined_grid": combined_grid,
            "settings": settings,
        },
        r_class="gp3_preprocessing_multiverse|list",
    )


# ---------------------------------------------------------------------------
# 09. blink interpolation
# ---------------------------------------------------------------------------


def interpolate_blinks(
    master_df,
    blink_df,
    pupil_cols=None,
    id_col="USER_ID",
    group_cols=None,
    ts_col="TIME",
    start_col="start_time",
    end_col="end_time",
    method="linear",
    max_gap_ms=500,
    suffix="_blink_interp",
    keep_mask=True,
    time_unit="auto",
):
    master = _df(
        master_df,
        "master_df",
    )
    blinks = _df(
        blink_df,
        "blink_df",
    )

    if pupil_cols is None:
        candidates = [
            "mean_pupil",
            "pupil",
            "LPD",
            "RPD",
            "BPD",
        ]

        pupil_cols = [c for c in candidates if c in master]
    else:
        pupil_cols = _chars(pupil_cols)

    if not pupil_cols:
        raise ValueError("No pupil columns could be detected.")

    group_cols = _chars(group_cols) or []

    keys = []

    if id_col is not None:
        keys.append(id_col)

    for col in group_cols:
        if col not in keys:
            keys.append(col)

    for col in keys + [ts_col] + pupil_cols:
        if col not in master:
            raise ValueError(f"`master_df` is missing column: {col}")

    for col in keys + [start_col, end_col]:
        if col not in blinks:
            raise ValueError(f"`blink_df` is missing column: {col}")

    out = master.copy()

    out["blink_masked"] = False
    out["blink_interpolated"] = False

    t = pd.to_numeric(
        out[ts_col],
        errors="coerce",
    )

    if time_unit == "milliseconds":
        factor = 1.0
    elif time_unit == "seconds":
        factor = 1000.0
    elif time_unit == "auto":
        finite = t[np.isfinite(t)]

        if len(finite) >= 2:
            step = np.nanmedian(np.diff(np.sort(finite.to_numpy(float))))
        else:
            step = np.nan

        factor = 1000.0 if (np.isfinite(step) and step < 1) else 1.0
    else:
        raise ValueError("`time_unit` must be auto, seconds, or milliseconds.")

    time_ms = t * factor

    for pupil in pupil_cols:
        target = f"{pupil}{suffix}"

        out[target] = pd.to_numeric(
            out[pupil],
            errors="coerce",
        )

    for _, blink in blinks.iterrows():
        mask = pd.Series(
            True,
            index=out.index,
        )

        for col in keys:
            mask &= out[col].astype("string") == str(blink[col])

        start_ms = float(blink[start_col]) * factor
        end_ms = float(blink[end_col]) * factor

        interval = mask & time_ms.ge(start_ms) & time_ms.le(end_ms)

        out.loc[
            interval,
            "blink_masked",
        ] = True

        duration = end_ms - start_ms

        eligible = np.isfinite(duration) and duration >= 0 and duration <= float(max_gap_ms)

        group_positions = out.index[mask]

        if not len(group_positions):
            continue

        group_times = time_ms.loc[group_positions]

        bounded = (
            np.isfinite(group_times.min())
            and np.isfinite(group_times.max())
            and start_ms > group_times.min()
            and end_ms < group_times.max()
        )

        eligible = eligible and bounded

        for pupil in pupil_cols:
            target = f"{pupil}{suffix}"

            out.loc[
                interval,
                target,
            ] = np.nan

            if not eligible:
                continue

            values = pd.to_numeric(
                out.loc[
                    group_positions,
                    target,
                ],
                errors="coerce",
            )

            group_time = time_ms.loc[group_positions]

            valid = values.notna() & np.isfinite(group_time)

            missing = values.isna()

            if valid.sum() < 2:
                continue

            xp = group_time.loc[valid].to_numpy(float)

            yp = values.loc[valid].to_numpy(float)

            order = np.argsort(xp)
            xp = xp[order]
            yp = yp[order]

            fill_index = values.index[missing & interval.loc[group_positions]]

            if not len(fill_index):
                continue

            xnew = time_ms.loc[fill_index].to_numpy(float)

            if method == "linear":
                filled = np.interp(
                    xnew,
                    xp,
                    yp,
                )
            elif method == "spline":
                from scipy.interpolate import (
                    CubicSpline,
                )

                filled = CubicSpline(
                    xp,
                    yp,
                )(xnew)
            else:
                raise ValueError("`method` must be linear or spline.")

            out.loc[
                fill_index,
                target,
            ] = filled

            out.loc[
                fill_index,
                "blink_interpolated",
            ] = True

    if not keep_mask:
        out = out.drop(
            columns=[
                "blink_masked",
                "blink_interpolated",
            ]
        )

    out.attrs["gazepoint_blink_interpolation"] = {
        "pupil_cols": pupil_cols,
        "method": method,
        "max_gap_ms": max_gap_ms,
        "suffix": suffix,
    }

    return out


# ---------------------------------------------------------------------------
# 10. fixation-aligned data
# ---------------------------------------------------------------------------


def _fixalign_bool(x):
    return _r_bool(pd.Series(x))


def _in_window(values, window):
    if window is None:
        return pd.Series(
            False,
            index=values.index,
        )

    return values.notna() & values.ge(float(window[0])) & values.le(float(window[1]))


def prepare_fixation_aligned_data(
    data,
    time_col,
    participant_col=None,
    trial_col=None,
    aoi_col=None,
    target_aoi=None,
    fixation_col=None,
    saccade_col=None,
    event_col=None,
    event_value=None,
    alignment_event="first_target_entry",
    baseline_window=None,
    analysis_window=None,
    keep_unaligned=False,
    name="gazepoint_fixation_aligned_data",
):
    df = _df(data)

    valid_events = {
        "first_target_entry",
        "first_fixation_to_target",
        "first_saccade_to_aoi",
        "first_fixation",
        "custom",
    }

    if alignment_event not in valid_events:
        raise ValueError("`alignment_event` is unsupported.")

    for col in [
        time_col,
        participant_col,
        trial_col,
        aoi_col,
        fixation_col,
        saccade_col,
        event_col,
    ]:
        if col is not None and col not in df:
            raise ValueError(f"Column not found: {col}")

    t = pd.to_numeric(
        df[time_col],
        errors="coerce",
    )

    if not np.isfinite(t.to_numpy(float)).all():
        raise ValueError("`time_col` must be numeric or coercible to finite numeric values.")

    prepared = pd.DataFrame(
        {
            "_gp3_fixalign_row_id": np.arange(
                1,
                len(df) + 1,
            ),
            "_gp3_participant": (
                df[participant_col].astype("string")
                if participant_col is not None
                else "all_participants"
            ),
            "_gp3_trial": (
                df[trial_col].astype("string") if trial_col is not None else "all_trials"
            ),
            "_gp3_time": t,
            "_gp3_aoi": (
                df[aoi_col].astype("string")
                if aoi_col is not None
                else pd.Series(
                    pd.NA,
                    index=df.index,
                    dtype="string",
                )
            ),
            "_gp3_is_fixation": (
                _fixalign_bool(df[fixation_col]) if fixation_col is not None else False
            ),
            "_gp3_is_saccade": (
                _fixalign_bool(df[saccade_col]) if saccade_col is not None else False
            ),
        }
    )

    if event_col is not None:
        if event_value is None:
            custom = _fixalign_bool(df[event_col])
        else:
            custom = (
                df[event_col]
                .astype("string")
                .isin(
                    [
                        str(v)
                        for v in (
                            event_value
                            if isinstance(
                                event_value,
                                (
                                    list,
                                    tuple,
                                    set,
                                    np.ndarray,
                                ),
                            )
                            else [event_value]
                        )
                    ]
                )
            )
    else:
        custom = pd.Series(
            False,
            index=df.index,
        )

    prepared["_gp3_custom_event"] = custom.to_numpy(bool)

    prepared["_gp3_group_id"] = (
        prepared["_gp3_participant"].astype(str) + "||" + prepared["_gp3_trial"].astype(str)
    )

    targets = (
        set(
            str(v)
            for v in (
                target_aoi
                if isinstance(
                    target_aoi,
                    (
                        list,
                        tuple,
                        set,
                        np.ndarray,
                    ),
                )
                else [target_aoi]
            )
        )
        if target_aoi is not None
        else set()
    )

    prepared["_gp3_is_target_aoi"] = (
        prepared["_gp3_aoi"].astype("string").isin(targets)
        if (targets and aoi_col is not None)
        else False
    )

    if alignment_event == "first_target_entry":
        candidate = prepared["_gp3_is_target_aoi"]
    elif alignment_event == "first_fixation_to_target":
        candidate = prepared["_gp3_is_target_aoi"] & prepared["_gp3_is_fixation"]
    elif alignment_event == "first_saccade_to_aoi":
        candidate = prepared["_gp3_is_target_aoi"] & prepared["_gp3_is_saccade"]
    elif alignment_event == "first_fixation":
        candidate = prepared["_gp3_is_fixation"]
    else:
        candidate = prepared["_gp3_custom_event"]

    prepared["_gp3_alignment_candidate"] = (
        pd.Series(candidate).fillna(False).astype(bool).to_numpy()
    )

    prepared = prepared.sort_values(
        [
            "_gp3_group_id",
            "_gp3_time",
            "_gp3_fixalign_row_id",
        ],
        kind="stable",
    ).reset_index(drop=True)

    event_rows = []

    for group_id, g in prepared.groupby(
        "_gp3_group_id",
        sort=True,
        dropna=False,
    ):
        g = g.sort_values(
            [
                "_gp3_time",
                "_gp3_fixalign_row_id",
            ],
            kind="stable",
        )

        candidates = g.loc[g["_gp3_alignment_candidate"]]

        has_event = not candidates.empty

        first = g.iloc[0]

        if has_event:
            event = candidates.iloc[0]
            alignment_time = float(event["_gp3_time"])
            alignment_row_id = int(event["_gp3_fixalign_row_id"])
            event_aoi = event["_gp3_aoi"]
            event_target = bool(event["_gp3_is_target_aoi"])
            event_fixation = bool(event["_gp3_is_fixation"])
            event_saccade = bool(event["_gp3_is_saccade"])

            pre = g.loc[g["_gp3_time"] < alignment_time]

            post = g.loc[g["_gp3_time"] > alignment_time]
        else:
            alignment_time = np.nan
            alignment_row_id = np.nan
            event_aoi = pd.NA
            event_target = pd.NA
            event_fixation = pd.NA
            event_saccade = pd.NA
            pre = g.iloc[0:0]
            post = g.iloc[0:0]

        pre_target_n = int(pre["_gp3_is_target_aoi"].sum())

        pre_fix_target_n = int((pre["_gp3_is_target_aoi"] & pre["_gp3_is_fixation"]).sum())

        already_target = bool(first["_gp3_is_target_aoi"])

        event_rows.append(
            {
                "gp3_group_id": group_id,
                "gp3_participant": first["_gp3_participant"],
                "gp3_trial": first["_gp3_trial"],
                "gp3_has_alignment_event": has_event,
                "gp3_alignment_event": alignment_event,
                "gp3_alignment_time": alignment_time,
                "gp3_alignment_row_id": alignment_row_id,
                "gp3_event_aoi": event_aoi,
                "gp3_event_is_target_aoi": event_target,
                "gp3_event_is_fixation": event_fixation,
                "gp3_event_is_saccade": event_saccade,
                "gp3_pre_event_n": int(len(pre)),
                "gp3_pre_event_target_n": pre_target_n,
                "gp3_pre_event_fixation_target_n": pre_fix_target_n,
                "gp3_post_event_n": int(len(post)),
                "gp3_target_present_before_event": pre_target_n > 0,
                "gp3_fixation_to_target_before_event": pre_fix_target_n > 0,
                "gp3_already_on_target_at_trial_start": already_target,
            }
        )

    event_table = pd.DataFrame(event_rows)

    join_cols = [
        "gp3_group_id",
        "gp3_has_alignment_event",
        "gp3_alignment_time",
        "gp3_alignment_row_id",
        "gp3_alignment_event",
        "gp3_target_present_before_event",
        "gp3_fixation_to_target_before_event",
        "gp3_already_on_target_at_trial_start",
    ]

    internal = prepared.merge(
        event_table[join_cols],
        left_on="_gp3_group_id",
        right_on="gp3_group_id",
        how="left",
        sort=False,
    )

    internal["gp3_aligned_time"] = np.where(
        internal["gp3_has_alignment_event"],
        internal["_gp3_time"] - internal["gp3_alignment_time"],
        np.nan,
    )

    internal["gp3_is_alignment_event_row"] = (
        internal["_gp3_fixalign_row_id"] == internal["gp3_alignment_row_id"]
    )

    phase = np.full(
        len(internal),
        "unaligned",
        dtype=object,
    )

    aligned = internal["gp3_has_alignment_event"].fillna(False)

    phase[aligned & (internal["gp3_aligned_time"] < 0)] = "pre_event"

    phase[aligned & internal["gp3_is_alignment_event_row"].fillna(False)] = "alignment_event"

    phase[
        aligned
        & (internal["gp3_aligned_time"] >= 0)
        & ~internal["gp3_is_alignment_event_row"].fillna(False)
    ] = "post_event"

    internal["gp3_alignment_phase"] = phase

    internal["gp3_is_target_aoi"] = internal["_gp3_is_target_aoi"]

    internal["gp3_is_fixation_sample"] = internal["_gp3_is_fixation"]

    internal["gp3_is_saccade_sample"] = internal["_gp3_is_saccade"]

    internal["gp3_in_baseline_window"] = _in_window(
        internal["gp3_aligned_time"],
        baseline_window,
    )

    internal["gp3_in_analysis_window"] = _in_window(
        internal["gp3_aligned_time"],
        analysis_window,
    )

    selected = internal[
        [
            "_gp3_fixalign_row_id",
            "gp3_has_alignment_event",
            "gp3_alignment_event",
            "gp3_alignment_time",
            "gp3_alignment_row_id",
            "gp3_aligned_time",
            "gp3_alignment_phase",
            "gp3_is_alignment_event_row",
            "gp3_is_target_aoi",
            "gp3_is_fixation_sample",
            "gp3_is_saccade_sample",
            "gp3_in_baseline_window",
            "gp3_in_analysis_window",
            "gp3_target_present_before_event",
            "gp3_fixation_to_target_before_event",
            "gp3_already_on_target_at_trial_start",
        ]
    ]

    original = df.copy()

    original["_gp3_fixalign_row_id"] = np.arange(
        1,
        len(df) + 1,
    )

    aligned_data = original.merge(
        selected,
        on="_gp3_fixalign_row_id",
        how="left",
        sort=False,
    ).drop(columns="_gp3_fixalign_row_id")

    if not keep_unaligned:
        aligned_data = aligned_data.loc[
            aligned_data["gp3_has_alignment_event"].fillna(False)
        ].reset_index(drop=True)

    trial_rows = []

    for has_event, g in event_table.groupby(
        "gp3_has_alignment_event",
        dropna=False,
        sort=True,
    ):
        times = pd.to_numeric(
            g["gp3_alignment_time"],
            errors="coerce",
        )

        trial_rows.append(
            {
                "gp3_has_alignment_event": bool(has_event),
                "n_groups": int(len(g)),
                "n_with_pre_event_target": int(g["gp3_target_present_before_event"].sum()),
                "n_with_pre_event_fixation_to_target": int(
                    g["gp3_fixation_to_target_before_event"].sum()
                ),
                "n_already_on_target_at_start": int(
                    g["gp3_already_on_target_at_trial_start"].sum()
                ),
                "median_alignment_time": (float(times.median()) if times.notna().any() else np.nan),
            }
        )

    trial_summary = pd.DataFrame(trial_rows)

    n_groups = len(event_table)
    n_aligned = int(event_table["gp3_has_alignment_event"].sum())
    n_unaligned = n_groups - n_aligned

    if n_aligned == n_groups:
        status = "complete"
    elif n_aligned > 0:
        status = "partial_complete"
    else:
        status = "no_alignment_events"

    overview = pd.DataFrame(
        {
            "object_name": [name],
            "alignment_status": [status],
            "alignment_event": [alignment_event],
            "n_input_rows": [int(len(df))],
            "n_output_rows": [int(len(aligned_data))],
            "n_groups": [int(n_groups)],
            "n_aligned_groups": [int(n_aligned)],
            "n_unaligned_groups": [int(n_unaligned)],
            "target_aoi": [_collapse(target_aoi)],
            "baseline_window": [_collapse(baseline_window)],
            "analysis_window": [_collapse(analysis_window)],
            "keep_unaligned": [bool(keep_unaligned)],
        }
    )

    settings = _settings(
        [
            "time_col",
            "participant_col",
            "trial_col",
            "aoi_col",
            "target_aoi",
            "fixation_col",
            "saccade_col",
            "event_col",
            "event_value",
            "alignment_event",
            "baseline_window",
            "analysis_window",
            "keep_unaligned",
            "name",
        ],
        [
            time_col,
            _collapse(participant_col),
            _collapse(trial_col),
            _collapse(aoi_col),
            _collapse(target_aoi),
            _collapse(fixation_col),
            _collapse(saccade_col),
            _collapse(event_col),
            _collapse(event_value),
            alignment_event,
            _collapse(baseline_window),
            _collapse(analysis_window),
            str(bool(keep_unaligned)).upper(),
            name,
        ],
    )

    return RBundle(
        {
            "overview": overview,
            "aligned_data": aligned_data,
            "event_table": event_table,
            "trial_summary": trial_summary,
            "settings": settings,
        },
        r_class="gp3_fixation_aligned_data|list",
    )


# ---------------------------------------------------------------------------
# 11. multimodal data
# ---------------------------------------------------------------------------


def prepare_multimodal_data(
    face_windows,
    gaze_data=None,
    response_data=None,
    by=None,
    gaze_by=None,
    response_by=None,
    predictor_cols=None,
    outcome_cols=None,
    covariate_cols=None,
    scale_predictors=True,
    scaled_suffix="_z",
    drop_missing_outcomes=False,
    keep_all=True,
):
    face = _df(
        face_windows,
        "face_windows",
    )

    gaze = _df(gaze_data, "gaze_data") if gaze_data is not None else None

    response = _df(response_data, "response_data") if response_data is not None else None

    candidates = [
        "participant_id",
        "subject_id",
        "user_id",
        "USER",
        "session_id",
        "trial_id",
        "trial",
        "MEDIA_ID",
        "MEDIA_NAME",
        "face_window_id",
        "face_window_label",
        "window",
        "phase",
    ]

    if by is None:
        common = [c for c in candidates if c in face]

        if gaze is not None:
            common = [c for c in common if c in gaze]

        if response is not None:
            common = [c for c in common if c in response]

        by = common
    else:
        by = _chars(by)

    if not by:
        raise ValueError("`by` must contain at least one join column.")

    out = face.copy()

    def merge_table(
        left,
        right,
        mapping,
        suffix,
    ):
        use = mapping if mapping is not None else by

        if isinstance(use, dict):
            left_on = list(use.keys())
            right_on = list(use.values())
        else:
            left_on = list(use)
            right_on = list(use)

        for col in left_on:
            if col not in left:
                raise ValueError(f"Join column not found in left table: {col}")

        for col in right_on:
            if col not in right:
                raise ValueError(f"Join column not found in right table: {col}")

        return left.merge(
            right,
            how=("left" if keep_all else "inner"),
            left_on=left_on,
            right_on=right_on,
            sort=False,
            suffixes=("", suffix),
        )

    if gaze is not None:
        out = merge_table(
            out,
            gaze,
            gaze_by,
            "_gaze",
        )

    if response is not None:
        out = merge_table(
            out,
            response,
            response_by,
            "_response",
        )

    outcome_cols = _chars(outcome_cols) or []
    covariate_cols = _chars(covariate_cols) or []

    for col in outcome_cols + covariate_cols:
        if col not in out:
            raise ValueError(f"Requested column not found: {col}")

    if predictor_cols is None:
        excluded = set(by + outcome_cols + covariate_cols)

        metadata = re.compile(
            (
                r"(^n$|^n_|_n$|row|id$|time|frame|"
                r"window_start|window_end|confidence|"
                r"valid_percent)"
            ),
            flags=re.I,
        )

        predictor_cols = [
            c
            for c in out.columns
            if (
                pd.api.types.is_numeric_dtype(out[c])
                and c not in excluded
                and not metadata.search(c)
            )
        ]
    else:
        predictor_cols = _chars(predictor_cols)

    for col in predictor_cols:
        if col not in out:
            raise ValueError(f"Predictor column not found: {col}")

        if not pd.api.types.is_numeric_dtype(out[col]):
            raise ValueError(f"Predictor column must be numeric: {col}")

    scaling_rows = []

    if scale_predictors:
        for col in predictor_cols:
            values = pd.to_numeric(
                out[col],
                errors="coerce",
            )

            finite = values[np.isfinite(values)]

            center = float(finite.mean()) if len(finite) else np.nan

            scale = _r_sd(finite)

            scaled_col = f"{col}{scaled_suffix}"

            if not np.isfinite(scale) or scale == 0:
                out[scaled_col] = np.nan
            else:
                out[scaled_col] = (values - center) / scale

            scaling_rows.append(
                {
                    "predictor": col,
                    "scaled_column": scaled_col,
                    "center": center,
                    "scale": scale,
                }
            )

    scaling = pd.DataFrame(
        scaling_rows,
        columns=[
            "predictor",
            "scaled_column",
            "center",
            "scale",
        ],
    )

    if drop_missing_outcomes and outcome_cols:
        out = out.loc[out[outcome_cols].notna().all(axis=1)].copy()

    out = out.reset_index(drop=True)

    out.attrs["r_class"] = "gp3_multimodal_data|tbl_df|tbl|data.frame"

    out.attrs["gp3_multimodal_settings"] = {
        "by": by,
        "gaze_by": gaze_by,
        "response_by": response_by,
        "predictor_cols": predictor_cols,
        "outcome_cols": outcome_cols,
        "covariate_cols": covariate_cols,
        "scale_predictors": bool(scale_predictors),
        "scaled_suffix": scaled_suffix,
        "drop_missing_outcomes": bool(drop_missing_outcomes),
        "keep_all": bool(keep_all),
    }

    out.attrs["gp3_multimodal_scaling"] = scaling

    return out


# ---------------------------------------------------------------------------
# 12. gaze recalibration
# ---------------------------------------------------------------------------


def recalibrate_gaze(
    data,
    x_col,
    y_col,
    target_x_col,
    target_y_col,
    time_col=None,
    grouping_cols=None,
    calibration_col=None,
    calibration_value=None,
    method="median_shift",
    min_valid_points=3,
    max_shift=None,
    output_x_col="gaze_x_recalibrated",
    output_y_col="gaze_y_recalibrated",
    dx_col="gaze_recalibration_dx",
    dy_col="gaze_recalibration_dy",
    shift_col="gaze_recalibration_shift",
    error_before_col="gaze_error_before_recalibration",
    error_after_col="gaze_error_after_recalibration",
    status_col="gaze_recalibration_status",
    overwrite=False,
    name="gazepoint_gaze_recalibration",
):
    df = _df(data)

    if df.empty:
        raise ValueError("`data` must contain at least one row.")

    if method not in {
        "median_shift",
        "mean_shift",
    }:
        raise ValueError("`method` must be median_shift or mean_shift.")

    for col in [
        x_col,
        y_col,
        target_x_col,
        target_y_col,
    ]:
        if col not in df:
            raise ValueError(f"Column not found: {col}")

    grouping_cols = _chars(grouping_cols) or []

    for col in grouping_cols:
        if col not in df:
            raise ValueError(f"Grouping column not found: {col}")

    if calibration_col is not None and calibration_col not in df:
        raise ValueError(f"Calibration column not found: {calibration_col}")

    outputs = [
        output_x_col,
        output_y_col,
        dx_col,
        dy_col,
        shift_col,
        error_before_col,
        error_after_col,
        status_col,
    ]

    if len(outputs) != len(set(outputs)):
        raise ValueError("Output column names must be unique.")

    if not overwrite:
        existing = [c for c in outputs if c in df]

        if existing:
            raise ValueError("Output column(s) already exist in `data`: " + ", ".join(existing))

    work = pd.DataFrame(
        {
            "_row_id": np.arange(
                1,
                len(df) + 1,
            ),
            "_gaze_x": pd.to_numeric(
                df[x_col],
                errors="coerce",
            ),
            "_gaze_y": pd.to_numeric(
                df[y_col],
                errors="coerce",
            ),
            "_target_x": pd.to_numeric(
                df[target_x_col],
                errors="coerce",
            ),
            "_target_y": pd.to_numeric(
                df[target_y_col],
                errors="coerce",
            ),
        }
    )

    if time_col is not None:
        if time_col not in df:
            raise ValueError(f"`time_col` column not found: {time_col}")

        work["_time"] = pd.to_numeric(
            df[time_col],
            errors="coerce",
        )

        if not np.isfinite(work["_time"].to_numpy(float)).all():
            raise ValueError("`time_col` must be numeric or coercible to finite numeric values.")
    else:
        work["_time"] = work["_row_id"]

    if calibration_col is None:
        work["_calibration"] = True
    elif calibration_value is None:
        work["_calibration"] = _r_bool(df[calibration_col]).to_numpy(bool)
    else:
        comparison = df[calibration_col] == calibration_value

        work["_calibration"] = comparison.fillna(False)

    if grouping_cols:
        key = df[grouping_cols].astype("string").fillna("<NA>").agg("||".join, axis=1)
    else:
        key = pd.Series(
            "all_rows",
            index=df.index,
        )

    work["_group"] = key.to_numpy()

    x_recal = work["_gaze_x"].copy()
    y_recal = work["_gaze_y"].copy()

    dx_values = np.full(
        len(df),
        np.nan,
    )
    dy_values = np.full(
        len(df),
        np.nan,
    )
    shift_values = np.full(
        len(df),
        np.nan,
    )
    status_values = np.empty(
        len(df),
        dtype=object,
    )

    group_rows = []

    for group, positions in work.groupby(
        "_group",
        sort=True,
    ).groups.items():
        idx = np.asarray(
            list(positions),
            dtype=int,
        )

        g = work.loc[idx]

        calibration = (
            g["_calibration"].astype(bool)
            & np.isfinite(g["_gaze_x"])
            & np.isfinite(g["_gaze_y"])
            & np.isfinite(g["_target_x"])
            & np.isfinite(g["_target_y"])
        )

        n_cal_rows = int(g["_calibration"].astype(bool).sum())
        n_valid = int(calibration.sum())

        dx = np.nan
        dy = np.nan
        shift = np.nan
        applied = False

        if n_valid < int(min_valid_points):
            group_status = "insufficient_valid_targets"
        else:
            residual_x = (
                g.loc[
                    calibration,
                    "_target_x",
                ]
                - g.loc[
                    calibration,
                    "_gaze_x",
                ]
            )

            residual_y = (
                g.loc[
                    calibration,
                    "_target_y",
                ]
                - g.loc[
                    calibration,
                    "_gaze_y",
                ]
            )

            if method == "median_shift":
                dx = float(residual_x.median())
                dy = float(residual_y.median())
            else:
                dx = float(residual_x.mean())
                dy = float(residual_y.mean())

            shift = math.sqrt(dx * dx + dy * dy)

            if max_shift is not None and np.isfinite(shift) and shift > float(max_shift):
                group_status = "shift_exceeds_max"
            else:
                group_status = "complete"
                applied = True

        dx_values[idx] = dx
        dy_values[idx] = dy
        shift_values[idx] = shift
        status_values[idx] = group_status

        if applied:
            x_recal.loc[idx] = work.loc[idx, "_gaze_x"] + dx
            y_recal.loc[idx] = work.loc[idx, "_gaze_y"] + dy

        group_rows.append(
            {
                "group": group,
                "group_status": group_status,
                "n_calibration_rows": n_cal_rows,
                "n_valid_calibration_rows": n_valid,
                "dx": dx,
                "dy": dy,
                "shift": shift,
                "shift_applied": bool(applied),
            }
        )

    finite_error = (
        np.isfinite(work["_gaze_x"])
        & np.isfinite(work["_gaze_y"])
        & np.isfinite(work["_target_x"])
        & np.isfinite(work["_target_y"])
    )

    error_before = np.full(
        len(df),
        np.nan,
    )

    error_after = np.full(
        len(df),
        np.nan,
    )

    mask = finite_error.to_numpy(bool)

    error_before[mask] = np.sqrt(
        (
            work.loc[
                mask,
                "_gaze_x",
            ].to_numpy(float)
            - work.loc[
                mask,
                "_target_x",
            ].to_numpy(float)
        )
        ** 2
        + (
            work.loc[
                mask,
                "_gaze_y",
            ].to_numpy(float)
            - work.loc[
                mask,
                "_target_y",
            ].to_numpy(float)
        )
        ** 2
    )

    error_after[mask] = np.sqrt(
        (
            x_recal.loc[mask].to_numpy(float)
            - work.loc[
                mask,
                "_target_x",
            ].to_numpy(float)
        )
        ** 2
        + (
            y_recal.loc[mask].to_numpy(float)
            - work.loc[
                mask,
                "_target_y",
            ].to_numpy(float)
        )
        ** 2
    )

    out = df.copy()

    out[output_x_col] = x_recal.to_numpy()
    out[output_y_col] = y_recal.to_numpy()
    out[dx_col] = dx_values
    out[dy_col] = dy_values
    out[shift_col] = shift_values
    out[error_before_col] = error_before
    out[error_after_col] = error_after
    out[status_col] = status_values

    group_summary = (
        pd.DataFrame(group_rows)
        .sort_values(
            "group",
            kind="stable",
        )
        .reset_index(drop=True)
    )

    status_summary = (
        out.groupby(
            status_col,
            dropna=False,
            sort=True,
        )
        .size()
        .rename("n")
        .reset_index()
        .rename(columns={status_col: "status"})
    )

    complete_groups = group_summary["group_status"].eq("complete")

    finite_shift = pd.to_numeric(
        group_summary["shift"],
        errors="coerce",
    )

    overview = pd.DataFrame(
        {
            "object_name": [name],
            "recalibration_method": [method],
            "x_col": [x_col],
            "y_col": [y_col],
            "target_x_col": [target_x_col],
            "target_y_col": [target_y_col],
            "time_col": [_collapse(time_col)],
            "grouping_cols": [_collapse(grouping_cols)],
            "calibration_col": [_collapse(calibration_col)],
            "calibration_value": [_collapse(calibration_value)],
            "n_input_rows": [int(len(df))],
            "n_groups": [int(len(group_summary))],
            "n_complete_groups": [int(complete_groups.sum())],
            "n_problem_groups": [int((~complete_groups).sum())],
            "n_recalibrated_rows": [int(out[status_col].eq("complete").sum())],
            "n_problem_rows": [int(out[status_col].ne("complete").sum())],
            "min_valid_points": [int(min_valid_points)],
            "max_shift": [(float(max_shift) if max_shift is not None else np.nan)],
            "mean_shift": [(float(finite_shift.mean()) if finite_shift.notna().any() else np.nan)],
            "max_observed_shift": [
                (float(finite_shift.max()) if finite_shift.notna().any() else np.nan)
            ],
        }
    )

    settings = _settings(
        [
            "x_col",
            "y_col",
            "target_x_col",
            "target_y_col",
            "time_col",
            "grouping_cols",
            "calibration_col",
            "calibration_value",
            "method",
            "min_valid_points",
            "max_shift",
            "output_x_col",
            "output_y_col",
            "dx_col",
            "dy_col",
            "shift_col",
            "error_before_col",
            "error_after_col",
            "status_col",
            "overwrite",
            "name",
        ],
        [
            x_col,
            y_col,
            target_x_col,
            target_y_col,
            _collapse(time_col),
            _collapse(grouping_cols),
            _collapse(calibration_col),
            _collapse(calibration_value),
            method,
            str(int(min_valid_points)),
            (str(max_shift) if max_shift is not None else np.nan),
            output_x_col,
            output_y_col,
            dx_col,
            dy_col,
            shift_col,
            error_before_col,
            error_after_col,
            status_col,
            str(bool(overwrite)).upper(),
            name,
        ],
    )

    out.attrs["r_class"] = "gp3_gaze_recalibrated_data|tbl_df|tbl|data.frame"

    out.attrs["gp3_gaze_recalibration_overview"] = overview

    out.attrs["gp3_gaze_recalibration_group_summary"] = group_summary

    out.attrs["gp3_gaze_recalibration_status_summary"] = status_summary

    out.attrs["gp3_gaze_recalibration_settings"] = settings

    return out


# === R2 ORACLE REPAIR LAYER v2 ===


def _r2_fmt_number(value):
    if value is None:
        return ""

    number = float(value)

    if number.is_integer():
        return str(int(number))

    return format(number, "g")


# ---------------------------------------------------------------------------
# GAZE QUALITY
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PREPROCESSING MULTIVERSE
# ---------------------------------------------------------------------------

_create_preprocessing_multiverse_before_r2_repair = create_preprocessing_multiverse


def create_preprocessing_multiverse(
    *args,
    **kwargs,
):
    result = _create_preprocessing_multiverse_before_r2_repair(
        *args,
        **kwargs,
    )

    pupil = result["pupil_grid"].copy()

    if not pupil.empty:
        pupil = pupil.rename(
            columns={
                "baseline_start_ms": "baseline_window_start_ms",
                "baseline_end_ms": "baseline_window_end_ms",
            }
        )

        pupil["baseline_window_label"] = [
            (f"{_r2_fmt_number(start)}_to_{_r2_fmt_number(end)}ms")
            for start, end in zip(
                pupil["baseline_window_start_ms"],
                pupil["baseline_window_end_ms"],
                strict=False,
            )
        ]

        pupil["branch_label"] = [
            (
                f"pupil_gap"
                f"{_r2_fmt_number(gap)}"
                f"_smooth"
                f"{_r2_fmt_number(smooth)}"
                f"_baseline"
                f"{baseline_label}"
                f"_pad"
                f"{_r2_fmt_number(padding)}"
            )
            for (
                gap,
                smooth,
                baseline_label,
                padding,
            ) in zip(
                pupil["max_gap_ms"],
                pupil["smoothing_window_samples"],
                pupil["baseline_window_label"],
                pupil["artifact_padding_ms"],
                strict=False,
            )
        ]

        pupil = pupil[
            [
                "branch_id",
                "branch_label",
                "preprocessing_family",
                "decision_type",
                "artifact_padding_ms",
                "max_gap_ms",
                "smoothing_window_samples",
                "baseline_window_start_ms",
                "baseline_window_end_ms",
                "baseline_window_label",
                "branch_status",
            ]
        ]

    result["pupil_grid"] = pupil

    aoi = result["aoi_grid"].copy()

    combined_rows = []

    if not pupil.empty and not aoi.empty:
        index = 0

        for _, p_row in pupil.iterrows():
            for _, a_row in aoi.iterrows():
                index += 1

                combined_rows.append(
                    {
                        "combined_branch_id": (
                            f"{kwargs.get('label_prefix', 'gp3')}_combined_{index}"
                        ),
                        "pupil_branch_id": p_row["branch_id"],
                        "pupil_branch_label": p_row["branch_label"],
                        "aoi_branch_id": a_row["branch_id"],
                        "aoi_branch_label": a_row["branch_label"],
                        "branch_status": "defined",
                    }
                )

    combined = pd.DataFrame(
        combined_rows,
        columns=[
            "combined_branch_id",
            "pupil_branch_id",
            "pupil_branch_label",
            "aoi_branch_id",
            "aoi_branch_label",
            "branch_status",
        ],
    )

    result["combined_grid"] = combined

    overview = pd.DataFrame(
        {
            "include_pupil": [
                bool(
                    kwargs.get(
                        "include_pupil",
                        True,
                    )
                )
            ],
            "include_aoi": [
                bool(
                    kwargs.get(
                        "include_aoi",
                        True,
                    )
                )
            ],
            "n_pupil_branches": [int(len(pupil))],
            "n_aoi_branches": [int(len(aoi))],
            "n_combined_branches": [int(len(combined))],
            "multiverse_status": ["defined"],
        }
    )

    result["overview"] = overview

    settings = result["settings"].copy()

    baseline_windows = kwargs.get(
        "pupil_baseline_windows",
        ((-200, 0),),
    )

    baseline_labels = []

    for window in baseline_windows:
        baseline_labels.append("[" + ", ".join(_r2_fmt_number(value) for value in window) + "]")

    settings.loc[
        settings["setting"].eq("pupil_baseline_windows"),
        "value",
    ] = "; ".join(baseline_labels)

    result["settings"] = settings

    return result


# ---------------------------------------------------------------------------
# FIXATION ALIGNMENT
# ---------------------------------------------------------------------------

_prepare_fixation_aligned_data_before_r2_repair = prepare_fixation_aligned_data


# ---------------------------------------------------------------------------
# MULTIMODAL SETTINGS NORMALIZATION
# ---------------------------------------------------------------------------

_prepare_multimodal_data_before_r2_repair = prepare_multimodal_data


def _r2_scalar_character_setting(
    value,
):
    if value is None:
        return "<EMPTY>"

    if isinstance(
        value,
        str,
    ):
        return value

    if isinstance(
        value,
        (
            list,
            tuple,
            np.ndarray,
            pd.Index,
        ),
    ):
        values = list(value)

        if len(values) == 0:
            return "<EMPTY>"

        if len(values) == 1:
            return str(values[0])

        return values

    return value


def prepare_multimodal_data(
    *args,
    **kwargs,
):
    result = _prepare_multimodal_data_before_r2_repair(
        *args,
        **kwargs,
    )

    settings = dict(result.attrs["gp3_multimodal_settings"])

    settings["gaze_by"] = _r2_scalar_character_setting(settings.get("gaze_by"))

    settings["response_by"] = _r2_scalar_character_setting(settings.get("response_by"))

    settings["outcome_cols"] = _r2_scalar_character_setting(settings.get("outcome_cols"))

    settings["covariate_cols"] = _r2_scalar_character_setting(settings.get("covariate_cols"))

    result.attrs["gp3_multimodal_settings"] = settings

    return result


# ---------------------------------------------------------------------------
# RECALIBRATION LOGICAL STRING NORMALIZATION
# ---------------------------------------------------------------------------

_recalibrate_gaze_before_r2_repair = recalibrate_gaze


# === R2 ORACLE REPAIR LAYER v3 ===


# ===========================================================================
# EXACT R 2.3.0 GAZE-SIGNAL QUALITY SEMANTICS
# ===========================================================================


def _r2_gaze_standardise_name(name):
    if name == "MEDIA_ID":
        return "media_id"
    if name == "USER_FILE":
        return "subject"
    return name


def _r2_gaze_cols(value):
    if value is None:
        return []

    if isinstance(value, str):
        values = [value]
    else:
        values = list(value)

    return [_r2_gaze_standardise_name(str(item)) for item in values]


def _r2_gaze_collapse_nullable(value):
    if value is None:
        return np.nan

    if isinstance(value, str):
        if value == "":
            return np.nan
        return value

    if isinstance(
        value,
        (
            list,
            tuple,
            np.ndarray,
            pd.Index,
        ),
    ):
        values = list(value)

        if len(values) == 0:
            return np.nan

        return ", ".join(str(item) for item in values)

    if pd.isna(value):
        return np.nan

    return str(value)


def _r2_gaze_setting_number(value):
    number = float(value)

    if number.is_integer():
        return str(int(number))

    return format(
        number,
        ".15g",
    )


def _r2_gaze_range_text(value):
    return ", ".join(_r2_gaze_setting_number(x) for x in value)


def _r2_gaze_validity_vector(series):
    values = pd.Series(
        series,
        copy=False,
    )

    if pd.api.types.is_bool_dtype(values.dtype):
        return values.astype("boolean")

    if pd.api.types.is_numeric_dtype(values.dtype):
        numbers = pd.to_numeric(
            values,
            errors="coerce",
        )

        out = pd.Series(
            pd.NA,
            index=values.index,
            dtype="boolean",
        )

        out.loc[numbers.notna() & numbers.gt(0)] = True

        out.loc[numbers.notna() & numbers.le(0)] = False

        return out

    text = values.astype("string").str.lower()

    out = pd.Series(
        pd.NA,
        index=values.index,
        dtype="boolean",
    )

    true_values = {
        "true",
        "t",
        "yes",
        "y",
        "1",
        "valid",
        "ok",
    }

    false_values = {
        "false",
        "f",
        "no",
        "n",
        "0",
        "invalid",
        "missing",
        "bad",
    }

    out.loc[text.isin(true_values)] = True

    out.loc[text.isin(false_values)] = False

    return out


def _r2_gaze_status(
    *,
    has_xy,
    has_validity,
    has_pupil,
    gaze_valid_prop,
    missing_gaze_prop,
    offscreen_prop,
    pupil_valid_prop,
    min_gaze_valid_prop,
    max_missing_gaze_prop,
    max_offscreen_prop,
    min_pupil_valid_prop,
):
    if not has_xy and not has_validity:
        return "gaze_columns_not_available"

    if pd.notna(gaze_valid_prop) and gaze_valid_prop < min_gaze_valid_prop:
        return "low_gaze_validity"

    if pd.notna(missing_gaze_prop) and missing_gaze_prop > max_missing_gaze_prop:
        return "high_missing_gaze"

    if pd.notna(offscreen_prop) and offscreen_prop > max_offscreen_prop:
        return "high_offscreen_gaze"

    if has_pupil and pd.notna(pupil_valid_prop) and pupil_valid_prop < min_pupil_valid_prop:
        return "low_pupil_validity"

    return "ok"


def _r2_gaze_mean(series):
    values = pd.to_numeric(
        series,
        errors="coerce",
    )

    finite = values[np.isfinite(values.to_numpy(dtype=float))]

    if len(finite) == 0:
        return np.nan

    return float(finite.mean())


def _r2_gaze_aggregate_summary(
    unit_summary,
    group_col,
    status_col,
):
    columns = [
        group_col,
        "n_units",
        "n_flagged_units",
        "mean_gaze_valid_prop",
        "mean_missing_gaze_prop",
        "mean_offscreen_prop",
        "mean_pupil_valid_prop",
        status_col,
    ]

    if group_col is None or group_col not in unit_summary.columns:
        return pd.DataFrame(columns=columns)

    rows = []

    for value, part in unit_summary.groupby(
        group_col,
        dropna=False,
        sort=True,
    ):
        flagged = int(part["gaze_signal_status"].ne("ok").sum())

        rows.append(
            {
                group_col: value,
                "n_units": int(len(part)),
                "n_flagged_units": flagged,
                "mean_gaze_valid_prop": _r2_gaze_mean(part["gaze_valid_prop"]),
                "mean_missing_gaze_prop": _r2_gaze_mean(part["missing_gaze_prop"]),
                "mean_offscreen_prop": _r2_gaze_mean(part["offscreen_prop"]),
                "mean_pupil_valid_prop": _r2_gaze_mean(part["pupil_valid_prop"]),
                status_col: ("review" if flagged > 0 else "ok"),
            }
        )

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def audit_gaze_signal_quality(
    data,
    subject_col="subject",
    condition_col=None,
    group_cols=(
        "subject",
        "media_id",
        "trial_global",
    ),
    x_col=None,
    y_col=None,
    validity_cols=None,
    pupil_col=None,
    screen_x_range=(0, 1),
    screen_y_range=(0, 1),
    min_gaze_valid_prop=0.7,
    max_missing_gaze_prop=0.3,
    max_offscreen_prop=0.3,
    min_pupil_valid_prop=0.7,
):
    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise ValueError("`data` must be a data frame.")

    if len(data) == 0:
        raise ValueError("`data` must contain at least one row.")

    frame = data.copy()

    # R .gp3_gaze_signal_standardise_aliases()
    if "MEDIA_ID" in frame.columns and "media_id" not in frame.columns:
        frame["media_id"] = frame["MEDIA_ID"]

    if "USER_FILE" in frame.columns and "subject" not in frame.columns:
        frame["subject"] = frame["USER_FILE"]

    subject_col = _r2_gaze_standardise_name(subject_col)

    if subject_col not in frame.columns:
        raise ValueError("`subject_col` must be present in `data`.")

    if condition_col is not None:
        condition_col = _r2_gaze_standardise_name(condition_col)

        if condition_col not in frame.columns:
            raise ValueError("`condition_col` must be present in `data`.")

    groups = _r2_gaze_cols(group_cols)

    groups = [col for col in groups if col in frame.columns]

    if len(groups) == 0:
        raise ValueError("At least one usable `group_cols` column must be present in `data`.")

    def resolve_or_detect(
        supplied,
        candidates,
    ):
        if supplied is not None:
            supplied = _r2_gaze_standardise_name(supplied)

            if supplied not in frame.columns:
                raise ValueError(f"Column {supplied!r} must be present in data.")

            return supplied

        for candidate in candidates:
            if candidate in frame.columns:
                return candidate

        return None

    x_col = resolve_or_detect(
        x_col,
        [
            "gaze_x",
            "x",
            "X",
            "FPOGX",
            "BPOGX",
            "LPOGX",
            "RPOGX",
        ],
    )

    y_col = resolve_or_detect(
        y_col,
        [
            "gaze_y",
            "y",
            "Y",
            "FPOGY",
            "BPOGY",
            "LPOGY",
            "RPOGY",
        ],
    )

    pupil_col = resolve_or_detect(
        pupil_col,
        [
            "mean_pupil",
            "pupil",
            "pupil_clean",
            "pupil_smoothed",
            "LPMM",
            "RPMM",
            "LPD",
            "RPD",
        ],
    )

    if validity_cols is None:
        validity = [
            candidate
            for candidate in [
                "valid_gaze",
                "gaze_valid",
                "gaze_validity",
                "FPOGV",
                "BPOGV",
                "LPOGV",
                "RPOGV",
                "LPV",
                "RPV",
            ]
            if candidate in frame.columns
        ]
    else:
        validity = (
            [validity_cols]
            if isinstance(
                validity_cols,
                str,
            )
            else list(validity_cols)
        )

        missing = [col for col in validity if col not in frame.columns]

        if missing:
            raise ValueError("All `validity_cols` must be present in `data`.")

    screen_x_range = tuple(float(x) for x in screen_x_range)

    screen_y_range = tuple(float(x) for x in screen_y_range)

    if (
        len(screen_x_range) != 2
        or not all(np.isfinite(screen_x_range))
        or screen_x_range[0] >= screen_x_range[1]
    ):
        raise ValueError("`screen_x_range` must be an increasing finite numeric range.")

    if (
        len(screen_y_range) != 2
        or not all(np.isfinite(screen_y_range))
        or screen_y_range[0] >= screen_y_range[1]
    ):
        raise ValueError("`screen_y_range` must be an increasing finite numeric range.")

    for value, name in [
        (
            min_gaze_valid_prop,
            "min_gaze_valid_prop",
        ),
        (
            max_missing_gaze_prop,
            "max_missing_gaze_prop",
        ),
        (
            max_offscreen_prop,
            "max_offscreen_prop",
        ),
        (
            min_pupil_valid_prop,
            "min_pupil_valid_prop",
        ),
    ]:
        if not np.isfinite(float(value)) or float(value) < 0 or float(value) > 1:
            raise ValueError(f"`{name}` must be between 0 and 1.")

    has_xy = x_col is not None and y_col is not None

    has_validity = len(validity) > 0

    has_pupil = pupil_col is not None

    # R:
    # id_cols <- unique(
    #     c(group_cols, condition_col)
    # )
    id_cols = list(dict.fromkeys(groups + ([condition_col] if condition_col is not None else [])))

    unit_rows = []

    grouper = id_cols[0] if len(id_cols) == 1 else id_cols

    for keys, part in frame.groupby(
        grouper,
        dropna=False,
        sort=True,
    ):
        if len(id_cols) == 1:
            keys = (keys,)

        row = dict(
            zip(
                id_cols,
                keys,
                strict=True,
            )
        )

        n_samples = int(len(part))

        # Coordinate availability
        if has_xy:
            x = pd.to_numeric(
                part[x_col],
                errors="coerce",
            ).to_numpy(dtype=float)

            y = pd.to_numeric(
                part[y_col],
                errors="coerce",
            ).to_numpy(dtype=float)

            missing_gaze = ~np.isfinite(x) | ~np.isfinite(y)

            offscreen_gaze = (
                np.isfinite(x)
                & np.isfinite(y)
                & (
                    (x < screen_x_range[0])
                    | (x > screen_x_range[1])
                    | (y < screen_y_range[0])
                    | (y > screen_y_range[1])
                )
            )

            valid_gaze = (~missing_gaze & ~offscreen_gaze).astype(object)

        else:
            missing_gaze = np.full(
                n_samples,
                np.nan,
                dtype=object,
            )

            offscreen_gaze = np.full(
                n_samples,
                np.nan,
                dtype=object,
            )

            valid_gaze = np.full(
                n_samples,
                np.nan,
                dtype=object,
            )

        # R validity-column override:
        # any TRUE across validity columns,
        # then AND coordinate availability.
        if has_validity:
            vectors = []

            for col in validity:
                vector = _r2_gaze_validity_vector(part[col]).reset_index(drop=True)

                vectors.append(vector)

            matrix = pd.concat(
                vectors,
                axis=1,
            )

            valid_from_cols = matrix.fillna(False).astype(bool).any(axis=1)

            validity_available = matrix.notna().any(axis=1)

            valid_series = pd.Series(
                pd.NA,
                index=range(n_samples),
                dtype="boolean",
            )

            valid_series.loc[validity_available] = valid_from_cols.loc[
                validity_available
            ].to_numpy()

            if has_xy:
                coordinate_valid = ~missing_gaze & ~offscreen_gaze

                available = valid_series.notna()

                valid_series.loc[available] = (
                    valid_series.loc[available].astype(bool).to_numpy()
                    & coordinate_valid[available.to_numpy()]
                )

            valid_gaze = valid_series

        if has_pupil:
            pupil = pd.to_numeric(
                part[pupil_col],
                errors="coerce",
            ).to_numpy(dtype=float)

            pupil_valid = np.isfinite(pupil) & (pupil > 0)
        else:
            pupil_valid = np.full(
                n_samples,
                np.nan,
                dtype=object,
            )

        def count_true(values):
            series = pd.Series(values)

            return int(series.eq(True).fillna(False).sum())

        def has_observed(values):
            return bool(pd.Series(values).notna().any())

        n_valid_gaze = count_true(valid_gaze)

        n_missing_gaze = count_true(missing_gaze)

        n_offscreen_gaze = count_true(offscreen_gaze)

        n_valid_pupil = count_true(pupil_valid)

        gaze_valid_prop = n_valid_gaze / n_samples if has_observed(valid_gaze) else np.nan

        missing_gaze_prop = n_missing_gaze / n_samples if has_observed(missing_gaze) else np.nan

        offscreen_prop = n_offscreen_gaze / n_samples if has_observed(offscreen_gaze) else np.nan

        pupil_valid_prop = n_valid_pupil / n_samples if has_observed(pupil_valid) else np.nan

        status = _r2_gaze_status(
            has_xy=has_xy,
            has_validity=has_validity,
            has_pupil=has_pupil,
            gaze_valid_prop=(gaze_valid_prop),
            missing_gaze_prop=(missing_gaze_prop),
            offscreen_prop=(offscreen_prop),
            pupil_valid_prop=(pupil_valid_prop),
            min_gaze_valid_prop=float(min_gaze_valid_prop),
            max_missing_gaze_prop=float(max_missing_gaze_prop),
            max_offscreen_prop=float(max_offscreen_prop),
            min_pupil_valid_prop=float(min_pupil_valid_prop),
        )

        row.update(
            {
                "n_samples": n_samples,
                "n_valid_gaze": n_valid_gaze,
                "gaze_valid_prop": gaze_valid_prop,
                "n_missing_gaze": n_missing_gaze,
                "missing_gaze_prop": missing_gaze_prop,
                "n_offscreen_gaze": n_offscreen_gaze,
                "offscreen_prop": offscreen_prop,
                "n_valid_pupil": (n_valid_pupil if has_pupil else np.nan),
                "pupil_valid_prop": pupil_valid_prop,
                "gaze_signal_status": status,
            }
        )

        unit_rows.append(row)

    unit_columns = id_cols + [
        "n_samples",
        "n_valid_gaze",
        "gaze_valid_prop",
        "n_missing_gaze",
        "missing_gaze_prop",
        "n_offscreen_gaze",
        "offscreen_prop",
        "n_valid_pupil",
        "pupil_valid_prop",
        "gaze_signal_status",
    ]

    unit_summary = pd.DataFrame(
        unit_rows,
        columns=unit_columns,
    )

    subject_summary = _r2_gaze_aggregate_summary(
        unit_summary,
        subject_col,
        "subject_signal_status",
    )

    condition_summary = (
        _r2_gaze_aggregate_summary(
            unit_summary,
            condition_col,
            "condition_signal_status",
        )
        if condition_col is not None
        else pd.DataFrame(
            columns=[
                "condition",
                "n_units",
                "n_flagged_units",
                "mean_gaze_valid_prop",
                "mean_missing_gaze_prop",
                "mean_offscreen_prop",
                "mean_pupil_valid_prop",
                "condition_signal_status",
            ]
        )
    )

    counts = unit_summary["gaze_signal_status"].value_counts(sort=False)

    statuses = sorted(counts.index.astype(str).tolist())

    signal_issue_summary = pd.DataFrame(
        {
            "gaze_signal_status": statuses,
            "n_units": [int(counts.loc[status]) for status in statuses],
        }
    )

    signal_issue_summary["unit_prop"] = signal_issue_summary["n_units"] / len(unit_summary)

    flagged_units = (
        unit_summary.loc[unit_summary["gaze_signal_status"].ne("ok")].copy().reset_index(drop=True)
    )

    overview = pd.DataFrame(
        [
            {
                "n_rows": int(len(frame)),
                "n_units": int(len(unit_summary)),
                "n_subjects": int(unit_summary[subject_col].nunique(dropna=False)),
                "n_flagged_units": int(len(flagged_units)),
                "x_col": _r2_gaze_collapse_nullable(x_col),
                "y_col": _r2_gaze_collapse_nullable(y_col),
                "validity_cols": _r2_gaze_collapse_nullable(validity),
                "pupil_col": _r2_gaze_collapse_nullable(pupil_col),
                "has_gaze_coordinates": bool(has_xy),
                "has_validity_cols": bool(has_validity),
                "has_pupil_col": bool(has_pupil),
                "gaze_signal_quality_status": (
                    "gaze_columns_not_available"
                    if (x_col is None and y_col is None and not has_validity)
                    else ("ok" if len(flagged_units) == 0 else "review")
                ),
            }
        ]
    )

    settings = pd.DataFrame(
        {
            "setting": [
                "subject_col",
                "condition_col",
                "group_cols",
                "x_col",
                "y_col",
                "validity_cols",
                "pupil_col",
                "screen_x_range",
                "screen_y_range",
                "min_gaze_valid_prop",
                "max_missing_gaze_prop",
                "max_offscreen_prop",
                "min_pupil_valid_prop",
            ],
            "value": [
                subject_col,
                _r2_gaze_collapse_nullable(condition_col),
                ", ".join(groups),
                _r2_gaze_collapse_nullable(x_col),
                _r2_gaze_collapse_nullable(y_col),
                _r2_gaze_collapse_nullable(validity),
                _r2_gaze_collapse_nullable(pupil_col),
                _r2_gaze_range_text(screen_x_range),
                _r2_gaze_range_text(screen_y_range),
                _r2_gaze_setting_number(min_gaze_valid_prop),
                _r2_gaze_setting_number(max_missing_gaze_prop),
                _r2_gaze_setting_number(max_offscreen_prop),
                _r2_gaze_setting_number(min_pupil_valid_prop),
            ],
        }
    )

    return {
        "overview": overview,
        "unit_summary": unit_summary,
        "subject_summary": subject_summary,
        "condition_summary": condition_summary,
        "signal_issue_summary": signal_issue_summary,
        "flagged_units": flagged_units,
        "settings": settings,
        "_gp3_class": "gp3_gaze_signal_quality_audit",
    }


# ===========================================================================
# FIXATION ALIGNMENT: R LOGICAL NA + EXACT EVENT TABLE
# ===========================================================================


def prepare_fixation_aligned_data(
    data,
    *args,
    **kwargs,
):
    # Call the original R2 implementation directly.
    # The v2 wrapper cannot be called because pandas bool dtype
    # rejects insertion of NA.
    result = _prepare_fixation_aligned_data_before_r2_repair(
        data,
        *args,
        **kwargs,
    )

    aligned = result["aligned_data"].copy()

    # Convert before inserting R logical NA.
    aligned["gp3_is_alignment_event_row"] = aligned["gp3_is_alignment_event_row"].astype(object)

    unaligned = ~aligned["gp3_has_alignment_event"].fillna(False).astype(bool)

    aligned.loc[
        unaligned,
        "gp3_is_alignment_event_row",
    ] = np.nan

    result["aligned_data"] = aligned

    event_table = result["event_table"].copy()

    participant_col = kwargs.get("participant_col")

    trial_col = kwargs.get("trial_col")

    time_col = kwargs.get("time_col")

    n_rows_values = []
    start_values = []
    end_values = []

    for _, row in event_table.iterrows():
        mask = pd.Series(
            True,
            index=aligned.index,
        )

        if participant_col is not None and participant_col in aligned.columns:
            mask &= aligned[participant_col].astype("string") == str(row["gp3_participant"])

        if trial_col is not None and trial_col in aligned.columns:
            mask &= aligned[trial_col].astype("string") == str(row["gp3_trial"])

        subset = aligned.loc[mask]

        n_rows_values.append(int(len(subset)))

        if time_col is not None and time_col in subset.columns and len(subset):
            times = pd.to_numeric(
                subset[time_col],
                errors="coerce",
            )

            finite = times[np.isfinite(times.to_numpy(dtype=float))]

            if len(finite):
                start_values.append(float(finite.min()))

                end_values.append(float(finite.max()))
            else:
                start_values.append(np.nan)

                end_values.append(np.nan)
        else:
            start_values.append(np.nan)

            end_values.append(np.nan)

    event_table["gp3_n_rows"] = n_rows_values

    event_table["gp3_start_time"] = start_values

    event_table["gp3_end_time"] = end_values

    event_table["gp3_pre_event_n_samples"] = event_table["gp3_pre_event_n"]

    event_table["gp3_pre_event_target_n_samples"] = event_table["gp3_pre_event_target_n"]

    denominator = pd.to_numeric(
        event_table["gp3_pre_event_n_samples"],
        errors="coerce",
    )

    numerator = pd.to_numeric(
        event_table["gp3_pre_event_target_n_samples"],
        errors="coerce",
    )

    event_table["gp3_pre_event_target_prop"] = np.where(
        denominator > 0,
        numerator / denominator,
        np.nan,
    )

    event_table["gp3_post_event_n_samples"] = event_table["gp3_post_event_n"]

    event_table = event_table[
        [
            "gp3_group_id",
            "gp3_participant",
            "gp3_trial",
            "gp3_alignment_event",
            "gp3_has_alignment_event",
            "gp3_alignment_time",
            "gp3_alignment_row_id",
            "gp3_event_aoi",
            "gp3_event_is_target_aoi",
            "gp3_event_is_fixation",
            "gp3_event_is_saccade",
            "gp3_n_rows",
            "gp3_start_time",
            "gp3_end_time",
            "gp3_pre_event_n_samples",
            "gp3_pre_event_target_n_samples",
            "gp3_pre_event_target_prop",
            "gp3_post_event_n_samples",
            "gp3_target_present_before_event",
            "gp3_fixation_to_target_before_event",
            "gp3_already_on_target_at_trial_start",
        ]
    ]

    result["event_table"] = event_table

    trial_summary = result["trial_summary"].copy()

    result["trial_summary"] = trial_summary[
        [
            "gp3_has_alignment_event",
            "n_groups",
            "n_with_pre_event_target",
            "n_already_on_target_at_start",
            "median_alignment_time",
        ]
    ]

    return result


# ===========================================================================
# RECALIBRATION: R LOGICAL TEXT
# ===========================================================================


def _r2_setting_text_v3(value):
    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        return "TRUE" if bool(value) else "FALSE"

    if isinstance(
        value,
        str,
    ):
        normalized = value.strip().lower()

        if normalized == "true":
            return "TRUE"

        if normalized == "false":
            return "FALSE"

    return value


def recalibrate_gaze(
    *args,
    **kwargs,
):
    result = _recalibrate_gaze_before_r2_repair(
        *args,
        **kwargs,
    )

    overview = result.attrs["gp3_gaze_recalibration_overview"].copy()

    if "calibration_value" in overview.columns:
        overview["calibration_value"] = overview["calibration_value"].astype(object)

        overview["calibration_value"] = overview["calibration_value"].map(_r2_setting_text_v3)

    result.attrs["gp3_gaze_recalibration_overview"] = overview

    settings = result.attrs["gp3_gaze_recalibration_settings"].copy()

    settings["value"] = settings["value"].astype(object)

    mask = settings["setting"].eq("calibration_value")

    settings.loc[
        mask,
        "value",
    ] = settings.loc[
        mask,
        "value",
    ].map(_r2_setting_text_v3)

    result.attrs["gp3_gaze_recalibration_settings"] = settings

    return result


# === R2 ORACLE CLASS METADATA FIX v4 ===


_audit_gaze_signal_quality_before_r2_class_fix = audit_gaze_signal_quality


class _R2ClassedList(dict):
    """Dictionary carrying R-list class metadata without extra elements."""

    def __init__(
        self,
        values,
        r_class,
    ):
        super().__init__(values)

        # Support the metadata conventions used throughout
        # the Python port without adding an eighth list element.
        self.r_class = r_class
        self._r_class = r_class
        self.__r_class__ = r_class
        self._gp3_class = r_class

        self.attrs = {"r_class": r_class}


def audit_gaze_signal_quality(
    *args,
    **kwargs,
):
    result = _audit_gaze_signal_quality_before_r2_class_fix(
        *args,
        **kwargs,
    )

    values = dict(result)

    values.pop(
        "_gp3_class",
        None,
    )

    return _R2ClassedList(
        values,
        ("gp3_gaze_signal_quality_audit|list"),
    )


# === R2 FINAL FOUR ORACLE REPAIR v5 ===


# ===========================================================================
# SHARED R2 FINAL-FOUR HELPERS
# ===========================================================================


def _r2_v5_chars(value):
    if value is None:
        return []

    if isinstance(
        value,
        str,
    ):
        return [value]

    return [str(item) for item in value]


def _r2_v5_num_text(value):
    if value is None:
        return np.nan

    try:
        number = float(value)
    except (
        TypeError,
        ValueError,
    ):
        return str(value)

    if not np.isfinite(number):
        return np.nan

    if number.is_integer():
        return str(int(number))

    return format(
        number,
        ".15g",
    )


def _r2_v5_first_unique(values):
    out = []

    for value in values:
        if value not in out:
            out.append(value)

    return out


# ===========================================================================
# audit_gazepoint_event_sync
# ===========================================================================


def _r2_event_sync_resolve_col(
    col,
    names_data,
    arg,
):
    if (
        not isinstance(
            col,
            str,
        )
        or col == ""
    ):
        raise ValueError(f"`{arg}` must be a non-missing character scalar.")

    if col == "MEDIA_ID" and "media_id" in names_data:
        return "media_id"

    if col == "USER_FILE" and "subject" in names_data:
        return "subject"

    if col not in names_data:
        raise ValueError(f"`{arg}` must be present in `data`.")

    return col


def _r2_event_sync_status(
    *,
    n_samples,
    n_finite_time,
    has_event_col,
    n_events,
    n_missing_expected,
    onset_count,
    response_count,
    n_duplicate_time,
    has_large_gap,
    min_samples_per_unit,
):
    if n_samples < min_samples_per_unit:
        return "too_few_samples"

    if n_finite_time == 0:
        return "missing_time"

    if n_duplicate_time > 0:
        return "duplicate_time_values"

    if has_large_gap:
        return "large_time_gap"

    if not has_event_col:
        return "event_column_not_available"

    if n_events == 0:
        return "no_events_observed"

    if n_missing_expected > 0:
        return "missing_expected_events"

    if onset_count is not None and not pd.isna(onset_count) and int(onset_count) == 0:
        return "missing_onset_event"

    if response_count is not None and not pd.isna(response_count) and int(response_count) == 0:
        return "missing_response_event"

    return "ok"


def audit_event_sync(
    data,
    time_col="time",
    event_col=None,
    group_cols=(
        "subject",
        "media_id",
        "trial_global",
    ),
    condition_col=None,
    expected_event_labels=None,
    onset_event_label=None,
    response_event_label=None,
    min_samples_per_unit=1,
    max_time_gap_ms=None,
):
    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise ValueError("`data` must be a data frame.")

    if len(data) == 0:
        raise ValueError("`data` must contain at least one row.")

    original_names = list(data.columns)

    time_col = _r2_event_sync_resolve_col(
        time_col,
        original_names,
        "time_col",
    )

    if condition_col is not None:
        condition_col = _r2_event_sync_resolve_col(
            condition_col,
            original_names,
            "condition_col",
        )

    frame = data.copy()

    if "MEDIA_ID" in frame.columns and "media_id" not in frame.columns:
        frame["media_id"] = frame["MEDIA_ID"]

    if "USER_FILE" in frame.columns and "subject" not in frame.columns:
        frame["subject"] = frame["USER_FILE"]

    groups = _r2_v5_chars(group_cols)

    groups = [
        ("media_id" if col == "MEDIA_ID" else ("subject" if col == "USER_FILE" else col))
        for col in groups
    ]

    groups = [col for col in groups if col in frame.columns]

    if len(groups) == 0:
        raise ValueError("At least one usable `group_cols` column must be present in `data`.")

    if event_col is not None:
        event_col = _r2_event_sync_resolve_col(
            event_col,
            list(frame.columns),
            "event_col",
        )
    else:
        candidates = [
            "event",
            "event_label",
            "event_name",
            "EVENT",
            "EVENT_LABEL",
            "Event",
            "EventLabel",
            "marker",
            "MARKER",
            "message",
            "MESSAGE",
        ]

        event_col = next(
            (candidate for candidate in candidates if candidate in frame.columns),
            None,
        )

    if (
        isinstance(
            min_samples_per_unit,
            (
                bool,
                np.bool_,
            ),
        )
        or not isinstance(
            min_samples_per_unit,
            (
                int,
                float,
                np.number,
            ),
        )
        or not np.isfinite(min_samples_per_unit)
        or min_samples_per_unit <= 0
    ):
        raise ValueError("`min_samples_per_unit` must be a positive numeric scalar.")

    if max_time_gap_ms is not None:
        if (
            isinstance(
                max_time_gap_ms,
                (
                    bool,
                    np.bool_,
                ),
            )
            or not isinstance(
                max_time_gap_ms,
                (
                    int,
                    float,
                    np.number,
                ),
            )
            or not np.isfinite(max_time_gap_ms)
            or max_time_gap_ms <= 0
        ):
            raise ValueError("`max_time_gap_ms` must be a positive numeric scalar.")

    expected = None

    if expected_event_labels is not None:
        expected = _r2_v5_chars(expected_event_labels)

        if len(expected) == 0 or any(item == "" for item in expected):
            raise ValueError("`expected_event_labels` must be a non-empty character vector.")

    for value, name in [
        (
            onset_event_label,
            "onset_event_label",
        ),
        (
            response_event_label,
            "response_event_label",
        ),
    ]:
        if value is not None and (
            not isinstance(
                value,
                str,
            )
            or value == ""
        ):
            raise ValueError(f"`{name}` must be a non-missing character scalar.")

    unit_rows = []

    grouper = groups[0] if len(groups) == 1 else groups

    for keys, block in frame.groupby(
        grouper,
        dropna=True,
        sort=True,
    ):
        if len(groups) == 1:
            keys = (keys,)

        group_row = dict(
            zip(
                groups,
                keys,
                strict=True,
            )
        )

        if condition_col is not None:
            group_row[condition_col] = block.iloc[0][condition_col]

        time_values = pd.to_numeric(
            block[time_col],
            errors="coerce",
        ).to_numpy(dtype=float)

        finite_time = time_values[np.isfinite(time_values)]

        event_values = []

        if event_col is not None:
            raw = block[event_col]

            for value in raw:
                if pd.isna(value):
                    continue

                text = str(value)

                if text != "":
                    event_values.append(text)

        event_unique = sorted(set(event_values))

        expected_missing = []

        if expected is not None and event_col is not None:
            expected_missing = [label for label in expected if label not in event_unique]

        onset_count = np.nan

        if event_col is not None and onset_event_label is not None:
            onset_count = int(sum(value == onset_event_label for value in event_values))

        response_count = np.nan

        if event_col is not None and response_event_label is not None:
            response_count = int(sum(value == response_event_label for value in event_values))

        sorted_time = np.sort(finite_time)

        if len(sorted_time) > 1:
            gaps = np.diff(sorted_time)

            max_gap = float(np.max(gaps))
        else:
            max_gap = np.nan

        has_large_gap = bool(
            max_time_gap_ms is not None
            and np.isfinite(max_gap)
            and max_gap > float(max_time_gap_ms)
        )

        finite_series = pd.Series(finite_time)

        n_duplicate_time = int(finite_series.duplicated().sum())

        status = _r2_event_sync_status(
            n_samples=int(len(block)),
            n_finite_time=int(len(finite_time)),
            has_event_col=(event_col is not None),
            n_events=len(event_values),
            n_missing_expected=len(expected_missing),
            onset_count=onset_count,
            response_count=(response_count),
            n_duplicate_time=(n_duplicate_time),
            has_large_gap=(has_large_gap),
            min_samples_per_unit=(float(min_samples_per_unit)),
        )

        group_row.update(
            {
                "n_samples": int(len(block)),
                "n_finite_time": int(len(finite_time)),
                "time_start": (float(np.min(finite_time)) if len(finite_time) else np.nan),
                "time_end": (float(np.max(finite_time)) if len(finite_time) else np.nan),
                "time_span": (
                    float(np.max(finite_time) - np.min(finite_time)) if len(finite_time) else np.nan
                ),
                "max_time_gap": max_gap,
                "n_duplicate_time": n_duplicate_time,
                "n_event_samples": int(len(event_values)),
                "n_unique_events": int(len(event_unique)),
                "event_labels": "; ".join(event_unique),
                "n_missing_expected_events": int(len(expected_missing)),
                "missing_expected_events": "; ".join(expected_missing),
                "onset_event_count": onset_count,
                "response_event_count": response_count,
                "event_sync_status": status,
            }
        )

        unit_rows.append(group_row)

    unit_columns = (
        groups
        + ([condition_col] if (condition_col is not None and condition_col not in groups) else [])
        + [
            "n_samples",
            "n_finite_time",
            "time_start",
            "time_end",
            "time_span",
            "max_time_gap",
            "n_duplicate_time",
            "n_event_samples",
            "n_unique_events",
            "event_labels",
            "n_missing_expected_events",
            "missing_expected_events",
            "onset_event_count",
            "response_event_count",
            "event_sync_status",
        ]
    )

    unit_summary = pd.DataFrame(
        unit_rows,
        columns=unit_columns,
    )

    # R table(condition, event_label):
    # first dimension varies fastest.
    if event_col is None:
        event_summary = pd.DataFrame(
            columns=[
                "event_col",
                "event_label",
                "n_event_samples",
                "event_summary_status",
            ]
        )

    else:
        observed = frame.loc[
            frame[event_col].notna() & frame[event_col].astype("string").ne("")
        ].copy()

        if len(observed) == 0:
            event_summary = pd.DataFrame(
                columns=[
                    "event_col",
                    "event_label",
                    "n_event_samples",
                    "event_summary_status",
                ]
            )

        elif condition_col is not None:
            conditions = sorted(observed[condition_col].astype(str).unique().tolist())

            events = sorted(observed[event_col].astype(str).unique().tolist())

            event_rows = []

            for event_value in events:
                for condition_value in conditions:
                    count = int(
                        (
                            observed[event_col].astype(str).eq(event_value)
                            & observed[condition_col].astype(str).eq(condition_value)
                        ).sum()
                    )

                    if count > 0:
                        event_rows.append(
                            {
                                "event_col": event_col,
                                "condition": condition_value,
                                "event_label": event_value,
                                "n_event_samples": count,
                                "event_summary_status": "ok",
                            }
                        )

            event_summary = pd.DataFrame(
                event_rows,
                columns=[
                    "event_col",
                    condition_col,
                    "event_label",
                    "n_event_samples",
                    "event_summary_status",
                ],
            )

        else:
            counts = observed[event_col].astype(str).value_counts(sort=False)

            event_rows = []

            for label in sorted(counts.index.tolist()):
                event_rows.append(
                    {
                        "event_col": event_col,
                        "event_label": label,
                        "n_event_samples": int(counts.loc[label]),
                        "event_summary_status": "ok",
                    }
                )

            event_summary = pd.DataFrame(event_rows)

    if event_col is None:
        expected_summary = pd.DataFrame(
            columns=[
                "expected_event_label",
                "n_units_missing",
                "expected_event_status",
            ]
        )

    elif expected is None:
        expected_summary = pd.DataFrame(
            columns=[
                "expected_event_label",
                "n_units_missing",
                "expected_event_status",
            ]
        )

    else:
        expected_rows = []

        for label in expected:
            missing_count = 0

            for text in unit_summary["missing_expected_events"].astype(str):
                labels = text.split("; ") if text != "" else []

                if label in labels:
                    missing_count += 1

            expected_rows.append(
                {
                    "expected_event_label": label,
                    "n_units_missing": int(missing_count),
                    "expected_event_status": (
                        "ok" if missing_count == 0 else ("missing_in_some_units")
                    ),
                }
            )

        expected_summary = pd.DataFrame(
            expected_rows,
            columns=[
                "expected_event_label",
                "n_units_missing",
                "expected_event_status",
            ],
        )

    flagged_units = (
        unit_summary.loc[unit_summary["event_sync_status"].ne("ok")].copy().reset_index(drop=True)
    )

    audit_status = (
        "event_column_not_available"
        if event_col is None
        else ("ok" if len(flagged_units) == 0 else "review")
    )

    overview = pd.DataFrame(
        [
            {
                "n_rows": int(len(frame)),
                "n_units": int(len(unit_summary)),
                "n_flagged_units": int(len(flagged_units)),
                "event_col": (event_col if event_col is not None else np.nan),
                "has_event_col": bool(event_col is not None),
                "has_expected_events": bool(expected is not None),
                "audit_status": audit_status,
            }
        ]
    )

    settings = pd.DataFrame(
        {
            "setting": [
                "time_col",
                "event_col",
                "group_cols",
                "condition_col",
                "expected_event_labels",
                "onset_event_label",
                "response_event_label",
                "min_samples_per_unit",
                "max_time_gap_ms",
            ],
            "value": [
                time_col,
                (event_col if event_col is not None else np.nan),
                ", ".join(groups),
                (condition_col if condition_col is not None else np.nan),
                (", ".join(expected) if expected is not None else np.nan),
                (onset_event_label if onset_event_label is not None else np.nan),
                (response_event_label if response_event_label is not None else np.nan),
                _r2_v5_num_text(min_samples_per_unit),
                (_r2_v5_num_text(max_time_gap_ms) if max_time_gap_ms is not None else np.nan),
            ],
        }
    )

    return _R2ClassedList(
        {
            "overview": overview,
            "unit_summary": unit_summary,
            "event_summary": event_summary,
            "expected_event_summary": expected_summary,
            "flagged_units": flagged_units,
            "settings": settings,
        },
        "gp3_event_sync_audit|list",
    )


# ===========================================================================
# audit_gazepoint_face_sync
# ===========================================================================


def _r2_face_percent(
    x,
    n,
):
    if n <= 0:
        return np.nan

    return 100.0 * float(x) / float(n)


def _r2_face_status(
    *,
    n_rows,
    matched_percent,
    max_abs_diff_sec_observed,
    min_matched_percent,
    warning_matched_percent,
    max_abs_diff_sec,
):
    if n_rows < 1:
        return "fail"

    if pd.isna(matched_percent):
        return "unknown"

    if matched_percent < min_matched_percent:
        return "fail"

    if matched_percent < warning_matched_percent:
        return "warn"

    if (
        max_abs_diff_sec is not None
        and pd.notna(max_abs_diff_sec_observed)
        and max_abs_diff_sec_observed > max_abs_diff_sec
    ):
        return "warn"

    return "pass"


def _r2_face_message(
    status,
    matched_percent,
    min_matched_percent,
    warning_matched_percent,
):
    if status == "unknown":
        return "Face-data synchronisation quality could not be evaluated."

    matched_text = format(
        round(
            float(matched_percent),
            1,
        ),
        "g",
    )

    if status == "fail":
        minimum = format(
            float(min_matched_percent),
            "g",
        )

        return (
            "Face-data synchronisation is "
            "below the minimum threshold "
            f"({matched_text}% matched; "
            f"minimum {minimum}%)."
        )

    if status == "warn":
        warning = format(
            float(warning_matched_percent),
            "g",
        )

        return (
            "Face-data synchronisation "
            "should be reviewed before "
            "analysis "
            f"({matched_text}% matched; "
            f"warning threshold "
            f"{warning}%)."
        )

    return "Face-data synchronisation passed the configured checks."


def _r2_face_subset(
    frame,
    index,
    group_cols,
    min_matched_percent,
    warning_matched_percent,
    max_abs_diff_sec,
):
    block = frame.loc[index].copy()

    n_rows = int(len(block))

    row = {}

    if group_cols:
        labels = []

        first = block.iloc[0]

        for col in group_cols:
            value = first[col]

            if pd.isna(value):
                text = "missing"
            else:
                text = str(value)

            row[col] = text

            labels.append(f"{col}={text}")

        row["face_sync_group"] = " | ".join(labels)

    else:
        row["face_sync_group"] = "overall"

    status_series = block["face_sync_status"].astype("string")

    n_matched = int(status_series.eq("matched").fillna(False).sum())

    n_unmatched = int(status_series.eq("unmatched").fillna(False).sum())

    n_outside = int(status_series.eq("outside_tolerance").fillna(False).sum())

    n_missing_time = int(status_series.eq("missing_gaze_time").fillna(False).sum())

    n_missing_frame = int(status_series.eq("missing_gaze_frame").fillna(False).sum())

    n_unknown = int((status_series.isna() | status_series.fillna("").eq("")).sum())

    within = block["face_sync_within_tolerance"]

    n_within = int(within.eq(True).fillna(False).sum())

    if "face_sync_abs_diff_sec" in block.columns:
        abs_diff = pd.to_numeric(
            block["face_sync_abs_diff_sec"],
            errors="coerce",
        ).to_numpy(dtype=float)
    else:
        abs_diff = np.full(
            n_rows,
            np.nan,
            dtype=float,
        )

    finite = abs_diff[np.isfinite(abs_diff)]

    if len(finite):
        mean_diff = float(np.mean(finite))

        median_diff = float(np.median(finite))

        p95_diff = float(
            np.quantile(
                finite,
                0.95,
                method="linear",
            )
        )

        max_diff = float(np.max(finite))
    else:
        mean_diff = np.nan
        median_diff = np.nan
        p95_diff = np.nan
        max_diff = np.nan

    if max_abs_diff_sec is None:
        n_above = np.nan
    else:
        n_above = int(np.sum(np.isfinite(abs_diff) & (abs_diff > float(max_abs_diff_sec))))

    matched_percent = _r2_face_percent(
        n_matched,
        n_rows,
    )

    audit_status = _r2_face_status(
        n_rows=n_rows,
        matched_percent=(matched_percent),
        max_abs_diff_sec_observed=(max_diff),
        min_matched_percent=float(min_matched_percent),
        warning_matched_percent=float(warning_matched_percent),
        max_abs_diff_sec=(None if max_abs_diff_sec is None else float(max_abs_diff_sec)),
    )

    message = _r2_face_message(
        audit_status,
        matched_percent,
        min_matched_percent,
        warning_matched_percent,
    )

    row.update(
        {
            "n_rows": n_rows,
            "n_matched": n_matched,
            "matched_percent": matched_percent,
            "n_unmatched": n_unmatched,
            "unmatched_percent": _r2_face_percent(
                n_unmatched,
                n_rows,
            ),
            "n_outside_tolerance": n_outside,
            "outside_tolerance_percent": _r2_face_percent(
                n_outside,
                n_rows,
            ),
            "n_missing_gaze_time": n_missing_time,
            "n_missing_gaze_frame": n_missing_frame,
            "n_unknown_status": n_unknown,
            "n_within_tolerance": n_within,
            "within_tolerance_percent": _r2_face_percent(
                n_within,
                n_rows,
            ),
            "mean_abs_diff_sec": mean_diff,
            "median_abs_diff_sec": median_diff,
            "p95_abs_diff_sec": p95_diff,
            "max_abs_diff_sec": max_diff,
            "n_abs_diff_above_limit": n_above,
            "face_sync_audit_status": audit_status,
            "message": message,
        }
    )

    columns = (
        ["face_sync_group"]
        + (list(group_cols) if group_cols else [])
        + [
            "n_rows",
            "n_matched",
            "matched_percent",
            "n_unmatched",
            "unmatched_percent",
            "n_outside_tolerance",
            "outside_tolerance_percent",
            "n_missing_gaze_time",
            "n_missing_gaze_frame",
            "n_unknown_status",
            "n_within_tolerance",
            "within_tolerance_percent",
            "mean_abs_diff_sec",
            "median_abs_diff_sec",
            "p95_abs_diff_sec",
            "max_abs_diff_sec",
            "n_abs_diff_above_limit",
            "face_sync_audit_status",
            "message",
        ]
    )

    return pd.DataFrame(
        [row],
        columns=columns,
    )


def audit_face_sync(
    data,
    group_cols=None,
    min_matched_percent=70,
    warning_matched_percent=85,
    max_abs_diff_sec=None,
):
    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise ValueError("`data` must be a data frame returned by sync_gazepoint_face_data().")

    required = [
        "face_sync_method",
        "face_sync_status",
        "face_sync_within_tolerance",
    ]

    missing = [col for col in required if col not in data.columns]

    if missing:
        raise ValueError(
            "`data` does not look like "
            "synchronised face data. "
            "Missing column(s): " + ", ".join(missing)
        )

    frame = data.copy()

    if group_cols is None:
        groups = None
    else:
        groups = [col for col in _r2_v5_chars(group_cols) if col in frame.columns]

        if len(groups) == 0:
            groups = None

    if groups:
        indices = []

        grouper = groups[0] if len(groups) == 1 else groups

        for _, block in frame.groupby(
            grouper,
            dropna=False,
            sort=True,
        ):
            indices.append(block.index)
    else:
        indices = [frame.index]

    group_frames = [
        _r2_face_subset(
            frame,
            index,
            groups,
            min_matched_percent,
            warning_matched_percent,
            max_abs_diff_sec,
        )
        for index in indices
    ]

    group_summary = pd.concat(
        group_frames,
        ignore_index=True,
    )

    overall = _r2_face_subset(
        frame,
        frame.index,
        None,
        min_matched_percent,
        warning_matched_percent,
        max_abs_diff_sec,
    )

    overall = overall.drop(columns=["face_sync_group"])

    overview = pd.concat(
        [
            pd.DataFrame({"n_groups": [int(len(group_summary))]}),
            overall.reset_index(drop=True),
        ],
        axis=1,
    )

    n_groups = int(len(group_summary))

    conditions = [
        (
            "matched_percent_below_minimum",
            int((group_summary["matched_percent"] < float(min_matched_percent)).sum()),
            float(min_matched_percent),
        ),
        (
            "matched_percent_below_warning",
            int((group_summary["matched_percent"] < float(warning_matched_percent)).sum()),
            float(warning_matched_percent),
        ),
        (
            "unmatched_rows",
            int((group_summary["n_unmatched"] > 0).sum()),
            np.nan,
        ),
        (
            "outside_tolerance_rows",
            int((group_summary["n_outside_tolerance"] > 0).sum()),
            np.nan,
        ),
        (
            "missing_gaze_time_rows",
            int((group_summary["n_missing_gaze_time"] > 0).sum()),
            np.nan,
        ),
        (
            "missing_gaze_frame_rows",
            int((group_summary["n_missing_gaze_frame"] > 0).sum()),
            np.nan,
        ),
        (
            "large_time_differences",
            (
                np.nan
                if max_abs_diff_sec is None
                else int((group_summary["max_abs_diff_sec"] > float(max_abs_diff_sec)).sum())
            ),
            (np.nan if max_abs_diff_sec is None else float(max_abs_diff_sec)),
        ),
    ]

    issue_rows = []

    for issue, affected, threshold in conditions:
        if pd.isna(affected):
            status = "not_checked"
        elif int(affected) > 0:
            status = "review"
        else:
            status = "ok"

        issue_rows.append(
            {
                "issue": issue,
                "n_groups_affected": affected,
                "n_groups": n_groups,
                "threshold": threshold,
                "status": status,
            }
        )

    issue_summary = pd.DataFrame(
        issue_rows,
        columns=[
            "issue",
            "n_groups_affected",
            "n_groups",
            "threshold",
            "status",
        ],
    )

    settings = {
        "group_cols": (
            groups[0] if (groups and len(groups) == 1) else (list(groups) if groups else [])
        ),
        "min_matched_percent": float(min_matched_percent),
        "warning_matched_percent": float(warning_matched_percent),
        "max_abs_diff_sec": (float(max_abs_diff_sec) if max_abs_diff_sec is not None else None),
    }

    return _R2ClassedList(
        {
            "overview": overview,
            "group_summary": group_summary,
            "issue_summary": issue_summary,
            "data": frame.reset_index(drop=True),
            "settings": settings,
        },
        "gp3_face_sync_audit|list",
    )


# ===========================================================================
# audit_gazepoint_timecourse_grid
# ===========================================================================


def audit_timecourse_grid(
    data,
    subject_col=".gp3_cluster_subject",
    condition_col=".gp3_cluster_condition",
    time_col=".gp3_cluster_time_bin",
    outcome_col=".gp3_cluster_outcome",
):
    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise ValueError("`data` must be a data frame.")

    resolved = [
        subject_col,
        condition_col,
        time_col,
        outcome_col,
    ]

    if any(
        not isinstance(
            col,
            str,
        )
        or col == ""
        for col in resolved
    ):
        raise ValueError("Timecourse-grid column names must be non-empty character scalars.")

    missing = [col for col in resolved if col not in data.columns]

    if missing:
        raise ValueError("Missing required column(s): " + ", ".join(missing))

    work = pd.DataFrame(
        {
            "subject": data[subject_col].astype("string"),
            "condition": data[condition_col].astype("string"),
            "time_bin": pd.to_numeric(
                data[time_col],
                errors="coerce",
            ),
            "outcome": pd.to_numeric(
                data[outcome_col],
                errors="coerce",
            ),
        }
    )

    n_input = int(len(work))

    valid = (
        work["subject"].notna()
        & work["condition"].notna()
        & np.isfinite(work["time_bin"].to_numpy(dtype=float))
        & np.isfinite(work["outcome"].to_numpy(dtype=float))
    )

    work = work.loc[valid].copy().reset_index(drop=True)

    subjects = sorted(work["subject"].astype(str).unique().tolist())

    conditions = sorted(work["condition"].astype(str).unique().tolist())

    time_bins = sorted(
        pd.to_numeric(
            work["time_bin"],
            errors="coerce",
        )
        .dropna()
        .unique()
        .tolist()
    )

    def make_key(
        subject,
        condition,
        time_bin,
    ):
        return f"{subject}\r{condition}\r{float(time_bin):.15g}"

    observed_keys = [
        make_key(
            row.subject,
            row.condition,
            row.time_bin,
        )
        for row in work.itertuples(index=False)
    ]

    count_series = pd.Series(
        observed_keys,
        dtype="object",
    ).value_counts(sort=True)

    duplicate_values = count_series.loc[count_series > 1]

    duplicate_cells = int(len(duplicate_values))

    expected_rows = []

    # R expand.grid():
    # first supplied factor varies fastest.
    for time_value in time_bins:
        for condition_value in conditions:
            for subject_value in subjects:
                expected_rows.append(
                    {
                        "subject": subject_value,
                        "condition": condition_value,
                        "time_bin": float(time_value),
                    }
                )

    expected = pd.DataFrame(
        expected_rows,
        columns=[
            "subject",
            "condition",
            "time_bin",
        ],
    )

    observed_unique = set(observed_keys)

    missing_rows = []

    for row in expected.itertuples(index=False):
        key = make_key(
            row.subject,
            row.condition,
            row.time_bin,
        )

        if key not in observed_unique:
            missing_rows.append(
                {
                    "subject": row.subject,
                    "condition": row.condition,
                    "time_bin": row.time_bin,
                }
            )

    missing_cells = pd.DataFrame(
        missing_rows,
        columns=[
            "subject",
            "condition",
            "time_bin",
        ],
    )

    paired = {}

    for row in work.itertuples(index=False):
        key = (
            str(row.subject),
            float(row.time_bin),
        )

        paired.setdefault(
            key,
            set(),
        ).add(str(row.condition))

    unpaired_cells = int(sum(len(values) < 2 for values in paired.values()))

    grid_summary = pd.DataFrame(
        [
            {
                "n_input_rows": n_input,
                "n_valid_rows": int(len(work)),
                "n_subjects": int(len(subjects)),
                "n_conditions": int(len(conditions)),
                "n_time_bins": int(len(time_bins)),
                "n_expected_cells": int(len(expected)),
                "n_observed_cells": int(len(observed_unique)),
                "n_missing_cells": int(len(missing_cells)),
                "n_duplicate_cells": duplicate_cells,
                "n_unpaired_subject_time_cells": unpaired_cells,
            }
        ]
    )

    readiness = pd.DataFrame(
        {
            "check": [
                "exactly_two_conditions",
                "no_missing_grid_cells",
                "no_duplicate_cells",
                "no_unpaired_subject_time_cells",
                "numeric_outcome",
            ],
            "passed": [
                len(conditions) == 2,
                len(missing_cells) == 0,
                duplicate_cells == 0,
                unpaired_cells == 0,
                True,
            ],
        }
    )

    audit_status = "ready" if bool(readiness["passed"].all()) else "review"

    duplicate_counts = duplicate_values.astype(int).tolist()

    if len(duplicate_counts) == 0:
        duplicate_output = []
    elif len(duplicate_counts) == 1:
        duplicate_output = int(duplicate_counts[0])
    else:
        duplicate_output = duplicate_counts

    return _R2ClassedList(
        {
            "grid_summary": grid_summary,
            "readiness": readiness,
            "missing_cells": missing_cells,
            "duplicate_cell_count": duplicate_output,
            "audit_status": audit_status,
        },
        "gp3_timecourse_grid_audit|list",
    )


# ===========================================================================
# compare_gazepoint_event_detectors
# ===========================================================================


_compare_event_detectors_before_r2_v5 = compare_event_detectors


def _r2_detector_threshold(
    detector,
    family,
):
    if family != "velocity":
        return np.nan

    prefix = "velocity_"

    if not str(detector).startswith(prefix):
        return np.nan

    text = str(detector)[len(prefix) :]

    try:
        return float(text)
    except ValueError:
        return np.nan


def _r2_detector_best_overlap(
    events_a,
    events_b,
):
    if len(events_a) == 0:
        return np.array(
            [],
            dtype=float,
        )

    if len(events_b) == 0:
        return np.zeros(
            len(events_a),
            dtype=float,
        )

    start_b = pd.to_numeric(
        events_b["start_time"],
        errors="coerce",
    ).to_numpy(dtype=float)

    end_b = pd.to_numeric(
        events_b["end_time"],
        errors="coerce",
    ).to_numpy(dtype=float)

    result = []

    for row in events_a.itertuples(index=False):
        start_a = float(row.start_time)

        end_a = float(row.end_time)

        intersection = np.maximum(
            0.0,
            np.minimum(
                end_a,
                end_b,
            )
            - np.maximum(
                start_a,
                start_b,
            ),
        )

        union = np.maximum(
            end_a,
            end_b,
        ) - np.minimum(
            start_a,
            start_b,
        )

        iou = np.where(
            np.isfinite(union) & (union > 0),
            intersection / union,
            0.0,
        )

        result.append(float(np.nanmax(iou)))

    return np.asarray(
        result,
        dtype=float,
    )


def _r2_detector_sequence_keys(
    frame,
    sequence_cols,
):
    if len(frame) == 0:
        return []

    if not sequence_cols:
        return [".all"] * len(frame)

    keys = []

    for row in frame[sequence_cols].itertuples(
        index=False,
        name=None,
    ):
        parts = [("<NA>" if pd.isna(value) else str(value)) for value in row]

        keys.append("\r".join(parts))

    return keys


def compare_event_detectors(
    *args,
    **kwargs,
):
    base = _compare_event_detectors_before_r2_v5(
        *args,
        **kwargs,
    )

    events = base["events"].copy()

    family_col = "family" if "family" in events.columns else "detector_family"

    status_col = "source_status" if "source_status" in events.columns else "detector_status"

    if family_col not in events.columns:
        events["family"] = [
            (
                str(detector).split(
                    "_",
                    1,
                )[0]
            )
            for detector in events["detector"]
        ]
    elif family_col != "family":
        events["family"] = events[family_col].astype(str)

    if status_col in events.columns:
        events["source_status"] = events[status_col]
    else:
        events["source_status"] = "ok"

    events["threshold"] = [
        _r2_detector_threshold(
            detector,
            family,
        )
        for detector, family in zip(
            events["detector"],
            events["family"],
            strict=False,
        )
    ]

    settings = dict(base.get("settings", {}))

    sequence_cols = settings.get("sequence_cols")

    if sequence_cols is None:
        id_col = settings.get(
            "id_col",
            kwargs.get(
                "id_col",
                "USER_ID",
            ),
        )

        trial_col = settings.get(
            "trial_col",
            kwargs.get(
                "trial_col",
                "trial",
            ),
        )

        sequence_cols = [
            col
            for col in [
                id_col,
                trial_col,
            ]
            if col in events.columns
        ]

    elif isinstance(
        sequence_cols,
        str,
    ):
        sequence_cols = [sequence_cols]
    else:
        sequence_cols = list(sequence_cols)

    event_columns = sequence_cols + [
        "detector",
        "family",
        "threshold",
        "event_id",
        "start_time",
        "end_time",
        "duration_ms",
        "mean_x",
        "mean_y",
        "n_samples",
        "source_status",
    ]

    for column in event_columns:
        if column not in events.columns:
            events[column] = np.nan

    events = events[event_columns].copy().reset_index(drop=True)

    detectors = _r2_v5_first_unique(events["detector"].astype(str).tolist())

    detector_rows = []

    for detector in detectors:
        block = events.loc[events["detector"].astype(str).eq(detector)]

        duration = pd.to_numeric(
            block["duration_ms"],
            errors="coerce",
        ).to_numpy(dtype=float)

        finite = duration[np.isfinite(duration)]

        threshold_values = pd.to_numeric(
            block["threshold"],
            errors="coerce",
        ).to_numpy(dtype=float)

        finite_threshold = threshold_values[np.isfinite(threshold_values)]

        detector_rows.append(
            {
                "detector": detector,
                "family": str(block.iloc[0]["family"]),
                "threshold": (float(finite_threshold[0]) if len(finite_threshold) else np.nan),
                "n_fixations": int(len(block)),
                "mean_duration_ms": (float(np.mean(finite)) if len(finite) else np.nan),
                "median_duration_ms": (float(np.median(finite)) if len(finite) else np.nan),
                "total_duration_ms": (float(np.sum(finite)) if len(finite) else np.nan),
            }
        )

    detector_summary = pd.DataFrame(
        detector_rows,
        columns=[
            "detector",
            "family",
            "threshold",
            "n_fixations",
            "mean_duration_ms",
            "median_duration_ms",
            "total_duration_ms",
        ],
    )

    runs_base = base["runs"].copy()

    if "family" not in runs_base.columns and "detector_family" in runs_base.columns:
        runs_base["family"] = runs_base["detector_family"]

    if "message" not in runs_base.columns:
        runs_base["message"] = np.nan

    if "n_events" not in runs_base.columns:
        runs_base["n_events"] = [
            int(events["detector"].astype(str).eq(str(detector)).sum())
            for detector in runs_base["detector"]
        ]

    run_rows = []

    for detector in detectors:
        hit = runs_base.loc[runs_base["detector"].astype(str).eq(detector)]

        if len(hit):
            source = hit.iloc[0]

            run_rows.append(
                {
                    "detector": detector,
                    "family": source["family"],
                    "status": source["status"],
                    "n_events": int(source["n_events"]),
                    "message": (np.nan if pd.isna(source["message"]) else source["message"]),
                }
            )

    runs = pd.DataFrame(
        run_rows,
        columns=[
            "detector",
            "family",
            "status",
            "n_events",
            "message",
        ],
    )

    min_overlap = float(
        settings.get(
            "min_overlap",
            kwargs.get(
                "min_overlap",
                0.5,
            ),
        )
    )

    pair_rows = []
    unmatched_rows = []

    for a_index in range(len(detectors)):
        for b_index in range(
            a_index + 1,
            len(detectors),
        ):
            detector_a = detectors[a_index]

            detector_b = detectors[b_index]

            events_a = events.loc[events["detector"].astype(str).eq(detector_a)].copy()

            events_b = events.loc[events["detector"].astype(str).eq(detector_b)].copy()

            keys_a = _r2_detector_sequence_keys(
                events_a,
                sequence_cols,
            )

            keys_b = _r2_detector_sequence_keys(
                events_b,
                sequence_cols,
            )

            sequence_keys = _r2_v5_first_unique(keys_a + keys_b)

            events_a["_r2_sequence_key"] = keys_a

            events_b["_r2_sequence_key"] = keys_b

            for sequence_key in sequence_keys:
                block_a = events_a.loc[events_a["_r2_sequence_key"].eq(sequence_key)].drop(
                    columns=["_r2_sequence_key"]
                )

                block_b = events_b.loc[events_b["_r2_sequence_key"].eq(sequence_key)].drop(
                    columns=["_r2_sequence_key"]
                )

                overlap_a = _r2_detector_best_overlap(
                    block_a,
                    block_b,
                )

                overlap_b = _r2_detector_best_overlap(
                    block_b,
                    block_a,
                )

                matched_a = int(np.sum(overlap_a >= min_overlap))

                matched_b = int(np.sum(overlap_b >= min_overlap))

                source_block = block_a if len(block_a) else block_b

                key_values = {}

                if len(source_block):
                    first = source_block.iloc[0]

                    for column in sequence_cols:
                        key_values[column] = first[column]

                pair_row = dict(key_values)

                pair_row.update(
                    {
                        "detector_a": detector_a,
                        "detector_b": detector_b,
                        "n_a": int(len(block_a)),
                        "n_b": int(len(block_b)),
                        "matched_a": matched_a,
                        "matched_b": matched_b,
                        "agreement_a": (matched_a / len(block_a) if len(block_a) else np.nan),
                        "agreement_b": (matched_b / len(block_b) if len(block_b) else np.nan),
                        "mean_best_overlap_a": (
                            float(np.mean(overlap_a)) if len(overlap_a) else np.nan
                        ),
                        "mean_best_overlap_b": (
                            float(np.mean(overlap_b)) if len(overlap_b) else np.nan
                        ),
                        "min_overlap": min_overlap,
                    }
                )

                pair_rows.append(pair_row)

                for (
                    block,
                    overlap,
                    compared_with,
                ) in [
                    (
                        block_a,
                        overlap_a,
                        detector_b,
                    ),
                    (
                        block_b,
                        overlap_b,
                        detector_a,
                    ),
                ]:
                    for position, (
                        _,
                        event_row,
                    ) in enumerate(block.iterrows()):
                        best = float(overlap[position])

                        if best >= min_overlap:
                            continue

                        row = {column: event_row[column] for column in event_columns}

                        row["compared_with"] = compared_with

                        row["best_overlap"] = best

                        unmatched_rows.append(row)

    pairwise_columns = sequence_cols + [
        "detector_a",
        "detector_b",
        "n_a",
        "n_b",
        "matched_a",
        "matched_b",
        "agreement_a",
        "agreement_b",
        "mean_best_overlap_a",
        "mean_best_overlap_b",
        "min_overlap",
    ]

    pairwise_agreement = pd.DataFrame(
        pair_rows,
        columns=pairwise_columns,
    )

    unmatched_events = pd.DataFrame(
        unmatched_rows,
        columns=(
            event_columns
            + [
                "compared_with",
                "best_overlap",
            ]
        ),
    )

    # R list semantics: scalar character for one
    # method, character(0) for NULL group_cols.
    methods = settings.get(
        "methods",
        kwargs.get("methods"),
    )

    if isinstance(
        methods,
        (
            list,
            tuple,
            np.ndarray,
            pd.Index,
        ),
    ):
        methods_list = list(methods)

        if len(methods_list) == 1:
            methods = str(methods_list[0])
        else:
            methods = methods_list

    settings["methods"] = methods

    if settings.get("group_cols") is None:
        settings["group_cols"] = "<EMPTY>"

    settings["sequence_cols"] = sequence_cols

    return _R2ClassedList(
        {
            "events": events,
            "runs": runs,
            "detector_summary": detector_summary,
            "pairwise_agreement": pairwise_agreement,
            "unmatched_events": unmatched_events,
            "settings": settings,
        },
        ("gp3_event_detector_comparison|list"),
    )
