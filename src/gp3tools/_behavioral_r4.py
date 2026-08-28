from __future__ import annotations

import inspect
import math
import re
from collections.abc import Callable, Mapping, Sequence
from functools import wraps
from typing import Any

import numpy as np
import pandas as pd


class R4List(dict):
    """Mapping with explicit R class metadata."""

    def __init__(
        self,
        *args: Any,
        r_class: str = "list",
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.r_class = r_class
        self._r_class = r_class
        self.__r_class__ = r_class
        self._gp3_class = r_class
        self.attrs = {
            "r_class": r_class,
        }

    @property
    def gp3_r_class(
        self,
    ) -> str:
        return self.r_class


def _set_r_class(
    value: pd.DataFrame,
    r_class: str,
) -> pd.DataFrame:
    result = value.copy()

    result.attrs["r_class"] = r_class

    return result


def _tibble(
    value: pd.DataFrame,
) -> pd.DataFrame:
    return _set_r_class(
        value,
        "tbl_df|tbl|data.frame",
    )


def _bind(
    function: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    signature = inspect.signature(function)

    bound = signature.bind_partial(
        *args,
        **kwargs,
    )

    result: dict[str, Any] = {}

    for name, value in bound.arguments.items():
        parameter = signature.parameters[name]

        if parameter.kind is inspect.Parameter.VAR_KEYWORD and isinstance(
            value,
            dict,
        ):
            result.update(value)

        else:
            result[name] = value

    return result


def _numeric(
    value: Any,
) -> pd.Series:
    return pd.to_numeric(
        pd.Series(value),
        errors="coerce",
    )


def _safe_min(
    value: Any,
) -> float:
    values = _numeric(value).to_numpy(float)

    values = values[np.isfinite(values)]

    if not len(values):
        return float("nan")

    return float(np.min(values))


def _safe_max(
    value: Any,
) -> float:
    values = _numeric(value).to_numpy(float)

    values = values[np.isfinite(values)]

    if not len(values):
        return float("nan")

    return float(np.max(values))


def _safe_mean(
    value: Any,
) -> float:
    values = _numeric(value).to_numpy(float)

    if not len(values) or np.isnan(values).all():
        return float("nan")

    return float(np.nanmean(values))


def _safe_median(
    value: Any,
) -> float:
    values = _numeric(value).to_numpy(float)

    if not len(values) or np.isnan(values).all():
        return float("nan")

    return float(np.nanmedian(values))


def _safe_sd(
    value: Any,
) -> float:
    values = _numeric(value).to_numpy(float)

    values = values[~np.isnan(values)]

    if len(values) < 2:
        return float("nan")

    return float(
        np.std(
            values,
            ddof=1,
        )
    )


def _prop_true_pct(
    value: Any,
) -> float:
    series = pd.Series(value)

    keep = series.notna()

    if not keep.any():
        return float("nan")

    return float(series[keep].astype(bool).mean() * 100.0)


def _r_character(
    value: Any,
) -> Any:
    if value is None:
        return pd.NA

    if value is pd.NA:
        return pd.NA

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
        (
            int,
            np.integer,
        ),
    ):
        return str(int(value))

    if isinstance(
        value,
        (
            float,
            np.floating,
        ),
    ):
        value_float = float(value)

        if math.isnan(value_float):
            return pd.NA

        if value_float.is_integer():
            return str(int(value_float))

        return format(
            value_float,
            ".15g",
        )

    return str(value)


def _collapse_nullable(
    value: Any,
) -> Any:
    if value is None:
        return pd.NA

    if isinstance(
        value,
        str,
    ):
        return value

    try:
        values = list(value)

    except TypeError:
        return _r_character(value)

    if not values:
        return pd.NA

    return ", ".join(str(item) for item in values)


# ============================================================================
# Pupil-response features
# ============================================================================


def _trapz_r(
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    keep = np.isfinite(x) & np.isfinite(y)

    x = x[keep]
    y = y[keep]

    if len(x) < 2:
        return float("nan")

    order = np.argsort(
        x,
        kind="stable",
    )

    x = x[order]
    y = y[order]

    return float(np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2.0))


def _pupil_response(
    data: pd.DataFrame,
    *,
    pupil: str,
    time: str,
    subject: str,
    trial: str,
    baseline_window: Sequence[float],
    response_window: Sequence[float],
    condition: str | None,
    interpolated: str | None,
) -> pd.DataFrame:
    required = [
        pupil,
        time,
        subject,
        trial,
    ]

    if condition is not None:
        required.append(condition)

    if interpolated is not None:
        required.append(interpolated)

    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    if len(baseline_window) != 2:
        raise ValueError("`baseline_window` must contain two values.")

    if len(response_window) != 2:
        raise ValueError("`response_window` must contain two values.")

    baseline_lower = min(baseline_window)

    baseline_upper = max(baseline_window)

    response_lower = min(response_window)

    response_upper = max(response_window)

    rows: list[dict[str, Any]] = []

    grouped = data.groupby(
        [
            subject,
            trial,
        ],
        sort=True,
        dropna=False,
    )

    for _, frame in grouped:
        t = pd.to_numeric(
            frame[time],
            errors="coerce",
        ).to_numpy(float)

        p = pd.to_numeric(
            frame[pupil],
            errors="coerce",
        ).to_numpy(float)

        baseline = np.isfinite(t) & (t >= baseline_lower) & (t <= baseline_upper)

        response = np.isfinite(t) & (t >= response_lower) & (t <= response_upper)

        baseline_values = p[baseline]

        if not len(baseline_values) or np.isnan(baseline_values).all():
            baseline_mean = float("nan")

        else:
            baseline_mean = float(np.nanmean(baseline_values))

        corrected = p - baseline_mean

        response_corrected = corrected[response]

        response_time = t[response]

        if not len(response_corrected) or np.isnan(response_corrected).all():
            peak = float("nan")

            latency = float("nan")

        else:
            peak_index = int(np.nanargmax(response_corrected))

            peak = float(response_corrected[peak_index])

            latency = float(response_time[peak_index])

        response_pupil = p[response]

        if not len(response_pupil):
            missing_percent = float("nan")

        else:
            missing_percent = float(np.mean(np.isnan(response_pupil)) * 100.0)

        interpolated_percent = float("nan")

        if interpolated is not None:
            positions = np.flatnonzero(response)

            values = frame[interpolated].iloc[positions].dropna()

            if len(values):
                interpolated_percent = float(values.astype(bool).mean() * 100.0)

        row = {
            "subject": frame[subject].iloc[0],
            "trial": frame[trial].iloc[0],
            "baseline_mean": baseline_mean,
            "peak_dilation": peak,
            "latency_to_peak": latency,
            "auc": _trapz_r(
                response_time,
                response_corrected,
            ),
            "missing_percent": missing_percent,
            "interpolated_percent": interpolated_percent,
        }

        if condition is not None:
            row["condition"] = frame[condition].iloc[0]

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# Master audit
# ============================================================================


def _detect_master_columns(
    master: pd.DataFrame,
) -> dict[str, str | None]:
    candidates = {
        "subject": (
            "subject",
            "pID",
            "participant",
        ),
        "media_id": (
            "media_id",
            "MEDIA_ID",
        ),
        "time_ms": (
            "time_ms",
            "time",
            "time_orig",
            "time_orig_ms",
        ),
        "x": (
            "x",
            "gaze_x",
        ),
        "y": (
            "y",
            "gaze_y",
        ),
        "valid_sample": ("valid_sample",),
        "missing_gaze": ("missing_gaze",),
        "missing_pupil": ("missing_pupil",),
        "gaze_offscreen": ("gaze_offscreen",),
        "mean_pupil": (
            "mean_pupil",
            "pupil",
            "pupil_raw",
        ),
        "aoi_current": (
            "aoi_current",
            "AOI",
        ),
        "aoi_count": ("aoi_count",),
        "raw_x": ("raw_x",),
        "raw_y": ("raw_y",),
    }

    result: dict[
        str,
        str | None,
    ] = {}

    for role, choices in candidates.items():
        result[role] = next(
            (choice for choice in choices if choice in master.columns),
            None,
        )

    return result


def _canonical_master(
    master: Any,
) -> bool:
    if not isinstance(
        master,
        pd.DataFrame,
    ):
        return False

    mapping = _detect_master_columns(master)

    required = [
        "subject",
        "media_id",
        "time_ms",
        "x",
        "y",
        "valid_sample",
        "missing_gaze",
        "missing_pupil",
        "gaze_offscreen",
        "mean_pupil",
        "aoi_current",
        "aoi_count",
    ]

    return all(mapping[name] is not None for name in required)


def _is_real_aoi(
    values: pd.Series,
) -> pd.Series:
    values = values.astype("string")

    return (
        values.notna()
        & values.ne("")
        & ~values.isin(
            [
                "missing",
                "offscreen",
                "non_aoi",
                "unclassified",
            ]
        )
    )


def _audit_master(
    source: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    mapping = _detect_master_columns(source)

    def column(
        role: str,
    ) -> pd.Series:
        name = mapping[role]

        if name is None:
            return pd.Series(
                np.nan,
                index=source.index,
            )

        return source[name]

    master = pd.DataFrame(
        {
            "subject": column("subject").astype("string"),
            "media_id": column("media_id").astype("string"),
            "time_ms": pd.to_numeric(
                column("time_ms"),
                errors="coerce",
            ),
            "x": pd.to_numeric(
                column("x"),
                errors="coerce",
            ),
            "y": pd.to_numeric(
                column("y"),
                errors="coerce",
            ),
            "valid_sample": column("valid_sample").astype("boolean"),
            "missing_gaze": column("missing_gaze").astype("boolean"),
            "missing_pupil": column("missing_pupil").astype("boolean"),
            "gaze_offscreen": column("gaze_offscreen").astype("boolean"),
            "mean_pupil": pd.to_numeric(
                column("mean_pupil"),
                errors="coerce",
            ),
            "aoi_current": column("aoi_current").astype("string"),
            "aoi_count": pd.to_numeric(
                column("aoi_count"),
                errors="coerce",
            ).astype("Int64"),
            "raw_x": pd.to_numeric(
                column("raw_x"),
                errors="coerce",
            ),
            "raw_y": pd.to_numeric(
                column("raw_y"),
                errors="coerce",
            ),
        }
    )

    real_aoi = _is_real_aoi(master["aoi_current"])

    time_min = _safe_min(master["time_ms"])

    time_max = _safe_max(master["time_ms"])

    overview = pd.DataFrame(
        [
            {
                "n_rows": len(master),
                "n_subjects": master["subject"].nunique(dropna=True),
                "n_media": master["media_id"].nunique(dropna=True),
                "n_subject_media": len(
                    master[
                        [
                            "subject",
                            "media_id",
                        ]
                    ].drop_duplicates()
                ),
                "time_min_ms": time_min,
                "time_max_ms": time_max,
                "time_span_ms": time_max - time_min,
                "valid_sample_pct": _prop_true_pct(master["valid_sample"]),
                "missing_gaze_pct": _prop_true_pct(master["missing_gaze"]),
                "missing_pupil_pct": _prop_true_pct(master["missing_pupil"]),
                "offscreen_gaze_pct": _prop_true_pct(master["gaze_offscreen"]),
                "n_missing_gaze": int(master["missing_gaze"].fillna(False).sum()),
                "n_missing_pupil": int(master["missing_pupil"].fillna(False).sum()),
                "n_offscreen_gaze": int(master["gaze_offscreen"].fillna(False).sum()),
                "has_pupil": bool(master["mean_pupil"].notna().any()),
                "has_aoi": bool(real_aoi.any()),
                "n_aoi_samples": int(real_aoi.sum()),
                "n_missing_state": int((master["aoi_current"] == "missing").sum()),
                "n_offscreen_state": int((master["aoi_current"] == "offscreen").sum()),
            }
        ]
    )

    overview = _tibble(overview)

    def grouped_summary(
        groups: list[str],
    ) -> pd.DataFrame:
        rows = []

        for key, frame in master.groupby(
            groups,
            sort=True,
            dropna=False,
        ):
            if not isinstance(
                key,
                tuple,
            ):
                key = (key,)

            row = dict(
                zip(
                    groups,
                    key,
                    strict=False,
                )
            )

            frame_real = _is_real_aoi(frame["aoi_current"])

            group_min = _safe_min(frame["time_ms"])

            group_max = _safe_max(frame["time_ms"])

            row.update(
                {
                    "n_rows": len(frame),
                    "time_min_ms": group_min,
                    "time_max_ms": group_max,
                    "time_span_ms": group_max - group_min,
                    "valid_sample_pct": _prop_true_pct(frame["valid_sample"]),
                    "missing_gaze_pct": _prop_true_pct(frame["missing_gaze"]),
                    "missing_pupil_pct": _prop_true_pct(frame["missing_pupil"]),
                    "offscreen_gaze_pct": _prop_true_pct(frame["gaze_offscreen"]),
                    "n_missing_gaze": int(frame["missing_gaze"].fillna(False).sum()),
                    "n_missing_pupil": int(frame["missing_pupil"].fillna(False).sum()),
                    "n_offscreen_gaze": int(frame["gaze_offscreen"].fillna(False).sum()),
                    "n_aoi_samples": int(frame_real.sum()),
                    "n_missing_state": int((frame["aoi_current"] == "missing").sum()),
                    "n_offscreen_state": int((frame["aoi_current"] == "offscreen").sum()),
                    "aoi_count_sum": int(
                        pd.to_numeric(
                            frame["aoi_count"],
                            errors="coerce",
                        )
                        .fillna(0)
                        .sum()
                    ),
                    "has_pupil": bool(frame["mean_pupil"].notna().any()),
                }
            )

            rows.append(row)

        return _tibble(pd.DataFrame(rows))

    by_subject = grouped_summary(
        [
            "subject",
        ]
    )

    by_media = grouped_summary(
        [
            "media_id",
        ]
    )

    by_subject_media = grouped_summary(
        [
            "subject",
            "media_id",
        ]
    )

    states = master["aoi_current"].astype("string")

    states = states.fillna("unclassified")

    states = states.mask(
        states.eq(""),
        "unclassified",
    )

    aoi_states = (
        states.value_counts(sort=False).rename_axis("aoi_state").reset_index(name="n_samples")
    )

    aoi_states["prop_samples"] = aoi_states["n_samples"] / aoi_states["n_samples"].sum() * 100.0

    aoi_states = aoi_states.sort_values(
        "n_samples",
        ascending=False,
        kind="stable",
    ).reset_index(drop=True)

    aoi_states = _tibble(aoi_states)

    pupil_rows = []

    for (
        subject_value,
        media_value,
    ), frame in master.groupby(
        [
            "subject",
            "media_id",
        ],
        sort=True,
        dropna=False,
    ):
        pupil_rows.append(
            {
                "subject": subject_value,
                "media_id": media_value,
                "n_rows": len(frame),
                "n_pupil_samples": int(frame["mean_pupil"].notna().sum()),
                "missing_pupil_pct": _prop_true_pct(frame["missing_pupil"]),
                "mean_pupil": _safe_mean(frame["mean_pupil"]),
                "median_pupil": _safe_median(frame["mean_pupil"]),
                "sd_pupil": _safe_sd(frame["mean_pupil"]),
                "min_pupil": _safe_min(frame["mean_pupil"]),
                "max_pupil": _safe_max(frame["mean_pupil"]),
            }
        )

    pupil_summary = _tibble(pd.DataFrame(pupil_rows))

    coordinate_summary = _tibble(
        pd.DataFrame(
            [
                {
                    "n_rows": len(master),
                    "x_min": _safe_min(master["x"]),
                    "x_max": _safe_max(master["x"]),
                    "y_min": _safe_min(master["y"]),
                    "y_max": _safe_max(master["y"]),
                    "raw_x_min": _safe_min(master["raw_x"]),
                    "raw_x_max": _safe_max(master["raw_x"]),
                    "raw_y_min": _safe_min(master["raw_y"]),
                    "raw_y_max": _safe_max(master["raw_y"]),
                    "n_offscreen_gaze": int(master["gaze_offscreen"].fillna(False).sum()),
                    "offscreen_gaze_pct": _prop_true_pct(master["gaze_offscreen"]),
                }
            ]
        )
    )

    return {
        "overview": overview,
        "by_subject": by_subject,
        "by_media": by_media,
        "by_subject_media": by_subject_media,
        "aoi_states": aoi_states,
        "pupil_summary": pupil_summary,
        "coordinate_summary": coordinate_summary,
    }


# ============================================================================
# QC status
# ============================================================================


def _find_object_summary(
    value: Any,
) -> pd.DataFrame | None:
    if isinstance(
        value,
        pd.DataFrame,
    ):
        if {
            "object_name",
            "qc_status",
        }.issubset(value.columns):
            return value.copy()

        return None

    if isinstance(
        value,
        Mapping,
    ):
        direct = value.get("object_summary")

        if isinstance(
            direct,
            pd.DataFrame,
        ):
            if {
                "object_name",
                "qc_status",
            }.issubset(direct.columns):
                return direct.copy()

        for item in value.values():
            found = _find_object_summary(item)

            if found is not None:
                return found

    return None


def _status_priority(
    status: str,
) -> int:
    lookup = {
        "pass": 0,
        "info": 1,
        "unknown": 1,
        "warn": 2,
        "fail": 3,
    }

    return lookup.get(
        status,
        1,
    )


def _derive_object_summary(
    qc_bundle: Any,
) -> pd.DataFrame | None:
    if not isinstance(
        qc_bundle,
        Mapping,
    ):
        return None

    rows = []

    for name, value in qc_bundle.items():
        if name in {
            "overview",
            "status_counts",
            "object_summary",
        }:
            continue

        statuses: list[str] = []

        if isinstance(
            value,
            pd.DataFrame,
        ):
            for column in (
                "qc_status",
                "status",
            ):
                if column in value.columns:
                    statuses.extend(value[column].dropna().astype(str).str.lower().tolist())

        elif isinstance(
            value,
            Mapping,
        ):
            scalar = value.get("qc_status")

            if scalar is not None:
                statuses.append(str(scalar).lower())

            overview = value.get("overview")

            if isinstance(
                overview,
                pd.DataFrame,
            ):
                for column in overview.columns:
                    if column.endswith("_status"):
                        statuses.extend(overview[column].dropna().astype(str).str.lower().tolist())

        statuses = [
            status
            for status in statuses
            if status
            in {
                "pass",
                "warn",
                "fail",
                "info",
                "unknown",
            }
        ]

        if statuses:
            status = max(
                statuses,
                key=_status_priority,
            )

        else:
            status = "unknown"

        rows.append(
            {
                "object_name": str(name),
                "qc_status": status,
            }
        )

    if not rows:
        return None

    return pd.DataFrame(rows)


def _qc_status_summary(
    original_result: Any,
    qc_bundle: Any,
) -> R4List | None:
    object_summary = _find_object_summary(qc_bundle)

    if object_summary is None:
        object_summary = _find_object_summary(original_result)

    if object_summary is None:
        object_summary = _derive_object_summary(qc_bundle)

    if object_summary is None:
        return None

    status = object_summary["qc_status"].astype("string").str.lower()

    levels = [
        "pass",
        "warn",
        "fail",
        "info",
        "unknown",
    ]

    counts = {level: int((status == level).sum()) for level in levels}

    if counts["fail"] > 0:
        overall = "fail"

    elif counts["warn"] > 0:
        overall = "warn"

    elif counts["info"] > 0 or counts["unknown"] > 0:
        overall = "info"

    else:
        overall = "pass"

    overview = pd.DataFrame(
        [
            {
                "n_objects": len(object_summary),
                "n_pass": counts["pass"],
                "n_warn": counts["warn"],
                "n_fail": counts["fail"],
                "n_info": counts["info"],
                "n_unknown": counts["unknown"],
                "qc_overview_status": overall,
            }
        ]
    )

    status_counts = pd.DataFrame(
        {
            "qc_status": levels,
            "n_objects": [counts[level] for level in levels],
        }
    )

    return R4List(
        {
            "overview": overview,
            "status_counts": status_counts,
            "object_summary": object_summary.reset_index(drop=True),
        },
        r_class=("gp3_qc_status_summary|list"),
    )


# ============================================================================
# Workflow
# ============================================================================


def _nrow_safe(
    value: Any,
) -> int | None:
    if isinstance(
        value,
        pd.DataFrame,
    ):
        return int(len(value))

    return None


def _n_entries_safe(
    value: Any,
) -> int:
    if value is None:
        return 0

    if isinstance(
        value,
        pd.DataFrame,
    ):
        return int(len(value))

    try:
        return int(len(value))

    except TypeError:
        return 1


def _workflow_summary(
    results: Mapping[str, Any],
) -> pd.DataFrame:
    required = [
        "all_gaze",
        "all_fix",
        "sampling",
        "quality",
        "flagged_quality",
        "aoi_table",
    ]

    missing = [name for name in required if name not in results]

    if missing:
        raise ValueError("`results` is missing required elements: " + ", ".join(missing))

    review_required = None

    flagged = results["flagged_quality"]

    if (
        isinstance(
            flagged,
            pd.DataFrame,
        )
        and "review_required" in flagged.columns
    ):
        review_required = int(flagged["review_required"].fillna(False).astype(bool).sum())

    file_pair_rows = None
    complete_file_pairs = None
    problem_file_pairs = None

    file_pairs = results.get("file_pairs")

    if isinstance(
        file_pairs,
        pd.DataFrame,
    ):
        file_pair_rows = int(len(file_pairs))

        if "status" in file_pairs.columns:
            complete_file_pairs = int((file_pairs["status"] == "complete").sum())

            problem_file_pairs = int((file_pairs["status"] != "complete").sum())

    output = pd.DataFrame(
        [
            {
                "all_gaze_rows": _nrow_safe(results["all_gaze"]),
                "fixation_rows": _nrow_safe(results["all_fix"]),
                "sampling_rows": _nrow_safe(results["sampling"]),
                "tracking_quality_rows": _nrow_safe(results["quality"]),
                "flagged_quality_rows": _nrow_safe(results["flagged_quality"]),
                "aoi_rows": _nrow_safe(results["aoi_table"]),
                "review_required_rows": review_required,
                "file_pair_rows": file_pair_rows,
                "complete_file_pairs": complete_file_pairs,
                "problem_file_pairs": problem_file_pairs,
                "output_table_files": _n_entries_safe(results.get("written_files")),
                "output_plot_files": _n_entries_safe(results.get("written_plots")),
                "report_created": (results.get("written_report") is not None),
            }
        ]
    )

    return _tibble(output)


# ============================================================================
# Static rectangular AOI
# ============================================================================


def _resolve_aoi_columns(
    definitions: pd.DataFrame,
) -> dict[str, str] | None:
    aliases = {
        "name": (
            "name",
            "aoi_name",
            "AOI",
            "aoi",
            "label",
        ),
        "left": (
            "L",
            "left",
            "xmin",
            "x_min",
        ),
        "right": (
            "R",
            "right",
            "xmax",
            "x_max",
        ),
        "top": (
            "T",
            "top",
            "ymin",
            "y_min",
        ),
        "bottom": (
            "B",
            "bottom",
            "ymax",
            "y_max",
        ),
    }

    resolved = {}

    for role, candidates in aliases.items():
        found = next(
            (candidate for candidate in candidates if candidate in definitions.columns),
            None,
        )

        if found is None:
            return None

        resolved[role] = found

    return resolved


def _make_name(
    value: Any,
) -> str:
    value = re.sub(
        r"[^A-Za-z0-9_.]",
        ".",
        str(value),
    )

    if not value:
        value = "X"

    if value[0].isdigit():
        value = "X" + value

    return value


def _static_aoi(
    master: pd.DataFrame,
    definitions: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    aoi_name: Any,
    output: str,
    prefix: str,
    label_col: str,
    outside_label: str,
    overlap: str,
    include_overlap_count: bool,
) -> pd.DataFrame:
    resolved = _resolve_aoi_columns(definitions)

    if resolved is None:
        raise ValueError("Could not resolve AOI definition fields.")

    defs = pd.DataFrame(
        {
            "name": definitions[resolved["name"]].astype(str),
            "left": pd.to_numeric(
                definitions[resolved["left"]],
                errors="coerce",
            ),
            "right": pd.to_numeric(
                definitions[resolved["right"]],
                errors="coerce",
            ),
            "top": pd.to_numeric(
                definitions[resolved["top"]],
                errors="coerce",
            ),
            "bottom": pd.to_numeric(
                definitions[resolved["bottom"]],
                errors="coerce",
            ),
        }
    )

    if aoi_name is not None:
        if isinstance(
            aoi_name,
            str,
        ):
            requested = [
                aoi_name,
            ]

        else:
            requested = list(aoi_name)

        defs = defs[defs["name"].isin([str(value) for value in requested])].copy()

        if defs.empty:
            raise ValueError("`aoi_name` did not match any AOI definition.")

    if defs.empty:
        raise ValueError("`aoi_defs` must contain at least one AOI.")

    if defs["name"].duplicated().any():
        raise ValueError("AOI names must be unique.")

    numeric = defs[
        [
            "left",
            "right",
            "top",
            "bottom",
        ]
    ].to_numpy(float)

    if not np.isfinite(numeric).all():
        raise ValueError("AOI boundaries must be finite numeric values.")

    left = np.minimum(
        defs["left"],
        defs["right"],
    ).to_numpy(float)

    right = np.maximum(
        defs["left"],
        defs["right"],
    ).to_numpy(float)

    top = np.minimum(
        defs["top"],
        defs["bottom"],
    ).to_numpy(float)

    bottom = np.maximum(
        defs["top"],
        defs["bottom"],
    ).to_numpy(float)

    x = pd.to_numeric(
        master[x_col],
        errors="coerce",
    ).to_numpy(float)

    y = pd.to_numeric(
        master[y_col],
        errors="coerce",
    ).to_numpy(float)

    valid = np.isfinite(x) & np.isfinite(y)

    membership = np.column_stack(
        [
            valid
            & (x >= left[index])
            & (x <= right[index])
            & (y >= top[index])
            & (y <= bottom[index])
            for index in range(len(defs))
        ]
    )

    overlap_count = membership.sum(axis=1).astype(float)

    if overlap == "error" and np.any(overlap_count > 1):
        raise ValueError(
            f"{int(np.sum(overlap_count > 1))} sample(s) fall inside overlapping AOIs."
        )

    result = master.copy()

    logical_names = []

    used = set()

    for raw_name in defs["name"]:
        base = _make_name(raw_name)

        candidate = base
        suffix = 1

        while candidate in used:
            candidate = f"{base}.{suffix}"

            suffix += 1

        used.add(candidate)

        logical_names.append(candidate)

    if output in {
        "logical",
        "both",
    }:
        for index, logical_name in enumerate(logical_names):
            result[prefix + logical_name] = membership[:, index]

    if output in {
        "label",
        "both",
    }:
        labels = np.full(
            len(master),
            outside_label,
            dtype=object,
        )

        labels[~valid] = None

        for index, name in enumerate(defs["name"]):
            hit = membership[:, index]

            if overlap == "last":
                labels[hit] = name

            else:
                labels[hit & (labels == outside_label)] = name

        result[label_col] = labels

    if include_overlap_count:
        result["aoi_overlap_count"] = overlap_count

    result.attrs.update(master.attrs)

    result.attrs["gazepoint_aoi_definitions"] = defs.copy()

    return result


# ============================================================================
# AOI geometry
# ============================================================================


def _geometry_aliases(
    source: pd.DataFrame,
) -> pd.DataFrame:
    data = source.copy()

    if "AOI" in data.columns and "aoi" not in data.columns:
        data["aoi"] = data["AOI"]

    if "MEDIA_ID" in data.columns and "media_id" not in data.columns:
        data["media_id"] = data["MEDIA_ID"]

    return data


def _resolve_column(
    provided: str | None,
    data: pd.DataFrame,
    candidates: Sequence[str],
    *,
    required: bool,
) -> str | None:
    translations = {
        "AOI": "aoi",
        "MEDIA_ID": "media_id",
    }

    if provided is not None:
        provided = translations.get(
            provided,
            provided,
        )

        if provided not in data.columns:
            raise ValueError(f"Column not found: {provided}")

        return provided

    for candidate in candidates:
        candidate = translations.get(
            candidate,
            candidate,
        )

        if candidate in data.columns:
            return candidate

    if required:
        raise ValueError("Required AOI column could not be detected.")

    return None


def _geometry_audit(
    source: pd.DataFrame,
    *,
    aoi_col: str | None,
    stimulus_col: str | None,
    x_min_col: str | None,
    y_min_col: str | None,
    x_max_col: str | None,
    y_max_col: str | None,
    x_col: str | None,
    y_col: str | None,
    width_col: str | None,
    height_col: str | None,
    screen_x_range: Sequence[float],
    screen_y_range: Sequence[float],
    min_width: float,
    min_height: float,
    min_area: float,
    max_area_prop: float,
    require_within_screen: bool,
) -> R4List:
    if source.empty:
        raise ValueError("`data` must contain at least one row.")

    data = _geometry_aliases(source)

    aoi_col = _resolve_column(
        aoi_col,
        data,
        (
            "aoi",
            "aoi_name",
            "aoi_id",
            "AOI",
            "AOI_NAME",
            "AOI_ID",
        ),
        required=True,
    )

    if stimulus_col is not None:
        stimulus_col = _resolve_column(
            stimulus_col,
            data,
            (),
            required=True,
        )

    x_min_col = _resolve_column(
        x_min_col,
        data,
        (
            "x_min",
            "xmin",
            "left",
            "Left",
            "AOI_X_MIN",
            "AOI_LEFT",
        ),
        required=False,
    )

    y_min_col = _resolve_column(
        y_min_col,
        data,
        (
            "y_min",
            "ymin",
            "top",
            "Top",
            "AOI_Y_MIN",
            "AOI_TOP",
        ),
        required=False,
    )

    x_max_col = _resolve_column(
        x_max_col,
        data,
        (
            "x_max",
            "xmax",
            "right",
            "Right",
            "AOI_X_MAX",
            "AOI_RIGHT",
        ),
        required=False,
    )

    y_max_col = _resolve_column(
        y_max_col,
        data,
        (
            "y_max",
            "ymax",
            "bottom",
            "Bottom",
            "AOI_Y_MAX",
            "AOI_BOTTOM",
        ),
        required=False,
    )

    x_col = _resolve_column(
        x_col,
        data,
        (
            "x",
            "X",
            "aoi_x",
            "AOI_X",
        ),
        required=False,
    )

    y_col = _resolve_column(
        y_col,
        data,
        (
            "y",
            "Y",
            "aoi_y",
            "AOI_Y",
        ),
        required=False,
    )

    width_col = _resolve_column(
        width_col,
        data,
        (
            "width",
            "Width",
            "aoi_width",
            "AOI_WIDTH",
        ),
        required=False,
    )

    height_col = _resolve_column(
        height_col,
        data,
        (
            "height",
            "Height",
            "aoi_height",
            "AOI_HEIGHT",
        ),
        required=False,
    )

    has_bounds = all(
        value is not None
        for value in (
            x_min_col,
            y_min_col,
            x_max_col,
            y_max_col,
        )
    )

    has_origin_size = all(
        value is not None
        for value in (
            x_col,
            y_col,
            width_col,
            height_col,
        )
    )

    if not (has_bounds or has_origin_size):
        raise ValueError(
            "AOI geometry requires either x/y min-max columns or x/y plus width/height columns."
        )

    sx0 = float(screen_x_range[0])

    sx1 = float(screen_x_range[1])

    sy0 = float(screen_y_range[0])

    sy1 = float(screen_y_range[1])

    if has_bounds:
        xmin = pd.to_numeric(
            data[x_min_col],
            errors="coerce",
        ).to_numpy(float)

        ymin = pd.to_numeric(
            data[y_min_col],
            errors="coerce",
        ).to_numpy(float)

        xmax = pd.to_numeric(
            data[x_max_col],
            errors="coerce",
        ).to_numpy(float)

        ymax = pd.to_numeric(
            data[y_max_col],
            errors="coerce",
        ).to_numpy(float)

        coordinate_format = "bounds"

    else:
        xmin = pd.to_numeric(
            data[x_col],
            errors="coerce",
        ).to_numpy(float)

        ymin = pd.to_numeric(
            data[y_col],
            errors="coerce",
        ).to_numpy(float)

        widths = pd.to_numeric(
            data[width_col],
            errors="coerce",
        ).to_numpy(float)

        heights = pd.to_numeric(
            data[height_col],
            errors="coerce",
        ).to_numpy(float)

        xmax = xmin + widths

        ymax = ymin + heights

        coordinate_format = "origin_size"

    width = xmax - xmin

    height = ymax - ymin

    area = width * height

    screen_width = sx1 - sx0

    screen_height = sy1 - sy0

    screen_area = screen_width * screen_height

    area_prop = area / screen_area

    invalid_coordinate = (
        ~np.isfinite(xmin) | ~np.isfinite(ymin) | ~np.isfinite(xmax) | ~np.isfinite(ymax)
    )

    invalid_dimension = ~invalid_coordinate & ((width <= 0) | (height <= 0))

    outside_screen = ~invalid_coordinate & (
        (xmin < sx0) | (xmax > sx1) | (ymin < sy0) | (ymax > sy1)
    )

    too_small = (
        ~invalid_coordinate
        & ~invalid_dimension
        & ((width < min_width) | (height < min_height) | (area < min_area))
    )

    too_large = ~invalid_coordinate & ~invalid_dimension & (area_prop > max_area_prop)

    status = np.full(
        len(data),
        "ok",
        dtype=object,
    )

    status[too_large] = "too_large"

    status[too_small] = "too_small"

    if require_within_screen:
        status[outside_screen] = "outside_screen"

    status[invalid_dimension] = "invalid_dimension"

    status[invalid_coordinate] = "invalid_coordinate"

    keep_columns = [
        aoi_col,
    ]

    if stimulus_col is not None:
        keep_columns.append(stimulus_col)

    geometry_summary = data[keep_columns].copy()

    geometry_summary["x_min"] = xmin

    geometry_summary["y_min"] = ymin

    geometry_summary["x_max"] = xmax

    geometry_summary["y_max"] = ymax

    geometry_summary["width"] = width

    geometry_summary["height"] = height

    geometry_summary["area"] = area

    geometry_summary["area_prop"] = area_prop

    geometry_summary["center_x"] = xmin + width / 2

    geometry_summary["center_y"] = ymin + height / 2

    geometry_summary["outside_screen"] = outside_screen

    geometry_summary["aoi_geometry_status"] = status

    geometry_summary = _tibble(geometry_summary.reset_index(drop=True))

    size_summary = _tibble(
        pd.DataFrame(
            [
                {
                    "n_aois": len(geometry_summary),
                    "min_width": _safe_min(width),
                    "median_width": _safe_median(width),
                    "max_width": _safe_max(width),
                    "min_height": _safe_min(height),
                    "median_height": _safe_median(height),
                    "max_height": _safe_max(height),
                    "min_area": _safe_min(area),
                    "median_area": _safe_median(area),
                    "max_area": _safe_max(area),
                    "min_area_prop": _safe_min(area_prop),
                    "median_area_prop": _safe_median(area_prop),
                    "max_area_prop": _safe_max(area_prop),
                }
            ]
        )
    )

    duplicate_columns = []

    if stimulus_col is not None:
        duplicate_columns.append(stimulus_col)

    duplicate_columns.extend(
        [
            "x_min",
            "y_min",
            "x_max",
            "y_max",
        ]
    )

    duplicate_rows = []

    grouped = geometry_summary.groupby(
        duplicate_columns,
        dropna=False,
        sort=True,
    )

    for key, frame in grouped:
        if len(frame) <= 1:
            continue

        if not isinstance(
            key,
            tuple,
        ):
            key = (key,)

        row = dict(
            zip(
                duplicate_columns,
                key,
                strict=False,
            )
        )

        row["n_aois"] = len(frame)

        row["aoi_values"] = ", ".join(frame[aoi_col].astype(str))

        row["duplicate_geometry_status"] = "duplicate_geometry"

        duplicate_rows.append(row)

    duplicate_geometry = _tibble(pd.DataFrame(duplicate_rows))

    flagged_aois = _tibble(
        geometry_summary[geometry_summary["aoi_geometry_status"] != "ok"]
        .copy()
        .reset_index(drop=True)
    )

    n_duplicate_geometry = len(duplicate_geometry)

    if len(flagged_aois) > 0 or n_duplicate_geometry > 0:
        overview_status = "review"

    else:
        overview_status = "ok"

    overview = _tibble(
        pd.DataFrame(
            [
                {
                    "n_rows": len(data),
                    "n_aois": len(geometry_summary),
                    "n_stimuli": (
                        data[stimulus_col].nunique(dropna=False)
                        if stimulus_col is not None
                        else pd.NA
                    ),
                    "n_flagged_aois": len(flagged_aois),
                    "n_duplicate_geometry_groups": n_duplicate_geometry,
                    "coordinate_format": coordinate_format,
                    "screen_width": screen_width,
                    "screen_height": screen_height,
                    "screen_area": screen_area,
                    "aoi_geometry_status": overview_status,
                }
            ]
        )
    )

    settings = _tibble(
        pd.DataFrame(
            {
                "setting": [
                    "aoi_col",
                    "stimulus_col",
                    "x_min_col",
                    "y_min_col",
                    "x_max_col",
                    "y_max_col",
                    "x_col",
                    "y_col",
                    "width_col",
                    "height_col",
                    "screen_x_range",
                    "screen_y_range",
                    "min_width",
                    "min_height",
                    "min_area",
                    "max_area_prop",
                    "require_within_screen",
                ],
                "value": [
                    aoi_col,
                    _collapse_nullable(stimulus_col),
                    _collapse_nullable(x_min_col),
                    _collapse_nullable(y_min_col),
                    _collapse_nullable(x_max_col),
                    _collapse_nullable(y_max_col),
                    _collapse_nullable(x_col),
                    _collapse_nullable(y_col),
                    _collapse_nullable(width_col),
                    _collapse_nullable(height_col),
                    (f"{_r_character(sx0)}, {_r_character(sx1)}"),
                    (f"{_r_character(sy0)}, {_r_character(sy1)}"),
                    _r_character(min_width),
                    _r_character(min_height),
                    _r_character(min_area),
                    _r_character(max_area_prop),
                    _r_character(require_within_screen),
                ],
            }
        )
    )

    return R4List(
        {
            "overview": overview,
            "geometry_summary": geometry_summary,
            "size_summary": size_summary,
            "duplicate_geometry": duplicate_geometry,
            "flagged_aois": flagged_aois,
            "settings": settings,
        },
        r_class=("gp3_aoi_geometry_audit|list"),
    )


# ============================================================================
# Structural result normalization
# ============================================================================


def _detector_result(
    result: Any,
    *,
    result_class: str,
    events_class: str,
) -> Any:
    if not isinstance(
        result,
        Mapping,
    ):
        return result

    cleaned = {key: value for key, value in result.items() if key != "_gp3_class"}

    events = cleaned.get("events")

    if isinstance(
        events,
        pd.DataFrame,
    ):
        cleaned["events"] = _set_r_class(
            events,
            events_class,
        )

    return R4List(
        cleaned,
        r_class=(result_class),
    )


def _coding_matrix_result(
    result: Any,
) -> Any:
    if not isinstance(
        result,
        Mapping,
    ):
        return result

    names = [
        "overview",
        "geometry_summary",
        "sample_coding",
        "coding_matrix",
        "observed_summary",
        "derived_summary",
        "flagged_samples",
        "settings",
    ]

    if not all(name in result for name in names):
        return result

    cleaned = {}

    for name in names:
        value = result[name]

        if isinstance(
            value,
            pd.DataFrame,
        ):
            value = _tibble(value)

        cleaned[name] = value

    matrix = cleaned["coding_matrix"]

    if isinstance(
        matrix,
        pd.DataFrame,
    ) and {
        "n_samples",
        "sample_prop",
    }.issubset(matrix.columns):
        matrix = matrix.copy()

        samples = pd.to_numeric(
            matrix["n_samples"],
            errors="coerce",
        )

        denominator = samples.sum()

        if denominator > 0:
            matrix["sample_prop"] = samples / denominator

        cleaned["coding_matrix"] = _tibble(matrix)

    return R4List(
        cleaned,
        r_class=("gp3_aoi_coding_matrix_audit|list"),
    )


# ============================================================================
# Main dual-contract wrapper
# ============================================================================


def wrap_r4(
    function: Callable[..., Any],
    *,
    name: str,
) -> Callable[..., Any]:
    @wraps(function)
    def wrapped(
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        bound = _bind(
            function,
            args,
            kwargs,
        )

        if (
            name == "summarize_gazepoint_pupil_response_features"
            and isinstance(
                bound.get("data"),
                pd.DataFrame,
            )
            and all(
                key in bound
                for key in (
                    "pupil",
                    "time",
                    "subject",
                    "trial",
                    "baseline_window",
                    "response_window",
                )
            )
        ):
            return _pupil_response(
                bound["data"],
                pupil=bound["pupil"],
                time=bound["time"],
                subject=bound["subject"],
                trial=bound["trial"],
                baseline_window=bound["baseline_window"],
                response_window=bound["response_window"],
                condition=bound.get("condition"),
                interpolated=bound.get("interpolated"),
            )

        if name == "audit_gazepoint_master":
            master = bound.get("master")

            if master is None:
                master = bound.get("data")

            if _canonical_master(master):
                return _audit_master(master)

        if name == "summarise_gazepoint_workflow":
            results = bound.get("results")

            if results is None:
                results = bound.get("result")

            if isinstance(
                results,
                Mapping,
            ) and {
                "all_gaze",
                "all_fix",
                "sampling",
                "quality",
                "flagged_quality",
                "aoi_table",
            }.issubset(results.keys()):
                return _workflow_summary(results)

        if name == "add_gazepoint_aoi":
            master = bound.get("master_df")

            definitions = bound.get("aoi_defs")

            if (
                isinstance(
                    master,
                    pd.DataFrame,
                )
                and isinstance(
                    definitions,
                    pd.DataFrame,
                )
                and _resolve_aoi_columns(definitions) is not None
            ):
                return _static_aoi(
                    master,
                    definitions,
                    x_col=bound.get("x_col") or "FPOGX",
                    y_col=bound.get("y_col") or "FPOGY",
                    aoi_name=bound.get("aoi_name"),
                    output=bound.get("output") or "logical",
                    prefix=bound.get("prefix") or "aoi_",
                    label_col=bound.get("label_col") or "aoi_current",
                    outside_label=bound.get("outside_label") or "outside",
                    overlap=bound.get("overlap") or "first",
                    include_overlap_count=(
                        True
                        if bound.get("include_overlap_count") is None
                        else bool(bound["include_overlap_count"])
                    ),
                )

        if name == "audit_gazepoint_aoi_geometry" and isinstance(
            bound.get("data"),
            pd.DataFrame,
        ):
            return _geometry_audit(
                bound["data"],
                aoi_col=bound.get("aoi_col"),
                stimulus_col=bound.get("stimulus_col"),
                x_min_col=bound.get("x_min_col"),
                y_min_col=bound.get("y_min_col"),
                x_max_col=bound.get("x_max_col"),
                y_max_col=bound.get("y_max_col"),
                x_col=bound.get("x_col"),
                y_col=bound.get("y_col"),
                width_col=bound.get("width_col"),
                height_col=bound.get("height_col"),
                screen_x_range=bound.get("screen_x_range")
                or (
                    0,
                    1,
                ),
                screen_y_range=bound.get("screen_y_range")
                or (
                    0,
                    1,
                ),
                min_width=bound.get(
                    "min_width",
                    0,
                ),
                min_height=bound.get(
                    "min_height",
                    0,
                ),
                min_area=bound.get(
                    "min_area",
                    0,
                ),
                max_area_prop=bound.get(
                    "max_area_prop",
                    1,
                ),
                require_within_screen=bound.get(
                    "require_within_screen",
                    True,
                ),
            )

        result = function(
            *args,
            **kwargs,
        )

        if name in {
            "summarise_gazepoint_qc_status",
            "summarize_gazepoint_qc_status",
        }:
            qc_bundle = bound.get("qc_bundle")

            if qc_bundle is None:
                qc_bundle = bound.get("data")

            canonical = _qc_status_summary(
                result,
                qc_bundle,
            )

            if canonical is not None:
                return canonical

            return result

        if name == "detect_gazepoint_blinks":
            return_mode = bound.get("return") or bound.get("return_mode")

            if return_mode == "both":
                return _detector_result(
                    result,
                    result_class=("gp3_blink_detection_result|list"),
                    events_class=("gp3_blink_events|tbl_df|tbl|data.frame"),
                )

        if name == "detect_gazepoint_fixations_velocity":
            return_mode = bound.get("return") or bound.get("return_mode")

            if return_mode == "both":
                return _detector_result(
                    result,
                    result_class=("gp3_velocity_fixation_result|list"),
                    events_class=("gp3_velocity_fixations|tbl_df|tbl|data.frame"),
                )

        if (
            name
            in {
                "add_gazepoint_dynamic_aoi",
                "add_gazepoint_polygon_aoi",
            }
            and isinstance(
                result,
                pd.DataFrame,
            )
            and "aoi_overlap_count" in result.columns
        ):
            result = result.copy()

            result["aoi_overlap_count"] = pd.to_numeric(
                result["aoi_overlap_count"],
                errors="coerce",
            ).astype(float)

        if name == "audit_gazepoint_aoi_coding_matrix":
            return _coding_matrix_result(result)

        return result

    return wrapped


# === R4 VIRTUAL LEGACY CLASS KEY ===


def _r4list_legacy_getitem(
    self,
    key,
):
    """Expose historical Python class metadata without making it an R list element."""
    if key == "_gp3_class":
        value = getattr(
            self,
            "r_class",
            "list",
        )

        return str(value).split(
            "|",
            1,
        )[0]

    return dict.__getitem__(
        self,
        key,
    )


R4List.__getitem__ = _r4list_legacy_getitem

# === R4 LEGACY DETECTOR RESULT BRIDGE ===


def legacy_detector_result_bridge(
    function,
    *,
    result_class,
):
    """Restore historical Python detector bundle shape on legacy calls."""

    @wraps(function)
    def wrapper(
        *args,
        **kwargs,
    ):
        canonical_r_call = kwargs.get("all_gaze") is not None or "return" in kwargs

        result = function(
            *args,
            **kwargs,
        )

        return_mode = kwargs.get("return_mode") or kwargs.get("return")

        if (
            canonical_r_call
            or return_mode != "both"
            or not isinstance(
                result,
                Mapping,
            )
        ):
            return result

        if "_gp3_class" in dict(result):
            return result

        legacy = dict(result)

        legacy["_gp3_class"] = result_class

        return legacy

    return wrapper


# === R4 GEOMETRY VALIDATION BRIDGE ===


def geometry_validation_bridge(
    function,
):
    """Apply canonical R scalar/range checks before either geometry backend."""

    @wraps(function)
    def wrapper(
        *args,
        **kwargs,
    ):
        for name in (
            "min_width",
            "min_height",
            "min_area",
        ):
            if name not in kwargs:
                continue

            value = kwargs[name]

            if isinstance(
                value,
                bool,
            ) or not np.isscalar(value):
                raise ValueError(f"`{name}` must be a non-negative numeric scalar.")

            try:
                number = float(value)

            except (
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(f"`{name}` must be a non-negative numeric scalar.") from exc

            if not np.isfinite(number) or number < 0:
                raise ValueError(f"`{name}` must be a non-negative numeric scalar.")

        if "max_area_prop" in kwargs:
            value = kwargs["max_area_prop"]

            if isinstance(
                value,
                bool,
            ) or not np.isscalar(value):
                raise ValueError("`max_area_prop` must be between 0 and 1.")

            try:
                number = float(value)

            except (
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError("`max_area_prop` must be between 0 and 1.") from exc

            if not np.isfinite(number) or number < 0 or number > 1:
                raise ValueError("`max_area_prop` must be between 0 and 1.")

        for name in (
            "screen_x_range",
            "screen_y_range",
        ):
            if name not in kwargs:
                continue

            value = kwargs[name]

            try:
                values = np.asarray(
                    value,
                    dtype=float,
                ).reshape(-1)

            except (
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(f"`{name}` must contain two finite increasing values.") from exc

            if len(values) != 2 or not np.isfinite(values).all() or values[0] >= values[1]:
                raise ValueError(f"`{name}` must contain two finite increasing values.")

        if "require_within_screen" in kwargs and not isinstance(
            kwargs["require_within_screen"],
            (
                bool,
                np.bool_,
            ),
        ):
            raise ValueError("`require_within_screen` must be a logical scalar.")

        return function(
            *args,
            **kwargs,
        )

    return wrapper
