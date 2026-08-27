from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

# =====================================================================
# GENERIC HELPERS
# =====================================================================


def _as_frame(
    data: Any,
    *,
    arg: str = "data",
) -> pd.DataFrame:
    if not isinstance(
        data,
        pd.DataFrame,
    ):
        raise TypeError(f"`{arg}` must be a data frame.")

    return data.copy()


def _first_not_none(
    *values: Any,
) -> Any:
    for value in values:
        if value is not None:
            return value

    return None


def _as_list(
    value: Any,
) -> list[Any]:
    if value is None:
        return []

    if isinstance(
        value,
        str,
    ):
        return [value]

    if isinstance(
        value,
        Sequence,
    ):
        return list(value)

    return [value]


def _check_cols(
    data: pd.DataFrame,
    cols: Iterable[str | None],
) -> None:
    required = [str(column) for column in cols if column is not None and str(column)]

    missing = [column for column in required if column not in data.columns]

    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))


def _num(
    series: pd.Series,
) -> pd.Series:
    return pd.to_numeric(
        series,
        errors="coerce",
    )


def _mean_pair(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:
    values = pd.concat(
        [
            _num(left),
            _num(right),
        ],
        axis=1,
    )

    return values.mean(
        axis=1,
        skipna=True,
    )


def _r_sd(
    values: pd.Series,
) -> float:
    clean = _num(values).dropna()

    if len(clean) < 2:
        return math.nan

    return float(clean.std(ddof=1))


def _median(
    values: pd.Series,
) -> float:
    clean = _num(values).dropna()

    if clean.empty:
        return math.nan

    return float(clean.median())


def _mean(
    values: pd.Series,
) -> float:
    clean = _num(values).dropna()

    if clean.empty:
        return math.nan

    return float(clean.mean())


def _min(
    values: pd.Series,
) -> float:
    clean = _num(values).dropna()

    if clean.empty:
        return math.nan

    return float(clean.min())


def _max(
    values: pd.Series,
) -> float:
    clean = _num(values).dropna()

    if clean.empty:
        return math.nan

    return float(clean.max())


def _safe_cor(
    left: pd.Series,
    right: pd.Series,
) -> tuple[int, float]:
    raise RuntimeError("Superseded internal R3-A implementation")


def _attach_gp3_attr(
    frame: pd.DataFrame,
    name: str,
    value: Any,
) -> pd.DataFrame:
    frame.attrs = dict(frame.attrs)

    frame.attrs[name] = value

    return frame


# =====================================================================
# CREATE GAZEPOINT MASTER
# =====================================================================


_MASTER_COLUMNS = [
    "subject",
    "pID",
    "USER_FILE",
    "MEDIA_ID",
    "MEDIA_NAME",
    "trial",
    "trial_global",
    "condition",
    "group",
    "item_id",
    "stimulus_id",
    "stimulus_file",
    "time",
    "time_orig",
    "sample_index",
    "sampling_rate_hz",
    "time_bin_25ms",
    "time_bin_50ms",
    "time_bin_100ms",
    "baseline_window",
    "analysis_window",
    "x",
    "y",
    "left_x",
    "left_y",
    "right_x",
    "right_y",
    "left_pupil",
    "right_pupil",
    "mean_pupil",
    "pupil",
    "pupil_raw",
    "pupil_unit",
    "gaze_unit",
    "valid_sample",
    "left_valid",
    "right_valid",
    "missing_gaze",
    "missing_pupil",
    "trackloss",
    "Trackloss",
    "blink",
    "gaze_offscreen",
    "interpolated",
    "filtered",
    "artifact_flag",
    "artifact_reason",
    "AOI",
    "aoi_current",
    "aoi_count",
    "message",
    "event_type",
    "event_label",
    "event_latency_offset_ms",
    "stimulus_onset_time",
    "target_onset_time",
    "response_time_orig",
    "response_time",
    "tracker_model",
    "tracker_sampling_rate",
    "calibration_quality",
    "validation_error_deg",
    "drift_correction_error",
    "response",
    "correct_response",
    "accuracy",
    "rt",
    "choice",
    "rating",
    "trust_rating",
    "risk_rating",
    "purchase_intention",
    "excluded_trial",
    "exclusion_reason",
]


def _empty_na(
    n: int,
) -> pd.Series:
    return pd.Series(
        [np.nan] * n,
        dtype=object,
    )


def _detect_sampling_rate(
    time_seconds: pd.Series,
) -> float:
    values = _num(time_seconds).dropna()

    if len(values) < 2:
        return math.nan

    diffs = values.sort_values().diff().dropna()

    diffs = diffs[diffs > 0]

    if diffs.empty:
        return math.nan

    delta = float(diffs.median())

    if delta <= 0:
        return math.nan

    return float(1.0 / delta)


def _r3a_create_gazepoint_master(
    data: pd.DataFrame | None = None,
    gaze_data: pd.DataFrame | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    source = _first_not_none(
        gaze_data,
        data,
    )

    gaze = _as_frame(
        source,
        arg="gaze_data",
    )

    required = [
        "USER_FILE",
        "MEDIA_ID",
        "TIME",
    ]

    missing = [column for column in required if column not in gaze.columns]

    if missing:
        raise ValueError("Missing required columns in `gaze_data`: " + ", ".join(missing))

    n = len(gaze)

    out = pd.DataFrame(index=np.arange(n))

    user_file = gaze["USER_FILE"].astype("string")

    user = gaze["USER"].astype("string") if "USER" in gaze.columns else user_file

    media_id = gaze["MEDIA_ID"].astype("string")

    media_name = gaze["MEDIA_NAME"].astype("string") if "MEDIA_NAME" in gaze.columns else media_id

    time_orig = _num(gaze["TIME"])

    # Canonical master uses milliseconds internally.
    time_ms = time_orig * 1000.0

    out["subject"] = user_file.astype(object)

    out["pID"] = user.astype(object)

    out["USER_FILE"] = user_file.astype(object)

    out["MEDIA_ID"] = media_id.astype(object)

    out["MEDIA_NAME"] = media_name.astype(object)

    group_key = pd.DataFrame(
        {
            "subject": user_file,
            "media": media_id,
        }
    )

    trial_codes = (
        group_key.drop_duplicates()
        .groupby(
            "subject",
            sort=False,
            dropna=False,
        )
        .cumcount()
        + 1
    )

    trial_lookup = group_key.drop_duplicates().assign(trial=trial_codes.to_numpy())

    keyed = group_key.merge(
        trial_lookup,
        on=[
            "subject",
            "media",
        ],
        how="left",
        sort=False,
    )

    out["trial"] = keyed["trial"].astype(int)

    global_lookup = group_key.drop_duplicates().reset_index(drop=True)

    global_lookup["trial_global"] = np.arange(
        1,
        len(global_lookup) + 1,
    )

    keyed_global = group_key.merge(
        global_lookup,
        on=[
            "subject",
            "media",
        ],
        how="left",
        sort=False,
    )

    out["trial_global"] = keyed_global["trial_global"].astype(int)

    for column in [
        "condition",
        "group",
        "item_id",
        "stimulus_id",
        "stimulus_file",
    ]:
        if column in gaze.columns:
            out[column] = gaze[column].to_numpy()
        else:
            out[column] = _empty_na(n)

    out["time"] = time_ms

    out["time_orig"] = time_orig

    out["sample_index"] = (
        group_key.groupby(
            [
                "subject",
                "media",
            ],
            sort=False,
            dropna=False,
        )
        .cumcount()
        .to_numpy()
        + 1
    )

    sampling = np.full(
        n,
        np.nan,
        dtype=float,
    )

    for _, indices in group_key.groupby(
        [
            "subject",
            "media",
        ],
        sort=False,
        dropna=False,
    ).groups.items():
        indices = list(indices)

        hz = _detect_sampling_rate(time_orig.iloc[indices])

        sampling[indices] = hz

    out["sampling_rate_hz"] = sampling

    out["time_bin_25ms"] = np.floor(time_ms / 25.0) * 25.0

    out["time_bin_50ms"] = np.floor(time_ms / 50.0) * 50.0

    out["time_bin_100ms"] = np.floor(time_ms / 100.0) * 100.0

    out["baseline_window"] = False

    out["analysis_window"] = False

    x_col = next(
        (
            column
            for column in [
                "BPOGX",
                "FPOGX",
                "x",
            ]
            if column in gaze.columns
        ),
        None,
    )

    y_col = next(
        (
            column
            for column in [
                "BPOGY",
                "FPOGY",
                "y",
            ]
            if column in gaze.columns
        ),
        None,
    )

    out["x"] = _num(gaze[x_col]) if x_col else np.nan

    out["y"] = _num(gaze[y_col]) if y_col else np.nan

    eye_map = {
        "left_x": [
            "LPOGX",
            "left_x",
        ],
        "left_y": [
            "LPOGY",
            "left_y",
        ],
        "right_x": [
            "RPOGX",
            "right_x",
        ],
        "right_y": [
            "RPOGY",
            "right_y",
        ],
    }

    for target, candidates in eye_map.items():
        source_col = next(
            (candidate for candidate in candidates if candidate in gaze.columns),
            None,
        )

        out[target] = _num(gaze[source_col]) if source_col else np.nan

    left_pupil_col = next(
        (
            column
            for column in [
                "LPD",
                "left_pupil",
            ]
            if column in gaze.columns
        ),
        None,
    )

    right_pupil_col = next(
        (
            column
            for column in [
                "RPD",
                "right_pupil",
            ]
            if column in gaze.columns
        ),
        None,
    )

    left_pupil = (
        _num(gaze[left_pupil_col])
        if left_pupil_col
        else pd.Series(
            np.nan,
            index=gaze.index,
        )
    )

    right_pupil = (
        _num(gaze[right_pupil_col])
        if right_pupil_col
        else pd.Series(
            np.nan,
            index=gaze.index,
        )
    )

    mean_pupil = _mean_pair(
        left_pupil,
        right_pupil,
    )

    out["left_pupil"] = left_pupil

    out["right_pupil"] = right_pupil

    out["mean_pupil"] = mean_pupil

    out["pupil"] = mean_pupil

    out["pupil_raw"] = mean_pupil

    out["pupil_unit"] = "unknown"

    out["gaze_unit"] = "normalized"

    left_valid = (
        gaze["LPDV"].fillna(0).astype(float) > 0 if "LPDV" in gaze.columns else left_pupil.notna()
    )

    right_valid = (
        gaze["RPDV"].fillna(0).astype(float) > 0 if "RPDV" in gaze.columns else right_pupil.notna()
    )

    gaze_valid = (
        gaze["BPOGV"].fillna(0).astype(float) > 0
        if "BPOGV" in gaze.columns
        else (out["x"].notna() & out["y"].notna())
    )

    out["valid_sample"] = gaze_valid | left_valid | right_valid

    out["left_valid"] = left_valid.to_numpy()

    out["right_valid"] = right_valid.to_numpy()

    out["missing_gaze"] = (~gaze_valid).to_numpy()

    out["missing_pupil"] = mean_pupil.isna().to_numpy()

    out["trackloss"] = out["missing_gaze"] & out["missing_pupil"]

    out["Trackloss"] = out["trackloss"]

    out["blink"] = False

    x_numeric = _num(out["x"])

    y_numeric = _num(out["y"])

    out["gaze_offscreen"] = (
        (x_numeric < 0) | (x_numeric > 1) | (y_numeric < 0) | (y_numeric > 1)
    ).fillna(False)

    out["interpolated"] = False

    out["filtered"] = False

    out["artifact_flag"] = False

    out["artifact_reason"] = _empty_na(n)

    out["AOI"] = _empty_na(n)

    out["aoi_current"] = _empty_na(n)

    out["aoi_count"] = 0

    for column in [
        "message",
        "event_type",
        "event_label",
        "event_latency_offset_ms",
        "stimulus_onset_time",
        "target_onset_time",
        "response_time_orig",
        "response_time",
        "tracker_model",
        "tracker_sampling_rate",
        "calibration_quality",
        "validation_error_deg",
        "drift_correction_error",
        "response",
        "correct_response",
        "accuracy",
        "rt",
        "choice",
        "rating",
        "trust_rating",
        "risk_rating",
        "purchase_intention",
    ]:
        if column in gaze.columns:
            out[column] = gaze[column].to_numpy()
        else:
            out[column] = _empty_na(n)

    out["excluded_trial"] = False

    out["exclusion_reason"] = _empty_na(n)

    out = out[_MASTER_COLUMNS].reset_index(drop=True)

    out.attrs = {}

    return out


# =====================================================================
# IVT FIXATION DETECTION
# =====================================================================


def _ivt_one_group(
    block: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    time_col: str,
    velocity_threshold: float,
    min_duration_ms: float,
    distance_scale: float,
    time_scale: float,
    group_cols: list[str],
) -> pd.DataFrame:
    raise RuntimeError("Superseded internal R3-A implementation")


def _r3a_detect_gazepoint_fixations_ivt(
    data: pd.DataFrame,
    x_col: str = "x",
    y_col: str = "y",
    time_col: str = "time",
    group_cols: Sequence[str] | str | None = None,
    velocity_threshold: float = 30.0,
    min_duration_ms: float = 100.0,
    distance_scale: float = 1.0,
    time_scale: float = 1.0,
    **kwargs: Any,
) -> pd.DataFrame:
    frame = _as_frame(data)

    group_cols = _as_list(group_cols)

    _check_cols(
        frame,
        [
            x_col,
            y_col,
            time_col,
            *group_cols,
        ],
    )

    if not group_cols:
        return _ivt_one_group(
            frame,
            x_col=x_col,
            y_col=y_col,
            time_col=time_col,
            velocity_threshold=velocity_threshold,
            min_duration_ms=min_duration_ms,
            distance_scale=distance_scale,
            time_scale=time_scale,
            group_cols=[],
        )

    outputs = []

    grouper: Any = group_cols[0] if len(group_cols) == 1 else group_cols

    for _, block in frame.groupby(
        grouper,
        sort=False,
        dropna=False,
    ):
        outputs.append(
            _ivt_one_group(
                block,
                x_col=x_col,
                y_col=y_col,
                time_col=time_col,
                velocity_threshold=velocity_threshold,
                min_duration_ms=min_duration_ms,
                distance_scale=distance_scale,
                time_scale=time_scale,
                group_cols=group_cols,
            )
        )

    if not outputs:
        return pd.DataFrame()

    return pd.concat(
        outputs,
        ignore_index=True,
    )


# =====================================================================
# EVENT DETECTOR AGREEMENT
# =====================================================================


_EVENT_NON_SEQUENCE = {
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
}


def _event_iou(
    row: pd.Series,
    others: pd.DataFrame,
) -> float:
    if others.empty:
        return 0.0

    start_a = float(row["start_time"])

    end_a = float(row["end_time"])

    start_b = _num(others["start_time"]).to_numpy(dtype=float)

    end_b = _num(others["end_time"]).to_numpy(dtype=float)

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

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):
        iou = np.where(
            np.isfinite(union) & (union > 0),
            intersection / union,
            0.0,
        )

    if not len(iou):
        return 0.0

    return float(np.nanmax(iou))


def _detector_summary(
    events: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for detector in pd.unique(events["detector"].astype(str)):
        block = events[events["detector"].astype(str) == detector]

        durations = _num(block["duration_ms"])

        finite = durations[np.isfinite(durations)]

        threshold_values = _num(block["threshold"])

        threshold_values = threshold_values[np.isfinite(threshold_values)]

        rows.append(
            {
                "detector": detector,
                "family": str(block["family"].iloc[0]),
                "threshold": (
                    float(threshold_values.iloc[0]) if len(threshold_values) else math.nan
                ),
                "n_fixations": int(len(block)),
                "mean_duration_ms": (float(finite.mean()) if len(finite) else math.nan),
                "median_duration_ms": (float(finite.median()) if len(finite) else math.nan),
                "total_duration_ms": (float(finite.sum()) if len(finite) else math.nan),
            }
        )

    return pd.DataFrame(
        rows,
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


def _r3a_summarise_gazepoint_event_detector_agreement(
    data: pd.DataFrame | Mapping[str, Any] | None = None,
    x: pd.DataFrame | Mapping[str, Any] | None = None,
    min_overlap: float = 0.5,
    **kwargs: Any,
) -> dict[
    str,
    Any,
]:
    source = _first_not_none(
        x,
        data,
    )

    comparison_settings = None

    if isinstance(
        source,
        Mapping,
    ):
        if "events" not in source:
            raise ValueError("`x` must be a detector-comparison object or event data frame.")

        events = _as_frame(source["events"])

        comparison_settings = source.get("settings")
    else:
        events = _as_frame(
            source,
            arg="x",
        )

    required = [
        "detector",
        "start_time",
        "end_time",
        "duration_ms",
    ]

    _check_cols(
        events,
        required,
    )

    if not (0 <= float(min_overlap) <= 1):
        raise ValueError("`min_overlap` must be between 0 and 1.")

    for column, default in {
        "family": "",
        "threshold": np.nan,
        "event_id": np.arange(
            1,
            len(events) + 1,
        ),
        "mean_x": np.nan,
        "mean_y": np.nan,
        "n_samples": np.nan,
        "source_status": "",
    }.items():
        if column not in events.columns:
            events[column] = default

    sequence_cols = [column for column in events.columns if column not in _EVENT_NON_SEQUENCE]

    detector_summary = _detector_summary(events)

    detectors = list(pd.unique(events["detector"].astype(str)))

    pair_rows = []
    unmatched_rows = []

    for detector_a, detector_b in itertools.combinations(
        detectors,
        2,
    ):
        events_a = events[events["detector"].astype(str) == detector_a].copy()

        events_b = events[events["detector"].astype(str) == detector_b].copy()

        if sequence_cols:
            keys = (
                pd.concat(
                    [
                        events_a[sequence_cols],
                        events_b[sequence_cols],
                    ],
                    ignore_index=True,
                )
                .drop_duplicates()
                .reset_index(drop=True)
            )

            key_records = keys.to_dict("records")
        else:
            key_records = [{}]

        for key_values in key_records:
            mask_a = pd.Series(
                True,
                index=events_a.index,
            )

            mask_b = pd.Series(
                True,
                index=events_b.index,
            )

            for column, value in key_values.items():
                if pd.isna(value):
                    mask_a &= events_a[column].isna()

                    mask_b &= events_b[column].isna()
                else:
                    mask_a &= events_a[column] == value

                    mask_b &= events_b[column] == value

            block_a = events_a[mask_a].copy()

            block_b = events_b[mask_b].copy()

            overlap_a = np.array(
                [
                    _event_iou(
                        row,
                        block_b,
                    )
                    for _, row in block_a.iterrows()
                ],
                dtype=float,
            )

            overlap_b = np.array(
                [
                    _event_iou(
                        row,
                        block_a,
                    )
                    for _, row in block_b.iterrows()
                ],
                dtype=float,
            )

            matched_a = int(np.sum(overlap_a >= float(min_overlap)))

            matched_b = int(np.sum(overlap_b >= float(min_overlap)))

            row = dict(key_values)

            row.update(
                {
                    "detector_a": detector_a,
                    "detector_b": detector_b,
                    "n_a": int(len(block_a)),
                    "n_b": int(len(block_b)),
                    "matched_a": matched_a,
                    "matched_b": matched_b,
                    "agreement_a": (matched_a / len(block_a) if len(block_a) else math.nan),
                    "agreement_b": (matched_b / len(block_b) if len(block_b) else math.nan),
                    "mean_best_overlap_a": (
                        float(np.mean(overlap_a)) if len(overlap_a) else math.nan
                    ),
                    "mean_best_overlap_b": (
                        float(np.mean(overlap_b)) if len(overlap_b) else math.nan
                    ),
                    "min_overlap": float(min_overlap),
                }
            )

            pair_rows.append(row)

            for block, overlaps, compared_with in [
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
                for position, (_, event_row) in enumerate(block.iterrows()):
                    if overlaps[position] >= float(min_overlap):
                        continue

                    out_row = event_row.to_dict()

                    out_row["compared_with"] = compared_with

                    out_row["best_overlap"] = float(overlaps[position])

                    unmatched_rows.append(out_row)

    pair_columns = sequence_cols + [
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

    pairwise = pd.DataFrame(
        pair_rows,
        columns=pair_columns,
    )

    unmatched_columns = list(events.columns) + [
        "compared_with",
        "best_overlap",
    ]

    unmatched = pd.DataFrame(
        unmatched_rows,
        columns=unmatched_columns,
    )

    settings_sequence = sequence_cols

    if isinstance(
        comparison_settings,
        Mapping,
    ):
        candidate = comparison_settings.get("sequence_cols")

        if candidate is not None:
            settings_sequence = list(candidate)

    return {
        "detector_summary": detector_summary.reset_index(drop=True),
        "pairwise_agreement": pairwise.reset_index(drop=True),
        "unmatched_events": unmatched.reset_index(drop=True),
        "settings": {
            "sequence_cols": settings_sequence,
            "min_overlap": float(min_overlap),
        },
    }


# =====================================================================
# FACE WINDOW SUMMARIZATION
# =====================================================================


def _face_measure_columns(
    data: pd.DataFrame,
    supplied: Sequence[str] | str | None,
    exclude: set[str],
) -> list[str]:
    if supplied is not None:
        result = _as_list(supplied)

        _check_cols(
            data,
            result,
        )

        return [str(value) for value in result]

    result = []

    for column in data.columns:
        if column in exclude:
            continue

        numeric = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        if numeric.notna().any():
            result.append(column)

    return result


def _face_summary_row(
    block: pd.DataFrame,
    *,
    group_values: Mapping[str, Any],
    window_id: Any,
    window_label: Any,
    window_start: float,
    window_end: float,
    measure_cols: list[str],
    validity_col: str | None,
    confidence_col: str | None,
    require_valid: bool,
) -> dict[
    str,
    Any,
]:
    n_rows = int(len(block))

    if validity_col is not None:
        valid_mask = block[validity_col].fillna(False).astype(bool)
    else:
        valid_mask = pd.Series(
            True,
            index=block.index,
        )

    n_valid = int(valid_mask.sum())

    n_invalid = int(n_rows - n_valid)

    used = block[valid_mask] if require_valid else block

    row: dict[
        str,
        Any,
    ] = dict(group_values)

    row.update(
        {
            "face_window_id": window_id,
            "face_window_label": window_label,
            "window_start_sec": float(window_start),
            "window_end_sec": float(window_end),
            "n_rows": n_rows,
            "n_used": int(len(used)),
            "n_valid": n_valid,
            "n_invalid": n_invalid,
            "valid_percent": (100.0 * n_valid / n_rows if n_rows else math.nan),
        }
    )

    if confidence_col is not None and confidence_col in used.columns:
        confidence = _num(used[confidence_col])

        row["face_confidence_mean"] = _mean(confidence)

        row["face_confidence_median"] = _median(confidence)
    else:
        row["face_confidence_mean"] = math.nan

        row["face_confidence_median"] = math.nan

    for measure in measure_cols:
        values = _num(used[measure])

        clean = values.dropna()

        row[f"{measure}_n"] = int(len(clean))

        row[f"{measure}_mean"] = _mean(values)

        row[f"{measure}_median"] = _median(values)

        row[f"{measure}_sd"] = _r_sd(values)

        row[f"{measure}_min"] = _min(values)

        row[f"{measure}_max"] = _max(values)

    return row


def _r3a_summarize_gazepoint_face_windows(
    data: pd.DataFrame,
    windows: pd.DataFrame | None = None,
    time_col: str | None = None,
    window_start_col: str = "window_start_sec",
    window_end_col: str = "window_end_sec",
    group_cols: Sequence[str] | str | None = None,
    window_id_col: str | None = None,
    window_label_col: str | None = None,
    measure_cols: Sequence[str] | str | None = None,
    validity_col: str | None = None,
    confidence_col: str | None = None,
    require_valid: bool = True,
    include_empty_windows: bool = True,
    **kwargs: Any,
) -> pd.DataFrame:
    frame = _as_frame(data)

    group_cols = [str(value) for value in _as_list(group_cols)]

    if time_col is None:
        time_col = next(
            (
                column
                for column in [
                    "face_time_sec",
                    "time_sec",
                    "timestamp_sec",
                    "timestamp",
                    "trial_time_sec",
                    "relative_time_sec",
                    "time",
                ]
                if column in frame.columns
            ),
            None,
        )

    if windows is not None and time_col is None:
        raise ValueError("`time_col` could not be detected automatically. Please supply it.")

    if validity_col is None:
        validity_col = next(
            (
                column
                for column in [
                    "face_valid",
                    "valid",
                    "success_valid",
                ]
                if column in frame.columns
            ),
            None,
        )

    if confidence_col is None:
        confidence_col = next(
            (
                column
                for column in [
                    "face_confidence",
                    "confidence",
                    "detection_confidence",
                ]
                if column in frame.columns
            ),
            None,
        )

    exclude = set(group_cols)

    for value in [
        time_col,
        window_start_col,
        window_end_col,
        window_id_col,
        window_label_col,
        validity_col,
        confidence_col,
    ]:
        if value:
            exclude.add(value)

    measures = _face_measure_columns(
        frame,
        measure_cols,
        exclude,
    )

    if not measures:
        raise ValueError(
            "No numeric facial-behaviour measure columns were found. Supply `measure_cols`."
        )

    rows = []

    if windows is not None:
        window_table = _as_frame(
            windows,
            arg="windows",
        )

        _check_cols(
            window_table,
            [
                window_start_col,
                window_end_col,
                *group_cols,
            ],
        )

        for _, window in window_table.iterrows():
            mask = pd.Series(
                True,
                index=frame.index,
            )

            group_values: dict[
                str,
                Any,
            ] = {}

            for column in group_cols:
                value = window[column]

                group_values[column] = value

                if pd.isna(value):
                    mask &= frame[column].isna()
                else:
                    mask &= frame[column] == value

            start = float(window[window_start_col])

            end = float(window[window_end_col])

            numeric_time = _num(frame[time_col])

            mask &= numeric_time >= start

            mask &= numeric_time <= end

            block = frame[mask]

            if block.empty and not include_empty_windows:
                continue

            window_id = (
                window[window_id_col]
                if (window_id_col is not None and window_id_col in window.index)
                else np.nan
            )

            window_label = (
                window[window_label_col]
                if (window_label_col is not None and window_label_col in window.index)
                else window_id
            )

            rows.append(
                _face_summary_row(
                    block,
                    group_values=group_values,
                    window_id=window_id,
                    window_label=window_label,
                    window_start=start,
                    window_end=end,
                    measure_cols=measures,
                    validity_col=validity_col,
                    confidence_col=confidence_col,
                    require_valid=require_valid,
                )
            )
    else:
        raise ValueError("Canonical R3-A path requires `windows` for this implementation.")

    columns = group_cols + [
        "face_window_id",
        "face_window_label",
        "window_start_sec",
        "window_end_sec",
        "n_rows",
        "n_used",
        "n_valid",
        "n_invalid",
        "valid_percent",
        "face_confidence_mean",
        "face_confidence_median",
    ]

    for measure in measures:
        columns.extend(
            [
                f"{measure}_n",
                f"{measure}_mean",
                f"{measure}_median",
                f"{measure}_sd",
                f"{measure}_min",
                f"{measure}_max",
            ]
        )

    out = pd.DataFrame(
        rows,
        columns=columns,
    )

    settings = {
        "time_col": time_col,
        "window_start_col": window_start_col,
        "window_end_col": window_end_col,
        "group_cols": group_cols,
        "window_id_col": window_id_col,
        "window_label_col": window_label_col,
        "measure_cols": measures,
        "validity_col": validity_col,
        "confidence_col": confidence_col,
        "require_valid": bool(require_valid),
        "include_empty_windows": bool(include_empty_windows),
        "used_window_table": True,
    }

    return _attach_gp3_attr(
        out,
        "gp3_face_window_settings",
        settings,
    )


# =====================================================================
# FACE REACTIVITY
# =====================================================================


def _r3a_summarize_gazepoint_face_reactivity(
    data: pd.DataFrame,
    baseline_window: Sequence[str] | str,
    response_window: Sequence[str] | str,
    group_cols: Sequence[str] | str | None = None,
    window_col: str | None = None,
    measure_cols: Sequence[str] | str | None = None,
    statistic: str | Sequence[str] = "mean",
    **kwargs: Any,
) -> pd.DataFrame:
    frame = _as_frame(data)

    group_cols = [str(value) for value in _as_list(group_cols)]

    if isinstance(
        statistic,
        Sequence,
    ) and not isinstance(
        statistic,
        str,
    ):
        statistic = str(list(statistic)[0])

    statistic = str(statistic)

    if statistic not in {
        "mean",
        "median",
    }:
        raise ValueError("`statistic` must be 'mean' or 'median'.")

    suffix = "_mean" if statistic == "mean" else "_median"

    if window_col is None:
        window_col = next(
            (
                column
                for column in [
                    "face_window_label",
                    "window_label",
                    "window",
                    "phase",
                    "task_phase",
                    "face_window_id",
                    "window_id",
                ]
                if column in frame.columns
            ),
            None,
        )

    if window_col is None:
        raise ValueError("`window_col` could not be detected automatically. Please supply it.")

    _check_cols(
        frame,
        group_cols,
    )

    if measure_cols is not None:
        base_measures = [str(value) for value in _as_list(measure_cols)]

        value_cols = [
            (measure if measure.endswith(suffix) else (measure + suffix))
            for measure in base_measures
        ]

        _check_cols(
            frame,
            value_cols,
        )
    else:
        value_cols = [
            column
            for column in frame.columns
            if column.endswith(suffix)
            and not any(
                token in column.lower()
                for token in [
                    "confidence",
                    "valid",
                    "sampling",
                    "time",
                    "n_rows",
                    "n_used",
                ]
            )
        ]

    if not value_cols:
        raise ValueError("No reactivity measure columns were found. Supply `measure_cols`.")

    baseline_values = [str(value) for value in _as_list(baseline_window)]

    response_values = [str(value) for value in _as_list(response_window)]

    window_text = frame[window_col].astype(str)

    baseline = frame[window_text.isin(baseline_values)].copy()

    response = frame[window_text.isin(response_values)].copy()

    if baseline.empty:
        raise ValueError("No baseline-window rows were found.")

    if response.empty:
        raise ValueError("No response-window rows were found.")

    if group_cols:
        combined_groups = (
            pd.concat(
                [
                    baseline[group_cols],
                    response[group_cols],
                ],
                ignore_index=True,
            )
            .drop_duplicates()
            .reset_index(drop=True)
        )

        group_records = combined_groups.to_dict("records")
    else:
        group_records = [{}]

    rows = []

    for group_values in group_records:
        mask_base = pd.Series(
            True,
            index=baseline.index,
        )

        mask_response = pd.Series(
            True,
            index=response.index,
        )

        for column, value in group_values.items():
            if pd.isna(value):
                mask_base &= baseline[column].isna()

                mask_response &= response[column].isna()
            else:
                mask_base &= baseline[column] == value

                mask_response &= response[column] == value

        block_base = baseline[mask_base]

        block_response = response[mask_response]

        for value_col in value_cols:
            baseline_value = _mean(block_base[value_col])

            response_value = _mean(block_response[value_col])

            delta = response_value - baseline_value

            if math.isnan(baseline_value) or math.isnan(response_value) or baseline_value == 0:
                percent = math.nan
            else:
                percent = 100.0 * delta / abs(baseline_value)

            measure = value_col[: -len(suffix)]

            row: dict[
                str,
                Any,
            ] = dict(group_values)

            row.update(
                {
                    "measure": measure,
                    "statistic": statistic,
                    "baseline_window": " | ".join(baseline_values),
                    "response_window": " | ".join(response_values),
                    "baseline_value": baseline_value,
                    "response_value": response_value,
                    "reactivity": delta,
                    "absolute_reactivity": abs(delta),
                    "percent_reactivity": percent,
                    "n_baseline_windows": int(len(block_base)),
                    "n_response_windows": int(len(block_response)),
                }
            )

            rows.append(row)

    columns = group_cols + [
        "measure",
        "statistic",
        "baseline_window",
        "response_window",
        "baseline_value",
        "response_value",
        "reactivity",
        "absolute_reactivity",
        "percent_reactivity",
        "n_baseline_windows",
        "n_response_windows",
    ]

    out = pd.DataFrame(
        rows,
        columns=columns,
    )

    settings = {
        "baseline_window": (baseline_values[0] if len(baseline_values) == 1 else baseline_values),
        "response_window": (response_values[0] if len(response_values) == 1 else response_values),
        "group_cols": group_cols,
        "window_col": window_col,
        "measure_cols": value_cols,
        "statistic": statistic,
    }

    return _attach_gp3_attr(
        out,
        "gp3_face_reactivity_settings",
        settings,
    )


# =====================================================================
# BINOCULAR SENSITIVITY
# =====================================================================


def _binocular_policy(
    data: pd.DataFrame,
    *,
    left_col: str,
    right_col: str,
    prefix: str,
    policy: str,
    valid_min: float | None,
    valid_max: float | None,
) -> tuple[
    pd.Series,
    pd.Series,
]:
    raise RuntimeError("Superseded internal R3-A implementation")


def _binocular_summary_block(
    block: pd.DataFrame,
    values: pd.Series,
    policy: str,
    group_values: Mapping[str, Any],
) -> dict[
    str,
    Any,
]:
    raise RuntimeError("Superseded internal R3-A implementation")


def _r3a_analyse_gazepoint_binocular_sensitivity(
    data: pd.DataFrame,
    left_col: str,
    right_col: str,
    policies: Sequence[str] | str = (
        "complete_case",
        "available_eye",
        "reconstructed_mean",
        "left_only",
        "right_only",
    ),
    prefix: str = "gp3_binocular",
    group_cols: Sequence[str] | str | None = None,
    condition_col: str | None = None,
    valid_min: float | None = None,
    valid_max: float | None = None,
    **kwargs: Any,
) -> dict[
    str,
    Any,
]:
    raise RuntimeError("Superseded internal R3-A implementation")


# =====================================================================
# BINOCULAR REPORTING
# =====================================================================


_LIMITATIONS = [
    (
        "Cross-eye prediction is a preprocessing "
        "reconstruction and does not recover an "
        "independently observed biological truth."
    ),
    (
        "Artificial masking quantifies prediction "
        "performance on observed bilateral samples "
        "but cannot establish the missingness mechanism "
        "of naturally unavailable samples."
    ),
    (
        "Calibration diagnostics and reconstruction "
        "burden should be reported alongside downstream "
        "sensitivity analyses."
    ),
    (
        "Temporal interpolation and cross-eye "
        "reconstruction should remain separately "
        "declared analytical decisions."
    ),
]


def _fmt_pct(
    value: float,
) -> str:
    return f"{100.0 * value:.1f}%"


def _fmt_num(
    value: float,
) -> str:
    if math.isnan(value):
        return "NA"

    return (f"{value:.3f}").rstrip("0").rstrip(".")


def _r3a_summarise_gazepoint_binocular_reporting(
    data: pd.DataFrame,
    audit: Any = None,
    validation: Any = None,
    by: Sequence[str] | str | None = None,
    prefix: str = "gp3_binocular",
    **kwargs: Any,
) -> dict[
    str,
    Any,
]:
    frame = _as_frame(data)

    metadata = frame.attrs.get("gp3_binocular_reconstruction")

    if not isinstance(
        metadata,
        Mapping,
    ):
        raise ValueError("`data` must be output from `reconstruct_gazepoint_binocular_pupil()`.")

    reconstructed = frame[f"{prefix}_reconstructed"].fillna(False).astype(bool)

    left_observed = _num(frame[f"{prefix}_left_observed"]).notna()

    right_observed = _num(frame[f"{prefix}_right_observed"]).notna()

    n = int(len(frame))

    n_reconstructed = int(reconstructed.sum())

    reconstruction_fraction = n_reconstructed / n if n else math.nan

    bilateral_fraction = float((left_observed & right_observed).sum()) / n if n else math.nan

    monocular_unreconstructed = (left_observed ^ right_observed) & ~reconstructed

    monocular_fraction = float(monocular_unreconstructed.sum()) / n if n else math.nan

    method = str(
        metadata.get(
            "method",
            "unknown",
        )
    )

    max_gap_ms = metadata.get(
        "max_gap_ms",
        math.inf,
    )

    allow_extrapolation = bool(
        metadata.get(
            "allow_extrapolation",
            False,
        )
    )

    calibration = metadata.get("calibration")

    models = pd.DataFrame()

    if isinstance(
        calibration,
        Mapping,
    ):
        candidate = calibration.get("models")

        if isinstance(
            candidate,
            pd.DataFrame,
        ):
            models = candidate.copy()

    audit_status = "descriptive"

    summary = pd.DataFrame(
        [
            {
                "n_rows": n,
                "n_reconstructed": n_reconstructed,
                "reconstruction_fraction": reconstruction_fraction,
                "bilateral_observed_fraction": bilateral_fraction,
                "monocular_unreconstructed_fraction": monocular_fraction,
                "audit_status": audit_status,
                "method": method,
                "max_gap_ms": max_gap_ms,
                "allow_extrapolation": allow_extrapolation,
            }
        ]
    )

    validation_table = pd.DataFrame()

    model_sentence = ""

    if not models.empty:
        eligible = (
            models["eligible"].fillna(False).astype(bool)
            if "eligible" in models.columns
            else pd.Series(
                False,
                index=models.index,
            )
        )

        eligible_models = models[eligible]

        if not eligible_models.empty:
            n_pairs = _num(eligible_models["n_pairs"])

            r_squared = _num(eligible_models["r_squared"])

            model_sentence = (
                "Eligible cross-eye calibration models "
                "used a median of "
                f"{int(n_pairs.median())} "
                "paired observations; the median "
                "in-sample R-squared was "
                f"{_fmt_num(float(r_squared.median()))}."
            )

    first_sentence = (
        "Binocular pupil handling used the declared "
        f"`{method}` policy. Of {n} rows, "
        f"{n_reconstructed} "
        f"({_fmt_pct(reconstruction_fraction)}) "
        "contained model-based cross-eye "
        "reconstruction; "
        f"{_fmt_pct(bilateral_fraction)} "
        "were retained as directly observed bilateral "
        "samples and "
        f"{_fmt_pct(monocular_fraction)} "
        "remained monocular without reconstruction."
    )

    provenance_sentence = (
        "Reconstructed values were retained as predicted "
        "values with explicit row-level provenance; "
        "they were not treated as independently measured "
        "pupil observations."
    )

    text = " ".join(
        part
        for part in [
            first_sentence,
            model_sentence,
            provenance_sentence,
        ]
        if part
    )

    return {
        "summary": summary,
        "models": models.reset_index(drop=True),
        "validation": validation_table,
        "text": text,
        "limitations": list(_LIMITATIONS),
    }


# =====================================================================
# PUBLIC DELEGATE LOOKUP
# =====================================================================


R3A_IMPLEMENTATIONS = {
    "analyse_gazepoint_binocular_sensitivity": _r3a_analyse_gazepoint_binocular_sensitivity,
    "create_gazepoint_master": _r3a_create_gazepoint_master,
    "detect_gazepoint_fixations_ivt": _r3a_detect_gazepoint_fixations_ivt,
    "summarise_gazepoint_binocular_reporting": _r3a_summarise_gazepoint_binocular_reporting,
    "summarise_gazepoint_event_detector_agreement": _r3a_summarise_gazepoint_event_detector_agreement,
    "summarize_gazepoint_face_windows": _r3a_summarize_gazepoint_face_windows,
    "summarize_gazepoint_face_reactivity": _r3a_summarize_gazepoint_face_reactivity,
}


def _dispatch_r3a(
    function_name: str,
    arguments: Mapping[str, Any],
) -> Any:
    implementation = R3A_IMPLEMENTATIONS[function_name]

    call_args = dict(arguments)

    extra = call_args.pop(
        "kwargs",
        None,
    )

    if isinstance(
        extra,
        Mapping,
    ):
        call_args.update(extra)

    return implementation(**call_args)


# === R3A PASS 2 CANONICAL ORACLE REPAIRS ===


# ---------------------------------------------------------------------
# 1. BINOCULAR POLICY
#
# NumPy 2.x refuses np.where(str, np.nan) promotion.
# R uses a character vector with NA_character_, so construct
# Python object Series explicitly.
# ---------------------------------------------------------------------


def _r3a_binocular_policy_v2(
    data: pd.DataFrame,
    *,
    left_col: str,
    right_col: str,
    prefix: str,
    policy: str,
    valid_min: float | None,
    valid_max: float | None,
) -> tuple[pd.Series, pd.Series]:
    raise RuntimeError("Superseded internal R3-A implementation")


_binocular_policy = _r3a_binocular_policy_v2


# ---------------------------------------------------------------------
# 2. IVT
#
# Canonical R lightweight I-VT:
# - the first sample has no velocity and is NOT a fixation sample
# - the run begins at sample 2
# - detection_method is exactly I-VT_lightweight
# ---------------------------------------------------------------------


def _r3a_ivt_one_group_v2(
    block: pd.DataFrame,
    *,
    x_col: str,
    y_col: str,
    time_col: str,
    velocity_threshold: float,
    min_duration_ms: float,
    distance_scale: float,
    time_scale: float,
    group_cols: list[str],
) -> pd.DataFrame:
    block = block.sort_values(
        time_col,
        kind="mergesort",
    ).reset_index(drop=True)

    x = _num(block[x_col]).to_numpy(dtype=float)

    y = _num(block[y_col]).to_numpy(dtype=float)

    t = _num(block[time_col]).to_numpy(dtype=float)

    if not len(block):
        return pd.DataFrame()

    velocity = np.full(
        len(block),
        np.nan,
        dtype=float,
    )

    if len(block) > 1:
        dt = np.diff(t)

        distance = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)

        with np.errstate(
            divide="ignore",
            invalid="ignore",
        ):
            velocity[1:] = (distance * float(distance_scale)) / (dt * float(time_scale))

    fixation_sample = np.isfinite(velocity) & (velocity <= float(velocity_threshold))

    segments: list[tuple[int, int]] = []

    start: int | None = None

    for index, flag in enumerate(fixation_sample):
        if flag and start is None:
            start = index

        at_end = index == len(fixation_sample) - 1

        if start is not None and (not flag or at_end):
            end = index if (flag and at_end) else index - 1

            segments.append(
                (
                    start,
                    end,
                )
            )

            start = None

    rows = []

    fixation_index = 0

    for start, end in segments:
        duration = float(t[end] - t[start])

        if duration < float(min_duration_ms):
            continue

        fixation_index += 1

        segment = block.iloc[start : end + 1]

        segment_velocity = velocity[start : end + 1]

        finite_velocity = segment_velocity[np.isfinite(segment_velocity)]

        row: dict[
            str,
            Any,
        ] = {}

        for column in group_cols:
            row[column] = block[column].iloc[0]

        row.update(
            {
                "fixation_index": fixation_index,
                "start_time": float(t[start]),
                "end_time": float(t[end]),
                "duration_ms": duration,
                "n_samples": int(end - start + 1),
                "x": float(_num(segment[x_col]).mean()),
                "y": float(_num(segment[y_col]).mean()),
                "mean_velocity": (
                    float(np.mean(finite_velocity)) if len(finite_velocity) else math.nan
                ),
                "detection_method": "I-VT_lightweight",
                "detection_status": "ok",
            }
        )

        rows.append(row)

    columns = group_cols + [
        "fixation_index",
        "start_time",
        "end_time",
        "duration_ms",
        "n_samples",
        "x",
        "y",
        "mean_velocity",
        "detection_method",
        "detection_status",
    ]

    return pd.DataFrame(
        rows,
        columns=columns,
    )


_ivt_one_group = _r3a_ivt_one_group_v2


# ---------------------------------------------------------------------
# 3. CREATE MASTER — CANONICAL DEFAULTS REVEALED BY ORACLE
# ---------------------------------------------------------------------


_r3a_create_gazepoint_master_pass1 = _r3a_create_gazepoint_master


def _r3a_create_gazepoint_master_v2(
    data: pd.DataFrame | None = None,
    gaze_data: pd.DataFrame | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    out = _r3a_create_gazepoint_master_pass1(
        data=data,
        gaze_data=gaze_data,
        **kwargs,
    )

    n = len(out)

    # Canonical gp3tools master contract.
    out["aoi_current"] = "non_aoi"

    out["event_latency_offset_ms"] = 0.0

    out["gaze_unit"] = "tracker_units"

    # Screen bounds have not been supplied, so R
    # intentionally leaves this unknown rather than FALSE.
    out["gaze_offscreen"] = pd.Series(
        [np.nan] * n,
        dtype=object,
    )

    return out


_r3a_create_gazepoint_master = _r3a_create_gazepoint_master_v2

R3A_IMPLEMENTATIONS["create_gazepoint_master"] = _r3a_create_gazepoint_master_v2


# ---------------------------------------------------------------------
# 4. BINOCULAR REPORTING
# ---------------------------------------------------------------------


_r3a_reporting_pass1 = _r3a_summarise_gazepoint_binocular_reporting


def _r3a_summarise_gazepoint_binocular_reporting_v2(
    data: pd.DataFrame,
    audit: Any = None,
    validation: Any = None,
    by: Sequence[str] | str | None = None,
    prefix: str = "gp3_binocular",
    **kwargs: Any,
) -> dict[str, Any]:
    result = _r3a_reporting_pass1(
        data=data,
        audit=audit,
        validation=validation,
        by=by,
        prefix=prefix,
        **kwargs,
    )

    models = result["models"].copy()

    if "reason" in models.columns:
        models["reason"] = models["reason"].map(lambda value: np.nan if value is None else value)

    result["models"] = models

    metadata = data.attrs.get(
        "gp3_binocular_reconstruction",
        {},
    )

    method = str(
        metadata.get(
            "method",
            "unknown",
        )
    )

    summary = result["summary"].iloc[0]

    n = int(summary["n_rows"])

    n_rec = int(summary["n_reconstructed"])

    rec_fraction = float(summary["reconstruction_fraction"])

    bilateral_fraction = float(summary["bilateral_observed_fraction"])

    mono_fraction = float(summary["monocular_unreconstructed_fraction"])

    first_sentence = (
        "Binocular pupil handling used the declared "
        f"`{method}` policy. Of {n} rows, "
        f"{n_rec} "
        f"({_fmt_pct(rec_fraction)}) "
        "contained model-based cross-eye "
        "reconstruction; "
        f"{_fmt_pct(bilateral_fraction)} "
        "were retained as directly observed bilateral "
        "samples and "
        f"{_fmt_pct(mono_fraction)} "
        "remained monocular without reconstruction."
    )

    model_sentence = ""

    if not models.empty:
        if "eligible" in models.columns:
            eligible_mask = models["eligible"].fillna(False).astype(bool)

            eligible_models = models[eligible_mask]
        else:
            eligible_models = models.iloc[0:0]

        if not eligible_models.empty:
            pair_values = _num(eligible_models["n_pairs"])

            r2_values = _num(eligible_models["r_squared"])

            model_sentence = (
                "Eligible cross-eye calibration models "
                "used a median of "
                f"{int(pair_values.median())} "
                "paired observations; the median "
                "in-sample R-squared was "
                f"{_fmt_num(float(r2_values.median()))}."
            )

    validation_sentence = ""

    provenance_sentence = (
        "Reconstructed values were retained as predicted "
        "values with explicit row-level provenance; "
        "they were not treated as independently measured "
        "pupil observations."
    )

    # R's paste() retains the separator around an empty
    # validation sentence, yielding two spaces here.
    result["text"] = " ".join(
        [
            first_sentence,
            model_sentence,
            validation_sentence,
            provenance_sentence,
        ]
    )

    return result


_r3a_summarise_gazepoint_binocular_reporting = _r3a_summarise_gazepoint_binocular_reporting_v2

R3A_IMPLEMENTATIONS["summarise_gazepoint_binocular_reporting"] = (
    _r3a_summarise_gazepoint_binocular_reporting_v2
)
# === R3A PASS 3 FINAL TWO ===


# =====================================================================
# CREATE GAZEPOINT MASTER — FINAL CANONICAL FIELD CONTRACT
# =====================================================================


_r3a_master_pass2 = _r3a_create_gazepoint_master_v2


def _r3a_create_gazepoint_master_v3(
    data: pd.DataFrame | None = None,
    gaze_data: pd.DataFrame | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    source = _first_not_none(
        gaze_data,
        data,
    )

    original = _as_frame(
        source,
        arg="gaze_data",
    )

    out = _r3a_master_pass2(
        data=data,
        gaze_data=gaze_data,
        **kwargs,
    )

    user_file = original["USER_FILE"].astype(str).reset_index(drop=True)

    media_id = original["MEDIA_ID"].astype(str).reset_index(drop=True)

    if "MEDIA_NAME" in original.columns:
        media_name = original["MEDIA_NAME"].astype(str).reset_index(drop=True)
    else:
        media_name = media_id.copy()

    # Canonical gp3tools master aliases.
    out["subject"] = user_file

    out["pID"] = user_file

    out["trial"] = media_id

    out["trial_global"] = user_file + "_M" + media_id

    out["stimulus_id"] = media_id

    out["stimulus_file"] = media_name

    # Both canonical time fields are milliseconds.
    out["time_orig"] = out["time"].astype(float)

    # Gazepoint GP3 default metadata in create_master().
    out["sampling_rate_hz"] = 60.0

    out["tracker_sampling_rate"] = 60.0

    out["tracker_model"] = "Gazepoint"

    out["pupil_unit"] = "camera_image_pixels"

    out["gaze_unit"] = "tracker_units"

    return out


_r3a_create_gazepoint_master = _r3a_create_gazepoint_master_v3

R3A_IMPLEMENTATIONS["create_gazepoint_master"] = _r3a_create_gazepoint_master_v3


# =====================================================================
# BINOCULAR SENSITIVITY — FINAL CANONICAL CONTRACT
# =====================================================================


def _r3a_mad(
    values: pd.Series,
) -> float:
    clean = _num(values).dropna().to_numpy(dtype=float)

    if not len(clean):
        return math.nan

    center = float(np.median(clean))

    return float(np.median(np.abs(clean - center)))


def _r3a_group_key(
    group_values: Mapping[str, Any],
) -> str:
    if not group_values:
        return ""

    parts = []

    for name, value in group_values.items():
        if pd.isna(value):
            rendered = "NA"
        else:
            rendered = str(value)

        parts.append(f"{name}={rendered}")

    return "||".join(parts)


def _r3a_policy_values_v3(
    data: pd.DataFrame,
    *,
    left_col: str,
    right_col: str,
    prefix: str,
    policy: str,
    valid_min: float | None,
    valid_max: float | None,
) -> tuple[
    pd.Series,
    pd.Series,
]:
    left = _num(data[left_col])

    right = _num(data[right_col])

    if valid_min is not None:
        left = left.where(left >= float(valid_min))

        right = right.where(right >= float(valid_min))

    if valid_max is not None:
        left = left.where(left <= float(valid_max))

        right = right.where(right <= float(valid_max))

    both = left.notna() & right.notna()

    neither = left.isna() & right.isna()

    if policy == "complete_case":
        value = _mean_pair(
            left,
            right,
        ).where(both)

        source = pd.Series(
            "unavailable",
            index=data.index,
            dtype=object,
        )

        source.loc[both] = "bilateral_observed"

        return (
            value,
            source,
        )

    if policy == "available_eye":
        value = _mean_pair(
            left,
            right,
        )

        source = pd.Series(
            "unavailable",
            index=data.index,
            dtype=object,
        )

        source.loc[both] = "bilateral_observed"

        source.loc[left.notna() & right.isna()] = "left_only_observed"

        source.loc[left.isna() & right.notna()] = "right_only_observed"

        source.loc[neither] = "unavailable"

        return (
            value,
            source,
        )

    if policy == "reconstructed_mean":
        left_final = _num(data[f"{prefix}_left_final"])

        right_final = _num(data[f"{prefix}_right_final"])

        left_reconstructed = data[f"{prefix}_left_reconstructed"].fillna(False).astype(bool)

        right_reconstructed = data[f"{prefix}_right_reconstructed"].fillna(False).astype(bool)

        value = _mean_pair(
            left_final,
            right_final,
        )

        source = pd.Series(
            "unavailable",
            index=data.index,
            dtype=object,
        )

        source.loc[both] = "bilateral_observed"

        reconstructed = left_reconstructed | right_reconstructed

        source.loc[reconstructed & value.notna()] = "bilateral_with_reconstruction"

        return (
            value,
            source,
        )

    if policy == "left_only":
        value = left

        source = pd.Series(
            "unavailable",
            index=data.index,
            dtype=object,
        )

        source.loc[left.notna()] = "left_observed"

        return (
            value,
            source,
        )

    if policy == "right_only":
        value = right

        source = pd.Series(
            "unavailable",
            index=data.index,
            dtype=object,
        )

        source.loc[right.notna()] = "right_observed"

        return (
            value,
            source,
        )

    raise ValueError("Unsupported policy")


def _r3a_summary_values_v3(
    values: pd.Series,
    *,
    policy: str,
    group_values: Mapping[str, Any],
) -> dict[str, Any]:
    numeric = _num(values)

    usable = numeric.dropna()

    n_total = int(len(numeric))

    n_usable = int(len(usable))

    row: dict[
        str,
        Any,
    ] = {
        "policy": policy,
    }

    row.update(group_values)

    row["group_key"] = _r3a_group_key(group_values)

    row["n_total"] = n_total

    row["n_usable"] = n_usable

    row["missing_fraction"] = 1.0 - (n_usable / n_total) if n_total else math.nan

    row["mean"] = _mean(numeric)

    row["sd"] = _r_sd(numeric)

    row["median"] = _median(numeric)

    row["mad"] = _r3a_mad(numeric)

    return row


def _r3a_analyse_gazepoint_binocular_sensitivity_v3(
    data: pd.DataFrame,
    left_col: str,
    right_col: str,
    policies: Sequence[str] | str = (
        "complete_case",
        "available_eye",
        "reconstructed_mean",
        "left_only",
        "right_only",
    ),
    prefix: str = "gp3_binocular",
    group_cols: Sequence[str] | str | None = None,
    condition_col: str | None = None,
    valid_min: float | None = None,
    valid_max: float | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    frame = _as_frame(data)

    groups = [str(value) for value in _as_list(group_cols)]

    policy_names = [str(value) for value in _as_list(policies)]

    allowed = {
        "complete_case",
        "available_eye",
        "reconstructed_mean",
        "left_only",
        "right_only",
    }

    if not set(policy_names).issubset(allowed):
        raise ValueError("`policies` contains an unsupported pupil-construction policy.")

    _check_cols(
        frame,
        [
            left_col,
            right_col,
            *groups,
            condition_col,
        ],
    )

    _check_cols(
        frame,
        [
            f"{prefix}_left_final",
            f"{prefix}_right_final",
            f"{prefix}_left_reconstructed",
            f"{prefix}_right_reconstructed",
        ],
    )

    values: dict[
        str,
        pd.Series,
    ] = {}

    sources: dict[
        str,
        pd.Series,
    ] = {}

    for policy in policy_names:
        value, source = _r3a_policy_values_v3(
            frame,
            left_col=left_col,
            right_col=right_col,
            prefix=prefix,
            policy=policy,
            valid_min=valid_min,
            valid_max=valid_max,
        )

        values[policy] = value

        sources[policy] = source

    # ---------------------------------------------------------
    # SUMMARY
    #
    # R iterates policy first, grouping block second.
    # ---------------------------------------------------------

    if groups:
        grouper: Any = groups[0] if len(groups) == 1 else groups

        grouped_blocks = list(
            frame.groupby(
                grouper,
                sort=False,
                dropna=False,
            )
        )
    else:
        grouped_blocks = [
            (
                (),
                frame,
            )
        ]

    summary_rows = []

    for policy in policy_names:
        for group_key, block in grouped_blocks:
            if groups:
                if not isinstance(
                    group_key,
                    tuple,
                ):
                    group_key = (group_key,)

                group_values = dict(
                    zip(
                        groups,
                        group_key,
                        strict=False,
                    )
                )
            else:
                group_values = {}

            summary_rows.append(
                _r3a_summary_values_v3(
                    values[policy].loc[block.index],
                    policy=policy,
                    group_values=group_values,
                )
            )

    summary_columns = (
        ["policy"]
        + groups
        + [
            "group_key",
            "n_total",
            "n_usable",
            "missing_fraction",
            "mean",
            "sd",
            "median",
            "mad",
        ]
    )

    summary = pd.DataFrame(
        summary_rows,
        columns=summary_columns,
    )

    # ---------------------------------------------------------
    # SERIES
    # ---------------------------------------------------------

    series_rows = []

    for policy in policy_names:
        for row_number, index in enumerate(
            frame.index,
            start=1,
        ):
            row = {
                "row_id": row_number,
                "policy": policy,
                "pupil": values[policy].loc[index],
                "source": sources[policy].loc[index],
            }

            for group in groups:
                row[group] = frame.loc[
                    index,
                    group,
                ]

            series_rows.append(row)

    series = pd.DataFrame(
        series_rows,
        columns=(
            [
                "row_id",
                "policy",
                "pupil",
                "source",
            ]
            + groups
        ),
    )

    # ---------------------------------------------------------
    # CORRELATIONS / DIFFERENCE SUMMARIES
    # ---------------------------------------------------------

    correlation_rows = []

    for policy_1, policy_2 in itertools.combinations(
        policy_names,
        2,
    ):
        first = _num(values[policy_1])

        second = _num(values[policy_2])

        ok = first.notna() & second.notna()

        n_complete = int(ok.sum())

        if n_complete > 2 and _r_sd(first[ok]) > 0 and _r_sd(second[ok]) > 0:
            correlation = float(first[ok].corr(second[ok]))
        else:
            correlation = math.nan

        if n_complete:
            difference = first[ok].to_numpy(dtype=float) - second[ok].to_numpy(dtype=float)

            mean_difference = float(np.mean(difference))

            mean_absolute_difference = float(np.mean(np.abs(difference)))
        else:
            mean_difference = math.nan
            mean_absolute_difference = math.nan

        correlation_rows.append(
            {
                "policy_1": policy_1,
                "policy_2": policy_2,
                "n_complete": n_complete,
                "correlation": correlation,
                "mean_difference": mean_difference,
                "mean_absolute_difference": mean_absolute_difference,
            }
        )

    correlations = pd.DataFrame(
        correlation_rows,
        columns=[
            "policy_1",
            "policy_2",
            "n_complete",
            "correlation",
            "mean_difference",
            "mean_absolute_difference",
        ],
    )

    # ---------------------------------------------------------
    # CONDITION SUMMARIES
    # ---------------------------------------------------------

    condition_summary = pd.DataFrame()

    condition_contrasts = pd.DataFrame()

    if condition_col is not None:
        condition_rows = []

        for policy in policy_names:
            for condition, block in frame.groupby(
                condition_col,
                sort=False,
                dropna=False,
            ):
                condition_rows.append(
                    _r3a_summary_values_v3(
                        values[policy].loc[block.index],
                        policy=policy,
                        group_values={condition_col: condition},
                    )
                )

        condition_summary = pd.DataFrame(condition_rows)

        contrast_rows = []

        condition_values = list(pd.unique(frame[condition_col]))

        if condition_values:
            reference = condition_values[0]

            for policy in policy_names:
                policy_table = condition_summary[condition_summary["policy"] == policy]

                reference_row = policy_table[policy_table[condition_col] == reference]

                if reference_row.empty:
                    continue

                reference_mean = float(reference_row["mean"].iloc[0])

                for comparison in condition_values[1:]:
                    comparison_row = policy_table[policy_table[condition_col] == comparison]

                    if comparison_row.empty:
                        continue

                    current_mean = float(comparison_row["mean"].iloc[0])

                    contrast_rows.append(
                        {
                            "policy": policy,
                            "reference": reference,
                            "comparison": comparison,
                            "mean_difference": (current_mean - reference_mean),
                        }
                    )

        condition_contrasts = pd.DataFrame(contrast_rows)

    return {
        "summary": summary,
        "correlations": correlations,
        "condition_summary": condition_summary,
        "condition_contrasts": condition_contrasts,
        "series": series,
        "settings": {
            "policies": policy_names,
            "prefix": prefix,
            "group_cols": groups,
            "condition_col": condition_col,
            "valid_min": valid_min,
            "valid_max": valid_max,
        },
    }


_r3a_analyse_gazepoint_binocular_sensitivity = _r3a_analyse_gazepoint_binocular_sensitivity_v3

R3A_IMPLEMENTATIONS["analyse_gazepoint_binocular_sensitivity"] = (
    _r3a_analyse_gazepoint_binocular_sensitivity_v3
)
# === R3A DUAL CONTRACT SELECTOR ===


def _r3a_call_arguments(
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    result = dict(arguments)

    extra = result.get("kwargs")

    if isinstance(
        extra,
        Mapping,
    ):
        result.update(extra)

    return result


def _r3a_columns(
    value: Any,
) -> set[str]:
    columns = getattr(
        value,
        "columns",
        None,
    )

    if columns is None:
        return set()

    return {str(column) for column in columns}


def _should_use_r3a(
    function_name: str,
    arguments: Mapping[str, Any],
) -> bool:
    args = _r3a_call_arguments(arguments)

    data = args.get("data")

    columns = _r3a_columns(data)

    # ---------------------------------------------------------
    # create_gazepoint_master
    #
    # Canonical R contract starts from a raw Gazepoint export
    # with USER_FILE / MEDIA_ID / TIME.
    #
    # Historical Python convenience inputs may already be
    # standardized and intentionally do not require USER_FILE.
    # ---------------------------------------------------------

    if function_name == "create_gazepoint_master":
        source = args.get(
            "gaze_data",
            data,
        )

        source_columns = _r3a_columns(source)

        return {
            "USER_FILE",
            "MEDIA_ID",
            "TIME",
        }.issubset(source_columns)

    # ---------------------------------------------------------
    # detect_gazepoint_fixations_ivt
    #
    # Canonical R call explicitly declares x/y/time columns.
    # Existing Python API also supports auto-detection and the
    # older velocity wrapper.
    # ---------------------------------------------------------

    if function_name == "detect_gazepoint_fixations_ivt":
        required_names = (
            "x_col",
            "y_col",
            "time_col",
        )

        if not all(name in args and args[name] is not None for name in required_names):
            return False

        requested = {str(args[name]) for name in required_names}

        return requested.issubset(columns)

    # ---------------------------------------------------------
    # summarize_gazepoint_face_windows
    #
    # R3-A oracle exercises the explicit window-table contract.
    # Existing Python API also supports labelled/pre-synchronized
    # data without a separate windows table.
    # ---------------------------------------------------------

    if function_name == "summarize_gazepoint_face_windows":
        return args.get("windows") is not None

    # ---------------------------------------------------------
    # analyse_gazepoint_binocular_sensitivity
    #
    # Canonical R path consumes output from
    # reconstruct_gazepoint_binocular_pupil().
    #
    # Existing Python API historically accepted ordinary
    # binocular columns as well.
    # ---------------------------------------------------------

    if function_name == "analyse_gazepoint_binocular_sensitivity":
        prefix = str(
            args.get(
                "prefix",
                "gp3_binocular",
            )
        )

        required = {
            f"{prefix}_left_final",
            f"{prefix}_right_final",
            f"{prefix}_left_reconstructed",
            f"{prefix}_right_reconstructed",
        }

        return required.issubset(columns)

    # The other R3-A functions have not produced legacy
    # regressions and remain on the exact canonical path.
    return True


# === R3A DUAL CONTRACT SELECTOR V2 ===


_r3a_should_use_previous = _should_use_r3a


def _should_use_r3a_v2(
    function_name: str,
    arguments: Mapping[str, Any],
) -> bool:
    args = _r3a_call_arguments(arguments)

    data = args.get("data")

    # ---------------------------------------------------------
    # Event-detector agreement
    #
    # Canonical R contract:
    # - standardized event dataframe with detector/start/end/
    #   duration columns; or
    # - detector-comparison mapping containing such an events
    #   dataframe.
    #
    # Historical Python contract also accepted the older
    # pairwise-agreement product directly.
    # ---------------------------------------------------------

    if function_name == "summarise_gazepoint_event_detector_agreement":
        source = args.get(
            "x",
            data,
        )

        required = {
            "detector",
            "start_time",
            "end_time",
            "duration_ms",
        }

        if isinstance(
            source,
            pd.DataFrame,
        ):
            return required.issubset(_r3a_columns(source))

        if isinstance(
            source,
            Mapping,
        ):
            events = source.get("events")

            if isinstance(
                events,
                pd.DataFrame,
            ):
                return required.issubset(_r3a_columns(events))

        return False

    # ---------------------------------------------------------
    # Facial reactivity
    #
    # Canonical R contract requires explicit baseline and
    # response windows. Existing Python convenience behavior
    # accepted synchronized/raw face data without these.
    # ---------------------------------------------------------

    if function_name == "summarize_gazepoint_face_reactivity":
        return args.get("baseline_window") is not None and args.get("response_window") is not None

    # ---------------------------------------------------------
    # Binocular reporting
    #
    # Canonical R path strictly consumes the output of
    # reconstruct_gazepoint_binocular_pupil(), identified by
    # gp3_binocular_reconstruction metadata.
    #
    # Historical Python behavior also accepted ordinary raw
    # left/right pupil data.
    # ---------------------------------------------------------

    if function_name == "summarise_gazepoint_binocular_reporting":
        attrs = getattr(
            data,
            "attrs",
            {},
        )

        return isinstance(
            attrs,
            Mapping,
        ) and isinstance(
            attrs.get("gp3_binocular_reconstruction"),
            Mapping,
        )

    return _r3a_should_use_previous(
        function_name,
        arguments,
    )


_should_use_r3a = _should_use_r3a_v2
