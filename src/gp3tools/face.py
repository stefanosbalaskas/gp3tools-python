"""Facial-analysis and multimodal synchronisation helpers."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ._compat import r_aliases
from ._utils import ensure_dataframe, normalize_group_cols


def standardize_gazepoint_face_columns(
    data,
    source="auto",
    participant_id_col=None,
    frame_col=None,
    time_col=None,
    confidence_col=None,
    success_col=None,
    face_id_col=None,
    file_col=None,
    confidence_threshold=0.80,
    keep_original_columns=True,
) -> pd.DataFrame:
    """Standardise external face-analysis columns using R v2.3.0 semantics."""
    allowed = {"auto", "openface", "pyfeat", "mediapipe", "facereader", "generic"}
    if source not in allowed:
        raise ValueError(
            "source must be one of: auto, openface, pyfeat, mediapipe, facereader, generic"
        )
    if isinstance(data, (str, Path)):
        path = Path(data)
        if not path.exists():
            raise ValueError("data must be a data frame or a readable CSV path")
        frame = pd.read_csv(path)
    else:
        frame = ensure_dataframe(data)

    columns = []
    seen = {}
    unnamed = 0
    for raw in frame.columns:
        value = "" if pd.isna(raw) else str(raw).strip()
        if not value:
            unnamed += 1
            value = f"unnamed_face_column_{unnamed}"
        count = seen.get(value, 0)
        seen[value] = count + 1
        columns.append(value if count == 0 else f"{value}_{count}")
    frame.columns = columns

    def detect_source(table):
        names = [str(column) for column in table.columns]
        if all(value in names for value in ["frame", "timestamp", "confidence", "success"]) and any(
            re.fullmatch(r"AU[0-9]{2}_[rc]", value) for value in names
        ):
            return "openface"
        if any(
            re.search(r"blendshape|face_landmark|faceblendshape", value, re.I) for value in names
        ):
            return "mediapipe"
        if any(re.fullmatch(r"AU[0-9]{2}(?:_r)?", value) for value in names) and any(
            re.search(
                r"anger|happy|sad|fear|surprise|disgust|neutral|valence|arousal",
                value,
                re.I,
            )
            for value in names
        ):
            return "pyfeat"
        if any(
            re.search(r"valence|arousal|neutral|happy|sad|angry|surprised", value, re.I)
            for value in names
        ) and any(re.search(r"quality|model|fit|head|orientation", value, re.I) for value in names):
            return "facereader"
        return "generic"

    if source == "auto" and "gp3_face_source" in frame.columns:
        unique_sources = frame["gp3_face_source"].dropna().astype(str).unique().tolist()
        detected_source = (
            unique_sources[0]
            if len(unique_sources) == 1
            else "mixed"
            if len(unique_sources) > 1
            else detect_source(frame)
        )
    else:
        detected_source = detect_source(frame) if source == "auto" else source

    lower_names = {str(column).lower(): column for column in frame.columns}

    def choose(supplied, candidates):
        if supplied is not None:
            if supplied not in frame.columns:
                raise ValueError(f"Column not found: {supplied}")
            return supplied
        for candidate in candidates:
            match = lower_names.get(candidate.lower())
            if match is not None:
                return match
        return None

    participant_id_col = choose(
        participant_id_col,
        [
            "gp3_face_participant_id",
            "participant_id",
            "participant",
            "subject_id",
            "subject",
            "user",
            "USER",
        ],
    )
    frame_col = choose(
        frame_col, ["frame", "Frame", "FRAME", "frame_id", "video_frame", "VID_FRAME"]
    )
    time_col = choose(
        time_col,
        [
            "timestamp",
            "Timestamp",
            "time",
            "Time",
            "TIME",
            "time_sec",
            "seconds",
            "sec",
            "time_seconds",
        ],
    )
    confidence_col = choose(
        confidence_col,
        [
            "confidence",
            "Confidence",
            "detection_confidence",
            "face_confidence",
            "tracking_confidence",
            "score",
        ],
    )
    success_col = choose(
        success_col,
        ["success", "Success", "detected", "face_detected", "tracking_success", "valid"],
    )
    face_id_col = choose(face_id_col, ["face_id", "FaceID", "face", "face_index", "person_id"])
    file_col = choose(file_col, ["gp3_face_file", "file", "filename", "video", "input"])

    def numeric(column):
        if column is None:
            return pd.Series(np.nan, index=frame.index, dtype=float)
        return pd.to_numeric(frame[column], errors="coerce")

    def character(column):
        if column is None:
            return pd.Series(pd.NA, index=frame.index, dtype="string")
        return frame[column].astype("string")

    def logical(column):
        if column is None:
            return pd.Series(pd.NA, index=frame.index, dtype="boolean")
        values = frame[column]
        if pd.api.types.is_bool_dtype(values):
            return values.astype("boolean")
        if pd.api.types.is_numeric_dtype(values):
            numbers = pd.to_numeric(values, errors="coerce")
            out = numbers.gt(0).astype("boolean")
            out[numbers.isna()] = pd.NA
            return out
        text = values.astype("string").str.strip().str.lower()
        out = pd.Series(pd.NA, index=frame.index, dtype="boolean")
        out[text.isin(["1", "true", "t", "yes", "y", "success", "valid", "detected"])] = True
        out[text.isin(["0", "false", "f", "no", "n", "fail", "failed", "invalid", "missing"])] = (
            False
        )
        return out

    confidence = numeric(confidence_col)
    success = logical(success_col)
    has_confidence = confidence.notna().any()
    has_success = success.notna().any()
    if has_confidence and has_success:
        valid = (
            success.fillna(False) & confidence.notna() & confidence.ge(float(confidence_threshold))
        )
    elif has_confidence:
        valid = confidence.notna() & confidence.ge(float(confidence_threshold))
    elif has_success:
        valid = success.fillna(False)
    else:
        valid = pd.Series(pd.NA, index=frame.index, dtype="boolean")

    standard = pd.DataFrame(
        {
            "face_source": [detected_source] * len(frame),
            "face_file": character(file_col),
            "participant_id": character(participant_id_col),
            "face_id": character(face_id_col),
            "face_frame": pd.to_numeric(numeric(frame_col), errors="coerce").astype("Int64"),
            "face_time_sec": numeric(time_col),
            "face_time_ms": numeric(time_col) * 1000,
            "face_confidence": confidence,
            "face_success": success,
            "face_valid": valid.astype("boolean"),
        }
    )

    if keep_original_columns:
        original = frame[
            [column for column in frame.columns if column not in standard.columns]
        ].reset_index(drop=True)
        out = pd.concat([standard.reset_index(drop=True), original], axis=1)
    else:
        out = standard.reset_index(drop=True)

    pose_map = {
        "face_pose_tx": "pose_Tx",
        "face_pose_ty": "pose_Ty",
        "face_pose_tz": "pose_Tz",
        "face_pose_rx": "pose_Rx",
        "face_pose_ry": "pose_Ry",
        "face_pose_rz": "pose_Rz",
    }
    for output_name, candidate in pose_map.items():
        source_name = lower_names.get(candidate.lower())
        if source_name is not None and output_name not in out.columns:
            out[output_name] = pd.to_numeric(frame[source_name], errors="coerce").to_numpy()

    if keep_original_columns:
        for column in frame.columns:
            normalized = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
            if normalized and normalized not in out.columns:
                out[normalized] = frame[column].to_numpy()

    out.attrs["_gp3_class"] = "gp3_face_data"
    out.attrs["gp3_face_standardization"] = {
        "source": source,
        "detected_source": detected_source,
        "participant_id_col": participant_id_col,
        "frame_col": frame_col,
        "time_col": time_col,
        "confidence_col": confidence_col,
        "success_col": success_col,
        "face_id_col": face_id_col,
        "file_col": file_col,
        "confidence_threshold": confidence_threshold,
    }
    return out


def audit_gazepoint_face_quality(
    data,
    confidence_col=None,
    threshold=None,
    *,
    group_cols=("participant_id", "face_file"),
    confidence_threshold=0.80,
    min_valid_percent=70,
    warning_valid_percent=85,
    max_time_gap_sec=None,
    max_duplicate_frame_percent=1,
    standardize=True,
):
    """Audit external facial-behaviour data quality using R v2.3.0 semantics."""
    # Historical Python compatibility: supplying the old confidence/threshold
    # controls retains the compact one-row DataFrame result.
    if confidence_col is not None or threshold is not None:
        df = standardize_gazepoint_face_columns(data)
        if confidence_col is None:
            confidence_col = next((c for c in df.columns if "confidence" in c.lower()), None)
        threshold_value = 0.8 if threshold is None else float(threshold)
        if confidence_col and confidence_col in df:
            values = pd.to_numeric(df[confidence_col], errors="coerce")
            return pd.DataFrame(
                {
                    "n": [len(df)],
                    "n_valid": [int(values.notna().sum())],
                    "mean_confidence": [float(values.mean())],
                    "prop_below_threshold": [float((values < threshold_value).mean())],
                }
            )
        return pd.DataFrame(
            {
                "n": [len(df)],
                "n_valid": [len(df)],
                "mean_confidence": [np.nan],
                "prop_below_threshold": [np.nan],
            }
        )

    if isinstance(data, (str, Path)):
        frame = standardize_gazepoint_face_columns(
            data,
            confidence_threshold=confidence_threshold,
        )
    else:
        frame = ensure_dataframe(data)
        required = {
            "face_frame",
            "face_time_sec",
            "face_confidence",
            "face_success",
            "face_valid",
        }
        if standardize or not required.issubset(frame.columns):
            frame = standardize_gazepoint_face_columns(
                frame,
                confidence_threshold=confidence_threshold,
            )

    if len(frame) < 1:
        raise ValueError("data must contain at least one row")

    if group_cols is None:
        groups = []
    elif isinstance(group_cols, str):
        groups = [group_cols] if group_cols in frame.columns else []
    else:
        groups = [column for column in group_cols if column in frame.columns]

    def percent(value, denominator):
        return np.nan if denominator <= 0 else 100.0 * value / denominator

    def safe_stat(values, statistic):
        numeric = pd.to_numeric(values, errors="coerce")
        if numeric.notna().sum() == 0:
            return np.nan
        return float(getattr(numeric, statistic)())

    def summarise_subset(part, group_values=None):
        n_rows = len(part)
        valid = part["face_valid"].astype("boolean")
        confidence = pd.to_numeric(part["face_confidence"], errors="coerce")
        success = part["face_success"].astype("boolean")
        face_frame = pd.to_numeric(part["face_frame"], errors="coerce")
        time_values = pd.to_numeric(part["face_time_sec"], errors="coerce")

        n_valid = int(valid.eq(True).sum())
        n_invalid = int(valid.eq(False).sum())
        n_unknown = int(valid.isna().sum())
        valid_percent = percent(n_valid, n_rows)
        invalid_percent = percent(n_invalid, n_rows)
        unknown_percent = percent(n_unknown, n_rows)

        n_missing_confidence = int(confidence.isna().sum())
        confidence_missing_percent = percent(n_missing_confidence, n_rows)
        known_success = success.notna()
        n_success = int(success.eq(True).sum())
        success_percent = (
            percent(n_success, int(known_success.sum())) if known_success.any() else np.nan
        )

        nonmissing_frame = face_frame.dropna()
        if len(nonmissing_frame):
            n_duplicate_frames = int(nonmissing_frame.duplicated().sum())
            duplicate_frame_percent = percent(n_duplicate_frames, len(nonmissing_frame))
        else:
            n_duplicate_frames = np.nan
            duplicate_frame_percent = np.nan

        finite_time = time_values[np.isfinite(time_values.to_numpy(float))].sort_values()
        n_missing_time = int(n_rows - len(finite_time))
        if len(finite_time) > 1:
            steps = np.diff(finite_time.to_numpy(float))
        else:
            steps = np.array([], dtype=float)
        n_nonpositive_time_steps = int(np.sum(steps <= 0))
        positive = steps[steps > 0]
        max_gap = float(np.max(positive)) if len(positive) else np.nan
        median_step = float(np.median(positive)) if len(positive) else np.nan
        estimated_hz = 1.0 / median_step if np.isfinite(median_step) and median_step > 0 else np.nan

        if valid.isna().all():
            status = "unknown"
        elif np.isfinite(valid_percent) and valid_percent < min_valid_percent:
            status = "fail"
        elif np.isfinite(valid_percent) and valid_percent < warning_valid_percent:
            status = "warn"
        elif (
            np.isfinite(duplicate_frame_percent)
            and duplicate_frame_percent > max_duplicate_frame_percent
        ):
            status = "warn"
        elif (
            max_time_gap_sec is not None
            and np.isfinite(max_gap)
            and max_gap > float(max_time_gap_sec)
        ):
            status = "warn"
        else:
            status = "pass"

        if status == "unknown":
            message = (
                "Face-data validity could not be evaluated because no confidence "
                "or success information was available."
            )
        elif status == "fail":
            message = (
                "Face-data validity is below the minimum threshold "
                f"({valid_percent:.1f}% valid; minimum {min_valid_percent}%)."
            )
        elif status == "warn":
            message = (
                "Face-data quality should be reviewed before analysis "
                f"({valid_percent:.1f}% valid; warning threshold "
                f"{warning_valid_percent}%)."
            )
        else:
            message = "Face-data quality passed the configured validity checks."

        row = {
            "face_quality_group": "overall",
            "n_rows": n_rows,
            "n_valid": n_valid,
            "valid_percent": valid_percent,
            "n_invalid": n_invalid,
            "invalid_percent": invalid_percent,
            "n_unknown_validity": n_unknown,
            "unknown_validity_percent": unknown_percent,
            "n_missing_confidence": n_missing_confidence,
            "confidence_missing_percent": confidence_missing_percent,
            "mean_confidence": safe_stat(confidence, "mean"),
            "median_confidence": safe_stat(confidence, "median"),
            "min_confidence": safe_stat(confidence, "min"),
            "max_confidence": safe_stat(confidence, "max"),
            "n_success": n_success,
            "success_percent": success_percent,
            "n_duplicate_frames": n_duplicate_frames,
            "duplicate_frame_percent": duplicate_frame_percent,
            "n_missing_time": n_missing_time,
            "n_nonpositive_time_steps": n_nonpositive_time_steps,
            "max_time_gap_sec": max_gap,
            "median_time_step_sec": median_step,
            "estimated_sampling_rate_hz": estimated_hz,
            "face_quality_status": status,
            "message": message,
        }
        if group_values:
            labels = []
            for column, value in group_values.items():
                value = "missing" if pd.isna(value) else str(value)
                row[column] = value
                labels.append(f"{column}={value}")
            row["face_quality_group"] = " | ".join(labels)
        return row

    group_rows = []
    if groups:
        grouper = groups[0] if len(groups) == 1 else groups
        for keys, part in frame.groupby(grouper, dropna=False, sort=False):
            if len(groups) == 1:
                keys = (keys,)
            group_rows.append(summarise_subset(part, dict(zip(groups, keys, strict=True))))
    else:
        group_rows.append(summarise_subset(frame))
    group_summary = pd.DataFrame(group_rows)

    overview_row = summarise_subset(frame)
    overview_row.pop("face_quality_group", None)
    overview = pd.DataFrame([{"n_groups": len(group_summary), **overview_row}])

    def count_where(series):
        return int(pd.Series(series).fillna(False).sum())

    issue_names = [
        "valid_percent_below_minimum",
        "valid_percent_below_warning",
        "unknown_validity",
        "duplicate_frames",
        "large_time_gaps",
        "missing_confidence",
    ]
    affected = [
        count_where(
            pd.to_numeric(group_summary["valid_percent"], errors="coerce") < min_valid_percent
        ),
        count_where(
            pd.to_numeric(group_summary["valid_percent"], errors="coerce") < warning_valid_percent
        ),
        count_where(group_summary["face_quality_status"].eq("unknown")),
        count_where(
            pd.to_numeric(group_summary["duplicate_frame_percent"], errors="coerce")
            > max_duplicate_frame_percent
        ),
        (
            count_where(
                pd.to_numeric(group_summary["max_time_gap_sec"], errors="coerce")
                > float(max_time_gap_sec)
            )
            if max_time_gap_sec is not None
            else np.nan
        ),
        count_where(pd.to_numeric(group_summary["n_missing_confidence"], errors="coerce") > 0),
    ]
    thresholds = [
        min_valid_percent,
        warning_valid_percent,
        np.nan,
        max_duplicate_frame_percent,
        np.nan if max_time_gap_sec is None else float(max_time_gap_sec),
        np.nan,
    ]
    issue_summary = pd.DataFrame(
        {
            "issue": issue_names,
            "n_groups_affected": affected,
            "n_groups": len(group_summary),
            "threshold": thresholds,
        }
    )
    issue_summary["status"] = [
        "not_checked" if pd.isna(value) else "review" if value > 0 else "ok" for value in affected
    ]

    out = {
        "overview": overview,
        "group_summary": group_summary,
        "issue_summary": issue_summary,
        "data": frame.reset_index(drop=True),
        "settings": {
            "group_cols": groups or None,
            "confidence_threshold": confidence_threshold,
            "min_valid_percent": min_valid_percent,
            "warning_valid_percent": warning_valid_percent,
            "max_time_gap_sec": max_time_gap_sec,
            "max_duplicate_frame_percent": max_duplicate_frame_percent,
            "standardize": standardize,
        },
        "_gp3_class": "gp3_face_quality_audit",
    }
    return out


def summarize_gazepoint_face_quality(data, **kwargs):
    """Return the overview from a face-quality audit."""
    if isinstance(data, dict) and data.get("_gp3_class") == "gp3_face_quality_audit":
        out = data["overview"].copy()
    else:
        audited = audit_gazepoint_face_quality(data, **kwargs)
        out = audited["overview"].copy() if isinstance(audited, dict) else audited.copy()
    out.attrs["_gp3_class"] = "gp3_face_quality_summary"
    return out


def summarise_gazepoint_face_quality(data, **kwargs):
    return summarize_gazepoint_face_quality(data, **kwargs)


def sync_gazepoint_face_data(
    gaze,
    face,
    gaze_time_col=None,
    face_time_col=None,
    tolerance_ms=None,
    by=None,
    *,
    method="nearest_time",
    gaze_frame_col=None,
    face_frame_col=None,
    tolerance_sec=0.050,
    prefix="face_",
    keep_unmatched=True,
    standardize_face=True,
) -> pd.DataFrame:
    """Synchronise Gazepoint and face data by nearest time or exact frame."""
    if method not in {"nearest_time", "frame_exact"}:
        raise ValueError("method must be nearest_time or frame_exact")
    if tolerance_ms is not None:
        tolerance_sec = float(tolerance_ms) / 1000
    if not np.isfinite(tolerance_sec) or tolerance_sec < 0:
        raise ValueError("tolerance_sec must be a non-negative finite numeric scalar")

    g = ensure_dataframe(gaze).reset_index(drop=True)
    f = ensure_dataframe(face).reset_index(drop=True)
    required_face = {"face_frame", "face_time_sec", "face_confidence", "face_valid"}
    if standardize_face or not required_face.issubset(f.columns):
        f = standardize_gazepoint_face_columns(f)

    if by is None:
        mapping = {}
    elif isinstance(by, dict):
        mapping = {str(key): str(value) for key, value in by.items()}
    else:
        cols = [by] if isinstance(by, str) else list(by)
        mapping = {str(column): str(column) for column in cols}
    missing_gaze = [column for column in mapping if column not in g.columns]
    missing_face = [column for column in mapping.values() if column not in f.columns]
    if missing_gaze:
        raise ValueError("Gazepoint grouping column(s) not found: " + ", ".join(missing_gaze))
    if missing_face:
        raise ValueError("Facial-data grouping column(s) not found: " + ", ".join(missing_face))

    def choose(table, supplied, candidates, arg):
        if supplied is not None:
            if supplied not in table.columns:
                raise ValueError(f"{arg} was not found in the data")
            return supplied
        lower = {str(column).lower(): column for column in table.columns}
        for candidate in candidates:
            match = lower.get(candidate.lower())
            if match is not None:
                return match
        raise ValueError(f"{arg} could not be detected automatically. Please supply it explicitly")

    def group_key(table, columns):
        if not columns:
            return pd.Series(["overall"] * len(table), index=table.index)
        values = table[columns].copy()
        for column in columns:
            values[column] = values[column].astype("string").fillna("missing")
            values.loc[values[column].eq(""), column] = "missing"
        return values.astype(str).agg(" | ".join, axis=1)

    g[".gp3_face_sync_gaze_row"] = np.arange(1, len(g) + 1)
    f[".gp3_face_sync_face_row"] = np.arange(1, len(f) + 1)
    gaze_keys = group_key(g, list(mapping))
    face_keys = group_key(f, list(mapping.values()))
    matched = np.full(len(g), -1, dtype=int)
    diff = np.full(len(g), np.nan, dtype=float)
    status = np.full(len(g), "unmatched", dtype=object)

    if method == "nearest_time":
        gaze_time_col = choose(
            g,
            gaze_time_col,
            [
                "time_sec",
                "time_seconds",
                "time",
                "TIME",
                "timestamp",
                "timestamp_sec",
                "trial_time_sec",
                "relative_time_sec",
            ],
            "gaze_time_col",
        )
        face_time_col = choose(
            f,
            face_time_col,
            ["face_time_sec", "timestamp", "time_sec", "time_seconds", "time", "seconds"],
            "face_time_col",
        )
        gaze_time = pd.to_numeric(g[gaze_time_col], errors="coerce").to_numpy(float)
        face_time = pd.to_numeric(f[face_time_col], errors="coerce").to_numpy(float)
        for key in pd.unique(gaze_keys):
            gaze_idx = np.flatnonzero(gaze_keys.to_numpy() == key)
            face_idx = np.flatnonzero(face_keys.to_numpy() == key)
            face_idx = face_idx[np.isfinite(face_time[face_idx])]
            if not len(face_idx):
                continue
            for row in gaze_idx:
                if not np.isfinite(gaze_time[row]):
                    status[row] = "missing_gaze_time"
                    continue
                distances = np.abs(face_time[face_idx] - gaze_time[row])
                best = int(face_idx[int(np.argmin(distances))])
                matched[row] = best
                diff[row] = face_time[best] - gaze_time[row]
                status[row] = "matched" if abs(diff[row]) <= tolerance_sec else "outside_tolerance"
    else:
        gaze_frame_col = choose(
            g,
            gaze_frame_col,
            ["VID_FRAME", "video_frame", "frame", "frame_id", "face_frame"],
            "gaze_frame_col",
        )
        face_frame_col = choose(
            f,
            face_frame_col,
            ["face_frame", "frame", "Frame", "FRAME", "video_frame", "frame_id"],
            "face_frame_col",
        )
        gaze_frame = g[gaze_frame_col].astype("string")
        face_frame = f[face_frame_col].astype("string")
        lookup = {}
        for index, key in enumerate(
            zip(face_keys.astype(str), face_frame.astype(str), strict=True)
        ):
            lookup.setdefault(key, index)
        for row, key in enumerate(zip(gaze_keys.astype(str), gaze_frame.astype(str), strict=True)):
            if pd.isna(gaze_frame.iloc[row]):
                status[row] = "missing_gaze_frame"
                continue
            best = lookup.get(key)
            if best is not None:
                matched[row] = best
                status[row] = "matched"

    face_join = f.copy()
    renamed = []
    used = set(g.columns)
    for column in face_join.columns:
        name = str(column)
        if name != ".gp3_face_sync_face_row" and not name.startswith(prefix):
            name = prefix + name
        base = name
        suffix = 1
        while name in used or name in renamed:
            name = f"{base}_{suffix}"
            suffix += 1
        renamed.append(name)
    face_join.columns = renamed

    joined_face = pd.DataFrame(index=range(len(g)), columns=face_join.columns)
    valid_match = matched >= 0
    if valid_match.any():
        joined_face.loc[valid_match, :] = face_join.iloc[matched[valid_match]].to_numpy()
    out = pd.concat([g, joined_face], axis=1)
    out["face_sync_method"] = method
    out["face_sync_status"] = status
    out["face_sync_diff_sec"] = diff if method == "nearest_time" else np.nan
    out["face_sync_abs_diff_sec"] = np.abs(diff) if method == "nearest_time" else np.nan
    out["face_sync_within_tolerance"] = status == "matched"
    out["face_sync_tolerance_sec"] = tolerance_sec if method == "nearest_time" else np.nan
    if not keep_unmatched:
        out = out.loc[status == "matched"].copy()
    out = out.reset_index(drop=True)
    out.attrs["_gp3_class"] = "gp3_face_sync"
    out.attrs["gp3_face_sync_settings"] = {
        "method": method,
        "by": mapping or None,
        "gaze_time_col": gaze_time_col if method == "nearest_time" else None,
        "face_time_col": face_time_col if method == "nearest_time" else None,
        "gaze_frame_col": gaze_frame_col if method == "frame_exact" else None,
        "face_frame_col": face_frame_col if method == "frame_exact" else None,
        "tolerance_sec": tolerance_sec,
        "prefix": prefix,
        "keep_unmatched": keep_unmatched,
        "standardize_face": standardize_face,
    }
    return out


def audit_gazepoint_face_sync(
    gaze=None,
    face=None,
    *,
    data=None,
    group_cols=None,
    min_matched_percent=70,
    warning_matched_percent=85,
    max_abs_diff_sec=None,
    **kwargs,
):
    if data is None and face is not None:
        # Historical Python two-table synchronization route.
        synced = sync_gazepoint_face_data(gaze, face, **kwargs)
        face_cols = [c for c in synced.columns if c.endswith("_face")]
        matched = (
            ~synced[face_cols].isna().all(axis=1)
            if face_cols
            else pd.Series(True, index=synced.index)
        )
        return pd.DataFrame(
            {
                "n_gaze": [len(synced)],
                "n_matched": [int(matched.sum())],
                "match_rate": [float(matched.mean())],
            }
        )

    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unknown}")

    source = data if data is not None else gaze

    # An already synchronized table follows the frozen R audit.
    if isinstance(source, pd.DataFrame) and {
        "face_sync_method",
        "face_sync_status",
        "face_sync_within_tolerance",
    } <= set(source.columns):
        from ._behavioral_r2 import audit_face_sync

        return audit_face_sync(
            source,
            group_cols=group_cols,
            min_matched_percent=min_matched_percent,
            warning_matched_percent=warning_matched_percent,
            max_abs_diff_sec=max_abs_diff_sec,
        )

    # Otherwise preserve the old one-table summary.
    synced = ensure_dataframe(source)
    face_cols = [c for c in synced.columns if c.endswith("_face")]
    matched = (
        ~synced[face_cols].isna().all(axis=1) if face_cols else pd.Series(True, index=synced.index)
    )
    return pd.DataFrame(
        {
            "n_gaze": [len(synced)],
            "n_matched": [int(matched.sum())],
            "match_rate": [float(matched.mean())],
        }
    )


def audit_gazepoint_event_sync(
    gaze=None,
    events=None,
    *,
    data=None,
    time_col="time",
    event_col=None,
    group_cols=("subject", "media_id", "trial_global"),
    condition_col=None,
    expected_event_labels=None,
    onset_event_label=None,
    response_event_label=None,
    min_samples_per_unit=1,
    max_time_gap_ms=None,
    **kwargs,
):
    if data is None and events is not None:
        # Historical Python cross-table synchronization route.
        return audit_gazepoint_face_sync(gaze, events, **kwargs)

    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unknown}")

    source = data if data is not None else gaze

    from ._behavioral_r2 import audit_event_sync

    return audit_event_sync(
        source,
        time_col=time_col,
        event_col=event_col,
        group_cols=group_cols,
        condition_col=condition_col,
        expected_event_labels=expected_event_labels,
        onset_event_label=onset_event_label,
        response_event_label=response_event_label,
        min_samples_per_unit=min_samples_per_unit,
        max_time_gap_ms=max_time_gap_ms,
    )


def _face_numeric_cols(df):
    exclude = {"time", "timestamp", "subject", "trial", "condition", "frame"}
    return [c for c in df.select_dtypes(include=np.number).columns if c not in exclude]


def summarize_gazepoint_face_windows(
    data, group_cols=None, value_cols=None, **kwargs
) -> pd.DataFrame:

    from ._behavioral_r3a import _dispatch_r3a, _should_use_r3a

    if _should_use_r3a(
        "summarize_gazepoint_face_windows",
        locals(),
    ):
        return _dispatch_r3a(
            "summarize_gazepoint_face_windows",
            locals(),
        )

    df = standardize_gazepoint_face_columns(data)
    groups = normalize_group_cols(df, group_cols)
    vals = value_cols or _face_numeric_cols(df)
    if groups:
        return df.groupby(groups, dropna=False)[vals].mean().reset_index()
    return pd.DataFrame({c: [pd.to_numeric(df[c], errors="coerce").mean()] for c in vals})


def summarise_gazepoint_face_windows(data, **kwargs):
    return summarize_gazepoint_face_windows(data, **kwargs)


def summarize_gazepoint_face_reactivity(
    data, baseline=None, group_cols=None, value_cols=None, **kwargs
) -> pd.DataFrame:

    from ._behavioral_r3a import _dispatch_r3a, _should_use_r3a

    if _should_use_r3a(
        "summarize_gazepoint_face_reactivity",
        locals(),
    ):
        return _dispatch_r3a(
            "summarize_gazepoint_face_reactivity",
            locals(),
        )

    out = summarize_gazepoint_face_windows(data, group_cols=group_cols, value_cols=value_cols)
    num = out.select_dtypes(include=np.number).columns
    for c in num:
        out[c + "_reactivity"] = out[c] - float(
            baseline.get(c, 0) if isinstance(baseline, dict) else (baseline or 0)
        )
    return out


def summarise_gazepoint_face_reactivity(data, **kwargs):
    return summarize_gazepoint_face_reactivity(data, **kwargs)


def prepare_gazepoint_multimodal_data(
    gaze=None,
    face=None,
    *,
    face_windows=None,
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
    **kwargs,
):
    if face_windows is None:
        # Historical Python route.
        if kwargs:
            return (
                sync_gazepoint_face_data(gaze, face, **kwargs)
                if face is not None
                else ensure_dataframe(gaze)
            )
        return sync_gazepoint_face_data(gaze, face) if face is not None else ensure_dataframe(gaze)

    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unknown}")

    from ._behavioral_r2 import prepare_multimodal_data

    return prepare_multimodal_data(
        face_windows,
        gaze_data=gaze_data,
        response_data=response_data,
        by=by,
        gaze_by=gaze_by,
        response_by=response_by,
        predictor_cols=predictor_cols,
        outcome_cols=outcome_cols,
        covariate_cols=covariate_cols,
        scale_predictors=scale_predictors,
        scaled_suffix=scaled_suffix,
        drop_missing_outcomes=drop_missing_outcomes,
        keep_all=keep_all,
    )


def create_gazepoint_face_reporting_checklist(
    face_data=None,
    quality_audit=None,
    sync_audit=None,
    window_summary=None,
    reactivity_summary=None,
    multimodal_model=None,
    include_interpretation_cautions=True,
) -> pd.DataFrame:
    """Create the R v2.3.0 reviewer-facing face reporting checklist."""

    def row(section, item, status, evidence, recommendation):
        return {
            "section": section,
            "item": item,
            "status": status,
            "evidence": evidence,
            "recommendation": recommendation,
        }

    def data_evidence(value):
        if value is None:
            return "No object supplied."
        if isinstance(value, pd.DataFrame):
            return f"{len(value)} row(s), {len(value.columns)} column(s)."
        return f"Object supplied with class: {type(value).__name__}"

    def standard_status(value):
        if not isinstance(value, pd.DataFrame):
            return "not_available"
        required = {"face_time_sec", "face_confidence", "face_valid"}
        present = required.intersection(value.columns)
        if len(present) == len(required):
            return "pass"
        return "warn" if present else "not_available"

    def standard_evidence(value):
        if not isinstance(value, pd.DataFrame):
            return "No face-data table supplied."
        required = [
            "face_frame",
            "face_time_sec",
            "face_confidence",
            "face_success",
            "face_valid",
        ]
        present = [column for column in required if column in value.columns]
        missing = [column for column in required if column not in value.columns]
        return (
            "Present: "
            + (", ".join(present) if present else "none")
            + ". Missing: "
            + (", ".join(missing) if missing else "none")
            + "."
        )

    def status_map(value):
        if value is None or pd.isna(value) or str(value) == "":
            return "unknown"
        lowered = str(value).lower()
        if lowered in {"pass", "ok"}:
            return "pass"
        if lowered in {"warn", "warning", "review"}:
            return "warn"
        if lowered in {"fail", "failed"}:
            return "fail"
        if lowered in {"unknown", "not_available"}:
            return lowered
        return "review"

    def audit_status(value, status_column):
        if not isinstance(value, dict) or not isinstance(value.get("overview"), pd.DataFrame):
            return "not_available", "No audit overview supplied."
        overview = value["overview"]
        if len(overview) < 1 or status_column not in overview.columns:
            return "unknown", "Audit overview is missing the status column."
        raw = overview.iloc[0][status_column]
        evidence_cols = [
            column
            for column in [
                "n_rows",
                "valid_percent",
                "matched_percent",
                "face_quality_status",
                "face_sync_audit_status",
                "max_abs_diff_sec",
                "max_time_gap_sec",
            ]
            if column in overview.columns
        ]
        evidence = (
            "; ".join(f"{column}={overview.iloc[0][column]}" for column in evidence_cols)
            if evidence_cols
            else f"Status={raw}"
        )
        return status_map(raw), evidence

    def object_evidence(value, expected_class):
        if value is None:
            return "No object supplied."
        cls = value.get("_gp3_class") if isinstance(value, dict) else None
        text = cls or type(value).__name__
        return f"Class: {text}" + (
            " (expected class present)."
            if cls == expected_class
            else " (expected class not found)."
        )

    def issue_status(value):
        if not isinstance(value, dict) or not isinstance(value.get("issue_summary"), pd.DataFrame):
            return "not_available"
        issues = value["issue_summary"]
        if "n_groups_affected" not in issues.columns:
            return "unknown"
        affected = pd.to_numeric(issues["n_groups_affected"], errors="coerce")
        return "review" if (affected > 0).any() else "pass"

    def issue_evidence(value):
        if not isinstance(value, dict) or not isinstance(value.get("issue_summary"), pd.DataFrame):
            return "No issue summary supplied."
        issues = value["issue_summary"]
        if not {"issue", "n_groups_affected"}.issubset(issues.columns):
            return f"{len(issues)} issue-summary row(s) supplied."
        affected = issues.loc[pd.to_numeric(issues["n_groups_affected"], errors="coerce") > 0]
        if len(affected) < 1:
            return "No affected groups reported in issue summary."
        return "; ".join(
            f"{record.issue}={record.n_groups_affected}"
            for record in affected.itertuples(index=False)
        )

    def window_status(value):
        if not isinstance(value, pd.DataFrame):
            return "not_available"
        if len(value) < 1:
            return "fail"
        if (
            "n_used" in value.columns
            and (pd.to_numeric(value["n_used"], errors="coerce") < 1).any()
        ):
            return "review"
        return "pass"

    def window_evidence(value):
        if not isinstance(value, pd.DataFrame):
            return "No window-summary table supplied."
        text = f"{len(value)} window-summary row(s)."
        if "n_used" in value.columns and len(value):
            numeric = pd.to_numeric(value["n_used"], errors="coerce")
            if numeric.notna().any():
                text += f" n_used range: {numeric.min():g}-{numeric.max():g}."
        return text

    def reactivity_evidence(value):
        if not isinstance(value, pd.DataFrame):
            return "No reactivity-summary table supplied."
        measures = (
            value["measure"].dropna().astype(str).unique().tolist() if "measure" in value else []
        )
        suffix = f"; measure(s): {', '.join(measures)}" if measures else ""
        return f"{len(value)} reactivity row(s){suffix}."

    def model_evidence(value):
        if value is None:
            return "No model object supplied."
        if isinstance(value, dict) and isinstance(value.get("settings"), dict):
            settings = value["settings"]
            return (
                f"Outcome: {settings.get('outcome')}; model rows: "
                f"{settings.get('n_rows_model')}; class: "
                f"{value.get('_gp3_class', type(value).__name__)}."
            )
        return f"Model-like object supplied with class: {type(value).__name__}"

    quality_status, quality_evidence = audit_status(quality_audit, "face_quality_status")
    sync_status, sync_evidence = audit_status(sync_audit, "face_sync_audit_status")

    rows = [
        row(
            "Input and provenance",
            "External face-analysis data are available",
            "pass" if face_data is not None else "not_available",
            data_evidence(face_data),
            "Report the external face-analysis tool, version, input files, and exported columns."
            if face_data is not None
            else "Provide imported or standardised external face-analysis data when facial-behaviour analyses are reported.",
        ),
        row(
            "Input and provenance",
            "Standardised face columns are available",
            standard_status(face_data),
            standard_evidence(face_data),
            "Report standardised timing, frame, confidence, success, and validity fields where available.",
        ),
        row(
            "Quality control",
            "Face-data quality audit is available",
            "pass" if quality_audit is not None else "not_available",
            object_evidence(quality_audit, "gp3_face_quality_audit"),
            "Use audit_gazepoint_face_quality() before reporting facial-behaviour summaries.",
        ),
        row(
            "Quality control",
            "Face-data quality status is acceptable",
            quality_status,
            quality_evidence,
            "Report valid-row percentage, confidence coverage, duplicate-frame checks, and timing-gap checks.",
        ),
        row(
            "Quality control",
            "Quality issues are documented",
            issue_status(quality_audit),
            issue_evidence(quality_audit),
            "Document groups requiring review and explain any exclusions or sensitivity analyses.",
        ),
        row(
            "Synchronisation",
            "Face-data synchronisation audit is available",
            "pass" if sync_audit is not None else "not_available",
            object_evidence(sync_audit, "gp3_face_sync_audit"),
            "Use audit_gazepoint_face_sync() when face data are aligned to Gazepoint rows.",
        ),
        row(
            "Synchronisation",
            "Synchronisation status is acceptable",
            sync_status,
            sync_evidence,
            "Report matching method, tolerance, matched percentage, unmatched rows, and timing differences.",
        ),
        row(
            "Window summaries",
            "Face-window summary is available",
            "pass" if window_summary is not None else "not_available",
            data_evidence(window_summary),
            "Report window definitions, grouping variables, validity filtering, and summarised facial-behaviour measures.",
        ),
        row(
            "Window summaries",
            "Window-summary coverage is documented",
            window_status(window_summary),
            window_evidence(window_summary),
            "Report n_rows, n_used, valid_percent, confidence summaries, and measure summaries for each window.",
        ),
        row(
            "Reactivity summaries",
            "Baseline-to-response reactivity is available when used",
            "pass" if reactivity_summary is not None else "not_available",
            reactivity_evidence(reactivity_summary),
            "Define baseline and response windows and report reactivity as response minus baseline.",
        ),
        row(
            "Modelling",
            "Multimodal or face-window model object is available when models are reported",
            "pass" if multimodal_model is not None else "not_available",
            model_evidence(multimodal_model),
            "Report formula, predictors, covariates, random effects, family, missing-data handling, and model sample size.",
        ),
    ]

    if include_interpretation_cautions:
        rows.extend(
            [
                row(
                    "Interpretation",
                    "Facial-behaviour variables are not interpreted as direct emotion measures",
                    "review",
                    "Manual manuscript/reporting review required.",
                    "Use cautious language such as facial-behaviour measure, action-unit intensity, confidence, synchronisation coverage, or window-level feature.",
                ),
                row(
                    "Interpretation",
                    "Unsupported claims are avoided",
                    "review",
                    "Manual manuscript/reporting review required.",
                    "Avoid claims of true emotion detection, hidden affect, micro-expression evidence, diagnosis, or causal mechanism without design support.",
                ),
            ]
        )

    out = pd.DataFrame(rows)
    out.attrs["_gp3_class"] = "gp3_face_reporting_checklist"
    return out


# BEGIN R V2.3.0 CALL-SURFACE ALIASES
sync_gazepoint_face_data = r_aliases(
    sync_gazepoint_face_data, gazepoint_data="gaze", face_data="face"
)
# END R V2.3.0 CALL-SURFACE ALIASES
