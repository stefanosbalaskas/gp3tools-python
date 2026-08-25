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


def audit_gazepoint_face_quality(data, confidence_col=None, threshold: float = 0.8) -> pd.DataFrame:
    df = standardize_gazepoint_face_columns(data)
    if confidence_col is None:
        confidence_col = next((c for c in df.columns if "confidence" in c), None)
    if confidence_col and confidence_col in df:
        v = pd.to_numeric(df[confidence_col], errors="coerce")
        return pd.DataFrame(
            {
                "n": [len(df)],
                "n_valid": [int(v.notna().sum())],
                "mean_confidence": [float(v.mean())],
                "prop_below_threshold": [float((v < threshold).mean())],
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


def summarize_gazepoint_face_quality(data, **kwargs):
    return audit_gazepoint_face_quality(data, **kwargs)


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


def audit_gazepoint_face_sync(gaze, face=None, **kwargs) -> pd.DataFrame:
    synced = (
        sync_gazepoint_face_data(gaze, face, **kwargs)
        if face is not None
        else ensure_dataframe(gaze)
    )
    face_cols = [c for c in synced.columns if c.endswith("_face")]
    matched = (
        (~synced[face_cols].isna().all(axis=1))
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


def audit_gazepoint_event_sync(gaze, events=None, **kwargs):
    return audit_gazepoint_face_sync(gaze, events, **kwargs)


def _face_numeric_cols(df):
    exclude = {"time", "timestamp", "subject", "trial", "condition", "frame"}
    return [c for c in df.select_dtypes(include=np.number).columns if c not in exclude]


def summarize_gazepoint_face_windows(
    data, group_cols=None, value_cols=None, **kwargs
) -> pd.DataFrame:
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
    out = summarize_gazepoint_face_windows(data, group_cols=group_cols, value_cols=value_cols)
    num = out.select_dtypes(include=np.number).columns
    for c in num:
        out[c + "_reactivity"] = out[c] - float(
            baseline.get(c, 0) if isinstance(baseline, dict) else (baseline or 0)
        )
    return out


def summarise_gazepoint_face_reactivity(data, **kwargs):
    return summarize_gazepoint_face_reactivity(data, **kwargs)


def prepare_gazepoint_multimodal_data(gaze, face=None, **kwargs):
    return (
        sync_gazepoint_face_data(gaze, face, **kwargs)
        if face is not None
        else ensure_dataframe(gaze)
    )


def create_gazepoint_face_reporting_checklist(data=None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "item": [
                "face software/version",
                "confidence threshold",
                "synchronisation method",
                "missing-face handling",
                "aggregation window",
            ],
            "reported": [False] * 5,
        }
    )


# BEGIN R V2.3.0 CALL-SURFACE ALIASES
sync_gazepoint_face_data = r_aliases(
    sync_gazepoint_face_data, gazepoint_data="gaze", face_data="face"
)
# END R V2.3.0 CALL-SURFACE ALIASES
