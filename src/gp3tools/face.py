"""Facial-analysis and multimodal synchronisation helpers."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ._utils import ensure_dataframe, infer_column, normalize_group_cols, time_to_seconds


def standardize_gazepoint_face_columns(data) -> pd.DataFrame:
    df = ensure_dataframe(data)
    ren = {}
    for c in df.columns:
        n = re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")
        ren[c] = n
    return df.rename(columns=ren)


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
    gaze, face, gaze_time_col=None, face_time_col=None, tolerance_ms: float = 50.0, by=None
) -> pd.DataFrame:
    g = ensure_dataframe(gaze)
    f = standardize_gazepoint_face_columns(face)
    gaze_time_col = gaze_time_col or infer_column(g, "time")
    face_time_col = (
        face_time_col or infer_column(f, "time") or next((c for c in f if "time" in c), None)
    )
    if not gaze_time_col or not face_time_col:
        raise ValueError("time columns required")
    g = g.copy()
    f = f.copy()
    g["_sync_t"] = time_to_seconds(g[gaze_time_col])
    f["_sync_t"] = time_to_seconds(f[face_time_col])
    by_cols = normalize_group_cols(g, by)
    out = pd.merge_asof(
        g.sort_values("_sync_t"),
        f.sort_values("_sync_t"),
        on="_sync_t",
        by=by_cols or None,
        direction="nearest",
        tolerance=tolerance_ms / 1000,
        suffixes=("", "_face"),
    )
    return out.drop(columns="_sync_t")


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
