"""Additional native utility functions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ._utils import ensure_dataframe, infer_column


def prepare_gazepoint_heatmap_data(data, x_col=None, y_col=None, bins=40, **kwargs) -> pd.DataFrame:
    df = ensure_dataframe(data)
    x = x_col or infer_column(df, "x")
    y = y_col or infer_column(df, "y")
    h, xe, ye = np.histogram2d(
        pd.to_numeric(df[x], errors="coerce").dropna(),
        pd.to_numeric(df.loc[pd.to_numeric(df[x], errors="coerce").notna(), y], errors="coerce"),
        bins=bins,
    )
    rows = []
    for i in range(h.shape[0]):
        for j in range(h.shape[1]):
            rows.append(
                {
                    "x_bin": i,
                    "y_bin": j,
                    "x_center": (xe[i] + xe[i + 1]) / 2,
                    "y_center": (ye[j] + ye[j + 1]) / 2,
                    "count": h[i, j],
                }
            )
    return pd.DataFrame(rows)


def recalibrate_gazepoint_gaze(
    data,
    x_col=None,
    y_col=None,
    target_x=0.5,
    target_y=0.5,
    method="offset",
    output_x="x_recalibrated",
    output_y="y_recalibrated",
    **kwargs,
):
    df = ensure_dataframe(data)
    x = x_col or infer_column(df, "x")
    y = y_col or infer_column(df, "y")
    out = df.copy()
    xv = pd.to_numeric(out[x], errors="coerce")
    yv = pd.to_numeric(out[y], errors="coerce")
    if method == "offset":
        out[output_x] = xv + (target_x - xv.mean())
        out[output_y] = yv + (target_y - yv.mean())
    elif method == "scale":
        out[output_x] = target_x + (xv - xv.mean())
        out[output_y] = target_y + (yv - yv.mean())
    else:
        raise ValueError("method must be 'offset' or 'scale'")
    return out


def filter_gazepoint_cnn_uncertainty(
    data, uncertainty_col="uncertainty", threshold=0.5, keep_flag=True, **kwargs
):
    df = ensure_dataframe(data)
    out = df.copy()
    out["cnn_uncertainty_pass"] = pd.to_numeric(out[uncertainty_col], errors="coerce") <= threshold
    return out if keep_flag else out[out["cnn_uncertainty_pass"]].copy()


def select_gazepoint_adaptive_trial(data, score_col=None, strategy="highest_uncertainty", **kwargs):
    df = ensure_dataframe(data)
    score_col = score_col or next(
        (c for c in ["uncertainty", "score", "information"] if c in df), None
    )
    if not score_col:
        return df.iloc[[0]].copy() if len(df) else df.copy()
    idx = (
        pd.to_numeric(df[score_col], errors="coerce").idxmax()
        if strategy in {"highest_uncertainty", "max", "highest"}
        else pd.to_numeric(df[score_col], errors="coerce").idxmin()
    )
    return df.loc[[idx]].copy()
