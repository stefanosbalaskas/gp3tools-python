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
    data,
    x=None,
    y=None,
    uncertainty=None,
    max_uncertainty=None,
    weight_output="cnn_uncertainty_weight",
    valid_output="cnn_valid_frame",
    **kwargs,
):
    """Apply R-v2.3.0 uncertainty weighting to external CNN gaze predictions."""
    import numpy as np
    import pandas as pd

    # Retain the pre-parity convenience call when x/y are omitted.
    if x is None or y is None:
        uncertainty_col = kwargs.pop("uncertainty_col", uncertainty or "uncertainty")
        threshold = kwargs.pop("threshold", 0.5 if max_uncertainty is None else max_uncertainty)
        keep_flag = kwargs.pop("keep_flag", True)
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")
        if uncertainty_col not in data:
            raise ValueError(f"Missing columns: {uncertainty_col}")
        out = data.copy()
        out["cnn_uncertainty_pass"] = (
            pd.to_numeric(out[uncertainty_col], errors="coerce") <= threshold
        )
        return out if keep_flag else out.loc[out["cnn_uncertainty_pass"]].copy()

    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unknown}")
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a data frame.")
    required = [x, y] + ([] if uncertainty is None else [uncertainty])
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    out = data.copy()
    xv = pd.to_numeric(out[x], errors="coerce").to_numpy(float)
    yv = pd.to_numeric(out[y], errors="coerce").to_numpy(float)
    valid = np.isfinite(xv) & np.isfinite(yv)
    if uncertainty is None:
        out[weight_output] = valid.astype(float)
        out[valid_output] = valid
        return out

    u = pd.to_numeric(out[uncertainty], errors="coerce").to_numpy(float)
    valid = valid & np.isfinite(u)
    if max_uncertainty is not None:
        valid = valid & (u <= max_uncertainty)
    finite_u = u[np.isfinite(u)]
    scale = float(np.median(finite_u)) if len(finite_u) else np.nan
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    with np.errstate(over="ignore", invalid="ignore"):
        weight = np.exp(-u / scale)
    weight[~valid] = 0.0
    out[weight_output] = weight
    out[valid_output] = valid
    return out


def select_gazepoint_adaptive_trial(
    candidates=None,
    mean=None,
    sd=None,
    acquisition="ucb",
    kappa=2,
    best_observed=None,
    maximize=True,
    *,
    data=None,
    score_col=None,
    strategy=None,
    **kwargs,
):
    """Select the next adaptive trial using R gp3tools 2.3.0 acquisition rules."""
    import numpy as np
    import pandas as pd
    from scipy import stats as scipy_stats

    if candidates is None:
        candidates = data
    if mean is None or sd is None:
        # Legacy convenience path.
        df = candidates.copy()
        selected_strategy = strategy or "highest_uncertainty"
        col = score_col or next(
            (c for c in ("uncertainty", "score", "information") if c in df.columns),
            None,
        )
        if col is None:
            return df.iloc[[0]].copy() if len(df) else df.copy()
        values = pd.to_numeric(df[col], errors="coerce")
        idx = (
            values.idxmax()
            if selected_strategy in {"highest_uncertainty", "max", "highest"}
            else values.idxmin()
        )
        return df.loc[[idx]].copy()

    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unknown}")
    if not isinstance(candidates, pd.DataFrame):
        raise ValueError("candidates must be a data frame.")
    if acquisition not in {"ucb", "uncertainty", "expected_improvement"}:
        raise ValueError("`acquisition` must be one of: ucb, uncertainty, expected_improvement.")
    if mean not in candidates.columns or sd not in candidates.columns:
        raise ValueError("mean and sd columns must be present in candidates.")
    mu = candidates[mean]
    sigma = candidates[sd]
    if not pd.api.types.is_numeric_dtype(mu) or not pd.api.types.is_numeric_dtype(sigma):
        raise ValueError("mean and sd columns must be numeric.")
    mu = pd.to_numeric(mu, errors="coerce").to_numpy(float)
    sigma = pd.to_numeric(sigma, errors="coerce").to_numpy(float)

    if acquisition == "ucb":
        score = mu + kappa * sigma
    elif acquisition == "uncertainty":
        score = sigma.copy()
    else:
        if best_observed is None:
            best_observed = np.nanmax(mu) if maximize else np.nanmin(mu)
        improvement = mu - best_observed if maximize else best_observed - mu
        with np.errstate(divide="ignore", invalid="ignore"):
            z = improvement / sigma
        z[~np.isfinite(z)] = 0.0
        score = improvement * scipy_stats.norm.cdf(z) + sigma * scipy_stats.norm.pdf(z)

    out = candidates.copy()
    out["acquisition_score"] = score
    finite = np.isfinite(score)
    if not finite.any():
        return out.iloc[0:0].copy()
    idx_pos = int(np.nanargmax(score) if maximize else np.nanargmin(score))
    return out.iloc[[idx_pos]].copy()
