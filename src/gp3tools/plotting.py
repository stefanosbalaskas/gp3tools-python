"""Matplotlib visualisations corresponding to gp3tools plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ._utils import ensure_dataframe, infer_column


def _figax(ax=None):
    if ax is None:
        return plt.subplots(figsize=(7, 4))
    return ax.figure, ax


def _xy(data, x=None, y=None):
    df = ensure_dataframe(data, copy=False)
    x = x or infer_column(df, "time") or df.columns[0]
    y = (
        y
        or infer_column(df, "pupil")
        or next((c for c in df.select_dtypes(include=np.number).columns if c != x), df.columns[-1])
    )
    return df, x, y


def plot_gazepoint_time_series(data, x_col=None, y_col=None, group_col=None, ax=None, **kwargs):
    df, x, y = _xy(data, x_col, y_col)
    fig, ax = _figax(ax)
    if group_col and group_col in df:
        for label, g in df.groupby(group_col, dropna=False):
            ax.plot(g[x], g[y], label=str(label))
            ax.legend()
    else:
        ax.plot(df[x], df[y])
    ax.set_xlabel(str(x))
    ax.set_ylabel(str(y))
    ax.set_title("Gazepoint time series")
    return fig


def plot_gazepoint_pupil_timecourse(data, pupil_col=None, **kwargs):
    if pupil_col is None:
        pupil_col = kwargs.pop("y_col", None)
    else:
        kwargs.pop("y_col", None)
    return plot_gazepoint_time_series(data, y_col=pupil_col, **kwargs)


def plot_gazepoint_pupil_preprocessing(data, **kwargs):
    return plot_gazepoint_pupil_timecourse(data, **kwargs)


def plot_gazepoint_pupil_status(data, status_col="pupil_flag", ax=None, **kwargs):
    df = ensure_dataframe(data)
    fig, ax = _figax(ax)
    df[status_col].astype(str).value_counts().plot(kind="bar", ax=ax)
    ax.set_title("Pupil status")
    ax.set_ylabel("Samples")
    return fig


def plot_sampling_rate(data, ax=None, **kwargs):
    from .qc import check_sampling_rate

    s = check_sampling_rate(data, **kwargs)
    fig, ax = _figax(ax)
    ax.bar(np.arange(len(s)), s["sampling_hz"])
    ax.axhline(kwargs.get("expected_hz", 60), linestyle="--")
    ax.set_ylabel("Hz")
    ax.set_title("Sampling rate")
    return fig


def plot_tracking_quality(data, ax=None, **kwargs):
    from .qc import summarise_tracking_quality

    q = summarise_tracking_quality(data, **kwargs)
    fig, ax = _figax(ax)
    y = "valid_prop" if "valid_prop" in q else next(c for c in q.select_dtypes(include=np.number))
    ax.bar(np.arange(len(q)), q[y])
    ax.set_ylim(0, 1 if "prop" in y else None)
    ax.set_title("Tracking quality")
    return fig


def plot_gazepoint_missingness_profile(data, ax=None, **kwargs):
    df = ensure_dataframe(data)
    fig, ax = _figax(ax)
    miss = df.isna().mean().sort_values(ascending=False)
    ax.barh(np.arange(len(miss)), miss.values)
    ax.set_yticks(np.arange(len(miss)), miss.index)
    ax.set_xlabel("Missing proportion")
    ax.set_title("Missingness profile")
    return fig


def plot_gazepoint_qc_overview(data, **kwargs):
    return plot_gazepoint_missingness_profile(data, **kwargs)


def plot_gazepoint_phase_timeline(data, **kwargs):
    return plot_gazepoint_time_series(data, **kwargs)


def plot_gazepoint_heatmap(data, x_col=None, y_col=None, bins=40, ax=None, **kwargs):
    df = ensure_dataframe(data)
    x = x_col or infer_column(df, "x")
    y = y_col or infer_column(df, "y")
    fig, ax = _figax(ax)
    ax.hist2d(
        pd.to_numeric(df[x], errors="coerce"), pd.to_numeric(df[y], errors="coerce"), bins=bins
    )
    ax.set_title("Gaze heatmap")
    ax.set_xlabel(str(x))
    ax.set_ylabel(str(y))
    ax.invert_yaxis()
    return fig


def plot_gazepoint_heatmap_overlay(data, **kwargs):
    return plot_gazepoint_heatmap(data, **kwargs)


def export_gazepoint_heatmap_png(data, path="gazepoint_heatmap.png", dpi=150, **kwargs):
    fig = plot_gazepoint_heatmap(data, **kwargs)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return Path(path)


def plot_gazepoint_aoi_timeline(data, aoi_col=None, time_col=None, ax=None, **kwargs):
    df = ensure_dataframe(data)
    aoi_col = aoi_col or infer_column(df, "aoi")
    time_col = time_col or infer_column(df, "time")
    fig, ax = _figax(ax)
    cats = pd.Categorical(df[aoi_col])
    ax.scatter(df[time_col], cats.codes, s=8)
    ax.set_yticks(range(len(cats.categories)), cats.categories)
    ax.set_title("AOI timeline")
    return fig


def plot_gazepoint_aoi_transition_matrix(data, ax=None, **kwargs):
    from .aoi import compute_gazepoint_aoi_transition_matrix

    m = (
        compute_gazepoint_aoi_transition_matrix(data, **kwargs)
        if not (
            isinstance(data, pd.DataFrame)
            and data.index.name is not None
            and data.shape[0] == data.shape[1]
        )
        else data
    )
    fig, ax = _figax(ax)
    im = ax.imshow(np.asarray(m, float), aspect="auto")
    ax.set_xticks(range(len(m.columns)), m.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(m.index)), m.index)
    fig.colorbar(im, ax=ax)
    ax.set_title("AOI transition matrix")
    return fig


def plot_transition_heatmap(data, **kwargs):
    return plot_gazepoint_aoi_transition_matrix(data, **kwargs)


def plot_gazepoint_aoi_verification(data, **kwargs):
    return plot_gazepoint_heatmap(data, **kwargs)


def plot_gazepoint_stimulus_layout_qc(data, **kwargs):
    return plot_gazepoint_heatmap(data, **kwargs)


def plot_gazepoint_scanpath(data, x_col=None, y_col=None, ax=None, **kwargs):
    df = ensure_dataframe(data)
    x = x_col or infer_column(df, "x")
    y = y_col or infer_column(df, "y")
    fig, ax = _figax(ax)
    ax.plot(df[x], df[y], marker="o", markersize=3)
    ax.invert_yaxis()
    ax.set_title("Scanpath")
    return fig


def plot_gazepoint_scanpaths(data, group_col=None, ax=None, **kwargs):
    df = ensure_dataframe(data)
    group_col = group_col or infer_column(df, "subject")
    fig, ax = _figax(ax)
    if group_col:
        for _, g in df.groupby(group_col):
            x = infer_column(g, "x")
            y = infer_column(g, "y")
            ax.plot(g[x], g[y], alpha=0.5)
    else:
        return plot_gazepoint_scanpath(df, ax=ax, **kwargs)
    ax.invert_yaxis()
    ax.set_title("Scanpaths")
    return fig


def plot_gazepoint_scanpath_clusters(data, cluster_col="cluster", **kwargs):
    return plot_gazepoint_scanpaths(data, group_col=cluster_col, **kwargs)


def plot_gazepoint_scanpath_cluster_stability(data, ax=None, **kwargs):
    df = ensure_dataframe(data)
    fig, ax = _figax(ax)
    x = df.get("n_clusters", np.arange(len(df)))
    y = df.get("stability", df.select_dtypes(include=np.number).iloc[:, -1])
    ax.plot(x, y, marker="o")
    ax.set_title("Scanpath cluster stability")
    return fig


def plot_gazepoint_event_detector_agreement(data, ax=None, **kwargs):
    from .events import compare_gazepoint_event_detectors

    df = (
        data
        if isinstance(data, pd.DataFrame) and "agreement" in data
        else compare_gazepoint_event_detectors(data, **kwargs)
    )
    fig, ax = _figax(ax)
    vals = df["agreement"].value_counts()
    ax.bar(vals.index.astype(str), vals.values)
    ax.set_title("Event-detector agreement")
    return fig


def plot_gazepoint_event_detector_benchmark(data, ax=None, **kwargs):
    df = ensure_dataframe(data)
    fig, ax = _figax(ax)
    s = df.groupby("detector")["elapsed_seconds"].mean()
    ax.bar(s.index, s.values)
    ax.set_ylabel("Seconds")
    ax.set_title("Event-detector benchmark")
    return fig


def plot_gazepoint_binocular_diagnostics(data, left_col=None, right_col=None, ax=None, **kwargs):
    df = ensure_dataframe(data)
    left_col = left_col or next(
        (c for c in ["LPMM", "left_pupil", "left_pupil_reconstructed"] if c in df), None
    )
    right_col = right_col or next(
        (c for c in ["RPMM", "right_pupil", "right_pupil_reconstructed"] if c in df), None
    )
    fig, ax = _figax(ax)
    ax.scatter(df[left_col], df[right_col], s=8, alpha=0.5)
    ax.set_xlabel(str(left_col))
    ax.set_ylabel(str(right_col))
    ax.set_title("Binocular pupil diagnostics")
    return fig


def plot_gazepoint_model_predictions(model=None, data=None, ax=None, **kwargs):
    """Plot model predictions; model is optional for R-compatible call surfaces."""
    if model is None:
        raise ValueError("model is required by the current native Python plotting backend")
    fig, ax = _figax(ax)
    pred = np.asarray(
        model.predict(data) if data is not None else getattr(model, "fittedvalues", [])
    )
    ax.plot(np.arange(len(pred)), pred)
    ax.set_title("Model predictions")
    return fig


def plot_gazepoint_model_residuals(
    model=None,
    data=None,
    fitted_col=None,
    residual_col=None,
    type="residuals_fitted",
    title=None,
    ax=None,
    **kwargs,
):
    """Plot residual-vs-fitted or QQ diagnostics from a model or data frame."""
    if type not in {"residuals_fitted", "qq"}:
        raise ValueError("type must be residuals_fitted or qq")
    if data is not None:
        df = ensure_dataframe(data, copy=False)
        if fitted_col is None:
            fitted_col = next((c for c in ("fitted", "fitted_value", ".fitted") if c in df), None)
        if residual_col is None:
            residual_col = next(
                (c for c in ("residual", "residuals", ".resid", ".residual") if c in df),
                None,
            )
        if fitted_col is None or residual_col is None:
            raise ValueError("Could not identify fitted and residual columns")
        fitted = pd.to_numeric(df[fitted_col], errors="coerce").to_numpy(float)
        resid = pd.to_numeric(df[residual_col], errors="coerce").to_numpy(float)
    else:
        if model is None:
            raise ValueError("Supply either model or data")
        resid = np.asarray(getattr(model, "resid", []), dtype=float)
        fitted = np.asarray(
            getattr(model, "fittedvalues", np.arange(len(resid))),
            dtype=float,
        )
    finite = np.isfinite(fitted) & np.isfinite(resid)
    fitted = fitted[finite]
    resid = resid[finite]
    if not len(resid):
        raise ValueError("No finite fitted/residual pairs are available")
    fig, ax = _figax(ax)
    if type == "residuals_fitted":
        ax.scatter(fitted, resid, s=10)
        ax.axhline(0, linestyle="--")
        ax.set_xlabel("Fitted values")
        ax.set_ylabel("Residuals")
    else:
        from scipy import stats as _stats

        ordered = np.sort(resid)
        theoretical = _stats.norm.ppf((np.arange(1, len(ordered) + 1) - 0.5) / len(ordered))
        ax.scatter(theoretical, ordered, s=10)
        if len(ordered) >= 2:
            slope, intercept = np.polyfit(theoretical, ordered, 1)
            ax.plot(theoretical, intercept + slope * theoretical)
        ax.set_xlabel("Theoretical quantiles")
        ax.set_ylabel("Residual quantiles")
    ax.set_title("Model residuals" if title is None else str(title))
    return fig


def plot_gazepoint_gca(model_or_data, **kwargs):
    return (
        plot_gazepoint_model_predictions(model_or_data, **kwargs)
        if hasattr(model_or_data, "predict")
        else plot_gazepoint_time_series(model_or_data, **kwargs)
    )


def plot_gazepoint_aoi_gamm(model_or_data, **kwargs):
    return plot_gazepoint_gca(model_or_data, **kwargs)


def plot_gazepoint_time_varying_effect(data, **kwargs):
    return plot_gazepoint_time_series(data, **kwargs)


def plot_gazepoint_cluster_results(result, ax=None, **kwargs):
    obs = result["observed"] if isinstance(result, dict) else ensure_dataframe(result)
    fig, ax = _figax(ax)
    x = obs.columns[0]
    y = "difference" if "difference" in obs else obs.select_dtypes(include=np.number).columns[-1]
    ax.plot(obs[x], obs[y])
    ax.axhline(0, linestyle="--")
    ax.set_title("Cluster results")
    return fig


def plot_gazepoint_cluster_permutation(result, **kwargs):
    return plot_gazepoint_cluster_results(result, **kwargs)


def plot_gazepoint_cluster_null_distribution(result, ax=None, **kwargs):
    vals = result.get("null_distribution", []) if isinstance(result, dict) else np.asarray(result)
    fig, ax = _figax(ax)
    ax.hist(vals, bins=30)
    ax.set_title("Cluster null distribution")
    return fig


def plot_gazepoint_multiverse_results(data, ax=None, **kwargs):
    df = ensure_dataframe(data)
    fig, ax = _figax(ax)
    num = df.select_dtypes(include=np.number).columns
    y = "mean_pupil" if "mean_pupil" in df else (num[-1] if len(num) else None)
    ax.plot(np.arange(len(df)), df[y], marker="o") if y else None
    ax.set_title("Multiverse results")
    return fig


def plot_gazepoint_face_quality(data, **kwargs):
    from .face import audit_gazepoint_face_quality

    q = audit_gazepoint_face_quality(data)
    fig, ax = _figax(None)
    ax.bar(
        ["Valid", "Below threshold"],
        [
            float(q["n_valid"].iloc[0]),
            float(q["prop_below_threshold"].iloc[0]) * float(q["n"].iloc[0])
            if np.isfinite(q["prop_below_threshold"].iloc[0])
            else 0,
        ],
    )
    ax.set_title("Face quality")
    return fig
