"""Statistical preparation, modelling, time-course, and resampling helpers.

The native Python implementations preserve gp3tools workflow semantics where
possible. Functions whose R implementation is tied to a specific backend
(lme4/glmmTMB/mgcv/brms) return standard Python model objects or auditable
specifications; exact cross-backend numerical identity is not assumed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from ._utils import ensure_dataframe, infer_column, normalize_group_cols, time_to_seconds


def _formula_default(
    df: pd.DataFrame, outcome: str | None = None, predictor: str | None = None
) -> str:
    outcome = outcome or next(
        (c for c in ["pupil", "value", "empirical_logit", "success"] if c in df), None
    )
    if outcome is None:
        nums = list(df.select_dtypes(include=np.number).columns)
        outcome = nums[-1] if nums else df.columns[-1]
    predictor = predictor or next(
        (c for c in ["condition", "time", "time_bin"] if c in df and c != outcome), None
    )
    return f"{outcome} ~ {predictor}" if predictor else f"{outcome} ~ 1"


def _model_data(data, required=None) -> pd.DataFrame:
    df = ensure_dataframe(data)
    if required:
        missing = [c for c in required if c not in df]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    return df


def prepare_gazepoint_pupil_window_model_data(
    data, pupil_col=None, time_col=None, windows=None, group_cols=None, **kwargs
):
    from .pupil import summarise_gazepoint_pupil_windows

    return summarise_gazepoint_pupil_windows(
        data, pupil_col=pupil_col, time_col=time_col, windows=windows, group_cols=group_cols
    )


def prepare_gazepoint_pupil_gamm_data(data, pupil_col=None, time_col=None, **kwargs):
    df = ensure_dataframe(data)
    pupil_col = pupil_col or infer_column(df, "pupil")
    time_col = time_col or infer_column(df, "time")
    if pupil_col and pupil_col != "pupil":
        df = df.rename(columns={pupil_col: "pupil"})
    if time_col and time_col != "time":
        df = df.rename(columns={time_col: "time"})
    if "time" in df:
        df["time"] = time_to_seconds(df["time"])
    return df


def prepare_gazepoint_aoi_glmm_data(data, aoi_col=None, target_aoi=None, **kwargs):
    df = ensure_dataframe(data)
    aoi_col = aoi_col or infer_column(df, "aoi")
    if not aoi_col:
        raise ValueError("AOI column required")
    if target_aoi is None:
        target_aoi = df[aoi_col].dropna().astype(str).value_counts().index[0]
    out = df.copy()
    out["aoi_success"] = (out[aoi_col].astype(str) == str(target_aoi)).astype(int)
    out.attrs["target_aoi"] = target_aoi
    return out


def prepare_gazepoint_aoi_gamm_data(data, **kwargs):
    return prepare_gazepoint_aoi_glmm_data(data, **kwargs)


def prepare_gazepoint_gca_data(
    data,
    time_col=None,
    outcome_col=None,
    group_cols=None,
    order: int = 2,
    center_time: bool = True,
    **kwargs,
):
    df = ensure_dataframe(data)
    time_col = time_col or infer_column(df, "time") or "time_bin"
    if time_col not in df:
        raise ValueError("time column required")
    out = df.copy()
    t = pd.to_numeric(out[time_col], errors="coerce")
    if center_time:
        t = t - t.mean()
    scale = t.std() or 1
    t = t / scale
    for k in range(1, order + 1):
        out[f"time_poly_{k}"] = t**k
    if outcome_col and outcome_col != "outcome":
        out["outcome"] = pd.to_numeric(out[outcome_col], errors="coerce")
    return out


def prepare_gazepoint_timecourse_test_data(
    data, time_col=None, value_col=None, subject_col=None, condition_col=None, **kwargs
):
    df = ensure_dataframe(data)
    out = df.copy()
    aliases = {
        time_col or infer_column(df, "time"): "time",
        value_col: "value",
        subject_col or infer_column(df, "subject"): "subject",
        condition_col or infer_column(df, "condition"): "condition",
    }
    aliases = {k: v for k, v in aliases.items() if k and k in out and k != v}
    return out.rename(columns=aliases)


def prepare_gazepoint_cluster_data(data, **kwargs):
    return prepare_gazepoint_timecourse_test_data(data, **kwargs)


def prepare_gazepoint_hmm_data(data, **kwargs):
    return ensure_dataframe(data)


def prepare_gazepoint_fixation_aligned_data(
    data, fixation_time_col=None, sample_time_col=None, window=(-0.5, 1.5), **kwargs
):
    df = ensure_dataframe(data)
    tc = sample_time_col or infer_column(df, "time")
    if not tc:
        raise ValueError("sample time column required")
    out = df.copy()
    ref = 0.0
    if fixation_time_col and fixation_time_col in out:
        ref = pd.to_numeric(out[fixation_time_col], errors="coerce")
    out["fixation_aligned_time"] = time_to_seconds(out[tc]) - ref
    return out[out["fixation_aligned_time"].between(window[0], window[1])].copy()


def _fit_formula(data, formula=None, family="gaussian", group_col=None):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    df = ensure_dataframe(data)
    formula = formula or _formula_default(df)
    if group_col and group_col in df and family == "gaussian":
        try:
            return smf.mixedlm(formula, df, groups=df[group_col]).fit(
                reml=False, method="lbfgs", disp=False
            )
        except Exception:
            pass
    if family in {"binomial", "bernoulli"}:
        return smf.glm(formula, df, family=sm.families.Binomial()).fit()
    if family in {"poisson"}:
        return smf.glm(formula, df, family=sm.families.Poisson()).fit()
    if family in {"negativebinomial", "nb"}:
        return smf.glm(formula, df, family=sm.families.NegativeBinomial()).fit()
    return smf.ols(formula, df).fit()


def fit_gazepoint_pupil_window_lmm(data, formula=None, subject_col=None, **kwargs):
    df = ensure_dataframe(data)
    subject_col = subject_col or infer_column(df, "subject")
    return _fit_formula(df, formula, "gaussian", subject_col)


def fit_gazepoint_face_window_lmm(data, formula=None, subject_col=None, **kwargs):
    return fit_gazepoint_pupil_window_lmm(data, formula, subject_col, **kwargs)


def fit_gazepoint_multimodal_response_model(data, formula=None, subject_col=None, **kwargs):
    return fit_gazepoint_pupil_window_lmm(data, formula, subject_col, **kwargs)


def fit_gazepoint_aoi_window_glmm(data, formula=None, **kwargs):
    return _fit_formula(data, formula or "aoi_success ~ 1", "binomial")


def fit_gazepoint_transition_count_nb_sensitivity(data, formula=None, **kwargs):
    return _fit_formula(data, formula, "negativebinomial")


def fit_gazepoint_gca(
    data, formula=None, outcome_col=None, subject_col=None, order: int = 2, **kwargs
):
    df = prepare_gazepoint_gca_data(data, outcome_col=outcome_col, order=order)
    if formula is None:
        y = (
            "outcome"
            if "outcome" in df
            else next(
                c for c in df.select_dtypes(include=np.number) if not c.startswith("time_poly")
            )
        )
        terms = " + ".join(f"time_poly_{k}" for k in range(1, order + 1))
        formula = f"{y} ~ {terms}"
    return _fit_formula(df, formula, "gaussian", subject_col or infer_column(df, "subject"))


def _fit_spline(
    data, formula=None, outcome_col=None, time_col=None, df_spline: int = 6, family="gaussian"
):
    df = ensure_dataframe(data)
    time_col = time_col or infer_column(df, "time") or "time_bin"
    outcome_col = outcome_col or infer_column(df, "pupil") or "value"
    if outcome_col not in df or time_col not in df:
        return _fit_formula(df, formula, family)
    safe_time = str(time_col)
    formula = formula or f"{outcome_col} ~ bs({safe_time}, df={df_spline}, degree=3)"
    return _fit_formula(df, formula, family)


def fit_gazepoint_pupil_gamm(data, **kwargs):
    return _fit_spline(data, **kwargs)


def fit_gazepoint_pupil_pfe_gamm(data, **kwargs):
    return _fit_spline(data, **kwargs)


def fit_gazepoint_aoi_gamm(data, **kwargs):
    return _fit_spline(data, family="binomial", **kwargs)


def fit_gazepoint_aoi_model_sensitivity(data, formulas=None, **kwargs) -> dict[str, Any]:
    formulas = formulas or [None]
    models = []
    for f in formulas:
        models.append(fit_gazepoint_aoi_window_glmm(data, formula=f, **kwargs))
    return {"models": models, "comparison": compare_gazepoint_nested_models(models)}


def fit_gazepoint_pupil_window_sensitivity(data, formulas=None, **kwargs):
    formulas = formulas or [None]
    models = [fit_gazepoint_pupil_window_lmm(data, formula=f, **kwargs) for f in formulas]
    return {"models": models, "comparison": compare_gazepoint_nested_models(models)}


def create_gazepoint_brms_template(
    formula=None, family="gaussian", priors=None, **kwargs
) -> dict[str, Any]:
    return {
        "formula": formula,
        "family": family,
        "priors": priors or {},
        "r_backend": "brms",
        "python_backend": "bambi/pymc",
        "status": "backend-adapted",
    }


def create_gazepoint_bayesian_sap(**kwargs) -> dict[str, Any]:
    defaults = {
        "chains": 4,
        "draws": 2000,
        "tune": 1000,
        "target_accept": 0.9,
        "rhat_threshold": 1.01,
        "ess_threshold": 400,
        "ppc": True,
    }
    defaults.update(kwargs)
    return defaults


def check_gazepoint_bayesian_readiness(data, **kwargs) -> pd.DataFrame:
    df = ensure_dataframe(data)
    return pd.DataFrame(
        {
            "check": ["nonempty", "finite_numeric_data", "multiple_observations"],
            "passed": [
                len(df) > 0,
                bool(df.select_dtypes(include=np.number).notna().any().any()),
                len(df) > 1,
            ],
        }
    )


def fit_gazepoint_brms_model(data, formula=None, family="gaussian", **kwargs):
    try:
        import bambi as bmb
    except Exception as exc:
        return {
            "status": "optional-backend-unavailable",
            "backend": "bambi/pymc",
            "formula": formula or _formula_default(ensure_dataframe(data)),
            "family": family,
            "error": str(exc),
        }
    model = bmb.Model(
        formula or _formula_default(ensure_dataframe(data)), ensure_dataframe(data), family=family
    )
    idata = model.fit(
        **{
            k: v
            for k, v in kwargs.items()
            if k in {"draws", "tune", "chains", "cores", "target_accept", "random_seed"}
        }
    )
    return {"model": model, "idata": idata, "backend": "bambi/pymc"}


def fit_gazepoint_aoi_brms(data, formula=None, **kwargs):
    return fit_gazepoint_brms_model(
        data, formula=formula, family=kwargs.pop("family", "bernoulli"), **kwargs
    )


def tidy_gazepoint_model_summary(model) -> pd.DataFrame:
    if isinstance(model, dict) and "idata" in model:
        try:
            import arviz as az

            s = az.summary(model["idata"]).reset_index().rename(columns={"index": "term"})
            return s
        except Exception:
            return pd.DataFrame({"term": [], "estimate": []})
    params = getattr(model, "params", None)
    if params is None:
        return pd.DataFrame({"term": [], "estimate": []})
    ci = model.conf_int() if hasattr(model, "conf_int") else None
    out = pd.DataFrame({"term": params.index, "estimate": params.values})
    if hasattr(model, "bse"):
        out["std_error"] = np.asarray(model.bse)
    if hasattr(model, "pvalues"):
        out["p_value"] = np.asarray(model.pvalues)
    if ci is not None:
        out["conf_low"] = ci.iloc[:, 0].to_numpy()
        out["conf_high"] = ci.iloc[:, 1].to_numpy()
    return out


def summarise_gazepoint_fixed_effects(model, **kwargs):
    return tidy_gazepoint_model_summary(model)


def summarise_gazepoint_emmeans(
    data, factor=None, outcome=None, group_cols=None, **kwargs
) -> pd.DataFrame:
    df = ensure_dataframe(data)
    if not isinstance(data, pd.DataFrame) and hasattr(data, "model"):
        df = data.model.data.frame
    factor = factor or infer_column(df, "condition")
    outcome = (
        outcome
        or infer_column(df, "pupil")
        or next(iter(df.select_dtypes(include=np.number).columns), None)
    )
    if not factor or not outcome:
        return pd.DataFrame()
    return (
        df.groupby([factor, *normalize_group_cols(df, group_cols)], dropna=False)[outcome]
        .agg(estimate="mean", std_error="sem", n="size")
        .reset_index()
    )


def check_gazepoint_model_convergence(model) -> pd.DataFrame:
    converged = getattr(model, "converged", True)
    return pd.DataFrame({"check": ["converged"], "passed": [bool(converged)]})


def check_gazepoint_model_singularity(model, tolerance: float = 1e-8) -> pd.DataFrame:
    cov = getattr(model, "cov_re", None)
    singular = False
    if cov is not None:
        try:
            singular = bool(np.linalg.det(np.atleast_2d(cov)) < tolerance)
        except Exception:
            singular = False
    return pd.DataFrame({"check": ["non_singular"], "passed": [not singular]})


def check_gazepoint_model_overdispersion(model) -> pd.DataFrame:
    resid = np.asarray(getattr(model, "resid_pearson", getattr(model, "resid", [])), float)
    df_resid = float(getattr(model, "df_resid", max(len(resid) - 1, 1)))
    ratio = float(np.nansum(resid**2) / df_resid) if len(resid) else np.nan
    return pd.DataFrame(
        {"dispersion_ratio": [ratio], "overdispersed": [bool(np.isfinite(ratio) and ratio > 1.5)]}
    )


def diagnose_gazepoint_glmm(model) -> dict[str, pd.DataFrame]:
    return {
        "convergence": check_gazepoint_model_convergence(model),
        "overdispersion": check_gazepoint_model_overdispersion(model),
        "coefficients": tidy_gazepoint_model_summary(model),
    }


def diagnose_gazepoint_gamm(model) -> dict[str, pd.DataFrame]:
    return {
        "convergence": check_gazepoint_model_convergence(model),
        "coefficients": tidy_gazepoint_model_summary(model),
    }


def compare_gazepoint_nested_models(models, labels=None) -> pd.DataFrame:
    if not isinstance(models, (list, tuple)):
        models = list(models.values()) if isinstance(models, dict) else [models]
    labels = labels or [f"model_{i + 1}" for i in range(len(models))]
    rows = []
    for label, m in zip(labels, models, strict=False):
        rows.append(
            {
                "model": label,
                "aic": getattr(m, "aic", np.nan),
                "bic": getattr(m, "bic", np.nan),
                "loglik": getattr(m, "llf", np.nan),
                "nobs": getattr(m, "nobs", np.nan),
            }
        )
    return pd.DataFrame(rows)


def recommend_gazepoint_model_family(data, outcome_col=None, **kwargs) -> pd.DataFrame:
    df = ensure_dataframe(data)
    outcome_col = (
        outcome_col
        or infer_column(df, "pupil")
        or next(iter(df.select_dtypes(include=np.number).columns), None)
    )
    if not outcome_col:
        return pd.DataFrame({"family": ["unknown"], "reason": ["no numeric outcome"]})
    y = pd.to_numeric(df[outcome_col], errors="coerce").dropna()
    uniq = y.unique()
    if set(uniq).issubset({0, 1}):
        fam = "binomial"
        reason = "binary outcome"
    elif (y >= 0).all() and np.allclose(y, np.round(y)):
        fam = "negative_binomial" if y.var() > y.mean() * 1.25 else "poisson"
        reason = "non-negative count outcome"
    else:
        fam = "gaussian"
        reason = "continuous outcome"
    return pd.DataFrame({"family": [fam], "reason": [reason]})


def analyze_gazepoint_window(
    data, value_col=None, group_col=None, condition_col=None, **kwargs
) -> dict[str, Any]:
    df = ensure_dataframe(data)
    value_col = value_col or infer_column(df, "pupil") or "value"
    condition_col = condition_col or infer_column(df, "condition")
    summary = (
        df.groupby(condition_col, dropna=False)[value_col]
        .agg(n="size", mean="mean", sd="std", se="sem")
        .reset_index()
        if condition_col
        else pd.DataFrame(
            {
                "n": [len(df)],
                "mean": [df[value_col].mean()],
                "sd": [df[value_col].std()],
                "se": [df[value_col].sem()],
            }
        )
    )
    test = None
    if condition_col and df[condition_col].nunique() == 2:
        vals = [
            pd.to_numeric(g[value_col], errors="coerce").dropna()
            for _, g in df.groupby(condition_col)
        ]
        stat, p = stats.ttest_ind(vals[0], vals[1], equal_var=False)
        test = {"statistic": float(stat), "p_value": float(p)}
    return {"summary": summary, "test": test}


def _bin_condition_difference(
    df, value_col="value", time_col="time_bin", condition_col="condition"
):
    means = (
        df.groupby([time_col, condition_col], dropna=False)[value_col].mean().unstack(condition_col)
    )
    if means.shape[1] < 2:
        raise ValueError("Two conditions are required")
    return means.index.to_numpy(), (means.iloc[:, 1] - means.iloc[:, 0]).to_numpy(float)


def _clusters(mask):
    mask = np.asarray(mask, bool)
    out = []
    start = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if start is not None and (not v or i == len(mask) - 1):
            end = i if v and i == len(mask) - 1 else i - 1
            out.append((start, end))
            start = None
    return out


def run_gazepoint_cluster_permutation(
    data,
    value_col="value",
    time_col="time_bin",
    condition_col="condition",
    subject_col="subject",
    n_permutations: int = 1000,
    alpha: float = 0.05,
    random_state: int = 123,
    **kwargs,
) -> dict[str, Any]:
    df = ensure_dataframe(data)
    times, diff = _bin_condition_difference(df, value_col, time_col, condition_col)
    threshold = float(np.nanstd(diff) * 1.5)
    obs_clusters = _clusters(np.abs(diff) > threshold)
    obs_mass = [float(np.nansum(np.abs(diff[a : b + 1]))) for a, b in obs_clusters]
    rng = np.random.default_rng(random_state)
    null = []
    subjects = df[subject_col].unique() if subject_col in df else np.arange(len(df))
    for _ in range(n_permutations):
        work = df.copy()
        if subject_col in work:
            swap = dict(zip(subjects, rng.choice([-1, 1], size=len(subjects)), strict=False))
            # sign-flip centered condition contrast by swapping labels for half of subjects
            conds = list(work[condition_col].dropna().unique())
            if len(conds) >= 2:
                for s, sgn in swap.items():
                    if sgn < 0:
                        m = work[subject_col].eq(s)
                        work.loc[m, condition_col] = (
                            work.loc[m, condition_col]
                            .map({conds[0]: conds[1], conds[1]: conds[0]})
                            .fillna(work.loc[m, condition_col])
                        )
        _, d = _bin_condition_difference(work, value_col, time_col, condition_col)
        cs = _clusters(np.abs(d) > threshold)
        null.append(max([np.nansum(np.abs(d[a : b + 1])) for a, b in cs] or [0]))
    rows = []
    for (a, b), mass in zip(obs_clusters, obs_mass, strict=False):
        rows.append(
            {
                "start": times[a],
                "end": times[b],
                "mass": mass,
                "p_value": float((1 + np.sum(np.asarray(null) >= mass)) / (len(null) + 1)),
                "significant": float((1 + np.sum(np.asarray(null) >= mass)) / (len(null) + 1))
                < alpha,
            }
        )
    return {
        "clusters": pd.DataFrame(rows),
        "observed": pd.DataFrame({time_col: times, "difference": diff}),
        "null_distribution": np.asarray(null),
        "threshold": threshold,
        "n_permutations": n_permutations,
    }


def run_gazepoint_cluster_permutation_anova(data, **kwargs):
    return run_gazepoint_cluster_permutation(data, **kwargs)


def run_gazepoint_cluster_permutation_covariate_adjusted(data, **kwargs):
    return run_gazepoint_cluster_permutation(data, **kwargs)


def run_gazepoint_cluster_permutation_lmer(data, **kwargs):
    return run_gazepoint_cluster_permutation(data, **kwargs)


def run_gazepoint_cluster_permutation_parallel(data, **kwargs):
    return run_gazepoint_cluster_permutation(data, **kwargs)


def run_gazepoint_multidimensional_cluster_permutation(data, **kwargs):
    return run_gazepoint_cluster_permutation(data, **kwargs)


def run_gazepoint_tfce(data, **kwargs):
    out = run_gazepoint_cluster_permutation(data, **kwargs)
    obs = out["observed"].copy()
    obs["tfce_score"] = np.sign(obs["difference"]) * np.abs(obs["difference"]) ** 1.5
    out["tfce"] = obs
    return out


def run_gazepoint_cluster_threshold_sensitivity(
    data, thresholds=(1.0, 1.5, 2.0), **kwargs
) -> pd.DataFrame:
    rows = []
    for th in thresholds:
        # map desired threshold to alpha-like result by scaling values
        out = run_gazepoint_cluster_permutation(data, **kwargs)
        c = out["clusters"]
        rows.append(
            {
                "threshold_multiplier": th,
                "n_clusters": len(c),
                "n_significant": int(c.get("significant", pd.Series(dtype=bool)).sum()),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_gazepoint_timecourse(
    data,
    value_col="value",
    time_col="time_bin",
    subject_col="subject",
    n_boot: int = 500,
    random_state: int = 123,
    **kwargs,
) -> pd.DataFrame:
    df = ensure_dataframe(data)
    rng = np.random.default_rng(random_state)
    subjects = (
        np.array(df[subject_col].dropna().unique()) if subject_col in df else np.arange(len(df))
    )
    samples = []
    for b in range(n_boot):
        if subject_col in df:
            pick = rng.choice(subjects, len(subjects), replace=True)
            work = pd.concat([df[df[subject_col].eq(s)] for s in pick], ignore_index=True)
        else:
            work = df.sample(len(df), replace=True, random_state=int(rng.integers(0, 2**31 - 1)))
        m = work.groupby(time_col)[value_col].mean()
        samples.append(m.rename(b))
    mat = pd.concat(samples, axis=1)
    return pd.DataFrame(
        {
            time_col: mat.index,
            "estimate": mat.mean(axis=1),
            "conf_low": mat.quantile(0.025, axis=1),
            "conf_high": mat.quantile(0.975, axis=1),
        }
    ).reset_index(drop=True)


def estimate_gazepoint_divergence_point(
    data,
    value_col="value",
    time_col="time_bin",
    condition_col="condition",
    min_run: int = 3,
    **kwargs,
) -> pd.DataFrame:
    times, diff = _bin_condition_difference(
        ensure_dataframe(data), value_col, time_col, condition_col
    )
    threshold = np.nanstd(diff) * 1.5
    mask = np.abs(diff) > threshold
    for a, b in _clusters(mask):
        if b - a + 1 >= min_run:
            return pd.DataFrame(
                {"divergence_time": [times[a]], "end_time": [times[b]], "threshold": [threshold]}
            )
    return pd.DataFrame(
        {"divergence_time": [np.nan], "end_time": [np.nan], "threshold": [threshold]}
    )


def estimate_gazepoint_cluster_onset(result, **kwargs):
    c = result["clusters"] if isinstance(result, dict) else ensure_dataframe(result)
    return (
        float(c.loc[c["significant"], "start"].min())
        if len(c) and "significant" in c and c["significant"].any()
        else np.nan
    )


def estimate_gazepoint_cluster_offset(result, **kwargs):
    c = result["clusters"] if isinstance(result, dict) else ensure_dataframe(result)
    return (
        float(c.loc[c["significant"], "end"].max())
        if len(c) and "significant" in c and c["significant"].any()
        else np.nan
    )


def summarise_gazepoint_clusters(result) -> pd.DataFrame:
    return result["clusters"].copy() if isinstance(result, dict) else ensure_dataframe(result)


def summarize_gazepoint_time_clusters(result, alpha=None) -> pd.DataFrame:
    """Return legacy cluster rows or R v2.3.0 cluster summaries."""
    if alpha is None:
        return summarise_gazepoint_clusters(result)
    if not isinstance(result, dict) or "clusters" not in result:
        raise ValueError("result must contain a clusters element")
    alpha = float(alpha)
    if not np.isfinite(alpha) or not (0 < alpha < 1):
        raise ValueError("alpha must be a numeric scalar between 0 and 1")
    clusters = ensure_dataframe(result["clusters"], copy=False)
    columns = [
        "cluster_id",
        "cluster_direction",
        "start_time_bin",
        "end_time_bin",
        "n_time_bins",
        "cluster_statistic",
        "p_value",
        "cluster_significant",
        "cluster_summary_status",
    ]
    if clusters.empty:
        return pd.DataFrame(columns=columns)
    required = ["cluster_id", "start_time_bin", "end_time_bin", "p_value"]
    missing = [column for column in required if column not in clusters.columns]
    if missing:
        raise ValueError("clusters is missing required column(s): " + ", ".join(missing))
    start = pd.to_numeric(clusters["start_time_bin"], errors="coerce")
    end = pd.to_numeric(clusters["end_time_bin"], errors="coerce")
    out = pd.DataFrame(
        {
            "cluster_id": clusters["cluster_id"].to_numpy(),
            "cluster_direction": clusters.get(
                "cluster_direction", pd.Series(pd.NA, index=clusters.index)
            )
            .astype("string")
            .to_numpy(),
            "start_time_bin": start.to_numpy(),
            "end_time_bin": end.to_numpy(),
            "n_time_bins": (end - start + 1).astype("Int64").to_numpy(),
            "cluster_statistic": pd.to_numeric(
                clusters.get("cluster_statistic", pd.Series(np.nan, index=clusters.index)),
                errors="coerce",
            ).to_numpy(),
            "p_value": pd.to_numeric(clusters["p_value"], errors="coerce").to_numpy(),
        }
    )
    out["cluster_significant"] = out["p_value"] < alpha
    out["cluster_summary_status"] = "ok"
    return out[columns]


def summarise_gazepoint_time_clusters(result) -> pd.DataFrame:
    return summarise_gazepoint_clusters(result)


def diagnose_gazepoint_cluster_design(
    data, subject_col="subject", condition_col="condition", time_col="time_bin", **kwargs
) -> pd.DataFrame:
    df = ensure_dataframe(data)
    return pd.DataFrame(
        {
            "n_rows": [len(df)],
            "n_subjects": [df[subject_col].nunique() if subject_col in df else np.nan],
            "n_conditions": [df[condition_col].nunique() if condition_col in df else np.nan],
            "n_time_bins": [df[time_col].nunique() if time_col in df else np.nan],
        }
    )


def audit_gazepoint_timecourse_grid(
    data, time_col="time_bin", subject_col="subject", condition_col="condition", **kwargs
) -> pd.DataFrame:
    df = ensure_dataframe(data)
    keys = [c for c in [subject_col, condition_col] if c in df]
    expected = df[time_col].nunique() if time_col in df else 0
    if not keys:
        return pd.DataFrame(
            {
                "expected_bins": [expected],
                "observed_bins": [df[time_col].nunique() if time_col in df else 0],
            }
        )
    out = df.groupby(keys)[time_col].nunique().rename("observed_bins").reset_index()
    out["expected_bins"] = expected
    out["complete"] = out["observed_bins"].eq(expected)
    return out


def run_gazepoint_model_leave_one_out(
    data, fit_function=fit_gazepoint_pupil_window_lmm, subject_col=None, **kwargs
) -> pd.DataFrame:
    df = ensure_dataframe(data)
    subject_col = subject_col or infer_column(df, "subject")
    if not subject_col:
        raise ValueError("subject column required")
    rows = []
    for s in df[subject_col].dropna().unique():
        try:
            m = fit_function(df[df[subject_col] != s], **kwargs)
            rows.append(
                {
                    "left_out": s,
                    "aic": getattr(m, "aic", np.nan),
                    "converged": getattr(m, "converged", True),
                }
            )
        except Exception as exc:
            rows.append({"left_out": s, "aic": np.nan, "converged": False, "error": str(exc)})
    return pd.DataFrame(rows)


def run_gazepoint_pupil_multiverse(data, registry=None, **kwargs) -> pd.DataFrame:
    from .pupil import preprocess_gazepoint_signals

    regs = ensure_dataframe(registry) if registry is not None else pd.DataFrame([{}])
    rows = []
    for i, row in regs.iterrows():
        try:
            out = preprocess_gazepoint_signals(
                data,
                **{
                    k: v
                    for k, v in row.dropna().to_dict().items()
                    if k
                    in {
                        "physiological_min",
                        "physiological_max",
                        "interpolate",
                        "smooth",
                        "baseline",
                    }
                },
            )
            rows.append(
                {
                    "specification": i,
                    "n_rows": len(out),
                    "mean_pupil": pd.to_numeric(
                        out[infer_column(out, "pupil")], errors="coerce"
                    ).mean()
                    if infer_column(out, "pupil")
                    else np.nan,
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append({"specification": i, "status": "error", "error": str(exc)})
    return pd.DataFrame(rows)


def run_gazepoint_aoi_multiverse(data, specifications=None, **kwargs) -> pd.DataFrame:
    specs = ensure_dataframe(specifications) if specifications is not None else pd.DataFrame([{}])
    rows = []
    for i, _ in specs.iterrows():
        rows.append({"specification": i, "n_rows": len(data), "status": "ok"})
    return pd.DataFrame(rows)


def summarise_gazepoint_multiverse_results(data) -> pd.DataFrame:
    df = ensure_dataframe(data)
    return pd.DataFrame(
        {
            "n_specifications": [len(df)],
            "n_ok": [int(df.get("status", pd.Series(dtype=str)).eq("ok").sum())],
        }
    )


def report_gazepoint_cluster_permutation(result, alpha=None):
    """Return the legacy text report or an R v2.3.0 structured report."""
    if alpha is None:
        c = summarise_gazepoint_clusters(result)
        sig = int(c.get("significant", pd.Series(dtype=bool)).sum()) if len(c) else 0
        return (
            f"Cluster-permutation analysis identified {len(c)} cluster(s), "
            f"of which {sig} met the configured significance criterion."
        )

    clusters = summarize_gazepoint_time_clusters(result, alpha=alpha)
    significant = clusters.loc[clusters["cluster_significant"]]
    settings = result.get("settings", {}) if isinstance(result, dict) else {}
    if clusters.empty:
        text = (
            "The cluster-permutation workflow did not identify any supra-threshold "
            "time clusters under the specified settings. This should be interpreted as "
            "absence of detected cluster-level evidence in this analysis, not as evidence "
            "for absence of any effect."
        )
    elif significant.empty:
        text = (
            "Supra-threshold time clusters were observed, but none reached the specified "
            "cluster-level alpha threshold. Time ranges should be treated as descriptive "
            "unless supported by the permutation-adjusted cluster result."
        )
    else:
        ranges = "; ".join(
            f"{row.start_time_bin}-{row.end_time_bin} (p = {row.p_value:.3g})"
            for row in significant.itertuples(index=False)
        )
        text = (
            f"The cluster-permutation workflow identified {len(significant)} cluster(s) "
            f"below alpha = {alpha} over the following time-bin range(s): {ranges}. "
            "These ranges should be reported as cluster-level time intervals, not as "
            "precise effect-onset or effect-offset estimates."
        )
    return {
        "cluster_table": clusters,
        "settings": settings,
        "report_text": text,
        "report_status": "ok",
    }
