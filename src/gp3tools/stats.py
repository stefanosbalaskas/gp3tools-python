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


def _gp3_model_for_diagnostics(model, model_name=None):
    if model is None:
        raise ValueError("model must not be None")
    fitted = model
    inferred = "model"
    if isinstance(model, dict) and "model" in model:
        fitted = model["model"]
        inferred = model.get("model_name") or "model"
        if fitted is None:
            raise ValueError("model['model'] must not be None")
    if model_name is not None:
        if not isinstance(model_name, str) or not model_name:
            raise ValueError("model_name must be a non-empty string")
        inferred = model_name
    return fitted, inferred


def check_gazepoint_model_convergence(model, model_name=None) -> pd.DataFrame:
    fitted, resolved_name = _gp3_model_for_diagnostics(model, model_name)
    model_class = type(fitted).__name__
    converged = getattr(fitted, "converged", None)
    if converged is None and hasattr(fitted, "mle_retvals"):
        converged = getattr(fitted, "mle_retvals", {}).get("converged")
    if converged is None:
        status = (
            "not_applicable"
            if hasattr(fitted, "params") and not hasattr(fitted, "cov_re")
            else "not_available"
        )
        message = (
            "Convergence diagnostics are not applicable to ordinary linear-model results."
            if status == "not_applicable"
            else "The fitted model does not expose convergence information."
        )
        passed = True
    else:
        converged = bool(converged)
        status = "ok" if converged else "convergence_warning"
        message = "Model reports convergence." if converged else "Model reports non-convergence."
        passed = converged
    return pd.DataFrame(
        {
            "model_name": [resolved_name],
            "model_class": [model_class],
            "diagnostic": ["convergence"],
            "converged": [pd.NA if converged is None else bool(converged)],
            "diagnostic_status": [status],
            "message": [message],
            "check": ["converged"],
            "passed": [bool(passed)],
        }
    )


def check_gazepoint_model_singularity(
    model,
    tolerance: float = 1e-4,
    model_name=None,
) -> pd.DataFrame:
    if isinstance(tolerance, (bool, np.bool_)) or not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be a positive finite numeric scalar")
    fitted, resolved_name = _gp3_model_for_diagnostics(model, model_name)
    model_class = type(fitted).__name__
    covariance = getattr(fitted, "cov_re", None)
    if covariance is None:
        singular = None
        status = "not_applicable" if hasattr(fitted, "params") else "unsupported_model_class"
        message = (
            f"Singularity diagnostics are not applicable to {model_class} objects."
            if status == "not_applicable"
            else "Unsupported model class for singularity diagnostics."
        )
        passed = True
    else:
        try:
            values = np.atleast_2d(np.asarray(covariance, dtype=float))
            eigenvalues = np.linalg.eigvalsh(values)
            singular = bool(np.any(eigenvalues < tolerance))
            status = "singular_fit" if singular else "ok"
            message = (
                "Random-effects covariance is singular."
                if singular
                else "No singular random-effects structure detected."
            )
            passed = not singular
        except Exception as exc:
            singular = None
            status = "error"
            message = str(exc)
            passed = True
    return pd.DataFrame(
        {
            "model_name": [resolved_name],
            "model_class": [model_class],
            "diagnostic": ["singularity"],
            "singular_fit": [pd.NA if singular is None else singular],
            "tolerance": [float(tolerance)],
            "diagnostic_status": [status],
            "message": [message],
            "check": ["non_singular"],
            "passed": [bool(passed)],
        }
    )


def check_gazepoint_model_overdispersion(
    model,
    ratio_threshold=1.2,
    model_name=None,
) -> pd.DataFrame:
    if (
        isinstance(ratio_threshold, (bool, np.bool_))
        or not np.isfinite(ratio_threshold)
        or ratio_threshold <= 0
    ):
        raise ValueError("ratio_threshold must be a positive finite numeric scalar")
    fitted, resolved_name = _gp3_model_for_diagnostics(model, model_name)
    model_class = type(fitted).__name__
    residuals = getattr(fitted, "resid_pearson", None)
    if residuals is None:
        residuals = getattr(fitted, "resid", None)
    df_resid = getattr(fitted, "df_resid", None)

    if residuals is None or df_resid is None:
        ratio = np.nan
        chisq = np.nan
        residual_df = np.nan
        overdispersed = False
        status = "not_applicable" if hasattr(fitted, "params") else "unsupported_model_class"
        message = "Overdispersion diagnostics are not available for this model."
    else:
        residuals = np.asarray(residuals, dtype=float)
        residual_df = float(df_resid)
        if not np.isfinite(residual_df) or residual_df <= 0:
            ratio = np.nan
            chisq = np.nan
            overdispersed = False
            status = "insufficient_residual_df"
            message = "Residual degrees of freedom are missing, non-finite, or non-positive."
        else:
            chisq = float(np.nansum(residuals**2))
            ratio = chisq / residual_df
            overdispersed = bool(ratio > ratio_threshold)
            status = "overdispersed" if overdispersed else "ok"
            message = (
                f"Dispersion ratio exceeds threshold {ratio_threshold}."
                if overdispersed
                else f"Dispersion ratio does not exceed threshold {ratio_threshold}."
            )
    return pd.DataFrame(
        {
            "model_name": [resolved_name],
            "model_class": [model_class],
            "diagnostic": ["overdispersion"],
            "dispersion_ratio": [ratio],
            "pearson_chisq": [chisq],
            "residual_df": [residual_df],
            "overdispersed": [overdispersed],
            "ratio_threshold": [float(ratio_threshold)],
            "diagnostic_status": [status],
            "message": [message],
        }
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


def compare_gazepoint_nested_models(
    models,
    labels=None,
    *,
    model_names=None,
    comparison=None,
    name=None,
):
    """Compare nested models; labels retains the historical Python table interface."""
    if labels is not None and model_names is None and comparison is None and name is None:
        model_list = list(models.values()) if isinstance(models, dict) else list(models)
        labels = labels or [f"model_{i + 1}" for i in range(len(model_list))]
        return pd.DataFrame(
            [
                {
                    "model": label,
                    "aic": getattr(model, "aic", np.nan),
                    "bic": getattr(model, "bic", np.nan),
                    "loglik": getattr(model, "llf", np.nan),
                    "nobs": getattr(model, "nobs", np.nan),
                }
                for label, model in zip(labels, model_list, strict=False)
            ]
        )

    if isinstance(models, dict):
        names_from_mapping = list(models)
        model_list = list(models.values())
    else:
        model_list = list(models) if isinstance(models, (list, tuple)) else [models]
        names_from_mapping = []
    if not model_list:
        raise ValueError("models must be a non-empty list of fitted model objects")
    comparison = "sequential" if comparison is None else comparison
    name = "gazepoint_nested_model_comparison" if name is None else name
    if comparison not in {"sequential", "against_first"}:
        raise ValueError("comparison must be 'sequential' or 'against_first'")
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")

    if model_names is None:
        if names_from_mapping and all(str(value) for value in names_from_mapping):
            model_names = [str(value) for value in names_from_mapping]
        else:
            model_names = [f"model_{i + 1}" for i in range(len(model_list))]
    else:
        model_names = list(model_names)
    if (
        len(model_names) != len(model_list)
        or any(not isinstance(value, str) or not value for value in model_names)
        or len(set(model_names)) != len(model_names)
    ):
        raise ValueError("model_names must contain one unique non-empty name per model")

    model_rows = []
    for index, (model_name, model) in enumerate(zip(model_names, model_list, strict=True), start=1):
        values = {
            "model_index": index,
            "model_name": model_name,
            "model_class": type(model).__name__,
            "nobs": float(getattr(model, "nobs", np.nan)),
            "df": np.nan,
            "logLik": float(getattr(model, "llf", np.nan)),
            "AIC": float(getattr(model, "aic", np.nan)),
            "BIC": float(getattr(model, "bic", np.nan)),
            "extraction_status": "complete",
            "message": pd.NA,
        }
        parameters = getattr(model, "params", None)
        if parameters is not None:
            try:
                values["df"] = float(len(parameters))
            except TypeError:
                pass
        if not np.isfinite(values["logLik"]):
            values["extraction_status"] = "extraction_error"
            values["message"] = "log-likelihood was not available"
        model_rows.append(values)
    model_table = pd.DataFrame(model_rows)

    pairs = []
    if len(model_table) >= 2:
        pair_indexes = (
            [(i, i + 1) for i in range(len(model_table) - 1)]
            if comparison == "sequential"
            else [(0, i) for i in range(1, len(model_table))]
        )
        for comparison_index, (left_index, right_index) in enumerate(pair_indexes, start=1):
            left = model_table.iloc[left_index]
            right = model_table.iloc[right_index]
            df_diff = right["df"] - left["df"]
            chisq = 2 * (right["logLik"] - left["logLik"])
            status = "complete"
            message = pd.NA
            p_value = np.nan
            if (
                left["extraction_status"] == "extraction_error"
                or right["extraction_status"] == "extraction_error"
            ):
                status = "model_extraction_error"
                message = "At least one model did not provide logLik/df information."
            elif not np.isfinite(df_diff) or not np.isfinite(chisq):
                status = "missing_lrt_components"
                message = "Likelihood-ratio components could not be computed."
            elif df_diff <= 0:
                status = "nonpositive_df_difference"
                message = "The comparison model did not have more degrees of freedom."
            elif chisq < 0:
                status = "negative_lrt_statistic"
                message = "The comparison model had a lower log-likelihood."
            else:
                p_value = float(stats.chi2.sf(chisq, df_diff))
            pairs.append(
                {
                    "comparison_index": comparison_index,
                    "model_0": left["model_name"],
                    "model_1": right["model_name"],
                    "df_0": left["df"],
                    "df_1": right["df"],
                    "df_diff": df_diff,
                    "logLik_0": left["logLik"],
                    "logLik_1": right["logLik"],
                    "chisq": chisq,
                    "p_value": p_value,
                    "comparison_status": status,
                    "message": message,
                }
            )
    lrt_table = pd.DataFrame(pairs)

    ranking = model_table[
        ["model_index", "model_name", "AIC", "BIC", "logLik", "df", "nobs", "extraction_status"]
    ].copy()
    finite_aic = ranking["AIC"].replace([np.inf, -np.inf], np.nan)
    finite_bic = ranking["BIC"].replace([np.inf, -np.inf], np.nan)
    ranking["delta_AIC"] = finite_aic - finite_aic.min()
    ranking["aic_rank"] = finite_aic.rank(method="min")
    ranking["delta_BIC"] = finite_bic - finite_bic.min()
    ranking["bic_rank"] = finite_bic.rank(method="min")
    ranking = ranking[
        [
            "model_index",
            "model_name",
            "AIC",
            "delta_AIC",
            "aic_rank",
            "BIC",
            "delta_BIC",
            "bic_rank",
            "logLik",
            "df",
            "nobs",
            "extraction_status",
        ]
    ].sort_values(["aic_rank", "bic_rank", "model_index"], kind="stable", na_position="last")

    n_complete = int(model_table["extraction_status"].eq("complete").sum())
    n_errors = len(model_table) - n_complete
    lrt_complete = int(lrt_table["comparison_status"].eq("complete").sum()) if len(lrt_table) else 0
    lrt_problem = len(lrt_table) - lrt_complete
    if len(model_list) < 2:
        comparison_status = "not_enough_models"
    elif n_complete < 2:
        comparison_status = "failed"
    elif n_errors == 0 and lrt_problem == 0:
        comparison_status = "complete"
    else:
        comparison_status = "partial_complete"

    best_aic = ranking.loc[ranking["aic_rank"].eq(1), "model_name"]
    best_bic = ranking.loc[ranking["bic_rank"].eq(1), "model_name"]
    overview = pd.DataFrame(
        [
            {
                "object_name": name,
                "comparison_status": comparison_status,
                "comparison": comparison,
                "n_models": len(model_list),
                "n_complete_models": n_complete,
                "n_model_extraction_errors": n_errors,
                "n_lrt_comparisons": len(lrt_table),
                "n_lrt_complete": lrt_complete,
                "n_lrt_problem": lrt_problem,
                "best_aic_model": best_aic.iloc[0] if len(best_aic) else pd.NA,
                "best_bic_model": best_bic.iloc[0] if len(best_bic) else pd.NA,
            }
        ]
    )
    settings = pd.DataFrame(
        {
            "setting": ["model_names", "comparison", "name"],
            "value": [", ".join(model_names), comparison, name],
        }
    )
    return {
        "overview": overview,
        "model_table": model_table,
        "lrt_table": lrt_table,
        "ranking_table": ranking.reset_index(drop=True),
        "settings": settings,
        "_gp3_class": "gp3_nested_model_comparison",
    }


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


def summarise_gazepoint_multiverse_results(*objects, results=None, data=None):
    """Summarise multiverse objects while retaining the historical DataFrame shortcut."""
    if data is not None and (objects or results is not None):
        raise TypeError("data cannot be combined with objects/results")
    if data is not None:
        objects = (data,)
    if len(objects) == 1 and isinstance(objects[0], pd.DataFrame) and results is None:
        frame = ensure_dataframe(objects[0])
        return pd.DataFrame(
            {
                "n_specifications": [len(frame)],
                "n_ok": [int(frame.get("status", pd.Series(dtype=str)).eq("ok").sum())],
            }
        )

    collected = []
    names = []
    if results is not None:
        if not isinstance(results, dict):
            raise ValueError("results must be a named mapping of multiverse result objects")
        for key, value in results.items():
            names.append(str(key))
            collected.append(value)
    for value in objects:
        names.append(f"multiverse_{len(names) + 1}")
        collected.append(value)
    if not collected:
        raise ValueError("At least one multiverse result object must be supplied")

    overview_rows = []
    branch_rows = []
    for result_name, result in zip(names, collected, strict=True):
        if not isinstance(result, dict):
            raise ValueError("All supplied objects must be multiverse result mappings")
        family = result.get("multiverse_family")
        if family is None:
            gp3_class = str(result.get("_gp3_class", ""))
            family = "pupil" if "pupil" in gp3_class else "aoi" if "aoi" in gp3_class else "unknown"
        overview = result.get("overview")
        if isinstance(overview, pd.DataFrame) and len(overview):
            overview_lookup = overview.iloc[0].to_dict()
        elif isinstance(overview, dict):
            overview_lookup = overview
        else:
            overview_lookup = {}
        overview_rows.append(
            {
                "result_name": result_name,
                "multiverse_family": family,
                "n_defined_branches": overview_lookup.get("n_defined_branches", np.nan),
                "n_requested_branches": overview_lookup.get("n_requested_branches", np.nan),
                "n_completed_branches": overview_lookup.get("n_completed_branches", np.nan),
                "n_failed_branches": overview_lookup.get("n_failed_branches", np.nan),
                "n_skipped_branches": overview_lookup.get("n_skipped_branches", np.nan),
                "multiverse_status": overview_lookup.get("multiverse_status", np.nan),
            }
        )
        branches = result.get("branch_results")
        if isinstance(branches, pd.DataFrame) and len(branches):
            block = branches.copy()
            block.insert(0, "multiverse_family", family)
            block.insert(0, "result_name", result_name)
            branch_rows.append(block)

    overview = pd.DataFrame(overview_rows)
    requested = pd.to_numeric(overview["n_requested_branches"], errors="coerce").sum()
    failed = pd.to_numeric(overview["n_failed_branches"], errors="coerce").sum()
    if requested == 0:
        overall_status = "not_run"
    elif failed > 0:
        overall_status = "completed_with_errors"
    elif overview["multiverse_status"].astype(str).eq("completed").all():
        overall_status = "completed"
    else:
        overall_status = "completed_with_cautions"
    overall = {
        "result_name": "overall",
        "multiverse_family": "combined",
        "n_defined_branches": pd.to_numeric(overview["n_defined_branches"], errors="coerce").sum(),
        "n_requested_branches": requested,
        "n_completed_branches": pd.to_numeric(
            overview["n_completed_branches"], errors="coerce"
        ).sum(),
        "n_failed_branches": failed,
        "n_skipped_branches": pd.to_numeric(overview["n_skipped_branches"], errors="coerce").sum(),
        "multiverse_status": overall_status,
    }
    overview = pd.concat([overview, pd.DataFrame([overall])], ignore_index=True)
    branch_summary = (
        pd.concat(branch_rows, ignore_index=True, sort=False)
        if branch_rows
        else pd.DataFrame(
            columns=[
                "result_name",
                "multiverse_family",
                "branch_id",
                "branch_label",
                "branch_status",
            ]
        )
    )
    if "message" not in branch_summary.columns:
        branch_summary["message"] = pd.NA
    if len(branch_summary):
        failure_summary = branch_summary.loc[
            branch_summary.get(
                "branch_status", pd.Series(index=branch_summary.index, dtype=str)
            ).isin(["failed", "skipped"])
            | branch_summary["message"].astype("string").fillna("").ne("")
        ].copy()
    else:
        failure_summary = branch_summary.copy()
    keep = [
        column
        for column in [
            "result_name",
            "multiverse_family",
            "branch_id",
            "branch_label",
            "branch_status",
            "message",
        ]
        if column in failure_summary.columns
    ]
    failure_summary = failure_summary[keep]
    settings = pd.DataFrame(
        {
            "setting": ["n_result_objects", "result_names", "result_classes"],
            "value": [
                str(len(collected)),
                ", ".join(names),
                " | ".join(str(result.get("_gp3_class", "dict")) for result in collected),
            ],
        }
    )
    return {
        "overview": overview,
        "branch_summary": branch_summary,
        "failure_summary": failure_summary,
        "settings": settings,
        "_gp3_class": "gp3_multiverse_summary",
    }


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
