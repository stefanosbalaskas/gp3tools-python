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
    data,
    subject_col=None,
    condition_col=None,
    time_col=None,
    outcome_col=None,
    condition_order=None,
    aggregate_fun=None,
    complete_only=True,
    **kwargs,
):
    """Prepare conservative two-condition time-course data (R v2.3.0 parity)."""
    import numpy as np
    import pandas as pd

    legacy_value_col = kwargs.pop("value_col", None)
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a data frame.")

    if (
        subject_col is None
        and condition_col is None
        and time_col is None
        and outcome_col is None
        and legacy_value_col is None
        and condition_order is None
        and aggregate_fun is None
        and complete_only is True
        and not kwargs
    ):
        out = data.copy()
        subject_source = infer_column(out, "subject")
        condition_source = infer_column(out, "condition")
        time_source = infer_column(out, "time")
        value_source = "value" if "value" in out.columns else infer_column(out, "pupil")
        aliases = {
            subject_source: "subject",
            condition_source: "condition",
            time_source: "time",
            value_source: "value",
        }
        aliases = {
            source: target
            for source, target in aliases.items()
            if source and source in out.columns and source != target
        }
        return out.rename(columns=aliases)

    if outcome_col is None and legacy_value_col is not None:
        # Preserve the historical Python rename-only helper.
        out = data.copy()
        aliases = {
            time_col: "time",
            legacy_value_col: "value",
            subject_col: "subject",
            condition_col: "condition",
        }
        aliases = {k: v for k, v in aliases.items() if k and k in out.columns and k != v}
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")
        return out.rename(columns=aliases)
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unknown}")
    for value, name in (
        (subject_col, "subject_col"),
        (condition_col, "condition_col"),
        (time_col, "time_col"),
        (outcome_col, "outcome_col"),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"`{name}` must be a non-empty character scalar.")
    missing = [
        c for c in (subject_col, condition_col, time_col, outcome_col) if c not in data.columns
    ]
    if missing:
        raise ValueError("`data` is missing required column(s): " + ", ".join(missing))
    if aggregate_fun is None:
        aggregate_fun = np.mean
    if not callable(aggregate_fun):
        raise ValueError("`aggregate_fun` must be a function.")
    if not isinstance(complete_only, (bool, np.bool_)):
        raise ValueError("`complete_only` must be TRUE or FALSE.")

    dat = pd.DataFrame(
        {
            ".gp3_cluster_subject": data[subject_col].astype("string"),
            ".gp3_cluster_condition": data[condition_col].astype("string"),
            ".gp3_cluster_time_bin": pd.to_numeric(data[time_col], errors="coerce"),
            ".gp3_cluster_outcome": pd.to_numeric(data[outcome_col], errors="coerce"),
        }
    )
    valid = (
        dat[".gp3_cluster_subject"].notna()
        & dat[".gp3_cluster_condition"].notna()
        & np.isfinite(dat[".gp3_cluster_time_bin"].to_numpy(float))
        & np.isfinite(dat[".gp3_cluster_outcome"].to_numpy(float))
    )
    dat = dat.loc[valid].copy()
    if dat.empty:
        raise ValueError("No valid time-course rows remained after preparation.")

    available_conditions = list(pd.unique(dat[".gp3_cluster_condition"].astype(str)))
    if condition_order is None:
        condition_order = available_conditions
    else:
        condition_order = (
            list(condition_order) if not isinstance(condition_order, str) else [condition_order]
        )
        if len(condition_order) != 2 or any(
            not isinstance(x, str) or not x for x in condition_order
        ):
            raise ValueError("`condition_order` must be NULL or a character vector of length two.")
    condition_order = list(dict.fromkeys(condition_order))
    if len(condition_order) != 2:
        raise ValueError(
            "Cluster-permutation preparation requires exactly two conditions. Found: "
            + ", ".join(available_conditions)
        )
    missing_conditions = [x for x in condition_order if x not in available_conditions]
    if missing_conditions:
        raise ValueError(
            "Requested condition(s) not found in `data`: " + ", ".join(missing_conditions)
        )
    dat = dat.loc[dat[".gp3_cluster_condition"].astype(str).isin(condition_order)].copy()
    dat[".gp3_cluster_condition"] = pd.Categorical(
        dat[".gp3_cluster_condition"].astype(str),
        categories=condition_order,
        ordered=True,
    )

    def aggregate_series(series):
        values = pd.to_numeric(series, errors="coerce")
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return np.nan
        try:
            return float(aggregate_fun(values))
        except TypeError:
            return float(aggregate_fun(values, na_rm=True))

    dat = (
        dat.groupby(
            [".gp3_cluster_subject", ".gp3_cluster_condition", ".gp3_cluster_time_bin"],
            observed=True,
            sort=True,
            dropna=False,
        )[".gp3_cluster_outcome"]
        .apply(aggregate_series)
        .reset_index()
    )
    if complete_only:
        counts = dat.groupby(
            [".gp3_cluster_subject", ".gp3_cluster_time_bin"],
            dropna=False,
        )[".gp3_cluster_condition"].transform("nunique")
        dat = dat.loc[counts.eq(2)].copy()
    if dat.empty:
        raise ValueError("No complete paired subject-by-time cells remained after preparation.")

    dat[".gp3_cluster_status"] = "ok"
    dat = dat.sort_values(
        [".gp3_cluster_subject", ".gp3_cluster_time_bin", ".gp3_cluster_condition"],
        kind="stable",
    ).reset_index(drop=True)
    dat.attrs["r_class"] = "gp3_timecourse_test_data"
    return dat


def prepare_gazepoint_cluster_data(data, **kwargs):
    return prepare_gazepoint_timecourse_test_data(data, **kwargs)


def prepare_gazepoint_hmm_data(data, **kwargs):
    return ensure_dataframe(data)


def prepare_gazepoint_fixation_aligned_data(
    data,
    fixation_time_col=None,
    sample_time_col=None,
    window=(-0.5, 1.5),
    *,
    time_col=None,
    participant_col=None,
    trial_col=None,
    aoi_col=None,
    target_aoi=None,
    fixation_col=None,
    saccade_col=None,
    event_col=None,
    event_value=None,
    alignment_event="first_target_entry",
    baseline_window=None,
    analysis_window=None,
    keep_unaligned=False,
    name="gazepoint_fixation_aligned_data",
    **kwargs,
):
    if time_col is None:
        # Historical Python row-alignment route.
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unknown}")
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

    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unknown}")

    from ._behavioral_r2 import prepare_fixation_aligned_data

    return prepare_fixation_aligned_data(
        data,
        time_col=time_col,
        participant_col=participant_col,
        trial_col=trial_col,
        aoi_col=aoi_col,
        target_aoi=target_aoi,
        fixation_col=fixation_col,
        saccade_col=saccade_col,
        event_col=event_col,
        event_value=event_value,
        alignment_event=alignment_event,
        baseline_window=baseline_window,
        analysis_window=analysis_window,
        keep_unaligned=keep_unaligned,
        name=name,
    )


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


_GP3_MODEL_R_UNSET = object()


def _gp3_model_r_requested(*values) -> bool:
    return any(value is not _GP3_MODEL_R_UNSET for value in values)


def _gp3_model_validate_bool(value, name: str) -> bool:
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be True or False")
    return bool(value)


def _gp3_model_validate_positive(value, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or float(value) <= 0:
        raise ValueError(f"{name} must be a positive finite numeric scalar")
    return float(value)


def _gp3_model_validate_seed(value) -> int:
    if isinstance(value, (bool, np.bool_)) or not np.isfinite(value):
        raise ValueError("seed must be a finite numeric scalar")
    return int(value)


def _gp3_model_class(model) -> str:
    return type(model).__name__


def _gp3_model_collection(model) -> list[tuple[str, Any]] | None:
    if isinstance(model, dict):
        if "model" in model or "idata" in model:
            return None
        if not model:
            return None
        return [(str(name), value) for name, value in model.items()]
    if isinstance(model, (list, tuple)) and model:
        return [(f"model_{i + 1}", value) for i, value in enumerate(model)]
    return None


def _gp3_model_family_link(model) -> tuple[object, object]:
    family_obj = getattr(model, "family", None)
    if family_obj is None:
        inner = getattr(model, "model", None)
        family_obj = getattr(inner, "family", None)
    if family_obj is None:
        return pd.NA, pd.NA
    family = getattr(family_obj, "family", None)
    if family is None:
        family = type(family_obj).__name__
    link_obj = getattr(family_obj, "link", None)
    link = getattr(link_obj, "__class__", type(None)).__name__ if link_obj is not None else pd.NA
    return str(family), link


def _gp3_model_formula(model):
    inner = getattr(model, "model", None)
    formula = getattr(inner, "formula", None)
    if formula is None:
        formula = getattr(model, "formula", None)
    return pd.NA if formula is None else str(formula)


def _gp3_model_info_table(model, model_name: str) -> pd.DataFrame:
    family, link = _gp3_model_family_link(model)

    def numeric_attr(name):
        value = getattr(model, name, np.nan)
        try:
            value = float(value)
        except Exception:
            return np.nan
        return value if np.isfinite(value) else np.nan

    nobs = numeric_attr("nobs")
    return pd.DataFrame(
        {
            "model_name": [model_name],
            "model_class": [_gp3_model_class(model)],
            "model_family": [family],
            "model_link": [link],
            "formula": [_gp3_model_formula(model)],
            "n_observations": [pd.NA if not np.isfinite(nobs) else int(nobs)],
            "df_residual": [numeric_attr("df_resid")],
            "aic": [numeric_attr("aic")],
            "bic": [numeric_attr("bic")],
            "log_lik": [numeric_attr("llf")],
        }
    )


def _gp3_model_legacy_summary(model) -> pd.DataFrame:
    if isinstance(model, dict) and "idata" in model:
        try:
            import arviz as az

            return az.summary(model["idata"]).reset_index().rename(columns={"index": "term"})
        except Exception:
            return pd.DataFrame({"term": [], "estimate": []})
    params = getattr(model, "params", None)
    if params is None:
        return pd.DataFrame({"term": [], "estimate": []})
    if isinstance(params, pd.Series):
        terms = list(params.index)
        estimates = params.to_numpy(dtype=float)
    else:
        estimates = np.asarray(params, dtype=float).reshape(-1)
        names = getattr(getattr(model, "model", None), "exog_names", None)
        terms = (
            list(names)
            if names is not None and len(names) == len(estimates)
            else [f"term_{i + 1}" for i in range(len(estimates))]
        )
    out = pd.DataFrame({"term": terms, "estimate": estimates})
    if hasattr(model, "bse"):
        out["std_error"] = np.asarray(model.bse, dtype=float).reshape(-1)
    if hasattr(model, "pvalues"):
        out["p_value"] = np.asarray(model.pvalues, dtype=float).reshape(-1)
    if hasattr(model, "conf_int"):
        try:
            ci = np.asarray(model.conf_int(), dtype=float)
            if ci.ndim == 2 and ci.shape[1] >= 2 and len(ci) == len(out):
                out["conf_low"] = ci[:, 0]
                out["conf_high"] = ci[:, 1]
        except Exception:
            pass
    return out


def _gp3_model_pstars(p):
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.10:
        return "."
    return ""


def _gp3_model_fixed_effects_r(
    model,
    model_name: str,
    conf_level: float,
    exponentiate: bool,
    drop_intercept: bool,
) -> pd.DataFrame:
    fitted, _ = _gp3_model_for_diagnostics(model, model_name)
    legacy = _gp3_model_legacy_summary(fitted)
    model_class = _gp3_model_class(fitted)
    columns = [
        "model_name",
        "model_class",
        "term",
        "estimate",
        "std_error",
        "statistic",
        "statistic_type",
        "df",
        "p_value",
        "conf_low",
        "conf_high",
        "response_scale",
        "significance",
        "diagnostic_status",
        "message",
    ]
    if legacy.empty or "estimate" not in legacy:
        row = {c: pd.NA for c in columns}
        row.update(
            {
                "model_name": model_name,
                "model_class": model_class,
                "response_scale": "exponentiated" if exponentiate else "link_or_original",
                "diagnostic_status": "unsupported_model_class",
                "message": "Unsupported model class for fixed-effect summaries.",
            }
        )
        return pd.DataFrame([row], columns=columns)
    out = pd.DataFrame()
    out["term"] = legacy.get(
        "term", pd.Series([f"term_{i + 1}" for i in range(len(legacy))])
    ).astype(str)
    out["estimate"] = pd.to_numeric(legacy["estimate"], errors="coerce")
    out["std_error"] = pd.to_numeric(legacy.get("std_error", np.nan), errors="coerce")
    statistic = getattr(fitted, "tvalues", None)
    if statistic is None:
        statistic = getattr(fitted, "zvalues", None)
    if statistic is None:
        with np.errstate(divide="ignore", invalid="ignore"):
            statistic = out["estimate"].to_numpy(float) / out["std_error"].to_numpy(float)
    statistic = np.asarray(statistic, dtype=float).reshape(-1)
    if len(statistic) != len(out):
        statistic = np.full(len(out), np.nan)
    out["statistic"] = statistic
    family, _ = _gp3_model_family_link(fitted)
    out["statistic_type"] = "z" if family is not pd.NA else "t"
    df_resid = getattr(fitted, "df_resid", np.nan)
    try:
        df_resid = float(df_resid)
    except Exception:
        df_resid = np.nan
    out["df"] = df_resid if np.isfinite(df_resid) else np.nan
    out["p_value"] = pd.to_numeric(legacy.get("p_value", np.nan), errors="coerce")
    alpha = 1.0 - conf_level
    ci = None
    if hasattr(fitted, "conf_int"):
        try:
            ci = np.asarray(fitted.conf_int(alpha=alpha), dtype=float)
        except TypeError:
            try:
                ci = np.asarray(fitted.conf_int(), dtype=float)
            except Exception:
                ci = None
        except Exception:
            ci = None
    if ci is not None and ci.ndim == 2 and ci.shape[1] >= 2 and len(ci) == len(out):
        out["conf_low"] = ci[:, 0]
        out["conf_high"] = ci[:, 1]
    else:
        z = float(stats.norm.ppf(1.0 - alpha / 2.0))
        out["conf_low"] = out["estimate"] - z * out["std_error"]
        out["conf_high"] = out["estimate"] + z * out["std_error"]
    if exponentiate:
        out[["estimate", "conf_low", "conf_high"]] = np.exp(
            out[["estimate", "conf_low", "conf_high"]]
        )
    out.insert(0, "model_class", model_class)
    out.insert(0, "model_name", model_name)
    out["response_scale"] = "exponentiated" if exponentiate else "link_or_original"
    out["significance"] = [
        _gp3_model_pstars(float(p)) if pd.notna(p) else "" for p in out["p_value"]
    ]
    out["diagnostic_status"] = "ok"
    out["message"] = "Fixed-effect summary extracted."
    if drop_intercept:
        out = out.loc[~out["term"].str.lower().isin({"intercept", "(intercept)"})].copy()
    if out.empty:
        row = {c: pd.NA for c in columns}
        row.update(
            {
                "model_name": model_name,
                "model_class": model_class,
                "response_scale": "exponentiated" if exponentiate else "link_or_original",
                "diagnostic_status": "not_available",
                "message": "No fixed-effect rows remained after filtering.",
            }
        )
        return pd.DataFrame([row], columns=columns)
    return out[columns].reset_index(drop=True)


def _gp3_model_skipped_row(model_name, model_class, diagnostic, status, message):
    base = {
        "model_name": [model_name],
        "model_class": [model_class],
        "diagnostic": [diagnostic],
        "diagnostic_status": [status],
        "message": [message],
    }
    if diagnostic == "convergence":
        base["converged"] = [pd.NA]
    elif diagnostic == "singularity":
        base["singular_fit"] = [pd.NA]
        base["tolerance"] = [np.nan]
    elif diagnostic == "overdispersion":
        base.update(
            {
                "dispersion_ratio": [np.nan],
                "pearson_chisq": [np.nan],
                "residual_df": [np.nan],
                "overdispersed": [pd.NA],
                "ratio_threshold": [np.nan],
            }
        )
    return pd.DataFrame(base)


def _gp3_model_dharma_row(model_name, model_class, use_dharma, simulations, seed):
    if not use_dharma:
        status = "skipped_disabled"
        message = "DHARMa diagnostics were disabled."
        dharma_status = "skipped_disabled"
    else:
        status = "skipped_missing_package"
        message = (
            "DHARMa diagnostics require the R DHARMa backend and are unavailable in native Python."
        )
        dharma_status = "skipped_missing_package"
    return pd.DataFrame(
        {
            "model_name": [model_name],
            "model_class": [model_class],
            "diagnostic": ["dharma"],
            "dharma_status": [dharma_status],
            "uniformity_p": [np.nan],
            "dispersion_p": [np.nan],
            "outlier_p": [np.nan],
            "diagnostic_status": [status],
            "message": [message],
            "simulations": [int(simulations)],
            "seed": [int(seed)],
        }
    )


def _gp3_model_status(statuses) -> str:
    values = [str(x) for x in statuses if pd.notna(x)]
    if any(x in {"error", "unsupported_model_class"} for x in values):
        return "error"
    if any(
        x
        in {
            "convergence_warning",
            "singular_fit",
            "overdispersed",
            "basis_warning",
            "diagnostic_warning",
        }
        for x in values
    ):
        return "diagnostic_warning"
    if any(x == "ok" for x in values):
        return "ok"
    for value in values:
        if value not in {"skipped_disabled", "skipped_missing_package", "not_applicable"}:
            return value
    return values[0] if values else "not_available"


def _gp3_model_messages(*tables) -> object:
    messages = []
    for table in tables:
        if isinstance(table, pd.DataFrame) and "message" in table:
            for value in table["message"].dropna().astype(str):
                if value and value not in messages:
                    messages.append(value)
    return " | ".join(messages) if messages else pd.NA


def _gp3_model_basis_table(model, model_name: str, enabled: bool) -> pd.DataFrame:
    model_class = _gp3_model_class(model)
    columns = [
        "model_name",
        "model_class",
        "diagnostic",
        "smooth",
        "k_index",
        "edf",
        "k_prime",
        "p_value",
        "basis_status",
        "diagnostic_status",
        "message",
    ]
    if not enabled:
        return pd.DataFrame(
            [
                [
                    model_name,
                    model_class,
                    "basis",
                    pd.NA,
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    "skipped_disabled",
                    "skipped_disabled",
                    "GAM basis diagnostics were disabled.",
                ]
            ],
            columns=columns,
        )
    raw = getattr(model, "basis_diagnostics", None)
    if raw is None:
        raw = getattr(model, "k_check", None)
    if isinstance(raw, pd.DataFrame) and len(raw):
        rows = []
        for idx, row in raw.iterrows():
            p = pd.to_numeric(
                pd.Series([row.get("p_value", row.get("p-value", np.nan))]), errors="coerce"
            ).iloc[0]
            status = "basis_warning" if np.isfinite(p) and float(p) < 0.05 else "ok"
            rows.append(
                {
                    "model_name": model_name,
                    "model_class": model_class,
                    "diagnostic": "basis",
                    "smooth": str(row.get("smooth", idx)),
                    "k_index": row.get("k_index", row.get("k-index", np.nan)),
                    "edf": row.get("edf", np.nan),
                    "k_prime": row.get("k_prime", row.get("k'", np.nan)),
                    "p_value": p,
                    "basis_status": status,
                    "diagnostic_status": status,
                    "message": "Basis-dimension check returned p < .05."
                    if status == "basis_warning"
                    else "Basis-dimension check did not return p < .05.",
                }
            )
        return pd.DataFrame(rows, columns=columns)
    return pd.DataFrame(
        [
            [
                model_name,
                model_class,
                "basis",
                pd.NA,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                "not_applicable",
                "not_applicable",
                "GAM basis diagnostics are not available for this native Python model.",
            ]
        ],
        columns=columns,
    )


def tidy_gazepoint_model_summary(
    model,
    model_name=_GP3_MODEL_R_UNSET,
    conf_level=_GP3_MODEL_R_UNSET,
    exponentiate=_GP3_MODEL_R_UNSET,
    drop_intercept=_GP3_MODEL_R_UNSET,
    include_diagnostics=_GP3_MODEL_R_UNSET,
    use_dharma=_GP3_MODEL_R_UNSET,
    dharma_simulations=_GP3_MODEL_R_UNSET,
    seed=_GP3_MODEL_R_UNSET,
):
    r_mode = _gp3_model_r_requested(
        model_name,
        conf_level,
        exponentiate,
        drop_intercept,
        include_diagnostics,
        use_dharma,
        dharma_simulations,
        seed,
    )
    if not r_mode:
        return _gp3_model_legacy_summary(model)
    model_name = None if model_name is _GP3_MODEL_R_UNSET else model_name
    conf_level = 0.95 if conf_level is _GP3_MODEL_R_UNSET else float(conf_level)
    exponentiate = (
        False
        if exponentiate is _GP3_MODEL_R_UNSET
        else _gp3_model_validate_bool(exponentiate, "exponentiate")
    )
    drop_intercept = (
        False
        if drop_intercept is _GP3_MODEL_R_UNSET
        else _gp3_model_validate_bool(drop_intercept, "drop_intercept")
    )
    include_diagnostics = (
        True
        if include_diagnostics is _GP3_MODEL_R_UNSET
        else _gp3_model_validate_bool(include_diagnostics, "include_diagnostics")
    )
    use_dharma = (
        False
        if use_dharma is _GP3_MODEL_R_UNSET
        else _gp3_model_validate_bool(use_dharma, "use_dharma")
    )
    dharma_simulations = (
        250
        if dharma_simulations is _GP3_MODEL_R_UNSET
        else int(_gp3_model_validate_positive(dharma_simulations, "dharma_simulations"))
    )
    seed = 123 if seed is _GP3_MODEL_R_UNSET else _gp3_model_validate_seed(seed)
    if not np.isfinite(conf_level) or not 0 < conf_level < 1:
        raise ValueError("conf_level must be a finite numeric scalar between 0 and 1")
    fitted, resolved_name = _gp3_model_for_diagnostics(model, model_name)
    fixed = _gp3_model_fixed_effects_r(
        fitted, resolved_name, conf_level, exponentiate, drop_intercept
    )
    info = _gp3_model_info_table(fitted, resolved_name)
    if include_diagnostics:
        diagnostics = diagnose_gazepoint_glmm(
            fitted,
            model_name=resolved_name,
            use_dharma=use_dharma,
            dharma_simulations=dharma_simulations,
            seed=seed,
        )
    else:
        diagnostics = {
            "overview": pd.DataFrame(
                {
                    "model_name": [resolved_name],
                    "model_class": [_gp3_model_class(fitted)],
                    "diagnostic_status": ["skipped_disabled"],
                    "message": ["Model diagnostics were disabled."],
                }
            )
        }
    diag_over = diagnostics.get("overview", pd.DataFrame())
    diag_status = (
        str(diag_over.iloc[0]["diagnostic_status"])
        if len(diag_over) and "diagnostic_status" in diag_over
        else "not_available"
    )
    diag_message = (
        diag_over.iloc[0].get("message", pd.NA)
        if len(diag_over)
        else "Diagnostics overview was not available."
    )
    fixed_statuses = (
        fixed["diagnostic_status"].dropna().astype(str).tolist()
        if "diagnostic_status" in fixed
        else []
    )
    fixed_status = (
        "error"
        if any(x in {"error", "unsupported_model_class"} for x in fixed_statuses)
        else (
            "ok" if fixed_statuses and all(x == "ok" for x in fixed_statuses) else "not_available"
        )
    )
    summary_status = fixed_status
    if fixed_status == "ok" and include_diagnostics:
        summary_status = (
            "diagnostic_warning"
            if diag_status == "diagnostic_warning"
            else (
                "diagnostic_error" if diag_status in {"error", "unsupported_model_class"} else "ok"
            )
        )
    overview = pd.DataFrame(
        {
            "model_name": [resolved_name],
            "model_class": [_gp3_model_class(fitted)],
            "model_family": [info.iloc[0]["model_family"]],
            "model_link": [info.iloc[0]["model_link"]],
            "n_observations": [info.iloc[0]["n_observations"]],
            "n_fixed_effects": [int((fixed["diagnostic_status"] == "ok").sum())],
            "fixed_effects_status": [fixed_status],
            "diagnostics_status": [diag_status],
            "summary_status": [summary_status],
            "message": [_gp3_model_messages(fixed, pd.DataFrame({"message": [diag_message]}))],
        }
    )
    return {
        "overview": overview,
        "model_info": info,
        "fixed_effects": fixed,
        "diagnostics": diagnostics,
        "settings": {
            "conf_level": conf_level,
            "exponentiate": exponentiate,
            "drop_intercept": drop_intercept,
            "include_diagnostics": include_diagnostics,
            "use_dharma": use_dharma,
            "dharma_simulations": int(dharma_simulations),
            "seed": int(seed),
        },
    }


def summarise_gazepoint_fixed_effects(
    model,
    model_name=_GP3_MODEL_R_UNSET,
    conf_level=_GP3_MODEL_R_UNSET,
    exponentiate=_GP3_MODEL_R_UNSET,
    drop_intercept=_GP3_MODEL_R_UNSET,
    **kwargs,
):
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected argument(s): {unknown}")
    r_mode = _gp3_model_r_requested(model_name, conf_level, exponentiate, drop_intercept)
    if not r_mode:
        return _gp3_model_legacy_summary(model)
    model_name = None if model_name is _GP3_MODEL_R_UNSET else model_name
    conf_level = 0.95 if conf_level is _GP3_MODEL_R_UNSET else float(conf_level)
    exponentiate = (
        False
        if exponentiate is _GP3_MODEL_R_UNSET
        else _gp3_model_validate_bool(exponentiate, "exponentiate")
    )
    drop_intercept = (
        False
        if drop_intercept is _GP3_MODEL_R_UNSET
        else _gp3_model_validate_bool(drop_intercept, "drop_intercept")
    )
    if not np.isfinite(conf_level) or not 0 < conf_level < 1:
        raise ValueError("conf_level must be a finite numeric scalar between 0 and 1")
    fitted, resolved_name = _gp3_model_for_diagnostics(model, model_name)
    return _gp3_model_fixed_effects_r(
        fitted, resolved_name, conf_level, exponentiate, drop_intercept
    )


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


def diagnose_gazepoint_glmm(
    model,
    model_name=_GP3_MODEL_R_UNSET,
    check_convergence=_GP3_MODEL_R_UNSET,
    check_singularity=_GP3_MODEL_R_UNSET,
    check_overdispersion=_GP3_MODEL_R_UNSET,
    use_dharma=_GP3_MODEL_R_UNSET,
    dharma_simulations=_GP3_MODEL_R_UNSET,
    seed=_GP3_MODEL_R_UNSET,
):
    r_mode = _gp3_model_r_requested(
        model_name,
        check_convergence,
        check_singularity,
        check_overdispersion,
        use_dharma,
        dharma_simulations,
        seed,
    )
    if not r_mode:
        return {
            "convergence": check_gazepoint_model_convergence(model),
            "overdispersion": check_gazepoint_model_overdispersion(model),
            "coefficients": _gp3_model_legacy_summary(model),
        }
    model_name = None if model_name is _GP3_MODEL_R_UNSET else model_name
    check_convergence = (
        True
        if check_convergence is _GP3_MODEL_R_UNSET
        else _gp3_model_validate_bool(check_convergence, "check_convergence")
    )
    check_singularity = (
        True
        if check_singularity is _GP3_MODEL_R_UNSET
        else _gp3_model_validate_bool(check_singularity, "check_singularity")
    )
    check_overdispersion = (
        True
        if check_overdispersion is _GP3_MODEL_R_UNSET
        else _gp3_model_validate_bool(check_overdispersion, "check_overdispersion")
    )
    use_dharma = (
        True
        if use_dharma is _GP3_MODEL_R_UNSET
        else _gp3_model_validate_bool(use_dharma, "use_dharma")
    )
    dharma_simulations = (
        250
        if dharma_simulations is _GP3_MODEL_R_UNSET
        else int(_gp3_model_validate_positive(dharma_simulations, "dharma_simulations"))
    )
    seed = 123 if seed is _GP3_MODEL_R_UNSET else _gp3_model_validate_seed(seed)
    collection = _gp3_model_collection(model)
    if collection is not None:
        parts = []
        for i, (name, fitted) in enumerate(collection):
            resolved = name if model_name is None else f"{model_name}_{name or i + 1}"
            parts.append(
                diagnose_gazepoint_glmm(
                    fitted,
                    model_name=resolved,
                    check_convergence=check_convergence,
                    check_singularity=check_singularity,
                    check_overdispersion=check_overdispersion,
                    use_dharma=use_dharma,
                    dharma_simulations=dharma_simulations,
                    seed=seed,
                )
            )
        return {
            key: pd.concat([p[key] for p in parts], ignore_index=True)
            for key in ["overview", "convergence", "singularity", "overdispersion", "dharma"]
        } | {
            "settings": {
                "check_convergence": check_convergence,
                "check_singularity": check_singularity,
                "check_overdispersion": check_overdispersion,
                "use_dharma": use_dharma,
                "dharma_simulations": dharma_simulations,
                "seed": seed,
                "n_models": len(parts),
            }
        }
    fitted, resolved_name = _gp3_model_for_diagnostics(model, model_name)
    model_class = _gp3_model_class(fitted)
    convergence = (
        check_gazepoint_model_convergence(fitted, resolved_name)
        if check_convergence
        else _gp3_model_skipped_row(
            resolved_name,
            model_class,
            "convergence",
            "skipped_disabled",
            "Convergence diagnostics were disabled.",
        )
    )
    singularity = (
        check_gazepoint_model_singularity(fitted, model_name=resolved_name)
        if check_singularity
        else _gp3_model_skipped_row(
            resolved_name,
            model_class,
            "singularity",
            "skipped_disabled",
            "Singularity diagnostics were disabled.",
        )
    )
    overdispersion = (
        check_gazepoint_model_overdispersion(fitted, model_name=resolved_name)
        if check_overdispersion
        else _gp3_model_skipped_row(
            resolved_name,
            model_class,
            "overdispersion",
            "skipped_disabled",
            "Overdispersion diagnostics were disabled.",
        )
    )
    dharma = _gp3_model_dharma_row(resolved_name, model_class, use_dharma, dharma_simulations, seed)
    statuses = [
        convergence.iloc[0].get("diagnostic_status"),
        singularity.iloc[0].get("diagnostic_status"),
        overdispersion.iloc[0].get("diagnostic_status"),
        dharma.iloc[0].get("diagnostic_status"),
    ]
    overview = pd.DataFrame(
        {
            "model_name": [resolved_name],
            "model_class": [model_class],
            "diagnostic_status": [_gp3_model_status(statuses)],
            "converged": [convergence.iloc[0].get("converged", pd.NA)],
            "singular_fit": [singularity.iloc[0].get("singular_fit", pd.NA)],
            "overdispersed": [overdispersion.iloc[0].get("overdispersed", pd.NA)],
            "dharma_status": [dharma.iloc[0].get("dharma_status", pd.NA)],
            "message": [_gp3_model_messages(convergence, singularity, overdispersion, dharma)],
        }
    )
    return {
        "overview": overview,
        "convergence": convergence,
        "singularity": singularity,
        "overdispersion": overdispersion,
        "dharma": dharma,
        "settings": {
            "check_convergence": check_convergence,
            "check_singularity": check_singularity,
            "check_overdispersion": check_overdispersion,
            "use_dharma": use_dharma,
            "dharma_simulations": dharma_simulations,
            "seed": seed,
            "n_models": 1,
        },
    }


def diagnose_gazepoint_gamm(
    model,
    model_name=_GP3_MODEL_R_UNSET,
    check_convergence=_GP3_MODEL_R_UNSET,
    check_basis=_GP3_MODEL_R_UNSET,
    check_overdispersion=_GP3_MODEL_R_UNSET,
    use_dharma=_GP3_MODEL_R_UNSET,
    dharma_simulations=_GP3_MODEL_R_UNSET,
    seed=_GP3_MODEL_R_UNSET,
):
    r_mode = _gp3_model_r_requested(
        model_name,
        check_convergence,
        check_basis,
        check_overdispersion,
        use_dharma,
        dharma_simulations,
        seed,
    )
    if not r_mode:
        return {
            "convergence": check_gazepoint_model_convergence(model),
            "coefficients": _gp3_model_legacy_summary(model),
        }
    model_name = None if model_name is _GP3_MODEL_R_UNSET else model_name
    check_convergence = (
        True
        if check_convergence is _GP3_MODEL_R_UNSET
        else _gp3_model_validate_bool(check_convergence, "check_convergence")
    )
    check_basis = (
        True
        if check_basis is _GP3_MODEL_R_UNSET
        else _gp3_model_validate_bool(check_basis, "check_basis")
    )
    check_overdispersion = (
        True
        if check_overdispersion is _GP3_MODEL_R_UNSET
        else _gp3_model_validate_bool(check_overdispersion, "check_overdispersion")
    )
    use_dharma = (
        False
        if use_dharma is _GP3_MODEL_R_UNSET
        else _gp3_model_validate_bool(use_dharma, "use_dharma")
    )
    dharma_simulations = (
        250
        if dharma_simulations is _GP3_MODEL_R_UNSET
        else int(_gp3_model_validate_positive(dharma_simulations, "dharma_simulations"))
    )
    seed = 123 if seed is _GP3_MODEL_R_UNSET else _gp3_model_validate_seed(seed)
    collection = _gp3_model_collection(model)
    if collection is not None:
        parts = []
        for i, (name, fitted) in enumerate(collection):
            resolved = name if model_name is None else f"{model_name}_{name or i + 1}"
            parts.append(
                diagnose_gazepoint_gamm(
                    fitted,
                    model_name=resolved,
                    check_convergence=check_convergence,
                    check_basis=check_basis,
                    check_overdispersion=check_overdispersion,
                    use_dharma=use_dharma,
                    dharma_simulations=dharma_simulations,
                    seed=seed,
                )
            )
        return {
            key: pd.concat([p[key] for p in parts], ignore_index=True)
            for key in ["overview", "convergence", "basis", "overdispersion", "dharma"]
        } | {
            "settings": {
                "check_convergence": check_convergence,
                "check_basis": check_basis,
                "check_overdispersion": check_overdispersion,
                "use_dharma": use_dharma,
                "dharma_simulations": dharma_simulations,
                "seed": seed,
                "n_models": len(parts),
            }
        }
    fitted, resolved_name = _gp3_model_for_diagnostics(model, model_name)
    model_class = _gp3_model_class(fitted)
    convergence = (
        check_gazepoint_model_convergence(fitted, resolved_name)
        if check_convergence
        else _gp3_model_skipped_row(
            resolved_name,
            model_class,
            "convergence",
            "skipped_disabled",
            "Convergence diagnostics were disabled.",
        )
    )
    basis = _gp3_model_basis_table(fitted, resolved_name, check_basis)
    overdispersion = (
        check_gazepoint_model_overdispersion(fitted, model_name=resolved_name)
        if check_overdispersion
        else _gp3_model_skipped_row(
            resolved_name,
            model_class,
            "overdispersion",
            "skipped_disabled",
            "Overdispersion diagnostics were disabled.",
        )
    )
    dharma = _gp3_model_dharma_row(resolved_name, model_class, use_dharma, dharma_simulations, seed)
    basis_statuses = basis["basis_status"].dropna().astype(str).tolist()
    basis_status = (
        "basis_warning"
        if "basis_warning" in basis_statuses
        else (
            "ok"
            if "ok" in basis_statuses
            else (basis_statuses[0] if basis_statuses else "not_available")
        )
    )
    statuses = [
        convergence.iloc[0].get("diagnostic_status"),
        basis_status,
        overdispersion.iloc[0].get("diagnostic_status"),
        dharma.iloc[0].get("diagnostic_status"),
    ]
    overview = pd.DataFrame(
        {
            "model_name": [resolved_name],
            "model_class": [model_class],
            "diagnostic_status": [_gp3_model_status(statuses)],
            "converged": [convergence.iloc[0].get("converged", pd.NA)],
            "basis_status": [basis_status],
            "overdispersed": [overdispersion.iloc[0].get("overdispersed", pd.NA)],
            "dharma_status": [dharma.iloc[0].get("dharma_status", pd.NA)],
            "message": [_gp3_model_messages(convergence, basis, overdispersion, dharma)],
        }
    )
    return {
        "overview": overview,
        "convergence": convergence,
        "basis": basis,
        "overdispersion": overdispersion,
        "dharma": dharma,
        "settings": {
            "check_convergence": check_convergence,
            "check_basis": check_basis,
            "check_overdispersion": check_overdispersion,
            "use_dharma": use_dharma,
            "dharma_simulations": dharma_simulations,
            "seed": seed,
            "n_models": 1,
        },
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
    et_data=None,
    window_size=50,
    step=10,
    summary_stats=("mean", "sd"),
    by="USER_ID",
    condition_col=None,
    value_cols=None,
    ts_col="TIME",
    window_unit="milliseconds",
    time_unit="auto",
    include_partial=False,
    **kwargs,
):
    """Summarise gaze or pupil measures in sliding time windows (R v2.3.0 parity)."""
    import numpy as np
    import pandas as pd

    # Legacy Python aliases retained without changing the R-style path.
    if et_data is None and "data" in kwargs:
        et_data = kwargs.pop("data")
    legacy_value_col = kwargs.pop("value_col", None)
    legacy_group_col = kwargs.pop("group_col", None)
    if value_cols is None and legacy_value_col is not None:
        value_cols = [legacy_value_col]
    if legacy_group_col is not None and by == "USER_ID":
        by = legacy_group_col
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unknown}")

    if not isinstance(et_data, pd.DataFrame):
        raise ValueError("`et_data` must be a data frame.")
    df = et_data.copy()

    legacy_window_mode = legacy_value_col is not None and (
        ts_col not in df.columns
        or (by == "USER_ID" and "USER_ID" not in df.columns and legacy_group_col is None)
    )
    if legacy_window_mode:
        from scipy import stats as scipy_stats

        value_col = legacy_value_col
        if value_col not in df.columns:
            raise ValueError(f"Missing value column: {value_col}")
        if condition_col is not None and condition_col not in df.columns:
            raise ValueError(f"Missing condition column: {condition_col}")

        if condition_col:
            summary = (
                df.groupby(condition_col, dropna=False)[value_col]
                .agg(n="size", mean="mean", sd="std", se="sem")
                .reset_index()
            )
        else:
            values = pd.to_numeric(df[value_col], errors="coerce")
            summary = pd.DataFrame(
                {
                    "n": [len(df)],
                    "mean": [values.mean()],
                    "sd": [values.std()],
                    "se": [values.sem()],
                }
            )

        test = None
        if condition_col and df[condition_col].nunique(dropna=True) == 2:
            vals = [
                pd.to_numeric(group[value_col], errors="coerce").dropna()
                for _, group in df.groupby(condition_col, dropna=True, sort=True)
            ]
            if len(vals) == 2:
                statistic, p_value = scipy_stats.ttest_ind(vals[0], vals[1], equal_var=False)
                test = {
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                }
        return {"summary": summary, "test": test}

    if window_unit not in {"milliseconds", "seconds", "native"}:
        raise ValueError("`window_unit` must be one of: milliseconds, seconds, native.")
    if time_unit not in {"auto", "seconds", "milliseconds"}:
        raise ValueError("`time_unit` must be one of: auto, seconds, milliseconds.")

    if isinstance(by, str):
        by_cols = [by]
    elif by is None:
        by_cols = []
    else:
        by_cols = list(by)
    if condition_col is not None:
        by_cols.append(condition_col)
    by_cols = list(dict.fromkeys(c for c in by_cols if c is not None and str(c) != ""))

    required = list(dict.fromkeys(by_cols + [ts_col]))
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"`et_data` is missing required column(s): {', '.join(map(str, missing))}."
        )

    if value_cols is None:
        candidates = [
            "FPOGX",
            "FPOGY",
            "x",
            "y",
            "mean_pupil",
            "pupil",
            "pupil_clean",
            "pupil_smoothed",
            "LPupil",
            "RPupil",
            "LPD",
            "RPD",
            "LPMM",
            "RPMM",
        ]
        value_cols = [
            c for c in candidates if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]
    elif isinstance(value_cols, str):
        value_cols = [value_cols]
    else:
        value_cols = list(value_cols)

    if not value_cols:
        raise ValueError("No numeric `value_cols` were supplied or detected.")
    missing_values = [c for c in value_cols if c not in df.columns]
    if missing_values:
        raise ValueError(
            f"`et_data` is missing required column(s): {', '.join(map(str, missing_values))}."
        )
    non_numeric = [c for c in value_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"`value_cols` must be numeric. Non-numeric: {', '.join(non_numeric)}.")

    if isinstance(summary_stats, str):
        summary_stats = [summary_stats]
    summary_stats = list(dict.fromkeys(summary_stats))
    supported = {"mean", "sd", "median", "min", "max", "sum", "valid_prop"}
    unsupported = [s for s in summary_stats if s not in supported]
    if unsupported:
        raise ValueError(f"Unsupported `summary_stats`: {', '.join(unsupported)}.")

    for name, value in (("window_size", window_size), ("step", step)):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"`{name}` must be one finite positive number.")

    def time_info(values):
        if time_unit == "seconds":
            return 1.0
        if time_unit == "milliseconds":
            return 0.001
        finite = np.asarray(values, dtype=float)
        finite = finite[np.isfinite(finite)]
        delta = np.diff(np.unique(np.sort(finite)))
        delta = delta[np.isfinite(delta) & (delta > 0)]
        typical = float(np.median(delta)) if len(delta) else np.nan
        return 0.001 if np.isfinite(typical) and typical >= 1 else 1.0

    def to_seconds(value, unit, input_to_seconds):
        if unit == "milliseconds":
            return float(value) / 1000.0
        if unit == "seconds":
            return float(value)
        return float(value) * input_to_seconds

    def stat_value(values, stat):
        x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
        finite = np.isfinite(x)
        if stat == "valid_prop":
            return float(finite.mean()) if len(x) else np.nan
        if not finite.any():
            return np.nan
        z = x[finite]
        if stat == "mean":
            return float(np.mean(z))
        if stat == "sd":
            return float(np.std(z, ddof=1)) if len(z) >= 2 else np.nan
        if stat == "median":
            return float(np.median(z))
        if stat == "min":
            return float(np.min(z))
        if stat == "max":
            return float(np.max(z))
        if stat == "sum":
            return float(np.sum(z))
        raise ValueError("Unknown summary statistic.")

    if by_cols:
        grouped = df.groupby(by_cols, dropna=False, sort=True)
        groups = [frame for _, frame in grouped]
    else:
        groups = [df]

    rows = []
    for frame in groups:
        frame = frame.assign(__gp3_time=pd.to_numeric(frame[ts_col], errors="coerce")).sort_values(
            "__gp3_time", kind="stable", na_position="last"
        )
        time_raw = frame["__gp3_time"].to_numpy(float)
        factor = time_info(time_raw)
        time_sec = time_raw * factor
        finite_time = np.isfinite(time_sec)
        if not finite_time.any():
            continue
        window_sec = to_seconds(window_size, window_unit, factor)
        step_sec = to_seconds(step, window_unit, factor)
        min_time = float(np.min(time_sec[finite_time]))
        max_time = float(np.max(time_sec[finite_time]))
        if include_partial:
            final_start = max_time
        else:
            final_start = max_time - window_sec
            if final_start < min_time:
                continue

        starts = []
        current = min_time
        tol = max(abs(step_sec), 1.0) * 1e-12
        while current <= final_start + tol:
            starts.append(current)
            current += step_sec

        for start_sec in starts:
            end_sec = start_sec + window_sec
            if include_partial and end_sec > max_time:
                mask = finite_time & (time_sec >= start_sec) & (time_sec <= max_time)
            else:
                mask = finite_time & (time_sec >= start_sec) & (time_sec < end_sec)
            selected = frame.loc[mask]
            if selected.empty:
                continue
            row = {c: selected.iloc[0][c] for c in by_cols}
            clipped_end = min(end_sec, max_time)
            row.update(
                {
                    "window_start": start_sec / factor,
                    "window_end": clipped_end / factor,
                    "window_mid": ((start_sec + clipped_end) / 2.0) / factor,
                    "window_size": window_size,
                    "window_step": step,
                    "window_unit": window_unit,
                    "n_samples": int(len(selected)),
                }
            )
            for column in value_cols:
                values = selected[column].to_numpy()
                for stat in summary_stats:
                    row[f"{column}_{stat}"] = stat_value(values, stat)
            rows.append(row)

    columns = (
        by_cols
        + [
            "window_start",
            "window_end",
            "window_mid",
            "window_size",
            "window_step",
            "window_unit",
            "n_samples",
        ]
        + [f"{c}_{s}" for c in value_cols for s in summary_stats]
    )
    out = pd.DataFrame(rows, columns=columns)
    out.attrs["r_class"] = "gp3_window_summary"
    return out


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

    # === R3B DIVERGENCE SELECTOR ===
    if any(
        key in kwargs
        for key in {
            "outcome_col",
            "participant_col",
            "trial_col",
            "comparison",
            "bootstrap_unit",
            "summary_function",
            "n_boot",
            "ci",
            "consecutive_points",
            "null_value",
            "min_abs_difference",
            "direction",
            "seed",
            "keep_bootstrap",
            "name",
        }
    ):
        from ._behavioral_r3b import (
            estimate_gazepoint_divergence_point as _r3b,
        )

        return _r3b(
            data=data,
            outcome_col=kwargs.pop(
                "outcome_col",
                value_col,
            ),
            time_col=time_col,
            condition_col=condition_col,
            participant_col=kwargs.pop(
                "participant_col",
                None,
            ),
            trial_col=kwargs.pop(
                "trial_col",
                None,
            ),
            comparison=kwargs.pop(
                "comparison",
                None,
            ),
            bootstrap_unit=kwargs.pop(
                "bootstrap_unit",
                "participant",
            ),
            summary_function=kwargs.pop(
                "summary_function",
                "mean",
            ),
            n_boot=kwargs.pop(
                "n_boot",
                1000,
            ),
            ci=kwargs.pop(
                "ci",
                0.95,
            ),
            consecutive_points=kwargs.pop(
                "consecutive_points",
                min_run,
            ),
            null_value=kwargs.pop(
                "null_value",
                0.0,
            ),
            min_abs_difference=kwargs.pop(
                "min_abs_difference",
                0.0,
            ),
            direction=kwargs.pop(
                "direction",
                "either",
            ),
            seed=kwargs.pop(
                "seed",
                None,
            ),
            keep_bootstrap=kwargs.pop(
                "keep_bootstrap",
                False,
            ),
            name=kwargs.pop(
                "name",
                "divergence_point",
            ),
        )

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


def summarise_gazepoint_clusters(
    result,
    alpha=None,
    round_digits=None,
    include_timecourse=None,
):
    """Summarise cluster-permutation output with a legacy DataFrame shortcut."""
    r_mode = alpha is not None or round_digits is not None or include_timecourse is not None
    if not r_mode:
        return result["clusters"].copy() if isinstance(result, dict) else ensure_dataframe(result)

    if not isinstance(result, dict):
        raise ValueError("result must be a cluster-permutation result object")
    required = {"timecourse", "clusters", "permutation_distribution", "settings", "model_status"}
    missing = [name for name in required if name not in result]
    if missing:
        raise ValueError("result is missing required element(s): " + ", ".join(sorted(missing)))
    alpha = 0.05 if alpha is None else float(alpha)
    if not 0 < alpha < 1:
        raise ValueError("alpha must be a numeric scalar between 0 and 1")
    if round_digits is not None:
        if (
            isinstance(round_digits, (bool, np.bool_))
            or not isinstance(round_digits, (int, float, np.integer, np.floating))
            or not np.isfinite(round_digits)
            or round_digits < 0
        ):
            raise ValueError("round_digits must be NULL or a non-negative numeric scalar")
        round_digits = int(round_digits)
    include_timecourse = True if include_timecourse is None else include_timecourse
    if not isinstance(include_timecourse, (bool, np.bool_)):
        raise ValueError("include_timecourse must be TRUE or FALSE")

    timecourse = ensure_dataframe(result["timecourse"])
    clusters = ensure_dataframe(result["clusters"])
    permutation = ensure_dataframe(result["permutation_distribution"])

    required_time = {
        ".gp3_cluster_time_bin",
        "n_subjects",
        "mean_difference",
        "statistic",
        "cluster_id",
        "point_candidate",
    }
    missing_time = required_time.difference(timecourse.columns)
    if missing_time:
        raise ValueError(
            "result$timecourse is missing required column(s): " + ", ".join(sorted(missing_time))
        )
    required_perm = {"permutation", "max_cluster_statistic"}
    missing_perm = required_perm.difference(permutation.columns)
    if missing_perm:
        raise ValueError(
            "result$permutation_distribution is missing required column(s): "
            + ", ".join(sorted(missing_perm))
        )
    required_cluster = {
        "cluster_id",
        "cluster_direction",
        "start_time_bin",
        "end_time_bin",
        "n_time_bins",
        "cluster_statistic",
        "max_abs_statistic",
        "mean_difference",
        "p_value",
    }
    if len(clusters):
        missing_cluster = required_cluster.difference(clusters.columns)
        if missing_cluster:
            raise ValueError(
                "result$clusters is missing required column(s): "
                + ", ".join(sorted(missing_cluster))
            )

    time_bins = np.sort(
        pd.to_numeric(timecourse[".gp3_cluster_time_bin"], errors="coerce").dropna().unique()
    )
    bin_step = float(np.median(np.diff(time_bins))) if len(time_bins) >= 2 else np.nan

    if len(clusters):
        clusters = clusters.copy()
        clusters["cluster_label"] = "Cluster " + clusters["cluster_id"].astype(str)
        duration = pd.to_numeric(clusters["end_time_bin"], errors="coerce") - pd.to_numeric(
            clusters["start_time_bin"], errors="coerce"
        )
        if np.isfinite(bin_step):
            duration = duration + bin_step
        clusters["cluster_duration_ms"] = duration
        clusters["significant_alpha"] = pd.to_numeric(clusters["p_value"], errors="coerce") < alpha
        clusters["report_status"] = np.where(
            clusters["significant_alpha"], "significant", "not_significant"
        )
        preferred = [
            "cluster_id",
            "cluster_label",
            "cluster_direction",
            "start_time_bin",
            "end_time_bin",
            "cluster_duration_ms",
            "n_time_bins",
            "cluster_statistic",
            "max_abs_statistic",
            "mean_difference",
            "p_value",
            "significant_alpha",
            "report_status",
        ]
        clusters = clusters[
            preferred + [column for column in clusters.columns if column not in preferred]
        ]
    else:
        clusters = pd.DataFrame(
            columns=[
                "cluster_id",
                "cluster_label",
                "cluster_direction",
                "start_time_bin",
                "end_time_bin",
                "cluster_duration_ms",
                "n_time_bins",
                "cluster_statistic",
                "max_abs_statistic",
                "mean_difference",
                "p_value",
                "significant_alpha",
                "report_status",
            ]
        )

    significant = clusters.loc[clusters["significant_alpha"].fillna(False)].copy()
    settings = result["settings"] if isinstance(result["settings"], dict) else {}
    n_observed = len(clusters)
    n_significant = len(significant)
    report_status = (
        "no_observed_clusters"
        if n_observed == 0
        else "significant_cluster_evidence"
        if n_significant
        else "observed_clusters_not_significant"
    )

    overview = pd.DataFrame(
        [
            {
                "model_status": str(result["model_status"]),
                "report_status": report_status,
                "alpha": alpha,
                "n_subjects": int(
                    result.get(
                        "n_subjects",
                        timecourse["n_subjects"].nunique(dropna=True),
                    )
                ),
                "n_time_bins": int(
                    result.get(
                        "n_time_bins",
                        timecourse[".gp3_cluster_time_bin"].nunique(dropna=True),
                    )
                ),
                "bin_step_ms": bin_step,
                "n_permutations": settings.get("n_permutations", np.nan),
                "condition_1": settings.get("condition_1", pd.NA),
                "condition_2": settings.get("condition_2", pd.NA),
                "difference": settings.get("difference", pd.NA),
                "cluster_threshold": settings.get("cluster_threshold", np.nan),
                "tail": settings.get("tail", pd.NA),
                "cluster_stat": settings.get("cluster_stat", pd.NA),
                "min_time_bins": settings.get("min_time_bins", np.nan),
                "n_observed_clusters": n_observed,
                "n_significant_clusters": n_significant,
            }
        ]
    )

    tc_stat = pd.to_numeric(timecourse["statistic"], errors="coerce")
    tc_diff = pd.to_numeric(timecourse["mean_difference"], errors="coerce")
    tc_n = pd.to_numeric(timecourse["n_subjects"], errors="coerce")
    timecourse_summary = pd.DataFrame(
        [
            {
                "n_time_bins": len(timecourse),
                "start_time_bin": pd.to_numeric(
                    timecourse[".gp3_cluster_time_bin"], errors="coerce"
                ).min(),
                "end_time_bin": pd.to_numeric(
                    timecourse[".gp3_cluster_time_bin"], errors="coerce"
                ).max(),
                "min_n_subjects": tc_n.min(),
                "max_n_subjects": tc_n.max(),
                "mean_difference_min": tc_diff.min(),
                "mean_difference_max": tc_diff.max(),
                "mean_difference_mean": tc_diff.mean(),
                "max_abs_statistic": tc_stat.abs().max(),
                "n_candidate_time_bins": int(
                    timecourse["point_candidate"].fillna(False).astype(bool).sum()
                ),
                "n_clustered_time_bins": int(timecourse["cluster_id"].notna().sum()),
            }
        ]
    )

    dist = pd.to_numeric(permutation["max_cluster_statistic"], errors="coerce")
    permutation_summary = pd.DataFrame(
        [
            {
                "n_permutations": len(permutation),
                "min_max_cluster_statistic": dist.min(),
                "median_max_cluster_statistic": dist.median(),
                "mean_max_cluster_statistic": dist.mean(),
                "p95_max_cluster_statistic": dist.quantile(0.95),
                "max_max_cluster_statistic": dist.max(),
            }
        ]
    )
    settings_table = pd.DataFrame(
        [
            {
                "parameter": key,
                "value": (
                    ", ".join(map(str, value)) if isinstance(value, (list, tuple)) else str(value)
                ),
            }
            for key, value in settings.items()
        ],
        columns=["parameter", "value"],
    )
    warning = pd.DataFrame(
        [
            {
                "warning": result.get(
                    "warning",
                    "Cluster-based permutation tests are for time-course inference; "
                    "do not use them to select a confirmatory window and then retest that same window.",
                )
            }
        ]
    )

    def round_frame(frame):
        if round_digits is None:
            return frame
        frame = frame.copy()
        for column in frame.select_dtypes(include=np.number).columns:
            frame[column] = frame[column].round(round_digits)
        return frame

    out = {
        "overview": round_frame(overview),
        "clusters": round_frame(clusters),
        "significant_clusters": round_frame(significant),
        "timecourse_summary": round_frame(timecourse_summary),
        "permutation_summary": round_frame(permutation_summary),
        "settings": settings_table,
        "warning": warning,
        "model_status": str(result["model_status"]),
        "_gp3_class": "gp3_cluster_summary",
    }
    if include_timecourse:
        out["timecourse"] = round_frame(timecourse)
    return out


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
    data,
    time_col="time_bin",
    subject_col="subject",
    condition_col="condition",
    outcome_col=None,
    **kwargs,
):
    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unknown}")

    if outcome_col is None:
        # Historical Python completeness table.
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

    from ._behavioral_r2 import audit_timecourse_grid

    return audit_timecourse_grid(
        data,
        subject_col=subject_col,
        condition_col=condition_col,
        time_col=time_col,
        outcome_col=outcome_col,
    )


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

    # === R3B PUPIL MULTIVERSE SELECTOR ===
    if any(
        key in kwargs
        for key in {
            "multiverse",
            "branch_ids",
            "pupil_col",
            "time_col",
            "group_cols",
            "summarise_windows",
            "windows",
            "keep_outputs",
            "stop_on_error",
        }
    ):
        from ._behavioral_r3b import (
            run_gazepoint_pupil_multiverse as _r3b,
        )

        return _r3b(
            data=data,
            multiverse=kwargs.pop(
                "multiverse",
                registry,
            ),
            branch_ids=kwargs.pop(
                "branch_ids",
                None,
            ),
            pupil_col=kwargs.pop(
                "pupil_col",
                "PUPIL",
            ),
            time_col=kwargs.pop(
                "time_col",
                "TIME",
            ),
            group_cols=kwargs.pop(
                "group_cols",
                None,
            ),
            summarise_windows=kwargs.pop(
                "summarise_windows",
                True,
            ),
            windows=kwargs.pop(
                "windows",
                None,
            ),
            keep_outputs=kwargs.pop(
                "keep_outputs",
                False,
            ),
            stop_on_error=kwargs.pop(
                "stop_on_error",
                False,
            ),
        )

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

    # === R3B AOI MULTIVERSE SELECTOR ===
    if any(
        key in kwargs
        for key in {
            "multiverse",
            "branch_ids",
            "windows",
            "time_col",
            "aoi_col",
            "subject_col",
            "condition_col",
            "group_cols",
            "target_aoi_values",
            "distractor_aoi_values",
            "success_col",
            "outcome_label",
            "keep_outputs",
            "stop_on_error",
        }
    ):
        from ._behavioral_r3b import (
            run_gazepoint_aoi_multiverse as _r3b,
        )

        return _r3b(
            data=data,
            multiverse=kwargs.pop(
                "multiverse",
                specifications,
            ),
            branch_ids=kwargs.pop(
                "branch_ids",
                None,
            ),
            windows=kwargs.pop("windows"),
            time_col=kwargs.pop(
                "time_col",
                "TIME",
            ),
            aoi_col=kwargs.pop(
                "aoi_col",
                "AOI",
            ),
            subject_col=kwargs.pop(
                "subject_col",
                "USER",
            ),
            condition_col=kwargs.pop(
                "condition_col",
                "condition",
            ),
            group_cols=kwargs.pop(
                "group_cols",
                None,
            ),
            target_aoi_values=kwargs.pop("target_aoi_values"),
            distractor_aoi_values=kwargs.pop(
                "distractor_aoi_values",
                None,
            ),
            success_col=kwargs.pop(
                "success_col",
                "success",
            ),
            outcome_label=kwargs.pop(
                "outcome_label",
                "target_aoi",
            ),
            keep_outputs=kwargs.pop(
                "keep_outputs",
                False,
            ),
            stop_on_error=kwargs.pop(
                "stop_on_error",
                False,
            ),
        )

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
