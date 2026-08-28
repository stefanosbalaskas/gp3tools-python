import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf

import gp3tools as gp3


def _ols():
    df = pd.DataFrame(
        {
            "y": [1.0, 2.1, 2.9, 4.2, 5.1, 5.9, 7.2, 8.1],
            "x": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )
    return smf.ols("y ~ x", data=df).fit()


def _glm():
    df = pd.DataFrame(
        {
            "y": [0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
            "x": np.linspace(-1.0, 1.0, 10),
        }
    )
    return smf.glm("y ~ x", data=df, family=sm.families.Binomial()).fit()


def test_legacy_model_summary_is_preserved():
    model = _ols()
    out = gp3.tidy_gazepoint_model_summary(model)
    assert isinstance(out, pd.DataFrame)
    assert {"term", "estimate", "std_error", "p_value"}.issubset(out.columns)


def test_r_tidy_model_summary_structure_and_settings():
    model = _ols()
    out = gp3.tidy_gazepoint_model_summary(
        model,
        model_name="ols_primary",
        conf_level=0.90,
        include_diagnostics=False,
        seed=321,
    )
    assert set(out) == {"overview", "model_info", "fixed_effects", "diagnostics", "settings"}
    assert out["overview"].loc[0, "model_name"] == "ols_primary"
    assert out["overview"].loc[0, "summary_status"] == "ok"
    assert out["settings"]["conf_level"] == pytest.approx(0.90)
    assert out["settings"]["seed"] == 321
    assert out["diagnostics"]["overview"].loc[0, "diagnostic_status"] == "skipped_disabled"


def test_r_fixed_effects_exponentiation_and_drop_intercept():
    model = _glm()
    out = gp3.summarise_gazepoint_fixed_effects(
        model,
        model_name="logit",
        exponentiate=True,
        drop_intercept=True,
    )
    assert isinstance(out, pd.DataFrame)
    assert set(
        [
            "model_name",
            "model_class",
            "term",
            "estimate",
            "std_error",
            "statistic",
            "p_value",
            "conf_low",
            "conf_high",
            "response_scale",
            "significance",
            "diagnostic_status",
        ]
    ).issubset(out.columns)
    assert "Intercept" not in out["term"].tolist()
    assert (out["estimate"] > 0).all()
    assert set(out["response_scale"]) == {"exponentiated"}
    assert set(out["diagnostic_status"]) == {"ok"}


def test_r_glmm_bundle_has_all_components_and_skips_dharma_cleanly():
    model = _ols()
    out = gp3.diagnose_gazepoint_glmm(
        model,
        model_name="ols_diag",
        check_convergence=True,
        check_singularity=True,
        check_overdispersion=True,
        use_dharma=False,
        dharma_simulations=101,
        seed=7,
    )
    assert set(out) == {
        "overview",
        "convergence",
        "singularity",
        "overdispersion",
        "dharma",
        "settings",
    }
    assert out["overview"].loc[0, "model_name"] == "ols_diag"
    assert out["dharma"].loc[0, "dharma_status"] == "skipped_disabled"
    assert out["settings"]["n_models"] == 1
    assert out["settings"]["dharma_simulations"] == 101


def test_r_glmm_disabled_checks_return_structured_skipped_rows():
    model = _ols()
    out = gp3.diagnose_gazepoint_glmm(
        model,
        check_convergence=False,
        check_singularity=False,
        check_overdispersion=False,
        use_dharma=False,
    )
    assert out["convergence"].loc[0, "diagnostic_status"] == "skipped_disabled"
    assert out["singularity"].loc[0, "diagnostic_status"] == "skipped_disabled"
    assert out["overdispersion"].loc[0, "diagnostic_status"] == "skipped_disabled"


def test_r_glmm_named_collection_is_combined():
    models = {"a": _ols(), "b": _glm()}
    out = gp3.diagnose_gazepoint_glmm(models, use_dharma=False)
    assert out["settings"]["n_models"] == 2
    assert out["overview"]["model_name"].tolist() == ["a", "b"]
    assert len(out["convergence"]) == 2


def test_r_gamm_bundle_basis_not_applicable_for_native_ols():
    model = _ols()
    out = gp3.diagnose_gazepoint_gamm(
        model,
        model_name="native_gam_adapter",
        use_dharma=False,
    )
    assert set(out) == {
        "overview",
        "convergence",
        "basis",
        "overdispersion",
        "dharma",
        "settings",
    }
    assert out["basis"].loc[0, "basis_status"] == "not_applicable"
    assert out["overview"].loc[0, "basis_status"] == "not_applicable"


def test_r_gamm_can_consume_native_basis_diagnostic_table():
    model = _ols()
    model.basis_diagnostics = pd.DataFrame(
        {
            "smooth": ["s(time)", "s(subject)"],
            "k_index": [0.8, 1.0],
            "edf": [4.2, 2.0],
            "k_prime": [5.0, 3.0],
            "p_value": [0.01, 0.50],
        }
    )
    out = gp3.diagnose_gazepoint_gamm(model, use_dharma=False)
    assert out["basis"]["basis_status"].tolist() == ["basis_warning", "ok"]
    assert out["overview"].loc[0, "basis_status"] == "basis_warning"
    assert out["overview"].loc[0, "diagnostic_status"] == "diagnostic_warning"


def test_r_tidy_with_diagnostics_uses_structured_bundle():
    model = _glm()
    out = gp3.tidy_gazepoint_model_summary(
        model,
        model_name="binomial",
        include_diagnostics=True,
        use_dharma=False,
    )
    assert isinstance(out["diagnostics"], dict)
    assert "overview" in out["diagnostics"]
    assert out["model_info"].loc[0, "model_family"].lower() == "binomial"


def test_r_model_argument_validation():
    model = _ols()
    with pytest.raises(ValueError, match="conf_level"):
        gp3.tidy_gazepoint_model_summary(model, conf_level=1.0)
    with pytest.raises(ValueError, match="check_convergence"):
        gp3.diagnose_gazepoint_glmm(model, check_convergence="yes")
    with pytest.raises(ValueError, match="dharma_simulations"):
        gp3.diagnose_gazepoint_gamm(model, dharma_simulations=0)


def test_legacy_diagnostics_interfaces_remain_available():
    model = _ols()
    glmm = gp3.diagnose_gazepoint_glmm(model)
    gamm = gp3.diagnose_gazepoint_gamm(model)
    assert set(glmm) == {"convergence", "overdispersion", "coefficients"}
    assert set(gamm) == {"convergence", "coefficients"}


def test_r_dharma_requested_and_basis_disabled_are_structured():
    model = _ols()
    glmm = gp3.diagnose_gazepoint_glmm(model, use_dharma=True, seed=99)
    assert glmm["dharma"].loc[0, "dharma_status"] == "skipped_missing_package"
    gamm = gp3.diagnose_gazepoint_gamm(model, check_basis=False, use_dharma=False)
    assert gamm["basis"].loc[0, "basis_status"] == "skipped_disabled"


def test_r_model_collection_list_names_and_invalid_seed():
    out = gp3.diagnose_gazepoint_glmm([_ols(), _ols()], use_dharma=False)
    assert out["overview"]["model_name"].tolist() == ["model_1", "model_2"]
    with pytest.raises(ValueError, match="seed"):
        gp3.diagnose_gazepoint_glmm(_ols(), seed=np.nan)


def test_r_fixed_effects_unsupported_model_has_structured_status():
    class Unsupported:
        pass

    out = gp3.summarise_gazepoint_fixed_effects(
        Unsupported(),
        model_name="unsupported",
    )
    assert out.loc[0, "diagnostic_status"] == "unsupported_model_class"
    assert out.loc[0, "model_name"] == "unsupported"


def test_r_fixed_effects_all_intercept_filter_yields_not_available():
    class InterceptOnly:
        params = pd.Series([0.5], index=["Intercept"])
        bse = pd.Series([0.1], index=["Intercept"])
        pvalues = pd.Series([0.01], index=["Intercept"])
        tvalues = pd.Series([5.0], index=["Intercept"])
        df_resid = 9.0

        def conf_int(self, alpha=0.05):
            return pd.DataFrame([[0.3, 0.7]], index=["Intercept"])

    out = gp3.summarise_gazepoint_fixed_effects(
        InterceptOnly(),
        drop_intercept=True,
    )
    assert out.loc[0, "diagnostic_status"] == "not_available"
