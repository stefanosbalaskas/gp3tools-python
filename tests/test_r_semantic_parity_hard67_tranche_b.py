from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3


class DummyModel:
    def __init__(self, llf=-10.0, aic=24.0, bic=26.0, nobs=20, n_params=2):
        self.llf = llf
        self.aic = aic
        self.bic = bic
        self.nobs = nobs
        self.params = pd.Series(
            np.arange(n_params, dtype=float), index=[f"b{i}" for i in range(n_params)]
        )
        self.bse = pd.Series(np.repeat(0.1, n_params), index=self.params.index)
        self.pvalues = pd.Series(np.repeat(0.05, n_params), index=self.params.index)
        self.df_resid = nobs - n_params
        self.resid = np.ones(nobs) * 0.5
        self.converged = True

    def conf_int(self, alpha=0.05):
        return pd.DataFrame(
            np.column_stack([self.params.to_numpy() - 0.2, self.params.to_numpy() + 0.2]),
            index=self.params.index,
        )


def test_trackloss_r_group_flag_and_filter():
    data = pd.DataFrame(
        {
            "id": ["a", "a", "b", "b"],
            "tracking": [1, 0, 1, 1],
        }
    )
    flagged = gp3.clean_gazepoint_by_trackloss(
        data,
        group_cols=["id"],
        tracking_col="tracking",
        max_trackloss=0.25,
        action="flag",
    )
    assert flagged[".gp3_trackloss_rate"].tolist() == [0.5, 0.5, 0.0, 0.0]
    assert flagged[".gp3_trackloss_exclude"].tolist() == [True, True, False, False]
    assert "gp3_trackloss_summary" in flagged.attrs

    filtered = gp3.clean_gazepoint_by_trackloss(
        data,
        group_cols=["id"],
        tracking_col="tracking",
        max_trackloss=0.25,
        action="filter",
    )
    assert filtered["id"].tolist() == ["b", "b"]


def test_trackloss_legacy_still_works():
    data = pd.DataFrame({"TRACKLOSS": [0, 1, 0], "x": [1, 2, 3]})
    out = gp3.clean_gazepoint_by_trackloss(data, validity_col="TRACKLOSS", drop=True)
    assert out["x"].tolist() == [1, 3]


def test_entropy_r_metrics_and_missing_bridge():
    data = pd.DataFrame(
        {
            "id": [1, 1, 1, 1],
            "time": [1, 2, 3, 4],
            "aoi": ["A", None, "B", "B"],
        }
    )
    out = gp3.compute_gazepoint_aoi_entropy(
        data=data,
        aoi_col="aoi",
        group_cols=["id"],
        time_col="time",
        collapse_repeats=True,
    )
    row = out.iloc[0]
    assert row["n_observations"] == 2
    assert row["n_aoi"] == 2
    assert row["n_transitions"] == 1
    assert row["entropy_status"] == "ok"
    assert np.isclose(row["spatial_entropy"], 1.0)


def test_entropy_legacy_sequence_still_works():
    out = gp3.compute_gazepoint_aoi_entropy(sequence=["A", "B", "A"])
    assert {"entropy", "normalized_entropy"} <= set(out.columns)


def test_empirical_logit_r_mode_and_attrs():
    data = pd.DataFrame({"hits": [0, 5, 7], "total": [10, 10, 5]})
    out = gp3.transform_gazepoint_aoi_empirical_logit(
        data,
        numerator_col="hits",
        denominator_col="total",
    )
    assert "aoi_empirical_logit" in out.columns
    assert out["aoi_empirical_logit_status"].tolist() == [
        "complete",
        "complete",
        "proportion_out_of_bounds",
    ]
    assert "gp3_empirical_logit_overview" in out.attrs


def test_empirical_logit_legacy_default():
    out = gp3.transform_gazepoint_aoi_empirical_logit(pd.DataFrame({"success": [1], "total": [4]}))
    assert "empirical_logit" in out.columns


def test_simulate_fixations_r_mode():
    out = gp3.simulate_gazepoint_fixations(
        n_subjects=2,
        n_fix=3,
        coordinate_system="normalized",
        seed=7,
    )
    assert len(out) == 6
    assert {"USER_ID", "FPOGID", "FPOGS", "FPOGD", "FPOGX", "FPOGY"} <= set(out.columns)
    assert out["FPOGX"].between(0, 1).all()


def test_simulate_fixations_legacy_mode():
    out = gp3.simulate_gazepoint_fixations(3, 4, 2)
    assert len(out) == 12
    assert "fixation" in out.columns


def test_simulate_data_r_mode():
    out = gp3.simulate_gazepoint_data(
        n_subjects=2,
        n_trials=2,
        trial_duration_ms=100,
        sampling_rate_hz=20,
        conditions=["control", "treatment"],
        aoi_labels=["target", "other"],
        seed=3,
        include_fixations=True,
    )
    assert isinstance(out, dict)
    assert {"all_gaze", "aoi_windows", "fixations", "metadata"} <= set(out)
    assert out["metadata"]["n_subjects"] == 2
    assert len(out["aoi_windows"]) == 4


def test_simulate_data_legacy_mode():
    out = gp3.simulate_gazepoint_data(
        n_subjects=2,
        n_trials=2,
        samples_per_trial=5,
        random_state=2,
    )
    assert isinstance(out, pd.DataFrame)
    assert len(out) == 20


def test_model_checks_accept_r_arguments():
    model = DummyModel()
    convergence = gp3.check_gazepoint_model_convergence(model, model_name="m1")
    assert convergence.iloc[0]["model_name"] == "m1"
    assert bool(convergence.iloc[0]["passed"])

    overdispersion = gp3.check_gazepoint_model_overdispersion(
        model,
        ratio_threshold=1.2,
        model_name="m1",
    )
    assert overdispersion.iloc[0]["ratio_threshold"] == 1.2

    singularity = gp3.check_gazepoint_model_singularity(
        model,
        tolerance=1e-4,
        model_name="m1",
    )
    assert singularity.iloc[0]["model_name"] == "m1"


def test_nested_models_legacy_and_r_mode():
    first = DummyModel(llf=-10, aic=24, bic=25, n_params=2)
    second = DummyModel(llf=-8, aic=22, bic=24, n_params=3)

    legacy = gp3.compare_gazepoint_nested_models([first, second], labels=["a", "b"])
    assert isinstance(legacy, pd.DataFrame)
    assert legacy["model"].tolist() == ["a", "b"]

    result = gp3.compare_gazepoint_nested_models(
        [first, second],
        model_names=["a", "b"],
        comparison="sequential",
    )
    assert result["_gp3_class"] == "gp3_nested_model_comparison"
    assert result["overview"].iloc[0]["n_models"] == 2
    assert len(result["lrt_table"]) == 1


def test_design_balance_r_mode():
    data = pd.DataFrame(
        {
            "subject": ["s1", "s1", "s2"],
            "condition": ["A", "B", "A"],
            "trial_global": [1, 2, 1],
        }
    )
    out = gp3.audit_gazepoint_design_balance(
        data,
        expected_conditions=["A", "B"],
    )
    assert out["_gp3_class"] == "gp3_design_balance_audit"
    assert out["overview"].iloc[0]["design_balance_status"] == "review"
    assert out["cell_summary"]["design_cell_status"].eq("missing_condition").sum() == 1


def test_design_balance_legacy_group_counts():
    data = pd.DataFrame({"subject": ["s1", "s1"], "condition": ["A", "B"]})
    out = gp3.audit_gazepoint_design_balance(
        data,
        group_cols=("subject", "condition"),
    )
    assert isinstance(out, pd.DataFrame)
    assert out["n"].sum() == 2


def test_multiverse_summary_legacy_and_structured():
    legacy = gp3.summarise_gazepoint_multiverse_results(pd.DataFrame({"status": ["ok", "failed"]}))
    assert legacy.iloc[0]["n_specifications"] == 2

    result = {
        "_gp3_class": "gp3_aoi_multiverse_results",
        "overview": pd.DataFrame(
            [
                {
                    "n_defined_branches": 2,
                    "n_requested_branches": 2,
                    "n_completed_branches": 1,
                    "n_failed_branches": 1,
                    "n_skipped_branches": 0,
                    "multiverse_status": "completed_with_errors",
                }
            ]
        ),
        "branch_results": pd.DataFrame(
            {
                "branch_id": ["a", "b"],
                "branch_label": ["A", "B"],
                "branch_status": ["completed", "failed"],
                "message": [None, "boom"],
            }
        ),
    }
    structured = gp3.summarise_gazepoint_multiverse_results(results={"aoi": result})
    assert structured["_gp3_class"] == "gp3_multiverse_summary"
    assert structured["overview"].iloc[-1]["multiverse_status"] == "completed_with_errors"
    assert len(structured["failure_summary"]) == 1


def test_multiverse_report_legacy_and_structured():
    legacy = gp3.report_gazepoint_multiverse(pd.DataFrame({"status": ["ok", "error"]}))
    assert isinstance(legacy, str)

    data = pd.DataFrame(
        {
            "branch": ["a", "b"],
            "term": ["condition", "condition"],
            "estimate": [0.2, 0.3],
            "p_value": [0.01, 0.2],
            "status": ["ok", "ok"],
        }
    )
    structured = gp3.report_gazepoint_multiverse(data, alpha=0.05)
    assert structured["_gp3_class"] == "gp3_multiverse_report"
    assert len(structured["branch_summary"]) == 2


def test_qc_overview_legacy_and_structured():
    qc = {"sampling": pd.DataFrame({"x": [1]}), "other": None}
    legacy = gp3.report_gazepoint_qc_overview(qc)
    assert isinstance(legacy, str)
    structured = gp3.report_gazepoint_qc_overview(qc, max_objects=2)
    assert structured["_gp3_class"] == "gp3_qc_overview_report"
    assert "report_text" in structured


def test_export_cluster_overwrite_argument(tmp_path):
    outdir = tmp_path / "cluster"
    result = pd.DataFrame({"start": [1], "end": [2]})
    first = gp3.export_gazepoint_cluster_results(result, output_dir=outdir)
    assert first
    with pytest.raises(FileExistsError):
        gp3.export_gazepoint_cluster_results(result, output_dir=outdir)
    second = gp3.export_gazepoint_cluster_results(
        result,
        output_dir=outdir,
        overwrite=True,
    )
    assert second


def test_trackloss_character_boolean_xy_and_validation():
    chars = pd.DataFrame({"tracking": ["valid", "lost", "", None]})
    out = gp3.clean_gazepoint_by_trackloss(
        chars,
        tracking_col="tracking",
        max_trackloss=0.5,
        action="flag",
    )
    assert out[".gp3_trackloss_rate"].iloc[0] == 0.75
    assert out[".gp3_trackloss_exclude"].all()

    booleans = pd.DataFrame({"tracking": pd.Series([True, False, None], dtype="boolean")})
    bool_out = gp3.clean_gazepoint_by_trackloss(
        booleans,
        tracking_col="tracking",
        max_trackloss=1,
    )
    assert np.isclose(bool_out[".gp3_trackloss_rate"].iloc[0], 2 / 3)

    xy = pd.DataFrame({"x": [0.0, 0.2, np.nan], "y": [0.0, 0.3, 0.4]})
    xy_out = gp3.clean_gazepoint_by_trackloss(
        xy,
        x_col="x",
        y_col="y",
        max_trackloss=0.9,
        treat_zero_zero_as_loss=True,
    )
    assert np.isclose(xy_out[".gp3_trackloss_rate"].iloc[0], 2 / 3)

    xy_keep_zero = gp3.clean_gazepoint_by_trackloss(
        xy,
        x_col="x",
        y_col="y",
        max_trackloss=0.9,
        treat_zero_zero_as_loss=False,
    )
    assert np.isclose(xy_keep_zero[".gp3_trackloss_rate"].iloc[0], 1 / 3)

    for kwargs in (
        {"tracking_col": "tracking", "max_trackloss": 2},
        {"tracking_col": "tracking", "action": "bad"},
        {"tracking_col": "missing"},
        {},
        {"x_col": "x", "y_col": "missing"},
        {"group_cols": ["missing"], "tracking_col": "tracking"},
    ):
        data = chars if "x_col" not in kwargs else xy
        with pytest.raises(ValueError):
            gp3.clean_gazepoint_by_trackloss(data, **kwargs)


def test_entropy_r_edge_statuses_and_validation():
    empty = gp3.compute_gazepoint_aoi_entropy(
        data=pd.DataFrame({"aoi": [None, ""]}),
        aoi_col="aoi",
    )
    assert empty.iloc[0]["entropy_status"] == "no_valid_aoi"

    single = gp3.compute_gazepoint_aoi_entropy(
        data=pd.DataFrame({"aoi": ["A"]}),
        aoi_col="aoi",
        log_base=np.e,
    )
    assert single.iloc[0]["entropy_status"] == "no_transitions"
    assert single.iloc[0]["spatial_entropy_norm"] == 0

    missing = gp3.compute_gazepoint_aoi_entropy(
        data=pd.DataFrame({"aoi": ["A", None, "A"]}),
        aoi_col="aoi",
        include_missing=True,
        missing_label="MISSING",
    )
    assert missing.iloc[0]["n_aoi"] == 2
    assert missing.iloc[0]["n_transitions"] == 2

    with pytest.raises(ValueError, match="aoi_col"):
        gp3.compute_gazepoint_aoi_entropy(data=pd.DataFrame({"aoi": ["A"]}))
    with pytest.raises(ValueError, match="Missing columns"):
        gp3.compute_gazepoint_aoi_entropy(
            data=pd.DataFrame({"aoi": ["A"]}), aoi_col="aoi", group_cols=["id"]
        )
    with pytest.raises(ValueError, match="log_base"):
        gp3.compute_gazepoint_aoi_entropy(
            data=pd.DataFrame({"aoi": ["A"]}), aoi_col="aoi", log_base=1
        )
    with pytest.raises(ValueError, match="include_missing"):
        gp3.compute_gazepoint_aoi_entropy(
            data=pd.DataFrame({"aoi": ["A"]}), aoi_col="aoi", include_missing="yes"
        )
    with pytest.raises(ValueError, match="collapse_repeats"):
        gp3.compute_gazepoint_aoi_entropy(
            data=pd.DataFrame({"aoi": ["A"]}), aoi_col="aoi", collapse_repeats="yes"
        )
    with pytest.raises(ValueError, match="missing_label"):
        gp3.compute_gazepoint_aoi_entropy(
            data=pd.DataFrame({"aoi": ["A"]}), aoi_col="aoi", missing_label=""
        )


def test_empirical_logit_proportion_modes_statuses_and_validation():
    observed = gp3.transform_gazepoint_aoi_empirical_logit(
        pd.DataFrame({"prop": [0.25, np.nan], "den": [20, 20]}),
        proportion_col="prop",
        denominator_col="den",
    )
    assert (
        observed.attrs["gp3_empirical_logit_overview"].iloc[0]["denominator_source"]
        == "observed_denominator_from_proportion"
    )
    assert observed["aoi_numerator"].iloc[0] == 5
    # R applies status rules sequentially; the non-finite numerator rule
    # overwrites the earlier non-finite proportion rule for this row.
    assert observed["aoi_empirical_logit_status"].iloc[1] == "missing_or_nonfinite_numerator"

    pseudo = gp3.transform_gazepoint_aoi_empirical_logit(
        pd.DataFrame({"prop": [0.0, 1.0]}),
        proportion_col="prop",
        pseudo_denominator=10,
    )
    assert pseudo["aoi_denominator"].tolist() == [10.0, 10.0]
    assert np.isfinite(pseudo["aoi_empirical_logit"]).all()

    statuses = gp3.transform_gazepoint_aoi_empirical_logit(
        pd.DataFrame(
            {
                "num": [-1.0, 2.0, 5.0, np.inf],
                "den": [5.0, 0.0, 4.0, 5.0],
            }
        ),
        numerator_col="num",
        denominator_col="den",
    )
    assert set(statuses["aoi_empirical_logit_status"]) >= {
        "numerator_exceeds_denominator",
        "proportion_out_of_bounds",
        "missing_or_nonfinite_numerator",
    }
    zero_den = gp3.transform_gazepoint_aoi_empirical_logit(
        pd.DataFrame({"num": [0.0], "den": [0.0]}),
        numerator_col="num",
        denominator_col="den",
    )
    assert zero_den["aoi_empirical_logit_status"].iloc[0] == "invalid_denominator"

    with pytest.raises(ValueError, match="Supply either"):
        gp3.transform_gazepoint_aoi_empirical_logit(pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError, match="positive finite"):
        gp3.transform_gazepoint_aoi_empirical_logit(
            pd.DataFrame({"prop": [0.5]}), proportion_col="prop", correction=0
        )
    with pytest.raises(ValueError, match="already exist"):
        gp3.transform_gazepoint_aoi_empirical_logit(
            pd.DataFrame({"prop": [0.5], "aoi_empirical_logit": [0.0]}),
            proportion_col="prop",
        )
    overwritten = gp3.transform_gazepoint_aoi_empirical_logit(
        pd.DataFrame({"prop": [0.5], "aoi_empirical_logit": [99.0]}),
        proportion_col="prop",
        overwrite=True,
    )
    assert overwritten["aoi_empirical_logit"].iloc[0] != 99


def test_simulate_fixations_pixels_and_validation():
    out = gp3.simulate_gazepoint_fixations(
        n_subjects=1,
        n_fix=2,
        coordinate_system="pixels",
        screen_width=800,
        screen_height=600,
        sd=5,
        seed=2,
    )
    assert len(out) == 2
    assert out["x"].between(0, 800).all()
    assert out["y"].between(0, 600).all()
    assert out["FPOGX"].between(0, 1).all()

    with pytest.raises(ValueError, match="coordinate_system"):
        gp3.simulate_gazepoint_fixations(n_subjects=1, coordinate_system="bad")
    with pytest.raises(ValueError, match="positive integer"):
        gp3.simulate_gazepoint_fixations(n_subjects=0, seed=1)
    with pytest.raises(ValueError, match="non-negative"):
        gp3.simulate_gazepoint_fixations(n_subjects=1, sd=-1, seed=1)
    with pytest.raises(ValueError, match="positive"):
        gp3.simulate_gazepoint_fixations(
            n_subjects=1, screen_width=0, screen_height=10, duration_mean=10, seed=1
        )


def test_simulate_data_no_fixations_and_validation():
    out = gp3.simulate_gazepoint_data(
        n_subjects=1,
        n_trials=1,
        trial_duration_ms=50,
        sampling_rate_hz=20,
        seed=1,
        include_fixations=False,
    )
    assert out["fixations"] is None
    assert len(out["aoi_windows"]) == 1

    bad_calls = [
        {"n_subjects": 0, "seed": 1},
        {"n_subjects": 1, "trial_duration_ms": 0},
        {"n_subjects": 1, "sampling_rate_hz": 0},
        {"n_subjects": 1, "conditions": []},
        {"n_subjects": 1, "aoi_labels": ["only"]},
        {"n_subjects": 1, "aoi_labels": ["A", "B"], "target_aoi": "C"},
    ]
    for kwargs in bad_calls:
        with pytest.raises(ValueError):
            gp3.simulate_gazepoint_data(**kwargs)


def test_model_diagnostic_alternative_paths_and_validation():
    ordinary = SimpleNamespace(params=pd.Series([1.0]))
    conv = gp3.check_gazepoint_model_convergence(ordinary)
    assert conv.iloc[0]["diagnostic_status"] == "not_applicable"

    wrapped = {"model": DummyModel(), "model_name": "wrapped"}
    assert gp3.check_gazepoint_model_convergence(wrapped).iloc[0]["model_name"] == "wrapped"

    unconverged = DummyModel()
    unconverged.converged = False
    assert (
        gp3.check_gazepoint_model_convergence(unconverged).iloc[0]["diagnostic_status"]
        == "convergence_warning"
    )

    with pytest.raises(ValueError):
        gp3.check_gazepoint_model_convergence(None)
    with pytest.raises(ValueError, match="model_name"):
        gp3.check_gazepoint_model_convergence(DummyModel(), model_name="")

    covariance_model = DummyModel()
    covariance_model.cov_re = np.array([[0.0, 0.0], [0.0, 1.0]])
    singular = gp3.check_gazepoint_model_singularity(covariance_model, tolerance=1e-4)
    assert singular.iloc[0]["diagnostic_status"] == "singular_fit"

    nonsingular_model = DummyModel()
    nonsingular_model.cov_re = np.eye(2)
    nonsingular = gp3.check_gazepoint_model_singularity(nonsingular_model)
    assert nonsingular.iloc[0]["diagnostic_status"] == "ok"

    with pytest.raises(ValueError, match="tolerance"):
        gp3.check_gazepoint_model_singularity(DummyModel(), tolerance=0)

    high_dispersion = DummyModel(nobs=5, n_params=1)
    high_dispersion.resid = np.repeat(5.0, 5)
    high_dispersion.df_resid = 4
    dispersion = gp3.check_gazepoint_model_overdispersion(high_dispersion, ratio_threshold=1.2)
    assert dispersion.iloc[0]["diagnostic_status"] == "overdispersed"

    bad_df = DummyModel()
    bad_df.df_resid = 0
    assert (
        gp3.check_gazepoint_model_overdispersion(bad_df).iloc[0]["diagnostic_status"]
        == "insufficient_residual_df"
    )

    unavailable = SimpleNamespace(params=pd.Series([1.0]))
    assert (
        gp3.check_gazepoint_model_overdispersion(unavailable).iloc[0]["diagnostic_status"]
        == "not_applicable"
    )

    with pytest.raises(ValueError, match="ratio_threshold"):
        gp3.check_gazepoint_model_overdispersion(DummyModel(), ratio_threshold=0)


def test_nested_models_statuses_mapping_and_validation():
    one = gp3.compare_gazepoint_nested_models([DummyModel()], model_names=["m1"])
    assert one["overview"].iloc[0]["comparison_status"] == "not_enough_models"

    mapping = gp3.compare_gazepoint_nested_models(
        {"base": DummyModel(llf=-10, n_params=2), "full": DummyModel(llf=-8, n_params=3)},
        comparison="against_first",
    )
    assert mapping["settings"].iloc[0]["value"] == "base, full"
    assert mapping["lrt_table"].iloc[0]["comparison_status"] == "complete"

    reversed_fit = gp3.compare_gazepoint_nested_models(
        [DummyModel(llf=-8, n_params=3), DummyModel(llf=-10, n_params=2)],
        model_names=["full", "small"],
    )
    assert reversed_fit["lrt_table"].iloc[0]["comparison_status"] in {
        "nonpositive_df_difference",
        "negative_lrt_statistic",
    }

    missing_fit = DummyModel()
    missing_fit.llf = np.nan
    result = gp3.compare_gazepoint_nested_models(
        [DummyModel(), missing_fit], model_names=["ok", "bad"]
    )
    assert result["overview"].iloc[0]["comparison_status"] in {"failed", "partial_complete"}

    with pytest.raises(ValueError, match="non-empty"):
        gp3.compare_gazepoint_nested_models([])
    with pytest.raises(ValueError, match="comparison"):
        gp3.compare_gazepoint_nested_models([DummyModel()], comparison="bad")
    with pytest.raises(ValueError, match="model_names"):
        gp3.compare_gazepoint_nested_models([DummyModel(), DummyModel()], model_names=["x", "x"])


def test_design_balance_balanced_alias_imbalance_and_validation():
    balanced = pd.DataFrame(
        {
            "USER_FILE": ["s1", "s1", "s2", "s2"],
            "condition": ["A", "B", "A", "B"],
            "MEDIA_ID": [1, 2, 1, 2],
        }
    )
    ok = gp3.audit_gazepoint_design_balance(
        balanced,
        subject_col="USER_FILE",
        unit_cols=["MEDIA_ID"],
        expected_conditions=["A", "B"],
    )
    assert ok["overview"].iloc[0]["design_balance_status"] == "ok"
    assert ok["settings"].set_index("setting").loc["subject_col", "value"] == "subject"

    imbalanced = pd.DataFrame(
        {
            "subject": ["s1"] * 4,
            "condition": ["A", "A", "A", "B"],
            "trial": [1, 2, 3, 4],
        }
    )
    review = gp3.audit_gazepoint_design_balance(
        imbalanced,
        unit_cols=["trial"],
        expected_conditions=["A", "B"],
        max_condition_ratio=2,
    )
    assert review["subject_summary"].iloc[0]["design_balance_status"] == "condition_count_imbalance"

    few = gp3.audit_gazepoint_design_balance(
        pd.DataFrame({"subject": ["s1", "s1"], "condition": ["A", "B"], "trial": [1, 2]}),
        unit_cols=["trial"],
        min_units_per_condition=2,
    )
    assert few["subject_summary"].iloc[0]["design_balance_status"] == "too_few_units"

    with pytest.raises(ValueError, match="at least one row"):
        gp3.audit_gazepoint_design_balance(pd.DataFrame(columns=["subject", "condition"]))
    with pytest.raises(ValueError, match="missing required"):
        gp3.audit_gazepoint_design_balance(pd.DataFrame({"subject": ["s1"]}))
    with pytest.raises(ValueError, match="positive"):
        gp3.audit_gazepoint_design_balance(
            pd.DataFrame({"subject": ["s1"], "condition": ["A"]}),
            min_units_per_condition=0,
        )
    with pytest.raises(ValueError, match="TRUE or FALSE"):
        gp3.audit_gazepoint_design_balance(
            pd.DataFrame({"subject": ["s1"], "condition": ["A"]}),
            require_all_conditions_per_subject="yes",
        )


def test_multiverse_summary_status_variants_and_validation():
    not_run = {
        "_gp3_class": "gp3_pupil_multiverse_results",
        "overview": {
            "n_defined_branches": 1,
            "n_requested_branches": 0,
            "n_completed_branches": 0,
            "n_failed_branches": 0,
            "n_skipped_branches": 0,
            "multiverse_status": "not_run",
        },
    }
    summary = gp3.summarise_gazepoint_multiverse_results(results={"pupil": not_run})
    assert summary["overview"].iloc[-1]["multiverse_status"] == "not_run"
    assert summary["branch_summary"].empty

    completed = {
        "multiverse_family": "custom",
        "overview": {
            "n_defined_branches": 1,
            "n_requested_branches": 1,
            "n_completed_branches": 1,
            "n_failed_branches": 0,
            "n_skipped_branches": 0,
            "multiverse_status": "completed",
        },
        "branch_results": pd.DataFrame(
            {"branch_id": ["x"], "branch_label": ["X"], "branch_status": ["completed"]}
        ),
    }
    summary2 = gp3.summarise_gazepoint_multiverse_results(completed)
    assert summary2["overview"].iloc[-1]["multiverse_status"] == "completed"
    assert summary2["failure_summary"].empty

    with pytest.raises(ValueError, match="At least one"):
        gp3.summarise_gazepoint_multiverse_results()
    with pytest.raises(ValueError, match="named mapping"):
        gp3.summarise_gazepoint_multiverse_results(results=[])
    with pytest.raises(ValueError, match="mappings"):
        gp3.summarise_gazepoint_multiverse_results("bad")
    with pytest.raises(TypeError):
        gp3.summarise_gazepoint_multiverse_results(pd.DataFrame(), data=pd.DataFrame())


def test_multiverse_report_detection_and_validation():
    missing_fields = pd.DataFrame({"value": [1, 2]})
    out = gp3.report_gazepoint_multiverse(missing_fields, alpha=0.1)
    assert out["branch_summary"].iloc[0]["branch"] == "all"
    assert out["status_summary"].iloc[0]["status"] == "unknown"
    assert out["inferential_summary"].empty

    wrapped = {
        "branch_summary": pd.DataFrame(
            {
                "analysis": ["a"],
                "parameter": ["condition"],
                "effect_size": [0.4],
                "p": [0.01],
                "fit_status": ["ok"],
            }
        )
    }
    out2 = gp3.report_gazepoint_multiverse(wrapped, alpha=0.05)
    assert bool(out2["inferential_summary"].iloc[0]["significant"])

    with pytest.raises(ValueError, match="alpha"):
        gp3.report_gazepoint_multiverse(pd.DataFrame({"x": [1]}), alpha=1)
    with pytest.raises(ValueError, match="not found"):
        gp3.report_gazepoint_multiverse(pd.DataFrame({"x": [1]}), branch_col="missing", alpha=0.05)


def test_qc_overview_statuses_and_validation():
    summary = pd.DataFrame(
        {
            "component": ["a", "b", "c", "d"],
            "status": ["pass", "warn", "fail", "not_available"],
        }
    )
    out = gp3.report_gazepoint_qc_overview(summary, max_objects=2)
    assert out["summary"]["overview"].iloc[0]["qc_overview_status"] in {"warn", "fail"}
    assert "a" in out["object_summary"]["object_name"].tolist()

    with pytest.raises(ValueError, match="positive integer"):
        gp3.report_gazepoint_qc_overview(summary, max_objects=0)
    with pytest.raises(ValueError, match="positive integer"):
        gp3.report_gazepoint_qc_overview(summary, max_objects="bad")


def test_export_cluster_structured_branch(tmp_path):
    outdir = tmp_path / "structured"
    structured = {
        "cluster_summary": pd.DataFrame(
            {"start_time_bin": [1], "end_time_bin": [2], "p_value": [0.01]}
        ),
        "null_distribution": np.array([1.0, 2.0, 3.0]),
        "settings": {"alpha": 0.05},
        "cluster_status": "ok",
    }
    manifest = gp3.export_gazepoint_cluster_results(structured, output_dir=outdir, overwrite=True)
    assert isinstance(manifest, pd.DataFrame)
    assert {"cluster_summary", "null_distribution", "settings"} <= set(manifest["file_type"])
    assert (outdir / "cluster_summary.csv").exists()
    assert (outdir / "null_distribution.csv").exists()
    assert (outdir / "cluster_settings.csv").exists()
