from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3
import gp3tools.aoi as aoi_mod
import gp3tools.pupil as pupil_mod
import gp3tools.qc as qc_mod
import gp3tools.stats as stats_mod


def test_dynamic_aoi_assignment_audit_and_validation():
    samples = pd.DataFrame(
        {
            "TIME": [0.0, 1.0, 2.0],
            "x": [0.5, 0.5, 2.0],
            "y": [0.5, 0.5, 0.5],
        }
    )
    geometry = pd.DataFrame(
        {
            "TIME": [0.0, 1.0, 2.0],
            "aoi": ["A", "B", "C"],
            "xmin": [0.0, 0.0, 0.0],
            "xmax": [1.0, 1.0, 1.0],
            "ymin": [0.0, 0.0, 0.0],
            "ymax": [1.0, 1.0, 1.0],
        }
    )

    out = gp3.add_gazepoint_dynamic_aoi(
        samples,
        geometry,
        time_col="TIME",
        aoi_time_col="TIME",
        x_col="x",
        y_col="y",
    )

    assert out["aoi_current"].tolist() == ["A", "B", "outside"]

    audit = gp3.audit_gazepoint_dynamic_aoi_coverage(out)
    assert audit.loc[0, "n_samples"] == 3
    assert audit.loc[0, "n_assigned"] == 3
    assert audit.loc[0, "n_outside"] == 1
    assert audit.loc[0, "assigned_prop"] == pytest.approx(1.0)

    missing = gp3.audit_gazepoint_dynamic_aoi_coverage(pd.DataFrame({"x": [1, 2]}))
    assert missing.loc[0, "n_assigned"] == 0
    assert missing.loc[0, "assigned_prop"] == pytest.approx(0.0)

    bad_geometry = geometry.drop(columns="xmax")

    with pytest.raises(ValueError, match="Dynamic AOI geometry"):
        gp3.add_gazepoint_dynamic_aoi(
            samples,
            bad_geometry,
            time_col="TIME",
            aoi_time_col="TIME",
            x_col="x",
            y_col="y",
        )


def test_grouped_time_varying_transitions_and_network_metrics():
    df = pd.DataFrame(
        {
            "TIME": [0, 100, 200, 600, 700, 800],
            "subject": ["S1"] * 6,
            "aoi_current": ["A", "B", "A", "B", "C", "B"],
        }
    )

    grouped = gp3.compute_gazepoint_aoi_transition_matrix(
        data=df,
        aoi_col="aoi_current",
        group_cols=["subject"],
        time_col="TIME",
    )

    assert not grouped.empty
    assert {"subject", "value"}.issubset(grouped.columns)

    time_varying = gp3.compute_gazepoint_time_varying_transition_matrix(
        df,
        aoi_col="aoi_current",
        time_col="TIME",
        bin_width=500,
        normalize=True,
    )

    assert not time_varying.empty
    assert "time_bin" in time_varying.columns
    assert set(time_varying["time_bin"].dropna().astype(int)) <= {0, 500}

    matrix = pd.DataFrame(
        [[0.0, 2.0], [1.0, 0.0]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    metrics = gp3.compute_gazepoint_transition_network_metrics(matrix)
    assert set(metrics.columns) == {"node", "in_degree", "out_degree", "pagerank"}
    assert set(metrics["node"]) == {"A", "B"}
    assert np.isfinite(metrics["pagerank"]).all()

    metrics_array = gp3.compute_gazepoint_transition_network_metrics(
        np.array([[0.0, 1.0], [1.0, 0.0]])
    )
    assert len(metrics_array) == 2


def test_scanpath_similarity_anomalies_and_trial_features(monkeypatch):
    coordinate_a = np.array([[0.0, 0.0], [1.0, 1.0]])
    coordinate_b = np.array([[0.0, 0.0], [2.0, 2.0]])

    similarity = gp3.compute_gazepoint_scanpath_similarity(
        coordinate_a,
        coordinate_b,
        method="coordinates",
    )
    assert 0 < similarity <= 1

    empty_similarity = gp3.compute_gazepoint_scanpath_similarity(
        np.empty((0, 2)),
        np.empty((0, 2)),
        method="coordinates",
    )
    assert np.isnan(empty_similarity)

    def varying_complexity(*args, **kwargs):
        return pd.DataFrame({"complexity_index": [1.0, 1.0, 10.0]})

    monkeypatch.setattr(
        aoi_mod,
        "compute_gazepoint_sequence_complexity",
        varying_complexity,
    )

    anomalies = gp3.flag_gazepoint_sequence_anomalies(
        sequence=["A", "B"],
        z_threshold=0.5,
    )
    assert anomalies["anomaly"].any()
    assert "anomaly_score" in anomalies

    def constant_complexity(*args, **kwargs):
        return pd.DataFrame({"complexity_index": [2.0, 2.0]})

    monkeypatch.setattr(
        aoi_mod,
        "compute_gazepoint_sequence_complexity",
        constant_complexity,
    )

    constant = gp3.flag_gazepoint_sequence_anomalies(sequence=["A", "B"])
    assert not constant["anomaly"].any()
    assert (constant["anomaly_score"] == 0).all()

    master = gp3.load_example_master().head(360)

    features = gp3.summarise_gazepoint_aoi_trial_features(
        master,
        aoi_col="aoi_current",
        trial_col="trial_global",
        group_cols=["subject"],
    )

    assert not features.empty
    assert "trial_global" in features.columns


@pytest.mark.parametrize(
    ("policy", "expected"),
    [
        ("mean", [1.0, 2.0, 3.0, np.nan]),
        ("available_eye", [1.0, 2.0, 3.0, np.nan]),
        ("bilateral_mean", [1.0, np.nan, np.nan, np.nan]),
        ("left_only", [1.0, np.nan, 3.0, np.nan]),
        ("right_only", [1.0, 2.0, np.nan, np.nan]),
        ("complete_case", [1.0, np.nan, np.nan, np.nan]),
    ],
)
def test_combine_eye_policies(policy, expected):
    df = pd.DataFrame(
        {
            "left": [1.0, np.nan, 3.0, np.nan],
            "right": [1.0, 2.0, np.nan, np.nan],
        }
    )

    out = gp3.combine_gazepoint_eyes(
        df,
        left_col="left",
        right_col="right",
        policy=policy,
    )

    np.testing.assert_allclose(
        out["pupil_combined"].to_numpy(float),
        np.asarray(expected, float),
        equal_nan=True,
    )

    assert out["pupil_eye_source"].tolist() == [
        "both",
        "right",
        "left",
        "missing",
    ]


def test_combine_eye_policy_validation():
    df = pd.DataFrame({"left": [1.0], "right": [1.0]})

    with pytest.raises(ValueError, match="Unknown eye-combination policy"):
        gp3.combine_gazepoint_eyes(
            df,
            left_col="left",
            right_col="right",
            policy="invalid",
        )


@pytest.mark.parametrize("method", ["moving_average", "mean", "median", "savgol"])
def test_pupil_smoothing_methods(method):
    df = pd.DataFrame(
        {
            "pupil": [1.0, 1.2, np.nan, 1.6, 1.8, 2.0, 2.2],
        }
    )

    out = gp3.smooth_gazepoint_pupil(
        df,
        pupil_col="pupil",
        window=5,
        method=method,
    )

    assert "pupil_smoothed" in out
    assert np.isfinite(out["pupil_smoothed"]).any()


def test_pupil_smoothing_invalid_method():
    with pytest.raises(ValueError, match="Unknown smoothing method"):
        gp3.smooth_gazepoint_pupil(
            pd.DataFrame({"pupil": [1.0, 2.0, 3.0]}),
            pupil_col="pupil",
            method="invalid",
        )


def test_pupil_reliability_without_and_with_subject():
    no_subject = pd.DataFrame({"pupil": [1.0, 2.0, 3.0, 4.0]})

    one = gp3.audit_gazepoint_pupil_reliability(
        no_subject,
        pupil_col="pupil",
    )

    assert len(one) == 1
    assert np.isnan(one.loc[0, "split_half_correlation"])
    assert one.loc[0, "mean_even"] == pytest.approx(2.0)
    assert one.loc[0, "mean_odd"] == pytest.approx(3.0)

    with_subject = pd.DataFrame(
        {
            "subject": ["S1"] * 4 + ["S2"] * 4,
            "pupil": [
                1.0,
                2.0,
                1.0,
                2.0,
                2.0,
                4.0,
                2.0,
                4.0,
            ],
        }
    )

    two = gp3.audit_gazepoint_pupil_reliability(
        with_subject,
        pupil_col="pupil",
        subject_col="subject",
    )

    assert two.loc[0, "n_subjects"] == 2
    assert two.loc[0, "split_half_correlation"] == pytest.approx(1.0)
    assert two.loc[0, "spearman_brown"] == pytest.approx(1.0)


def test_gp_pupil_imputation_short_and_model_paths():
    short = pd.DataFrame(
        {
            "TIME": [0.0, 1.0, 2.0, 3.0],
            "pupil": [1.0, np.nan, np.nan, 2.0],
        }
    )

    short_out = gp3.impute_gazepoint_pupil_gp(
        short,
        pupil_col="pupil",
        time_col="TIME",
    )

    assert short_out["pupil_gp"].isna().sum() == 2

    df = pd.DataFrame(
        {
            "TIME": np.arange(8, dtype=float),
            "pupil": [1.0, 1.2, np.nan, 1.6, 1.8, np.nan, 2.2, 2.4],
        }
    )

    out = gp3.impute_gazepoint_pupil_gp(
        df,
        pupil_col="pupil",
        time_col="TIME",
        max_points=3,
        random_state=1,
    )

    assert out["pupil_gp"].isna().sum() == 0
    assert np.isfinite(out.loc[[2, 5], "pupil_gp"]).all()

    observed = df["pupil"].notna()
    np.testing.assert_allclose(
        out.loc[observed, "pupil_gp"],
        df.loc[observed, "pupil"],
    )


def test_stats_formula_model_data_and_fixation_alignment():
    assert (
        stats_mod._formula_default(pd.DataFrame({"pupil": [1.0], "condition": ["A"]}))
        == "pupil ~ condition"
    )

    assert stats_mod._formula_default(pd.DataFrame({"x": [1.0], "y": [2.0]})) == "y ~ 1"

    assert (
        stats_mod._formula_default(
            pd.DataFrame({"a": [1.0]}),
            outcome="a",
            predictor="a",
        )
        == "a ~ a"
    )

    model_data = stats_mod._model_data(
        pd.DataFrame({"x": [1]}),
        required=["x"],
    )
    assert list(model_data.columns) == ["x"]

    with pytest.raises(ValueError, match="Missing required columns"):
        stats_mod._model_data(
            pd.DataFrame({"x": [1]}),
            required=["missing"],
        )

    aligned = gp3.prepare_gazepoint_fixation_aligned_data(
        pd.DataFrame({"TIME": [0.0, 0.5, 1.0, 2.0]}),
        sample_time_col="TIME",
        window=(-10, 10),
    )

    assert "fixation_aligned_time" in aligned

    aligned_ref = gp3.prepare_gazepoint_fixation_aligned_data(
        pd.DataFrame(
            {
                "TIME": [0.0, 0.5, 1.0],
                "fix_ref": [0.0, 0.25, 0.5],
            }
        ),
        sample_time_col="TIME",
        fixation_time_col="fix_ref",
        window=(-10, 10),
    )

    assert len(aligned_ref) == 3

    with pytest.raises(ValueError, match="sample time column required"):
        gp3.prepare_gazepoint_fixation_aligned_data(pd.DataFrame({"x": [1.0]}))


def test_gca_and_spline_formula_construction(monkeypatch):
    captured = []

    def fake_fit(df, formula=None, family="gaussian", *args, **kwargs):
        captured.append(
            {
                "formula": formula,
                "family": family,
                "args": args,
            }
        )
        return captured[-1]

    monkeypatch.setattr(stats_mod, "_fit_formula", fake_fit)

    prepared = pd.DataFrame(
        {
            "subject": ["S1", "S2", "S3"],
            "outcome": [1.0, 2.0, 3.0],
            "time_poly_1": [-1.0, 0.0, 1.0],
            "time_poly_2": [1.0, 0.0, 1.0],
        }
    )

    def fake_prepare(data, outcome_col=None, order=2, **kwargs):
        return prepared.copy()

    monkeypatch.setattr(
        stats_mod,
        "prepare_gazepoint_gca_data",
        fake_prepare,
    )

    gca = stats_mod.fit_gazepoint_gca(
        prepared,
        order=2,
        subject_col="subject",
    )

    assert gca["formula"] == "outcome ~ time_poly_1 + time_poly_2"
    assert gca["family"] == "gaussian"

    spline = stats_mod._fit_spline(
        pd.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "pupil": [1.0, 1.2, 1.4],
            }
        ),
        outcome_col="pupil",
        time_col="time",
        df_spline=4,
    )

    assert spline["formula"] == "pupil ~ bs(time, df=4, degree=3)"

    fallback = stats_mod._fit_spline(
        pd.DataFrame({"x": [1.0, 2.0]}),
        outcome_col="missing",
        time_col="time",
        formula="x ~ 1",
        family="gaussian",
    )

    assert fallback["formula"] == "x ~ 1"


def test_model_sensitivity_dispatch(monkeypatch):
    aoi_calls = []
    pupil_calls = []

    def fake_aoi(data, formula=None, **kwargs):
        aoi_calls.append(formula)
        return SimpleNamespace(aic=float(len(aoi_calls)))

    def fake_pupil(data, formula=None, **kwargs):
        pupil_calls.append(formula)
        return SimpleNamespace(aic=float(len(pupil_calls)))

    def fake_compare(models):
        return pd.DataFrame(
            {
                "model": np.arange(len(models)),
                "aic": [model.aic for model in models],
            }
        )

    monkeypatch.setattr(
        stats_mod,
        "fit_gazepoint_aoi_window_glmm",
        fake_aoi,
    )
    monkeypatch.setattr(
        stats_mod,
        "fit_gazepoint_pupil_window_lmm",
        fake_pupil,
    )
    monkeypatch.setattr(
        stats_mod,
        "compare_gazepoint_nested_models",
        fake_compare,
    )

    aoi = stats_mod.fit_gazepoint_aoi_model_sensitivity(
        pd.DataFrame({"x": [1]}),
        formulas=["y ~ 1", "y ~ x"],
    )

    pupil = stats_mod.fit_gazepoint_pupil_window_sensitivity(
        pd.DataFrame({"x": [1]}),
        formulas=["y ~ 1", "y ~ x"],
    )

    assert aoi_calls == ["y ~ 1", "y ~ x"]
    assert pupil_calls == ["y ~ 1", "y ~ x"]
    assert len(aoi["models"]) == 2
    assert len(pupil["models"]) == 2
    assert len(aoi["comparison"]) == 2
    assert len(pupil["comparison"]) == 2


def test_model_summary_singularity_and_overdispersion():
    class DummyModel:
        params = pd.Series(
            [1.0, 2.0],
            index=["Intercept", "x"],
        )
        bse = np.array([0.1, 0.2])
        pvalues = np.array([0.01, 0.02])

        def conf_int(self):
            return pd.DataFrame(
                [
                    [0.8, 1.2],
                    [1.6, 2.4],
                ]
            )

    tidy = gp3.tidy_gazepoint_model_summary(DummyModel())

    assert list(tidy["term"]) == ["Intercept", "x"]
    assert {
        "estimate",
        "std_error",
        "p_value",
        "conf_low",
        "conf_high",
    }.issubset(tidy.columns)

    empty = gp3.tidy_gazepoint_model_summary(SimpleNamespace())
    assert empty.empty

    singular = gp3.check_gazepoint_model_singularity(SimpleNamespace(cov_re=np.array([[0.0]])))
    assert not bool(singular.loc[0, "passed"])

    nonsingular = gp3.check_gazepoint_model_singularity(SimpleNamespace(cov_re=np.array([[1.0]])))
    assert bool(nonsingular.loc[0, "passed"])

    class BadCov:
        def __array__(self, *args, **kwargs):
            raise TypeError("bad covariance")

    fallback = gp3.check_gazepoint_model_singularity(SimpleNamespace(cov_re=BadCov()))
    assert bool(fallback.loc[0, "passed"])

    over = gp3.check_gazepoint_model_overdispersion(
        SimpleNamespace(
            resid_pearson=np.array([2.0, 2.0]),
            df_resid=1,
        )
    )
    assert bool(over.loc[0, "overdispersed"])
    assert over.loc[0, "dispersion_ratio"] == pytest.approx(8.0)

    none = gp3.check_gazepoint_model_overdispersion(SimpleNamespace())
    assert np.isnan(none.loc[0, "dispersion_ratio"])
    assert not bool(none.loc[0, "overdispersed"])


def test_cluster_boundaries_and_leave_one_out():
    clusters = pd.DataFrame(
        {
            "start": [10.0, 20.0, 30.0],
            "end": [15.0, 25.0, 35.0],
            "significant": [False, True, True],
        }
    )

    result = {"clusters": clusters}

    assert gp3.estimate_gazepoint_cluster_onset(result) == pytest.approx(20.0)
    assert gp3.estimate_gazepoint_cluster_offset(result) == pytest.approx(35.0)

    nonsig = clusters.assign(significant=False)
    assert np.isnan(gp3.estimate_gazepoint_cluster_onset(nonsig))
    assert np.isnan(gp3.estimate_gazepoint_cluster_offset(nonsig))

    data = pd.DataFrame(
        {
            "subject": ["A", "A", "B", "B", "C", "C"],
            "y": np.arange(6, dtype=float),
        }
    )

    all_subjects = {"A", "B", "C"}

    def fit_function(subset):
        missing = all_subjects - set(subset["subject"])

        if "B" in missing:
            raise RuntimeError("deliberate fit failure")

        return SimpleNamespace(
            aic=100.0 + len(subset),
            converged=True,
        )

    loo = gp3.run_gazepoint_model_leave_one_out(
        data,
        fit_function=fit_function,
        subject_col="subject",
    )

    assert len(loo) == 3

    failed = loo.loc[loo["left_out"] == "B"].iloc[0]
    assert not bool(failed["converged"])
    assert "deliberate fit failure" in failed["error"]

    succeeded = loo.loc[loo["left_out"] != "B"]
    assert succeeded["converged"].all()

    with pytest.raises(ValueError, match="subject column required"):
        gp3.run_gazepoint_model_leave_one_out(
            pd.DataFrame({"x": [1.0]}),
            fit_function=fit_function,
        )


def test_reporting_outputs_models_and_cluster_exports(tmp_path):
    class DummyModel:
        params = pd.Series([1.0], index=["Intercept"])
        bse = np.array([0.1])
        pvalues = np.array([0.05])

        def conf_int(self):
            return pd.DataFrame([[0.8, 1.2]])

    model_paths = gp3.export_gazepoint_model_tables(
        {"first": DummyModel(), "second": DummyModel()},
        tmp_path / "models",
        prefix="fit",
    )

    assert len(model_paths) == 2

    cyclic = []
    cyclic.append(cyclic)

    written = gp3.write_gazepoint_outputs(
        {
            "frame": pd.DataFrame({"x": [1, 2]}),
            "metadata": {"method": "test"},
            "list_value": [1, 2, 3],
            "cyclic": cyclic,
            "ignored": object(),
        },
        tmp_path / "outputs",
        prefix="demo",
    )

    assert {"frame", "metadata", "list_value"}.issubset(written)
    assert "cyclic" not in written
    assert "ignored" not in written

    direct = gp3.write_gazepoint_outputs(
        pd.DataFrame({"x": [1]}),
        tmp_path / "direct",
        prefix="single",
    )

    assert "result" in direct

    cluster_dict = gp3.export_gazepoint_cluster_results(
        {
            "clusters": pd.DataFrame({"start": [1], "end": [2]}),
            "metadata": {"n": 1},
        },
        output_dir=tmp_path / "cluster_dict",
    )

    assert cluster_dict

    cluster_df = gp3.export_gazepoint_cluster_results(
        pd.DataFrame({"start": [1], "end": [2]}),
        output_dir=tmp_path / "cluster_df",
    )

    assert cluster_df


def test_workflow_error_resilience(monkeypatch):
    master = gp3.load_example_master().head(60).copy()

    def fail_sampling(*args, **kwargs):
        raise RuntimeError("sampling failed")

    def fail_tracking(*args, **kwargs):
        raise RuntimeError("tracking failed")

    def fail_preprocessing(*args, **kwargs):
        raise RuntimeError("preprocessing failed")

    monkeypatch.setattr(
        qc_mod,
        "check_sampling_rate",
        fail_sampling,
    )
    monkeypatch.setattr(
        qc_mod,
        "summarise_tracking_quality",
        fail_tracking,
    )
    monkeypatch.setattr(
        pupil_mod,
        "preprocess_gazepoint_signals",
        fail_preprocessing,
    )

    result = gp3.run_gazepoint_workflow(
        data=master,
        create_report=False,
    )

    assert result["sampling_rate_error"] == "sampling failed"
    assert result["tracking_quality_error"] == "tracking failed"
    assert result["preprocessing_error"] == "preprocessing failed"

    with pytest.raises(ValueError, match="Provide `data` or `export_dir`"):
        gp3.run_gazepoint_workflow(
            data=None,
            export_dir=None,
            create_report=False,
        )


def test_uncovered_plotting_paths():
    stability = pd.DataFrame(
        {
            "n_clusters": [2, 3, 4],
            "stability": [0.6, 0.8, 0.7],
        }
    )

    fig = gp3.plot_gazepoint_scanpath_cluster_stability(stability)
    assert fig.axes[0].get_title() == "Scanpath cluster stability"
    plt.close(fig)

    class DummyModel:
        fittedvalues = np.array([1.0, 1.2, 1.1])
        resid = np.array([0.1, -0.1, 0.0])

        def predict(self, data):
            return np.arange(len(data), dtype=float)

    model = DummyModel()

    fig = gp3.plot_gazepoint_model_predictions(
        model,
        data=pd.DataFrame({"x": [1, 2, 3]}),
    )
    assert fig.axes[0].get_title() == "Model predictions"
    plt.close(fig)

    fig = gp3.plot_gazepoint_model_predictions(model)
    assert len(fig.axes[0].lines[0].get_ydata()) == 3
    plt.close(fig)

    fig = gp3.plot_gazepoint_model_residuals(model)
    assert fig.axes[0].get_title() == "Model residuals"
    assert len(fig.axes[0].collections) == 1
    plt.close(fig)

    fig = gp3.plot_gazepoint_multiverse_results(
        pd.DataFrame(
            {
                "model": ["a", "b", "c"],
                "mean_pupil": [1.0, 1.2, 0.9],
            }
        )
    )
    assert fig.axes[0].get_title() == "Multiverse results"
    plt.close(fig)

    fig = gp3.plot_gazepoint_multiverse_results(pd.DataFrame({"model": ["a", "b"]}))
    assert fig.axes[0].get_title() == "Multiverse results"
    assert len(fig.axes[0].lines) == 0
    plt.close(fig)
