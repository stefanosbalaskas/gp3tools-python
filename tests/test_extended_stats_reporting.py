from pathlib import Path

import numpy as np
import pandas as pd

import gp3tools as gp3


def _timecourse():
    return gp3.simulate_gazepoint_cluster_timecourse_data(
        n_subjects=6, n_time=20, effect_window=(7, 13), random_state=5
    )


def test_stats_preparation_family():
    master = gp3.load_example_master().head(480).copy()
    win = gp3.prepare_gazepoint_pupil_window_model_data(
        master,
        pupil_col="pupil",
        time_col="TIME",
        windows={"w": (0, 1)},
        group_cols=["subject", "condition"],
    )
    assert isinstance(win, pd.DataFrame)
    gamm = gp3.prepare_gazepoint_pupil_gamm_data(master, pupil_col="pupil", time_col="TIME")
    assert len(gamm) == len(master)
    aoi = gp3.prepare_gazepoint_aoi_glmm_data(master, aoi_col="aoi_current", target_aoi="center")
    assert "aoi_success" in aoi
    assert len(
        gp3.prepare_gazepoint_aoi_gamm_data(master, aoi_col="aoi_current", target_aoi="center")
    ) == len(master)
    gca = gp3.prepare_gazepoint_gca_data(
        master, time_col="TIME", outcome_col="pupil", group_cols=["subject"], order=2
    )
    assert any(c.startswith("time_") for c in gca.columns)
    tc = _timecourse()
    prepared = gp3.prepare_gazepoint_timecourse_test_data(tc)
    assert {"subject", "condition", "time_bin", "value"} <= set(prepared)
    assert len(gp3.prepare_gazepoint_cluster_data(tc)) == len(tc)
    assert len(gp3.prepare_gazepoint_hmm_data(master)) == len(master)


def test_model_helpers_and_diagnostics():
    windows = gp3.load_example_data("pupil_windows").copy()
    model = gp3.fit_gazepoint_pupil_window_lmm(
        windows, formula="mean_pupil ~ condition", subject_col="subject"
    )
    tidy = gp3.tidy_gazepoint_model_summary(model)
    assert {"term", "estimate"} <= set(tidy)
    assert len(gp3.summarise_gazepoint_fixed_effects(model)) == len(tidy)
    conv = gp3.check_gazepoint_model_convergence(model)
    singular = gp3.check_gazepoint_model_singularity(model)
    assert "passed" in conv and "passed" in singular
    diag = gp3.diagnose_gazepoint_gamm(model)
    assert "coefficients" in diag
    comp = gp3.compare_gazepoint_nested_models([model, model], labels=["a", "b"])
    assert len(comp) == 2

    em = gp3.summarise_gazepoint_emmeans(windows, factor="condition", outcome="mean_pupil")
    assert "estimate" in em
    fam_binary = gp3.recommend_gazepoint_model_family(
        pd.DataFrame({"y": [0, 1, 0, 1]}), outcome_col="y"
    )
    assert fam_binary["family"].iloc[0] == "binomial"
    fam_count = gp3.recommend_gazepoint_model_family(
        pd.DataFrame({"y": [0, 1, 2, 3]}), outcome_col="y"
    )
    assert fam_count["family"].iloc[0] in {"poisson", "negative_binomial"}
    fam_cont = gp3.recommend_gazepoint_model_family(
        pd.DataFrame({"y": [0.1, 0.4, 0.8]}), outcome_col="y"
    )
    assert fam_cont["family"].iloc[0] == "gaussian"


def test_bayesian_templates_and_readiness():
    template = gp3.create_gazepoint_brms_template(
        formula="y ~ x", family="gaussian", priors=["normal(0,1)"]
    )
    assert template["formula"] == "y ~ x"
    sap = gp3.create_gazepoint_bayesian_sap(chains=4, iter=2000)
    assert isinstance(sap, dict)
    ready = gp3.check_gazepoint_bayesian_readiness(
        pd.DataFrame({"y": np.arange(20), "x": np.arange(20)})
    )
    assert isinstance(ready, pd.DataFrame)


def test_cluster_permutation_family():
    tc = _timecourse()
    result = gp3.run_gazepoint_cluster_permutation(tc, n_permutations=9, random_state=2)
    assert {"clusters", "observed", "null_distribution"} <= set(result)
    for fun in [
        gp3.run_gazepoint_cluster_permutation_anova,
        gp3.run_gazepoint_cluster_permutation_covariate_adjusted,
        gp3.run_gazepoint_cluster_permutation_lmer,
        gp3.run_gazepoint_cluster_permutation_parallel,
        gp3.run_gazepoint_multidimensional_cluster_permutation,
    ]:
        out = fun(tc, n_permutations=3, random_state=2)
        assert "observed" in out
    tfce = gp3.run_gazepoint_tfce(tc, n_permutations=3)
    assert "tfce" in tfce
    sens = gp3.run_gazepoint_cluster_threshold_sensitivity(
        tc, thresholds=(1.0, 1.5), n_permutations=3
    )
    assert len(sens) == 2
    boot = gp3.bootstrap_gazepoint_timecourse(tc, n_boot=5)
    assert {"estimate", "conf_low", "conf_high"} <= set(boot)
    div = gp3.estimate_gazepoint_divergence_point(tc)
    assert "divergence_time" in div
    clusters = gp3.summarise_gazepoint_clusters(result)
    assert isinstance(clusters, pd.DataFrame)
    assert gp3.summarise_gazepoint_time_clusters(result).equals(clusters)
    assert gp3.summarize_gazepoint_time_clusters(result).equals(clusters)
    design = gp3.diagnose_gazepoint_cluster_design(tc)
    assert "n_subjects" in design
    grid = gp3.audit_gazepoint_timecourse_grid(tc)
    assert len(grid) >= 1
    text = gp3.report_gazepoint_cluster_permutation(result)
    assert "Cluster-permutation" in text


def test_window_analysis_and_multiverse():
    df = pd.DataFrame(
        {"condition": ["A"] * 10 + ["B"] * 10, "value": np.r_[np.arange(10), np.arange(10) + 1.0]}
    )
    out = gp3.analyze_gazepoint_window(df, value_col="value", condition_col="condition")
    assert "summary" in out and "test" in out
    master = gp3.load_example_master().head(100).copy()
    multi = gp3.run_gazepoint_pupil_multiverse(
        master, registry=pd.DataFrame([{"interpolate": True}, {"smooth": False}])
    )
    assert len(multi) == 2
    aoi_multi = gp3.run_gazepoint_aoi_multiverse(master, specifications=pd.DataFrame([{}, {}]))
    assert len(aoi_multi) == 2
    summ = gp3.summarise_gazepoint_multiverse_results(aoi_multi)
    assert "n_specifications" in summ


def test_reporting_exports_and_reports(tmp_path: Path):
    master = gp3.load_example_master().head(150).copy()
    tables = {"one": master.head(3), "two": pd.DataFrame({"x": [1, 2]})}
    exported = gp3.export_gazepoint_tables(tables, tmp_path / "tables", prefix="demo")
    assert len(exported) == 2
    written = gp3.write_gazepoint_outputs(tables, tmp_path / "written", prefix="demo")
    assert written
    audit_files = gp3.export_gazepoint_master_audit(master, tmp_path / "audit")
    assert audit_files

    checklist = gp3.create_gazepoint_reporting_checklist(master)
    assert len(checklist) >= 1
    decisions = gp3.create_gazepoint_analysis_decision_audit(
        interpolation="linear", baseline="subtract"
    )
    assert len(decisions) == 2
    report = gp3.create_gazepoint_report(
        {"master": master.head(), "meta": {"a": 1}},
        tmp_path / "report.html",
        metadata={"project": "test"},
    )
    assert report.exists()
    cross = gp3.create_gazepoint_cross_package_report(
        {"master": master.head()}, tmp_path / "cross.html"
    )
    assert cross.exists()

    assert isinstance(gp3.report_gazepoint_missingness(master), str)
    phases = gp3.segment_gazepoint_task_phases(master, time_col="TIME")
    assert isinstance(gp3.report_gazepoint_phase_coverage(phases), str)
    assert isinstance(gp3.report_gazepoint_qc_overview(master), str)
    face = pd.DataFrame({"face_confidence": [0.9, 0.7]})
    assert isinstance(gp3.report_gazepoint_face_qc(face), str)
    assert isinstance(
        gp3.report_gazepoint_multiverse(pd.DataFrame({"status": ["ok", "error"]})), str
    )

    perf = gp3.benchmark_gazepoint_export_performance(master.head(50), repeats=2)
    assert len(perf) == 2
    limits = gp3.gp3tools_performance_limits()
    assert len(limits) >= 1
    perf_path = gp3.write_gazepoint_performance_benchmark(perf, tmp_path / "perf.csv")
    assert perf_path.exists()
    regression = gp3.check_gazepoint_performance_regression(perf, perf, tolerance=0.2)
    assert isinstance(regression, pd.DataFrame)

    workflow = gp3.run_gazepoint_workflow(
        data=master, output_dir=tmp_path / "workflow", create_report=True
    )
    assert "master" in workflow
    summary = gp3.summarise_gazepoint_workflow(workflow)
    assert len(summary) >= 1


def test_plot_and_save_family(tmp_path: Path):
    import matplotlib.pyplot as plt

    plt.close("all")
    master = gp3.load_example_master().head(200).copy()
    master = gp3.flag_gazepoint_pupil(master, pupil_col="pupil")
    phase = gp3.segment_gazepoint_task_phases(master, time_col="TIME")
    tc = _timecourse()
    cluster = gp3.run_gazepoint_cluster_permutation(tc, n_permutations=5)
    events = gp3.compare_gazepoint_event_detectors(
        gp3.simulate_gazepoint_fixations(5, 8), velocity_threshold=2, min_duration_ms=10
    )
    bench = gp3.benchmark_gazepoint_event_detectors(
        gp3.simulate_gazepoint_fixations(5, 8), repeats=2, velocity_threshold=2, min_duration_ms=10
    )

    calls = [
        lambda: gp3.plot_gazepoint_heatmap(master),
        lambda: gp3.plot_gazepoint_scanpath(master),
        lambda: gp3.plot_gazepoint_scanpaths(master, group_col="condition"),
        lambda: gp3.plot_gazepoint_time_series(
            master, x_col="TIME", y_col="pupil", group_col="condition"
        ),
        lambda: gp3.plot_gazepoint_pupil_timecourse(master),
        lambda: gp3.plot_gazepoint_pupil_status(master),
        lambda: gp3.plot_gazepoint_missingness_profile(master),
        lambda: gp3.plot_gazepoint_aoi_timeline(master),
        lambda: gp3.plot_gazepoint_aoi_transition_matrix(master),
        lambda: gp3.plot_transition_heatmap(master),
        lambda: gp3.plot_gazepoint_binocular_diagnostics(master),
        lambda: gp3.plot_sampling_rate(master, time_col="TIME", group_cols=["subject"]),
        lambda: gp3.plot_tracking_quality(master),
        lambda: gp3.plot_gazepoint_phase_timeline(phase),
        lambda: gp3.plot_gazepoint_cluster_results(cluster),
        lambda: gp3.plot_gazepoint_cluster_null_distribution(cluster),
        lambda: gp3.plot_gazepoint_cluster_permutation(cluster),
        lambda: gp3.plot_gazepoint_event_detector_agreement(events),
        lambda: gp3.plot_gazepoint_event_detector_benchmark(bench),
    ]
    figs = []
    for call in calls:
        fig = call()
        assert fig is not None
        figs.append(fig)
    saved = gp3.save_gazepoint_plots(
        {f"p{i}": f for i, f in enumerate(figs[:3])}, tmp_path / "plots"
    )
    assert saved
    png = gp3.export_gazepoint_heatmap_png(master, tmp_path / "heat.png")
    assert Path(png).exists()
    for fig in figs:
        plt.close(fig)
