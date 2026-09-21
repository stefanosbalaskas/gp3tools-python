import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3tools.reporting as reporting
import gp3tools.stats as stats_mod


def test_reporting_final_validation_and_dispatch_paths(monkeypatch, tmp_path):
    with pytest.raises(TypeError, match="objects must"):
        reporting._gp3_reporting_r_object_summary(object())

    audit = reporting.create_gazepoint_analysis_decision_audit(
        results={"custom": object()},
        branch_roles=None,
        diagnostics_required=False,
    )
    assert audit["branch_audit"].loc[0, "object_type"] == "object"
    assert audit["branch_audit"].loc[0, "decision_type"] == "unknown"

    import gp3tools._behavioral_r3b as r3b

    sentinel = tmp_path / "r3b.html"
    monkeypatch.setattr(
        r3b,
        "create_gazepoint_report",
        lambda **kwargs: sentinel,
    )
    routed = reporting.create_gazepoint_report(
        {"sampling": pd.DataFrame({"x": [1]})},
        output_file=tmp_path / "ignored.html",
        overwrite=True,
    )
    assert routed == sentinel

    direct = reporting.report_gazepoint_qc_overview(
        pd.DataFrame(
            {
                "object_name": ["x"],
                "qc_status": ["pass"],
            }
        ),
        max_objects=1,
    )
    assert direct["summary"]["overview"].loc[0, "n_pass"] == 1

    import gp3tools.qc as qc

    monkeypatch.setattr(
        qc,
        "summarise_gazepoint_qc_status",
        lambda data: pd.DataFrame(index=[0]),
    )
    fallback = reporting.report_gazepoint_qc_overview(
        object(),
        max_objects=1,
    )
    assert fallback["object_summary"].loc[0, "object_name"] == "object_1"
    assert fallback["object_summary"].loc[0, "qc_status"] == "unknown"

    markdown = reporting.report_gazepoint_face_qc(
        checklist=pd.DataFrame(index=[0]),
        output="markdown",
    )
    assert "_No columns._" in markdown

    multiverse = reporting.report_gazepoint_multiverse(
        {
            "branch_results": pd.DataFrame(
                {
                    "estimate": [0.2],
                    "p_value": [0.2],
                }
            )
        },
        alpha=0.05,
    )
    assert len(multiverse["branch_summary"]) == 1

    with pytest.raises(ValueError, match="missing was not found"):
        reporting.report_gazepoint_multiverse(
            pd.DataFrame({"estimate": [1.0]}),
            branch_col="missing",
            alpha=0.05,
        )

    assert reporting._gp3_perf_summarise_trials(pd.DataFrame()).empty


def _r_performance_summary():
    return pd.DataFrame(
        {
            "operation": ["import"],
            "total_rows": [1000],
            "n_files": [1],
            "median_elapsed_s": [1.0],
            "median_heap_delta_mb": [10.0],
            "n_success": [1],
            "n_trials": [1],
        }
    )


def test_reporting_performance_final_paths(tmp_path):
    legacy = pd.DataFrame({"elapsed_seconds": [1.0]})

    with pytest.raises(TypeError, match="either current or x"):
        reporting.check_gazepoint_performance_regression(
            current=legacy,
            x=_r_performance_summary(),
        )

    with pytest.raises(TypeError, match="current and baseline"):
        reporting.check_gazepoint_performance_regression(
            current=pd.DataFrame({"other": [1]}),
        )

    with pytest.raises(TypeError):
        reporting.check_gazepoint_performance_regression(
            current=object(),
            baseline=legacy,
        )

    with pytest.raises(ValueError, match="current is missing metric"):
        reporting.check_gazepoint_performance_regression(
            current=pd.DataFrame({"x": [1]}),
            baseline=legacy,
        )

    with pytest.raises(ValueError, match="baseline is missing metric"):
        reporting.check_gazepoint_performance_regression(
            current=legacy,
            baseline=pd.DataFrame({"x": [1]}),
        )

    with pytest.raises(ValueError, match="x is missing required columns"):
        reporting.check_gazepoint_performance_regression(
            x=pd.DataFrame({"operation": ["import"]}),
        )

    with pytest.raises(ValueError, match="baseline is missing required columns"):
        reporting.check_gazepoint_performance_regression(
            x=_r_performance_summary(),
            baseline=pd.DataFrame({"operation": ["import"]}),
        )

    with pytest.raises(ValueError, match="supplied together"):
        reporting.write_gazepoint_performance_benchmark(
            x={},
            output_dir=None,
        )

    with pytest.raises(TypeError, match="data cannot be combined"):
        reporting.write_gazepoint_performance_benchmark(
            data=pd.DataFrame({"x": [1]}),
            x={},
            output_dir=tmp_path,
        )

    with pytest.raises(TypeError, match="benchmark dictionary"):
        reporting.write_gazepoint_performance_benchmark(
            x=[],
            output_dir=tmp_path,
        )

    with pytest.raises(TypeError, match="regression"):
        reporting.write_gazepoint_performance_benchmark(
            x={"regression": []},
            output_dir=tmp_path,
        )

    with pytest.raises(ValueError, match="benchmark object is missing"):
        reporting.write_gazepoint_performance_benchmark(
            x={"regression": {}},
            output_dir=tmp_path,
        )

    benchmark = {
        "trials": pd.DataFrame({"x": [1]}),
        "summary": pd.DataFrame({"x": [1]}),
        "regression": {
            "checks": pd.DataFrame({"x": [1]}),
            "evaluated": pd.DataFrame({"x": [1]}),
        },
    }
    with pytest.raises(ValueError, match="prefix"):
        reporting.write_gazepoint_performance_benchmark(
            x=benchmark,
            output_dir=tmp_path / "bench",
            prefix="",
        )

    with pytest.raises(TypeError, match="data is required"):
        reporting.write_gazepoint_performance_benchmark()


def test_reporting_workflow_cross_package_cluster_and_dashboard(
    monkeypatch,
    tmp_path,
):
    import gp3tools.io as io_mod
    import gp3tools.pupil as pupil
    import gp3tools.qc as qc
    import gp3tools.stats as stats_mod_local

    source = pd.DataFrame({"TIME": [0.0], "FPOGX": [0.5], "FPOGY": [0.5]})
    monkeypatch.setattr(io_mod, "read_gazepoint_folder", lambda *a, **k: source)
    monkeypatch.setattr(qc, "create_gazepoint_master", lambda data: data.copy())
    monkeypatch.setattr(qc, "audit_gazepoint_master", lambda data: {})
    monkeypatch.setattr(qc, "check_sampling_rate", lambda data: pd.DataFrame({"hz": [60]}))
    monkeypatch.setattr(
        qc,
        "summarise_tracking_quality",
        lambda data: pd.DataFrame({"valid": [1.0]}),
    )
    monkeypatch.setattr(pupil, "preprocess_gazepoint_signals", lambda data: data.copy())

    workflow = reporting.run_gazepoint_workflow(
        export_dir=tmp_path / "exports",
    )
    assert "master" in workflow

    with pytest.raises(TypeError, match="Unexpected keyword"):
        reporting.create_gazepoint_cross_package_report(
            {
                "audit": {},
                "report_text": "x",
            },
            impossible=True,
        )

    monkeypatch.setattr(
        stats_mod_local,
        "summarize_gazepoint_time_clusters",
        lambda result: pd.DataFrame({"cluster_id": [1]}),
    )
    monkeypatch.setattr(
        stats_mod_local,
        "report_gazepoint_cluster_permutation",
        lambda result: {"report_text": "cluster report"},
    )

    exported = reporting.export_gazepoint_cluster_results(
        {
            "cluster_summary": pd.DataFrame({"cluster_id": [1]}),
            "null_distribution": np.array([0.1, 0.2]),
            "settings": {"alpha": 0.05},
        },
        output_dir=tmp_path / "clusters",
        overwrite=True,
    )
    assert {"cluster_summary", "null_distribution", "report_text", "settings"}.issubset(
        set(exported["file_type"])
    )

    monkeypatch.setitem(sys.modules, "shiny", None)
    unavailable = reporting.launch_gazepoint_qc_dashboard(pd.DataFrame({"x": [1]}))
    assert unavailable["status"] == "optional-backend-unavailable"


def test_stats_final_wrapper_and_formula_paths(monkeypatch):
    frame = pd.DataFrame(
        {
            "x": [0.0, 1.0, 2.0, 3.0],
            "y": [0.0, 1.0, 2.0, 3.0],
            "time": [0.0, 1.0, 2.0, 3.0],
        }
    )

    with pytest.raises(TypeError, match="Unexpected keyword"):
        stats_mod.prepare_gazepoint_fixation_aligned_data(
            frame,
            sample_time_col="time",
            bad=True,
        )

    with pytest.raises(TypeError, match="Unexpected keyword"):
        stats_mod.prepare_gazepoint_fixation_aligned_data(
            frame,
            time_col="time",
            bad=True,
        )

    nb = stats_mod._fit_formula(
        pd.DataFrame(
            {
                "y": [0, 1, 2, 3, 2, 4, 1, 3],
                "x": np.arange(8),
            }
        ),
        formula="y ~ x",
        family="negativebinomial",
    )
    assert hasattr(nb, "params")

    monkeypatch.setattr(
        stats_mod,
        "fit_gazepoint_pupil_window_lmm",
        lambda *a, **k: "multimodal",
    )
    assert (
        stats_mod.fit_gazepoint_multimodal_response_model(
            pd.DataFrame({"x": [1]})
        )
        == "multimodal"
    )

    monkeypatch.setattr(
        stats_mod,
        "_fit_formula",
        lambda *a, **k: ("nb", a, k),
    )
    assert stats_mod.fit_gazepoint_transition_count_nb_sensitivity(
        pd.DataFrame({"x": [1]})
    )[0] == "nb"

    monkeypatch.setattr(
        stats_mod,
        "fit_gazepoint_brms_model",
        lambda *a, **k: k,
    )
    assert (
        stats_mod.fit_gazepoint_aoi_brms(
            pd.DataFrame({"y": [0, 1]}),
            family="binomial",
        )["family"]
        == "binomial"
    )


def test_stats_final_model_summary_paths(monkeypatch):
    fake_arviz = types.ModuleType("arviz")
    fake_arviz.summary = lambda idata: pd.DataFrame(
        {"mean": [1.0]},
        index=["beta"],
    )
    monkeypatch.setitem(sys.modules, "arviz", fake_arviz)

    arviz_summary = stats_mod._gp3_model_legacy_summary(
        {"idata": object()}
    )
    assert arviz_summary.loc[0, "term"] == "beta"

    class NoStatistic:
        params = pd.Series([1.0], index=["x"])
        bse = np.array([0.5])
        pvalues = np.array([0.1])
        df_resid = 10

    fallback = stats_mod._gp3_model_fixed_effects_r(
        NoStatistic(),
        "m",
        0.95,
        False,
        False,
    )
    assert np.isfinite(fallback.loc[0, "statistic"])

    class WrongStatistic(NoStatistic):
        tvalues = np.array([1.0, 2.0])

    wrong = stats_mod._gp3_model_fixed_effects_r(
        WrongStatistic(),
        "m",
        0.95,
        False,
        False,
    )
    assert pd.isna(wrong.loc[0, "statistic"])

    class TypeErrorCI(NoStatistic):
        def conf_int(self, alpha=None):
            if alpha is not None:
                raise TypeError("legacy signature")
            return np.array([[0.5, 1.5]])

    legacy_ci = stats_mod._gp3_model_fixed_effects_r(
        TypeErrorCI(),
        "m",
        0.95,
        False,
        False,
    )
    assert legacy_ci.loc[0, "conf_low"] == pytest.approx(0.5)

    fitted = stats_mod._fit_formula(
        pd.DataFrame(
            {
                "y": [1.0, 2.0, 1.5, 2.5, 3.0, 2.0],
                "x": [0, 1, 0, 1, 0, 1],
            }
        ),
        formula="y ~ x",
    )
    diagnosed = stats_mod.diagnose_gazepoint_gamm(
        {"a": fitted, "b": fitted},
        model_name="set",
        check_convergence=False,
        check_basis=False,
        check_overdispersion=False,
        use_dharma=False,
    )
    assert diagnosed["settings"]["n_models"] == 2


def _fake_comparison_model(llf, params):
    return SimpleNamespace(
        llf=llf,
        params=params,
        nobs=10,
        aic=1.0,
        bic=2.0,
    )


def test_stats_final_comparison_and_window_paths():
    extraction = stats_mod.compare_gazepoint_nested_models(
        [
            _fake_comparison_model(np.nan, [1.0]),
            _fake_comparison_model(2.0, [1.0, 2.0]),
        ],
        comparison="sequential",
    )
    assert extraction["lrt_table"].loc[0, "comparison_status"] == "model_extraction_error"

    missing = stats_mod.compare_gazepoint_nested_models(
        [
            _fake_comparison_model(1.0, 1.0),
            _fake_comparison_model(2.0, 2.0),
        ],
        comparison="sequential",
    )
    assert missing["lrt_table"].loc[0, "comparison_status"] == "missing_lrt_components"

    legacy = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    result = stats_mod.analyze_gazepoint_window(
        data=legacy,
        value_col="value",
        group_col="group",
    )
    assert result["summary"].shape[0] == 1

    with pytest.raises(ValueError, match="Missing value column"):
        stats_mod.analyze_gazepoint_window(
            data=pd.DataFrame({"x": [1]}),
            value_col="missing",
        )

    with pytest.raises(ValueError, match="Missing condition column"):
        stats_mod.analyze_gazepoint_window(
            data=pd.DataFrame({"value": [1.0]}),
            value_col="value",
            condition_col="condition",
        )

    normal = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S1", "S1"],
            "condition": ["A", "A", "A", "A"],
            "TIME": [0.0, 1.0, 2.0, 3.0],
            "value": [np.nan, np.nan, np.nan, np.nan],
        }
    )
    windows = stats_mod.analyze_gazepoint_window(
        normal,
        by=["subject"],
        condition_col="condition",
        value_cols=["value"],
        include_partial=True,
        window_size=1,
        step=1,
        window_unit="native",
    )
    assert windows["value_mean"].isna().all()

    no_divergence = stats_mod.estimate_gazepoint_divergence_point(
        pd.DataFrame(
            {
                "time_bin": [0, 0, 1, 1, 2, 2],
                "condition": ["A", "B"] * 3,
                "value": [1.0, 1.0] * 3,
            }
        ),
        min_run=2,
    )
    assert pd.isna(no_divergence.loc[0, "divergence_time"])


def test_stats_final_cluster_grid_and_multiverse_paths():
    with pytest.raises(ValueError, match="between 0 and 1"):
        stats_mod.summarize_gazepoint_time_clusters(
            {"clusters": pd.DataFrame()},
            alpha=2,
        )

    with pytest.raises(ValueError, match="missing required column"):
        stats_mod.summarize_gazepoint_time_clusters(
            {
                "clusters": pd.DataFrame(
                    {
                        "cluster_id": [1],
                        "start_time_bin": [0],
                    }
                )
            },
            alpha=0.05,
        )

    with pytest.raises(TypeError, match="Unexpected keyword"):
        stats_mod.audit_gazepoint_timecourse_grid(
            pd.DataFrame({"time_bin": [0]}),
            impossible=True,
        )

    cautious = stats_mod.summarise_gazepoint_multiverse_results(
        results={
            "x": {
                "overview": {
                    "n_defined_branches": 1,
                    "n_requested_branches": 1,
                    "n_completed_branches": 1,
                    "n_failed_branches": 0,
                    "n_skipped_branches": 0,
                    "multiverse_status": "partial",
                },
                "branch_results": pd.DataFrame(
                    {
                        "branch_id": [1],
                        "branch_label": ["a"],
                        "branch_status": ["ok"],
                    }
                ),
            }
        }
    )
    assert cautious["overview"].iloc[-1]["multiverse_status"] == "completed_with_cautions"

    completed = stats_mod.summarise_gazepoint_multiverse_results(
        results={
            "x": {
                "overview": pd.DataFrame(
                    [
                        {
                            "n_defined_branches": 1,
                            "n_requested_branches": 1,
                            "n_completed_branches": 1,
                            "n_failed_branches": 0,
                            "n_skipped_branches": 0,
                            "multiverse_status": "completed",
                        }
                    ]
                ),
                "branch_results": pd.DataFrame(
                    {
                        "branch_id": [1],
                        "branch_label": ["a"],
                        "branch_status": ["ok"],
                    }
                ),
            }
        }
    )
    assert completed["overview"].iloc[-1]["multiverse_status"] == "completed"


def test_reporting_and_stats_last_batch1_lines(monkeypatch):
    explicit = reporting.report_gazepoint_multiverse(
        pd.DataFrame(
            {
                "branch": ["a"],
                "status": ["ok"],
                "estimate": [0.2],
                "p_value": [0.5],
            }
        ),
        branch_col="branch",
        alpha=0.05,
    )
    assert explicit["branch_summary"].loc[0, "branch"] == "a"

    class DoubleFailCI:
        params = pd.Series([1.0], index=["x"])
        bse = np.array([0.2])
        pvalues = np.array([0.2])

        def conf_int(self, alpha=None):
            if alpha is not None:
                raise TypeError("legacy")
            raise RuntimeError("unavailable")

    fallback = stats_mod._gp3_model_fixed_effects_r(
        DoubleFailCI(),
        "m",
        0.95,
        False,
        False,
    )
    assert np.isfinite(fallback.loc[0, "conf_low"])

    too_short = stats_mod.analyze_gazepoint_window(
        pd.DataFrame(
            {
                "TIME": [0.0, 1.0],
                "value": [1.0, 2.0],
            }
        ),
        by=None,
        value_cols=["value"],
        window_size=100.0,
        step=1.0,
        window_unit="native",
        include_partial=False,
    )
    assert too_short.empty

    unknown_overview = stats_mod.summarise_gazepoint_multiverse_results(
        results={"x": {"_gp3_class": "custom"}}
    )
    assert unknown_overview["overview"].iloc[-1]["multiverse_status"] == "not_run"

    monkeypatch.setattr(
        stats_mod,
        "_fit_spline",
        lambda *a, **k: "pfe",
    )
    assert stats_mod.fit_gazepoint_pupil_pfe_gamm(
        pd.DataFrame({"x": [1]})
    ) == "pfe"


def test_reporting_dashboard_executes_summary(monkeypatch):
    fake = types.ModuleType("shiny")

    class FakeUI:
        @staticmethod
        def h2(value):
            return value

        @staticmethod
        def output_text_verbatim(value):
            return value

        @staticmethod
        def page_fluid(*args):
            return args

    class FakeRender:
        @staticmethod
        def text(fn):
            return fn

    class FakeApp:
        def __init__(self, ui, server):
            self.ui = ui
            self.server = server

    fake.App = FakeApp
    fake.render = FakeRender()
    fake.ui = FakeUI()
    monkeypatch.setitem(sys.modules, "shiny", fake)

    app = reporting.launch_gazepoint_qc_dashboard(
        pd.DataFrame({"x": [1.0, np.nan]})
    )

    captured = {}

    def output(fn):
        captured["summary"] = fn()
        return fn

    app.server(None, output, None)

    assert "Rows: 2" in captured["summary"]
    assert "Missing cells: 1" in captured["summary"]
