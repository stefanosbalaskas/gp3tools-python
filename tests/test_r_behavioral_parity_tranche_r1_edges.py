from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3
import gp3tools.aoi as aoi_mod


def test_r1_legacy_sequence_anomaly_contract(monkeypatch):
    def varying_complexity(*args, **kwargs):
        assert kwargs["sequence"] == ["A", "B"]
        return pd.DataFrame({"complexity_index": [1.0, 1.0, 10.0]})

    monkeypatch.setattr(aoi_mod, "compute_gazepoint_sequence_complexity", varying_complexity)
    out = gp3.flag_gazepoint_sequence_anomalies(sequence=["A", "B"], z_threshold=0.5)
    assert {"anomaly_score", "anomaly"} <= set(out.columns)
    assert out["anomaly"].any()


def test_r1_analyze_window_edge_and_validation_paths():
    base = pd.DataFrame(
        {
            "USER_ID": ["P1"] * 4,
            "TIME": [0.0, 0.05, 0.10, 0.15],
            "mean_pupil": [1.0, np.nan, 3.0, 4.0],
        }
    )
    out = gp3.analyze_gazepoint_window(
        base,
        by=None,
        value_cols=["mean_pupil"],
        summary_stats=["mean", "sd", "median", "min", "max", "sum", "valid_prop"],
        window_size=0.1,
        step=0.05,
        window_unit="native",
        time_unit="auto",
        include_partial=True,
    )
    assert {"mean_pupil_median", "mean_pupil_sum", "mean_pupil_valid_prop"} <= set(out)
    sec = gp3.analyze_gazepoint_window(
        base,
        by="USER_ID",
        value_cols="mean_pupil",
        summary_stats="mean",
        window_size=0.1,
        step=0.1,
        window_unit="seconds",
        time_unit="seconds",
    )
    assert not sec.empty
    empty = gp3.analyze_gazepoint_window(
        pd.DataFrame({"USER_ID": ["P1", "P1"], "TIME": [np.nan, np.nan], "mean_pupil": [1.0, 2.0]}),
        value_cols="mean_pupil",
    )
    assert empty.empty
    with pytest.raises(TypeError):
        gp3.analyze_gazepoint_window(base, value_cols="mean_pupil", nonsense=True)
    with pytest.raises(ValueError):
        gp3.analyze_gazepoint_window([1, 2])
    with pytest.raises(ValueError):
        gp3.analyze_gazepoint_window(base, window_unit="fortnights")
    with pytest.raises(ValueError):
        gp3.analyze_gazepoint_window(base, time_unit="ticks")
    with pytest.raises(ValueError):
        gp3.analyze_gazepoint_window(base.drop(columns="TIME"), value_cols="mean_pupil")
    with pytest.raises(ValueError):
        gp3.analyze_gazepoint_window(pd.DataFrame({"USER_ID": ["P1"], "TIME": [0], "label": ["x"]}))
    with pytest.raises(ValueError):
        gp3.analyze_gazepoint_window(base, value_cols="missing")
    with pytest.raises(ValueError):
        gp3.analyze_gazepoint_window(base.assign(label="x"), value_cols="label")
    with pytest.raises(ValueError):
        gp3.analyze_gazepoint_window(base, value_cols="mean_pupil", summary_stats=["wat"])
    with pytest.raises(ValueError):
        gp3.analyze_gazepoint_window(base, value_cols="mean_pupil", window_size=0)
    with pytest.raises(ValueError):
        gp3.analyze_gazepoint_window(base, value_cols="mean_pupil", step=np.inf)


def test_r1_sequence_metrics_edges_and_validation():
    out = gp3.compute_gazepoint_aoi_sequence_metrics(
        sequence=["A", "", None, "A", "B"], include_missing=True, collapse_repeats=False
    )
    assert out.iloc[0]["sequence_length"] == 5
    none_valid = gp3.compute_gazepoint_aoi_sequence_metrics(
        pd.DataFrame({"AOI": [None, ""]}), aoi_col="AOI", group_cols=[], include_missing=False
    )
    assert none_valid.iloc[0]["sequence_status"] == "no_valid_aoi"
    with pytest.raises(TypeError):
        gp3.compute_gazepoint_aoi_sequence_metrics(sequence=["A"], unknown=True)
    with pytest.raises(ValueError):
        gp3.compute_gazepoint_aoi_sequence_metrics(data=[1], aoi_col="AOI")
    with pytest.raises(ValueError):
        gp3.compute_gazepoint_aoi_sequence_metrics(pd.DataFrame({"AOI": ["A"]}))
    with pytest.raises(ValueError):
        gp3.compute_gazepoint_aoi_sequence_metrics(
            pd.DataFrame({"AOI": ["A"]}), aoi_col="AOI", group_cols=[""]
        )
    with pytest.raises(ValueError):
        gp3.compute_gazepoint_aoi_sequence_metrics(
            pd.DataFrame({"AOI": ["A"]}), aoi_col="AOI", time_col=""
        )
    with pytest.raises(ValueError):
        gp3.compute_gazepoint_aoi_sequence_metrics(
            pd.DataFrame({"AOI": ["A"]}), aoi_col="AOI", missing_label=""
        )
    with pytest.raises(ValueError):
        gp3.compute_gazepoint_aoi_sequence_metrics(
            pd.DataFrame({"AOI": ["A"]}), aoi_col="AOI", include_missing=1
        )
    with pytest.raises(ValueError):
        gp3.compute_gazepoint_aoi_sequence_metrics(
            pd.DataFrame({"AOI": ["A"]}), aoi_col="AOI", collapse_repeats=1
        )
    with pytest.raises(ValueError):
        gp3.compute_gazepoint_aoi_sequence_metrics(
            pd.DataFrame({"AOI": ["A"]}), aoi_col="AOI", group_cols=["subject"]
        )


def test_r1_event_review_edge_paths(tmp_path):
    canonical = gp3.create_gazepoint_event_review_template(
        pd.DataFrame({"TIME": [0.0, 1.0]}),
        id_col=None,
        rows_per_sequence=2,
        event_type="blink",
        reviewer=None,
        path=tmp_path / "canonical.csv",
    )
    assert len(canonical) == 2
    legacy = gp3.create_gazepoint_event_review_template(
        pd.DataFrame({"event_state": ["fixation", "saccade"]}), path=tmp_path / "legacy.csv"
    )
    assert {"review_state", "reviewer_note", "reviewed"} <= set(legacy)
    classified = gp3.create_gazepoint_event_review_template(
        pd.DataFrame({"x": [0.1, 0.1, 0.2], "y": [0.2, 0.2, 0.3], "time": [0.0, 0.01, 0.02]}),
        x_col="x",
        y_col="y",
        time_col="time",
    )
    assert "review_state" in classified
    with pytest.raises(TypeError):
        gp3.create_gazepoint_event_review_template(
            pd.DataFrame({"event_state": ["fixation"]}), bogus=True
        )
    with pytest.raises(ValueError):
        gp3.create_gazepoint_event_review_template("bad")
    with pytest.raises(ValueError):
        gp3.create_gazepoint_event_review_template(pd.DataFrame(columns=["USER_ID", "TIME"]))
    with pytest.raises(ValueError):
        gp3.create_gazepoint_event_review_template(
            pd.DataFrame({"USER_ID": ["P1"], "TIME": [0.0]}), rows_per_sequence=0
        )
    with pytest.raises(ValueError):
        gp3.create_gazepoint_event_review_template(
            pd.DataFrame({"USER_ID": ["P1"], "TIME": [0.0]}), event_type=""
        )
    with pytest.raises(ValueError):
        gp3.create_gazepoint_event_review_template(
            pd.DataFrame({"USER_ID": ["P1"], "TIME": [np.nan]})
        )


def test_r1_cnn_uncertainty_edge_paths():
    legacy = pd.DataFrame({"uncertainty": [0.1, 0.8, np.nan]})
    kept = gp3.filter_gazepoint_cnn_uncertainty(legacy, threshold=0.5, keep_flag=False)
    assert len(kept) == 1
    data = pd.DataFrame({"x": [0.1, np.nan, 0.3], "y": [0.2, 0.3, 0.4], "u": [0.0, 0.0, np.nan]})
    no_u = gp3.filter_gazepoint_cnn_uncertainty(data, x="x", y="y")
    assert no_u["cnn_uncertainty_weight"].tolist() == [1.0, 0.0, 1.0]
    zero_scale = gp3.filter_gazepoint_cnn_uncertainty(data, x="x", y="y", uncertainty="u")
    assert np.isfinite(zero_scale["cnn_uncertainty_weight"].iloc[0])
    with pytest.raises(ValueError):
        gp3.filter_gazepoint_cnn_uncertainty(pd.DataFrame({"q": [1.0]}), uncertainty_col="missing")
    with pytest.raises(TypeError):
        gp3.filter_gazepoint_cnn_uncertainty(legacy, bogus=True)
    with pytest.raises(TypeError):
        gp3.filter_gazepoint_cnn_uncertainty(data, x="x", y="y", bogus=True)
    with pytest.raises(ValueError):
        gp3.filter_gazepoint_cnn_uncertainty([1], x="x", y="y")
    with pytest.raises(ValueError):
        gp3.filter_gazepoint_cnn_uncertainty(data, x="missing", y="y")


def test_r1_tracking_quality_edge_paths():
    raw = pd.DataFrame({"FPOGV": [1, 0], "FPOGX": [0.1, np.nan], "FPOGY": [0.2, 0.3]})
    legacy = gp3.flag_tracking_quality(
        raw, validity_col="FPOGV", x_col="FPOGX", y_col="FPOGY", min_usable_prop=0.9
    )
    assert "quality_flag" in legacy
    quality = pd.DataFrame({"USER_FILE": ["P1"], "MEDIA_ID": ["M1"], "FPOGV_valid_pct": [80]})
    sampling = pd.DataFrame(
        {"USER_FILE": ["P1"], "MEDIA_ID": ["M1"], "estimated_hz": [60], "duration_sec": [10]}
    )
    out = gp3.flag_tracking_quality(quality, sampling)
    assert pd.isna(out["min_pupil_valid_pct_observed"].iloc[0])
    with pytest.raises(TypeError):
        gp3.flag_tracking_quality(quality, sampling, bogus=True)
    with pytest.raises(ValueError):
        gp3.flag_tracking_quality("bad", sampling)
    with pytest.raises(ValueError):
        gp3.flag_tracking_quality(quality, "bad")
    with pytest.raises(ValueError):
        gp3.flag_tracking_quality(quality.drop(columns="MEDIA_ID"), sampling)
    with pytest.raises(ValueError):
        gp3.flag_tracking_quality(quality, sampling.drop(columns="MEDIA_ID"))
    with pytest.raises(ValueError):
        gp3.flag_tracking_quality(quality.drop(columns="FPOGV_valid_pct"), sampling)
    with pytest.raises(ValueError):
        gp3.flag_tracking_quality(quality, sampling.drop(columns="estimated_hz"))


def test_r1_pchip_edge_and_validation_paths():
    observed = pd.DataFrame(
        {"subject": ["P1"] * 4, "time": [0.0, 100.0, 200.0, 300.0], "pupil": [1.0, 2.0, 3.0, 4.0]}
    )
    out = gp3.interpolate_gazepoint_pupil_pchip(
        observed, grouping_cols=[], max_gap_ms=None, max_gap_samples=None
    )
    assert not out["interpolated_pupil_pchip"].any()
    insufficient = gp3.interpolate_gazepoint_pupil_pchip(
        pd.DataFrame({"time": [0, 100, 200], "pupil": [1.0, np.nan, np.nan]}),
        grouping_cols=[],
        min_valid_points=3,
    )
    assert "missing_insufficient_valid_points" in set(insufficient["pchip_interpolation_status"])
    edges = gp3.interpolate_gazepoint_pupil_pchip(
        pd.DataFrame(
            {"time": [0, 100, 200, 300, 400], "pupil": [np.nan, 1.0, np.nan, 3.0, np.nan]}
        ),
        grouping_cols=[],
        max_gap_ms=50,
        min_valid_points=2,
    )
    statuses = set(edges["pchip_interpolation_status"])
    assert "missing_leading_or_trailing_gap" in statuses and "missing_long_gap" in statuses
    missing_time = gp3.interpolate_gazepoint_pupil_pchip(
        pd.DataFrame({"time": [0.0, np.nan, 200.0, 300.0], "pupil": [1.0, np.nan, 3.0, 4.0]}),
        grouping_cols=[],
        min_valid_points=3,
    )
    assert "missing_time" in set(missing_time["pchip_interpolation_status"])
    invalid_df = pd.DataFrame({"foo": [1.0]})
    with pytest.raises(TypeError):
        gp3.interpolate_gazepoint_pupil_pchip(observed, bogus=True)
    with pytest.raises(ValueError):
        gp3.interpolate_gazepoint_pupil_pchip("bad")
    with pytest.raises(ValueError):
        gp3.interpolate_gazepoint_pupil_pchip(pd.DataFrame())
    with pytest.raises(ValueError):
        gp3.interpolate_gazepoint_pupil_pchip(observed, min_valid_points=0)
    with pytest.raises(ValueError):
        gp3.interpolate_gazepoint_pupil_pchip(observed, output_col="")
    with pytest.raises(ValueError):
        gp3.interpolate_gazepoint_pupil_pchip(observed, max_gap_ms=0)
    with pytest.raises(ValueError):
        gp3.interpolate_gazepoint_pupil_pchip(observed, max_gap_samples=1.5)
    with pytest.raises(ValueError):
        gp3.interpolate_gazepoint_pupil_pchip(observed, pupil_col=1)
    with pytest.raises(ValueError):
        gp3.interpolate_gazepoint_pupil_pchip(observed, pupil_col="missing")
    with pytest.raises(ValueError):
        gp3.interpolate_gazepoint_pupil_pchip(invalid_df)
    with pytest.raises(ValueError):
        gp3.interpolate_gazepoint_pupil_pchip(observed, grouping_cols=[1])
    with pytest.raises(ValueError):
        gp3.interpolate_gazepoint_pupil_pchip(observed, grouping_cols=["missing"])


def test_r1_inspect_column_dtype_and_alias_paths():
    frame = pd.DataFrame(
        {
            "FPOGV": pd.Series([True, False], dtype=bool),
            "CNT": pd.Series([1, 2], dtype="int64"),
            "FPOGX": pd.Series([0.1, np.nan], dtype=float),
            "AOI": pd.Categorical(["a", "b"]),
            "when": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "other": ["x", None],
        }
    )
    out = gp3.inspect_gazepoint_columns(x=frame)
    assert {"fixation_gaze", "identification", "derived", "other"} <= set(out["semantic_group"])
    assert {"logical", "integer", "numeric", "factor", "POSIXct/POSIXt", "character"} <= set(
        out["dtype"]
    )


def test_r1_prepare_timecourse_edge_and_validation_paths():
    data = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S1", "S2"],
            "condition": ["A", "B", "A", "A"],
            "time": [0, 0, 1, 0],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    out = gp3.prepare_gazepoint_timecourse_test_data(
        data,
        subject_col="subject",
        condition_col="condition",
        time_col="time",
        outcome_col="value",
        complete_only=False,
    )
    assert len(out) == 4

    def aggregate_requires_na_rm(values, na_rm):
        assert na_rm is True
        return np.mean(values)

    dup = pd.DataFrame(
        {
            "subject": ["S1"] * 5,
            "condition": ["A", "A", "B", "A", "B"],
            "time": [0, 0, 0, 1, 1],
            "value": [1.0, 3.0, 2.0, 4.0, 5.0],
        }
    )
    agg = gp3.prepare_gazepoint_timecourse_test_data(
        dup,
        subject_col="subject",
        condition_col="condition",
        time_col="time",
        outcome_col="value",
        aggregate_fun=aggregate_requires_na_rm,
    )
    assert not agg.empty
    with pytest.raises(ValueError):
        gp3.prepare_gazepoint_timecourse_test_data("bad")
    with pytest.raises(TypeError):
        gp3.prepare_gazepoint_timecourse_test_data(
            data,
            subject_col="subject",
            condition_col="condition",
            time_col="time",
            outcome_col="value",
            bogus=True,
        )
    with pytest.raises(ValueError):
        gp3.prepare_gazepoint_timecourse_test_data(
            data, subject_col="", condition_col="condition", time_col="time", outcome_col="value"
        )
    with pytest.raises(ValueError):
        gp3.prepare_gazepoint_timecourse_test_data(
            data,
            subject_col="missing",
            condition_col="condition",
            time_col="time",
            outcome_col="value",
        )
    with pytest.raises(ValueError):
        gp3.prepare_gazepoint_timecourse_test_data(
            data,
            subject_col="subject",
            condition_col="condition",
            time_col="time",
            outcome_col="value",
            aggregate_fun=1,
        )
    with pytest.raises(ValueError):
        gp3.prepare_gazepoint_timecourse_test_data(
            data,
            subject_col="subject",
            condition_col="condition",
            time_col="time",
            outcome_col="value",
            complete_only=1,
        )
    with pytest.raises(ValueError):
        gp3.prepare_gazepoint_timecourse_test_data(
            data.assign(time=np.nan, value=np.nan),
            subject_col="subject",
            condition_col="condition",
            time_col="time",
            outcome_col="value",
        )
    with pytest.raises(ValueError):
        gp3.prepare_gazepoint_timecourse_test_data(
            data,
            subject_col="subject",
            condition_col="condition",
            time_col="time",
            outcome_col="value",
            condition_order=["A"],
        )
    with pytest.raises(ValueError):
        gp3.prepare_gazepoint_timecourse_test_data(
            data.loc[data["condition"].eq("A")],
            subject_col="subject",
            condition_col="condition",
            time_col="time",
            outcome_col="value",
        )
    with pytest.raises(ValueError):
        gp3.prepare_gazepoint_timecourse_test_data(
            data,
            subject_col="subject",
            condition_col="condition",
            time_col="time",
            outcome_col="value",
            condition_order=["A", "C"],
        )
    with pytest.raises(ValueError):
        gp3.prepare_gazepoint_timecourse_test_data(
            pd.DataFrame(
                {
                    "subject": ["S1", "S1"],
                    "condition": ["A", "B"],
                    "time": [0, 1],
                    "value": [1.0, 2.0],
                }
            ),
            subject_col="subject",
            condition_col="condition",
            time_col="time",
            outcome_col="value",
            complete_only=True,
        )


def test_r1_analysis_audit_edge_and_diagnostic_paths():
    with pytest.raises(ValueError):
        gp3.create_gazepoint_analysis_decision_audit(results=["bad"])
    with pytest.raises(ValueError):
        gp3.create_gazepoint_analysis_decision_audit(results={})
    with pytest.raises(ValueError):
        gp3.create_gazepoint_analysis_decision_audit(results={"x": 1}, diagnostics_required=1)
    with pytest.raises(ValueError):
        gp3.create_gazepoint_analysis_decision_audit(results={"x": 1}, require_clean_diagnostics=1)
    with pytest.raises(ValueError):
        gp3.create_gazepoint_analysis_decision_audit(results={"x": 1}, branch_roles=["bad"])
    with pytest.raises(ValueError):
        gp3.create_gazepoint_analysis_decision_audit(
            results={"x": 1}, branch_roles=pd.DataFrame({"branch_name": ["x"]})
        )
    with pytest.raises(ValueError):
        gp3.create_gazepoint_analysis_decision_audit(
            results={"x": 1},
            branch_roles=pd.DataFrame({"branch_name": ["x"], "decision_type": ["unsupported"]}),
        )
    roles = pd.DataFrame(
        {
            "branch_name": [
                "confirm_missing",
                "explore",
                "sensitivity",
                "diag_error",
                "report",
                "preprocess",
                "model_no_diag",
                "diag_empty",
                "overview",
            ],
            "decision_type": [
                "confirmatory",
                "exploratory",
                "sensitivity",
                "diagnostic",
                "reporting",
                "preprocessing",
                "confirmatory",
                "diagnostic",
                "reporting",
            ],
        }
    )
    results = {
        "confirm_missing": None,
        "explore": 3.0,
        "sensitivity": pd.DataFrame({"x": [1]}),
        "diag_error": {
            "status": ["ok", "error"],
            "diagnostics": {
                "checks": pd.DataFrame(
                    {
                        "diagnostic_status": ["warning", "error", "skipped", "not_applicable"],
                        "message": ["warn", "boom", "", ""],
                        "warning_note": ["", "", "skip", ""],
                    }
                )
            },
            "fallback_used": True,
            "singular_fit": True,
        },
        "report": pd.DataFrame({"summary_status": ["ok"]}),
        "preprocess": {"workflow_status": "ok"},
        "model_no_diag": {"model": {"x": 1}, "model_status": "ok"},
        "diag_empty": {"diagnostics": {}},
        "overview": {"overview": pd.DataFrame({"validation_status": ["ok"]})},
    }
    out = gp3.create_gazepoint_analysis_decision_audit(
        results=results,
        branch_roles=roles,
        required_confirmatory=["confirm_missing", "model_no_diag"],
        diagnostics_required=True,
        require_clean_diagnostics=True,
    )
    assert out["readiness"].iloc[0]["readiness_status"] == "not_ready"
    warning_only = gp3.create_gazepoint_analysis_decision_audit(
        results={
            "confirm": {
                "model": {"x": 1},
                "diagnostics": {
                    "checks": pd.DataFrame({"diagnostic_status": ["warning"], "message": ["warn"]})
                },
            }
        },
        branch_roles=pd.DataFrame({"branch_name": ["confirm"], "decision_type": ["confirmatory"]}),
        required_confirmatory=["confirm"],
        require_clean_diagnostics=True,
    )
    assert warning_only["readiness"].iloc[0]["readiness_status"] == "not_ready"
    ready = gp3.create_gazepoint_analysis_decision_audit(
        results={
            "confirm": {
                "model": {"x": 1},
                "diagnostics": {
                    "checks": pd.DataFrame({"diagnostic_status": ["ok"], "message": [""]})
                },
            }
        },
        branch_roles=pd.DataFrame({"branch_name": ["confirm"], "decision_type": ["confirmatory"]}),
        required_confirmatory=["confirm"],
    )
    assert ready["readiness"].iloc[0]["readiness_status"] == "ready"


def test_r1_exclusion_edge_validation_and_reason_paths():
    with pytest.raises(ValueError):
        gp3.recommend_gazepoint_exclusions("bad", participant_col="p", validity_col="v")
    with pytest.raises(ValueError):
        gp3.recommend_gazepoint_exclusions(pd.DataFrame(), participant_col="p", validity_col="v")
    base = pd.DataFrame(
        {
            "p": ["P1", "P1", "P1", "P2"],
            "trial": ["T1", "T1", "T2", "T1"],
            "cond": ["A", "A", "B", None],
            "valid": ["valid", "bad", "mystery", "good"],
            "x": [1.0, np.nan, 1.0, np.nan],
            "y": [1.0, 1.0, np.nan, np.nan],
            "pupil": [3.0, np.nan, 3.0, 3.0],
            "artifact": ["yes", "no", "1", "false"],
        }
    )
    out = gp3.recommend_gazepoint_exclusions(
        base,
        participant_col="p",
        trial_col="trial",
        condition_col="cond",
        validity_col="valid",
        x_col="x",
        y_col="y",
        pupil_col="pupil",
        artifact_col="artifact",
        min_trial_samples=3,
        max_trial_missing_prop=0.1,
        max_trial_artifact_prop=0.1,
        min_participant_trials=3,
        min_participant_valid_trials=2,
        max_participant_missing_prop=0.1,
        max_participant_artifact_prop=0.1,
    )
    reasons = " ".join(out["exclusion_table"]["exclusion_reason"].fillna(""))
    assert "too_few_trial_samples" in reasons and "high_trial_missingness" in reasons
    x_only = gp3.recommend_gazepoint_exclusions(
        base[["p", "x"]].copy(),
        participant_col="p",
        x_col="x",
        require_both_gaze_coordinates=False,
        min_trial_samples=1,
        min_participant_trials=1,
    )
    y_only = gp3.recommend_gazepoint_exclusions(
        base[["p", "y"]].copy(),
        participant_col="p",
        y_col="y",
        require_both_gaze_coordinates=False,
        min_trial_samples=1,
        min_participant_trials=1,
    )
    assert "exclusion_table" in x_only and "exclusion_table" in y_only
    with pytest.raises(ValueError):
        gp3.recommend_gazepoint_exclusions(base, participant_col="missing", validity_col="valid")
    with pytest.raises(ValueError):
        gp3.recommend_gazepoint_exclusions(base, participant_col="p")
    with pytest.raises(ValueError):
        gp3.recommend_gazepoint_exclusions(base, participant_col="p", x_col="x")
    with pytest.raises(ValueError):
        gp3.recommend_gazepoint_exclusions(
            base, participant_col="p", validity_col="valid", min_trial_samples=0
        )
    with pytest.raises(ValueError):
        gp3.recommend_gazepoint_exclusions(
            base, participant_col="p", validity_col="valid", max_trial_missing_prop=2
        )
    with pytest.raises(ValueError):
        gp3.recommend_gazepoint_exclusions(
            base, participant_col="p", validity_col="valid", require_both_gaze_coordinates=1
        )
    with pytest.raises(ValueError):
        gp3.recommend_gazepoint_exclusions(base, participant_col="p", validity_col="valid", name="")


def test_r1_adaptive_trial_edge_and_validation_paths():
    legacy = pd.DataFrame({"score": [1.0, 3.0, 2.0]})
    assert (
        gp3.select_gazepoint_adaptive_trial(legacy, score_col="score", strategy="lowest").iloc[0][
            "score"
        ]
        == 1.0
    )
    assert len(gp3.select_gazepoint_adaptive_trial(pd.DataFrame({"x": [1, 2]}))) == 1
    assert gp3.select_gazepoint_adaptive_trial(pd.DataFrame()).empty
    candidates = pd.DataFrame({"mu": [1.0, 2.0, 3.0], "sd": [0.2, 0.4, 0.1]})
    assert (
        len(
            gp3.select_gazepoint_adaptive_trial(
                candidates, mean="mu", sd="sd", acquisition="uncertainty"
            )
        )
        == 1
    )
    assert (
        len(
            gp3.select_gazepoint_adaptive_trial(
                candidates, mean="mu", sd="sd", acquisition="expected_improvement", maximize=False
            )
        )
        == 1
    )
    assert (
        len(
            gp3.select_gazepoint_adaptive_trial(
                candidates,
                mean="mu",
                sd="sd",
                acquisition="expected_improvement",
                best_observed=2.0,
            )
        )
        == 1
    )
    assert gp3.select_gazepoint_adaptive_trial(
        pd.DataFrame({"mu": [np.nan], "sd": [np.nan]}), mean="mu", sd="sd"
    ).empty
    with pytest.raises(TypeError):
        gp3.select_gazepoint_adaptive_trial(candidates, mean="mu", sd="sd", bogus=True)
    with pytest.raises(ValueError):
        gp3.select_gazepoint_adaptive_trial([1, 2], mean="mu", sd="sd")
    with pytest.raises(ValueError):
        gp3.select_gazepoint_adaptive_trial(candidates, mean="mu", sd="sd", acquisition="bad")
    with pytest.raises(ValueError):
        gp3.select_gazepoint_adaptive_trial(candidates, mean="missing", sd="sd")
    with pytest.raises(ValueError):
        gp3.select_gazepoint_adaptive_trial(
            pd.DataFrame({"mu": ["x"], "sd": [1.0]}), mean="mu", sd="sd"
        )
