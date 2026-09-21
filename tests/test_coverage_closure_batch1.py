from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import gp3tools.events as events
import gp3tools.reporting as reporting
import gp3tools.stats as stats_mod

# ============================================================================
# EVENTS
# ============================================================================


def test_events_detector_validation_and_empty_candidate_paths():
    with pytest.raises(ValueError, match="x, y, and time"):
        events.detect_gazepoint_fixations_velocity(pd.DataFrame({"x": [1.0]}))

    no_candidate = pd.DataFrame(
        {
            "USER_ID": ["S1", "S1", "S1"],
            "FPOGX": [np.nan, np.nan, np.nan],
            "FPOGY": [np.nan, np.nan, np.nan],
            "TIME": [0.0, 0.01, 0.02],
        }
    )

    out = events.detect_gazepoint_fixations_velocity(
        no_candidate,
        id_col="USER_ID",
        return_mode="events",
    )

    assert out.empty

    short = pd.DataFrame(
        {
            "USER_ID": ["S1", "S1"],
            "FPOGX": [0.5, 0.5],
            "FPOGY": [0.5, 0.5],
            "TIME": [0.0, 0.01],
        }
    )

    out = events.detect_gazepoint_fixations_velocity(
        short,
        id_col="USER_ID",
        min_duration_ms=500,
        return_mode="events",
    )

    assert out.empty

    with pytest.raises(ValueError, match="x, y, and time"):
        events.classify_gazepoint_events_hmm(pd.DataFrame({"x": [1.0]}))


def test_events_saccade_auto_classification_and_grouped_summary():
    frame = pd.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B"],
            "x": [0.0, 0.1, 0.9, 0.0, 0.1, 0.8],
            "y": [0.0, 0.1, 0.9, 0.0, 0.1, 0.8],
            "time": [0.0, 0.1, 0.2, 0.0, 0.1, 0.2],
        }
    )

    result = events.compute_gazepoint_saccade_metrics(
        frame,
        x_col="x",
        y_col="y",
        time_col="time",
        group_cols=["group"],
    )

    assert isinstance(result, pd.DataFrame)


def test_events_summarise_fixations_fallbacks():
    with pytest.raises(ValueError, match="Missing columns"):
        events.summarise_fixations(
            pd.DataFrame(
                {
                    "FPOGD": [0.1],
                    "FPOGS": [0.0],
                }
            )
        )

    raw = pd.DataFrame(
        {
            "x": [0.5, 0.5, 0.5],
            "y": [0.5, 0.5, 0.5],
            "time": [0.0, 0.2, 0.4],
        }
    )

    detected = events.summarise_fixations(
        raw,
        x_col="x",
        y_col="y",
        time_col="time",
    )

    assert isinstance(detected, pd.DataFrame)

    empty = events.summarise_fixations(
        pd.DataFrame(
            {
                "x": [0.5],
                "y": [0.5],
                "time": [0.0],
                "fixation_id": pd.Series([pd.NA], dtype="Int64"),
            }
        ),
        x_col="x",
        y_col="y",
        time_col="time",
    )

    assert empty.empty


def test_events_fixation_trial_legacy_and_validation_paths():
    with pytest.raises(KeyError, match="Missing required column"):
        events.summarise_gazepoint_fixation_trials(
            pd.DataFrame({"subject": ["S1"]}),
            subject_col="subject",
        )

    ungrouped = events.summarise_gazepoint_fixation_trials(
        pd.DataFrame({"duration_ms": [10.0, 20.0]}),
        subject_col="",
    )

    assert len(ungrouped) == 1

    with pytest.raises(ValueError, match="start_time_unit"):
        events.summarise_gazepoint_fixation_trials(
            pd.DataFrame(),
            group_cols=["subject"],
            start_time_unit="bad",
        )

    with pytest.raises(ValueError, match="duration_unit"):
        events.summarise_gazepoint_fixation_trials(
            pd.DataFrame(),
            group_cols=["subject"],
            duration_unit="bad",
        )

    with pytest.raises(ValueError, match="TRUE or FALSE"):
        events.summarise_gazepoint_fixation_trials(
            pd.DataFrame(),
            group_cols=["subject"],
            valid_only="yes",
        )

    with pytest.raises(ValueError, match="automatically detect required fixation"):
        events.summarise_gazepoint_fixation_trials(
            pd.DataFrame(
                {
                    "subject": ["S1"],
                    "trial": ["T1"],
                }
            ),
            group_cols=["subject", "trial"],
        )

    with pytest.raises(ValueError, match="Missing required column: missing_x"):
        events.summarise_gazepoint_fixation_trials(
            pd.DataFrame(
                {
                    "subject": ["S1"],
                    "trial": ["T1"],
                    "start": [0.0],
                    "duration": [0.1],
                }
            ),
            group_cols=["subject", "trial"],
            start_col="start",
            duration_col="duration",
            x_col="missing_x",
        )


def _trial_frame(aoi=None, valid=None):
    frame = pd.DataFrame(
        {
            "subject": ["S1", "S1"],
            "trial": ["T1", "T1"],
            "time": [0.1, 0.3],
            "duration": [0.1, 0.1],
        }
    )

    if aoi is not None:
        frame["aoi"] = aoi

    if valid is not None:
        frame["valid"] = valid

    return frame


def test_events_fixation_trial_status_contracts():
    no_aoi = events.summarise_gazepoint_fixation_trials(
        _trial_frame(),
        group_cols=["subject", "trial"],
        start_col="time",
        duration_col="duration",
    )

    assert no_aoi.loc[0, "fixation_trial_feature_status"] == "no_aoi_column"

    outside = events.summarise_gazepoint_fixation_trials(
        _trial_frame(["outside", "outside"]),
        group_cols=["subject", "trial"],
        start_col="time",
        duration_col="duration",
        aoi_col="aoi",
    )

    assert outside.loc[0, "fixation_trial_feature_status"] == "no_aoi_fixations"

    undefined = events.summarise_gazepoint_fixation_trials(
        _trial_frame(["A", "B"]),
        group_cols=["subject", "trial"],
        start_col="time",
        duration_col="duration",
        aoi_col="aoi",
    )

    assert undefined.loc[0, "fixation_trial_feature_status"] == ("no_target_or_distractor_defined")

    target_missing = events.summarise_gazepoint_fixation_trials(
        _trial_frame(["A", "B"]),
        group_cols=["subject", "trial"],
        start_col="time",
        duration_col="duration",
        aoi_col="aoi",
        target_aoi_values=["T"],
    )

    assert target_missing.loc[0, "fixation_trial_feature_status"] == ("target_not_observed")

    distractor_missing = events.summarise_gazepoint_fixation_trials(
        _trial_frame(["T", "T"]),
        group_cols=["subject", "trial"],
        start_col="time",
        duration_col="duration",
        aoi_col="aoi",
        target_aoi_values=["T"],
        distractor_aoi_values=["D"],
    )

    assert distractor_missing.loc[0, "fixation_trial_feature_status"] == ("distractor_not_observed")

    ok = events.summarise_gazepoint_fixation_trials(
        _trial_frame(["T", "D"]),
        group_cols=["subject", "trial"],
        start_col="time",
        duration_col="duration",
        aoi_col="aoi",
        target_aoi_values=["T"],
        distractor_aoi_values=["D"],
    )

    assert ok.loc[0, "fixation_trial_feature_status"] == "ok"


def test_events_fixation_trial_filter_to_empty():
    with pytest.raises(ValueError, match="No fixation rows remain"):
        events.summarise_gazepoint_fixation_trials(
            _trial_frame(["A", "B"], [0, 0]),
            group_cols=["subject", "trial"],
            start_col="time",
            duration_col="duration",
            aoi_col="aoi",
            valid_col="valid",
            valid_only=True,
        )

    with pytest.raises(ValueError, match="No fixation rows remain"):
        events.summarise_gazepoint_fixation_trials(
            _trial_frame(["outside", "outside"]),
            group_cols=["subject", "trial"],
            start_col="time",
            duration_col="duration",
            aoi_col="aoi",
            include_non_aoi=False,
        )


def test_events_reliability_legacy_and_validation_paths():
    legacy = events.audit_gazepoint_fixation_reliability(
        pd.DataFrame({"duration_ms": [10.0, 20.0]})
    )

    assert legacy.loc[0, "n"] == 2

    base = pd.DataFrame(
        {
            "subject": ["S1", "S1"],
            "trial": ["T1", "T2"],
            "duration": [1.0, 2.0],
            "aoi": ["A", "B"],
        }
    )

    with pytest.raises(ValueError, match="subject_col"):
        events.audit_gazepoint_fixation_reliability(
            base,
            subject_col=None,
            trial_col="trial",
        )

    with pytest.raises(ValueError, match="metric must be"):
        events.audit_gazepoint_fixation_reliability(
            base,
            subject_col="subject",
            trial_col="trial",
            metric="bad",
        )

    with pytest.raises(ValueError, match="split_method"):
        events.audit_gazepoint_fixation_reliability(
            base,
            subject_col="subject",
            trial_col="trial",
            split_method="bad",
        )

    with pytest.raises(ValueError, match="correlation_method"):
        events.audit_gazepoint_fixation_reliability(
            base,
            subject_col="subject",
            trial_col="trial",
            correlation_method="bad",
        )

    with pytest.raises(ValueError, match="min_trials"):
        events.audit_gazepoint_fixation_reliability(
            base,
            subject_col="subject",
            trial_col="trial",
            min_trials=1,
        )

    with pytest.raises(ValueError, match="aoi_col is required"):
        events.audit_gazepoint_fixation_reliability(
            base,
            subject_col="subject",
            trial_col="trial",
            metric="entropy_score",
        )

    with pytest.raises(ValueError, match="missing required column"):
        events.audit_gazepoint_fixation_reliability(
            base,
            subject_col="subject",
            trial_col="trial",
            duration_col="missing",
            metric="mean_fixation_duration",
        )


def test_events_reliability_empty_duration_aoi_and_entropy_paths():
    empty = events.audit_gazepoint_fixation_reliability(
        pd.DataFrame(
            {
                "subject": pd.Series(dtype=str),
                "trial": pd.Series(dtype=str),
            }
        ),
        subject_col="subject",
        trial_col="trial",
    )

    assert empty.loc[0, "reliability_status"] == "no_trials"

    duration_missing = pd.DataFrame(
        {
            "subject": ["A", "A", "B", "B", "C", "C"],
            "trial": ["1", "2"] * 3,
            "duration": [np.nan] * 6,
        }
    )

    out = events.audit_gazepoint_fixation_reliability(
        duration_missing,
        subject_col="subject",
        trial_col="trial",
        duration_col="duration",
        metric="mean_fixation_duration",
        min_trials=2,
    )

    assert out.loc[0, "reliability_status"] == "too_few_subjects"

    no_aoi = duration_missing.assign(aoi="")

    out = events.audit_gazepoint_fixation_reliability(
        no_aoi,
        subject_col="subject",
        trial_col="trial",
        aoi_col="aoi",
        target_aoi="T",
        metric="aoi_dwell_prop",
        min_trials=2,
    )

    assert out.loc[0, "reliability_status"] == "too_few_subjects"

    zero_duration = pd.DataFrame(
        {
            "subject": ["A", "A", "B", "B", "C", "C"],
            "trial": ["1", "2"] * 3,
            "aoi": ["T", "T"] * 3,
            "duration": [0.0] * 6,
        }
    )

    out = events.audit_gazepoint_fixation_reliability(
        zero_duration,
        subject_col="subject",
        trial_col="trial",
        aoi_col="aoi",
        target_aoi="T",
        duration_col="duration",
        metric="aoi_dwell_prop",
        min_trials=2,
    )

    assert out.loc[0, "reliability_status"] == "too_few_subjects"

    sequence = pd.DataFrame(
        {
            "subject": ["A"] * 4 + ["B"] * 4 + ["C"] * 4,
            "trial": ["1", "2", "3", "4"] * 3,
            "aoi": ["A", "A", "B", "B"] * 3,
            "time": list(range(4)) * 3,
        }
    )

    transition = events.audit_gazepoint_fixation_reliability(
        sequence,
        subject_col="subject",
        trial_col="trial",
        aoi_col="aoi",
        time_col="time",
        metric="transition_count",
        min_trials=4,
    )

    assert len(transition) == 1

    entropy = events.audit_gazepoint_fixation_reliability(
        sequence.assign(aoi="A"),
        subject_col="subject",
        trial_col="trial",
        aoi_col="aoi",
        metric="entropy_score",
        min_trials=4,
    )

    assert len(entropy) == 1


def test_events_reliability_negative_split_correlation():
    rows = []

    values = {
        "A": [1.0, 3.0, 1.0, 3.0],
        "B": [2.0, 2.0, 2.0, 2.0],
        "C": [3.0, 1.0, 3.0, 1.0],
    }

    for subject, durations in values.items():
        for trial, duration in enumerate(durations, start=1):
            rows.append(
                {
                    "subject": subject,
                    "trial": str(trial),
                    "duration": duration,
                }
            )

    out = events.audit_gazepoint_fixation_reliability(
        pd.DataFrame(rows),
        subject_col="subject",
        trial_col="trial",
        duration_col="duration",
        metric="mean_fixation_duration",
        min_trials=4,
    )

    assert len(out) == 1
    assert out.loc[0, "reliability_status"] in {"ok", "no_variance"}


def test_events_comparison_benchmark_and_review_validation(tmp_path):
    frame = pd.DataFrame(
        {
            "x": [0.0, 0.1, 0.2],
            "y": [0.0, 0.1, 0.2],
            "time": [0.0, 0.1, 0.2],
        }
    )

    with pytest.raises(TypeError, match="Unexpected keyword"):
        events.compare_gazepoint_event_detectors(
            frame,
            methods=["velocity"],
            impossible=True,
        )

    with pytest.raises(TypeError, match="either data or x"):
        events.summarise_gazepoint_event_detector_benchmark(
            pd.DataFrame(),
            x={},
        )

    benchmark = {
        "detector_metrics": pd.DataFrame(
            {
                "detector": ["b", "a"],
                "f1": [0.5, 0.9],
            }
        )
    }

    detector = events.summarise_gazepoint_event_detector_benchmark(
        x=benchmark,
        level="detector",
    )

    assert detector.iloc[0]["detector"] == "a"

    with pytest.raises(ValueError, match="level must"):
        events.summarise_gazepoint_event_detector_benchmark(
            x=benchmark,
            level="bad",
        )

    with pytest.raises(ValueError, match="does not contain"):
        events.summarise_gazepoint_event_detector_benchmark(
            x=benchmark,
            level="errors",
        )

    with pytest.raises(ValueError, match="at least one sample"):
        events.create_gazepoint_event_review_template(
            pd.DataFrame(
                {
                    "USER_ID": pd.Series(dtype=str),
                    "TIME": pd.Series(dtype=float),
                }
            )
        )

    with pytest.raises(
        ValueError,
        match="x, y, and time columns are required",
    ):
        events.create_gazepoint_event_review_template(
            pd.DataFrame(
                {
                    "USER_ID": ["S1"],
                    "x": [1.0],
                }
            )
        )

    review_data = pd.DataFrame(
        {
            "USER_ID": ["S1", "S1"],
            "TIME": [0.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="positive integer"):
        events.create_gazepoint_event_review_template(
            review_data,
            rows_per_sequence=0,
        )

    with pytest.raises(ValueError, match="event_type"):
        events.create_gazepoint_event_review_template(
            review_data,
            event_type="",
        )

    with pytest.raises(ValueError, match="finite timestamp"):
        events.create_gazepoint_event_review_template(review_data.assign(TIME=np.nan))

    mixed = pd.DataFrame(
        {
            "USER_ID": ["bad", "good", "good"],
            "TIME": [np.nan, 0.0, 1.0],
        }
    )

    review = events.create_gazepoint_event_review_template(
        mixed,
        rows_per_sequence=2,
        path=tmp_path / "review.csv",
    )

    assert len(review) == 2
    assert (tmp_path / "review.csv").exists()


# ============================================================================
# REPORTING
# ============================================================================


def test_reporting_object_summary_and_match_paths():
    fail_frame = pd.DataFrame(
        {
            "object_name": ["qc"],
            "qc_status": ["fail"],
            "message": ["problem"],
        }
    )

    single = reporting._gp3_reporting_r_object_summary(fail_frame)

    assert single.loc[0, "status"] == "fail"
    assert single.loc[0, "message"] == "problem"

    listed = reporting._gp3_reporting_r_object_summary(
        [
            pd.DataFrame({"status": ["warn"]}),
            pd.DataFrame({"status": ["pass"]}),
        ]
    )

    assert len(listed) == 2

    summary = pd.DataFrame(
        {
            "object_label": ["f", "w", "p", "i"],
            "class": ["x"] * 4,
            "object_name": ["fail_obj", "warn_obj", "pass_obj", "info_obj"],
            "status": ["fail", "warn", "pass", "info"],
            "message": ["", "", "", ""],
        }
    )

    assert reporting._gp3_reporting_r_match(summary, ["fail_obj"])[0] == "fail"

    assert reporting._gp3_reporting_r_match(summary, ["warn_obj"])[0] == "warn"

    assert reporting._gp3_reporting_r_match(summary, ["pass_obj"])[0] == "pass"

    assert reporting._gp3_reporting_r_match(summary, ["info_obj"])[0] == "info"


def test_reporting_checklist_validation_and_data_summary():
    with pytest.raises(ValueError, match="analysis_type"):
        reporting.create_gazepoint_reporting_checklist(
            objects={},
            analysis_type="bad",
        )

    with pytest.raises(TypeError, match="data must"):
        reporting.create_gazepoint_reporting_checklist(
            data=[1, 2],
            objects={},
        )

    no_structure = reporting.create_gazepoint_reporting_checklist(
        data=pd.DataFrame({"x": [1]}),
        objects={},
        study_title="Study",
        required_sections="study_title",
        include_optional=False,
    )

    assert isinstance(no_structure, dict)

    no_data = reporting.create_gazepoint_reporting_checklist(
        data=None,
        objects={},
        study_title="Study",
        include_optional=False,
    )

    assert no_data["data_summary"]["n_rows"].isna().all()


def test_reporting_analysis_decision_audit_validation_and_diagnostics():
    with pytest.raises(ValueError, match="named list"):
        reporting.create_gazepoint_analysis_decision_audit(results=[1, 2])

    with pytest.raises(ValueError, match="At least one"):
        reporting.create_gazepoint_analysis_decision_audit()

    with pytest.raises(ValueError, match="must be named"):
        reporting.create_gazepoint_analysis_decision_audit(results={"": pd.DataFrame({"x": [1]})})

    with pytest.raises(ValueError, match="diagnostics_required"):
        reporting.create_gazepoint_analysis_decision_audit(
            results={"x": pd.DataFrame({"x": [1]})},
            diagnostics_required="yes",
        )

    branches = {
        "error_branch": {
            "model": object(),
            "diagnostics": pd.DataFrame(
                {
                    "diagnostic_status": ["error"],
                    "message": ["bad"],
                }
            ),
        },
        "warning_branch": {
            "diagnostics": {
                "main": pd.DataFrame(
                    {
                        "diagnostic_status": ["warning"],
                    }
                )
            }
        },
        "skipped_branch": {
            "diagnostics": pd.DataFrame(
                {
                    "diagnostic_status": ["skipped_missing_package"],
                }
            )
        },
        "nostatus_branch": {"diagnostics": pd.DataFrame({"x": [1]})},
        "nodf_branch": {"diagnostics": {"x": "not-a-table"}},
        "none_branch": None,
        "list_branch": [1, 2],
        "bool_branch": True,
        "text_branch": "text",
        "number_branch": 1.0,
    }

    roles = pd.DataFrame(
        {
            "branch_name": ["error_branch"],
            "decision_type": ["confirmatory"],
        }
    )

    result = reporting.create_gazepoint_analysis_decision_audit(
        results=branches,
        branch_roles=roles,
        required_confirmatory=["error_branch"],
    )

    assert result.empty is False
    assert result["readiness"].loc[0, "readiness_status"] == "not_ready"

    statuses = set(result["diagnostics_summary"]["diagnostic_status"])

    assert "error" in statuses
    assert "diagnostic_warning" in statuses
    assert "skipped" in statuses
    assert "not_available" in statuses


def test_reporting_report_and_missingness_paths(tmp_path, monkeypatch):
    report = reporting.create_gazepoint_report(
        pd.DataFrame({"x": [1, 2]}),
        output_file=tmp_path / "report.html",
    )

    assert report.exists()

    import gp3tools.qc as qc

    monkeypatch.setattr(
        qc,
        "summarise_gazepoint_missingness",
        lambda *args, **kwargs: pd.DataFrame(),
    )

    assert (
        reporting.report_gazepoint_missingness(pd.DataFrame({"x": [1]}))
        == "No missingness summary available."
    )

    with pytest.raises(ValueError, match="digits"):
        reporting.report_gazepoint_missingness(
            pd.DataFrame({"x": [1]}),
            digits=-1,
        )

    summary = pd.DataFrame(
        {
            "group_id": ["all", "all"],
            "variable": ["a", "b"],
            "n_rows": [10, 0],
            "n_missing": [2, 0],
            "missing_rate": [0.2, np.nan],
        }
    )

    structured = reporting.report_gazepoint_missingness(
        summary,
        digits=1,
        max_variables=2,
    )

    assert "overall" in structured


def test_reporting_phase_qc_face_and_multiverse_paths():
    legacy = reporting.report_gazepoint_phase_coverage(
        pd.DataFrame(
            {
                "task_phase": ["a", "b", "a"],
            }
        )
    )

    assert "2 phase" in legacy

    with pytest.raises(ValueError, match="digits"):
        reporting.report_gazepoint_phase_coverage(
            pd.DataFrame(),
            group_cols=["subject"],
            digits=-1,
        )

    phase_summary = pd.DataFrame(
        {
            "group_id": ["S1", "S1"],
            "phase": ["a", "b"],
            "n_rows": [10, 5],
        }
    )

    structured = reporting.report_gazepoint_phase_coverage(
        phase_summary,
        group_cols=["subject"],
        digits=1,
    )

    assert np.isnan(
        structured["overall"].loc[
            0,
            "weighted_complete_value_rate",
        ]
    )

    qc = reporting.report_gazepoint_qc_overview(
        pd.DataFrame(
            {
                "component": ["a", "b", "c"],
                "status": ["available", "not_available", "strange"],
            }
        ),
        max_objects=3,
    )

    assert qc["summary"]["overview"].loc[0, "n_unknown"] >= 2

    checklist = pd.DataFrame(
        {
            "item": ["quality"],
            "status": ["pass"],
        }
    )

    multimodal = {
        "_gp3_class": "demo",
        "settings": {
            "outcome": "pupil",
            "predictors": ["face"],
            "covariates": ["condition"],
            "random_effects": "subject",
            "n_rows_input": 10,
            "n_rows_model": 8,
        },
    }

    as_list = reporting.report_gazepoint_face_qc(
        checklist=checklist,
        multimodal_model=multimodal,
        output="list",
    )

    assert as_list["model_summary"].loc[0, "outcome"] == "pupil"

    markdown = reporting.report_gazepoint_face_qc(
        checklist=checklist,
        multimodal_model=multimodal,
        output="markdown",
        include_cautions=False,
    )

    assert "External facial-behaviour QC report" in markdown

    frame = pd.DataFrame(
        {
            "estimate": [1.0, -1.0],
            "p_value": [0.01, 0.20],
        }
    )

    multi = reporting.report_gazepoint_multiverse(
        frame,
        alpha=0.05,
    )

    assert len(multi["branch_summary"]) == 1

    with pytest.raises(ValueError, match="alpha"):
        reporting.report_gazepoint_multiverse(
            frame,
            alpha=2,
        )

    with pytest.raises(ValueError, match="not found"):
        reporting.report_gazepoint_multiverse(
            frame,
            branch_col="missing",
            alpha=0.05,
        )


def _performance_trials():
    return pd.DataFrame(
        {
            "scale_id": ["a", "a", "b"],
            "total_rows": [100, 100, 200],
            "n_files": [1, 1, 2],
            "rows_per_file": [100, 100, 100],
            "operation": ["import", "import", "import"],
            "status": ["ok", "error", "ok"],
            "elapsed_s": [1.0, np.nan, 2.1],
            "heap_delta_mb": [10.0, np.nan, 21.0],
            "output_size_mb": [1.0, np.nan, 2.0],
            "trial": [1, 2, 1],
        }
    )


def test_reporting_performance_helpers_and_regression():
    trials = _performance_trials()

    summary = reporting._gp3_perf_summarise_trials(trials)

    assert len(summary) == 2

    with pytest.raises(ValueError, match="trial benchmark data is missing"):
        reporting._gp3_perf_summarise_trials(pd.DataFrame({"scale_id": ["a"]}))

    assert len(reporting._gp3_perf_as_summary({"summary": summary})) == 2

    with pytest.raises(TypeError, match="summary"):
        reporting._gp3_perf_as_summary({})

    with pytest.raises(ValueError, match="limits is missing"):
        reporting._gp3_perf_validate_limits(pd.DataFrame({"operation": ["x"]}))

    with pytest.raises(ValueError, match="positive finite"):
        reporting._gp3_perf_positive_scalar(
            "bad",
            "limit",
        )

    with pytest.raises(ValueError, match="positive finite"):
        reporting._gp3_perf_positive_scalar(
            0,
            "limit",
        )

    scaling = reporting._gp3_perf_scaling_exponents(summary)

    assert np.isfinite(scaling.loc[0, "scaling_exponent"])

    legacy = reporting.check_gazepoint_performance_regression(
        current=pd.DataFrame({"elapsed_seconds": [2.0]}),
        baseline=pd.DataFrame({"elapsed_seconds": [1.0]}),
    )

    assert bool(legacy.loc[0, "regression"])


def test_reporting_cluster_workflow_and_dashboard_paths(
    tmp_path,
    monkeypatch,
):
    simple = reporting.summarise_gazepoint_workflow("result")

    assert simple.loc[0, "status"] == "available"

    populated = tmp_path / "existing"
    populated.mkdir()
    (populated / "x.txt").write_text("x", encoding="utf-8")

    with pytest.raises(FileExistsError):
        reporting.export_gazepoint_cluster_results(
            pd.DataFrame({"x": [1]}),
            output_dir=populated,
        )

    fake = types.ModuleType("shiny")

    class FakeUI:
        @staticmethod
        def h2(value):
            return ("h2", value)

        @staticmethod
        def output_text_verbatim(value):
            return ("output", value)

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

    monkeypatch.setitem(
        sys.modules,
        "shiny",
        fake,
    )

    app = reporting.launch_gazepoint_qc_dashboard(
        pd.DataFrame(
            {
                "x": [1.0, np.nan],
            }
        )
    )

    assert isinstance(app, FakeApp)


# ============================================================================
# STATS
# ============================================================================


def test_stats_preparation_validation_and_alias_paths():
    pupil = stats_mod.prepare_gazepoint_pupil_gamm_data(
        pd.DataFrame(
            {
                "PUPIL": [1.0, 2.0],
                "TIME": [0.0, 1000.0],
            }
        ),
        pupil_col="PUPIL",
        time_col="TIME",
    )

    assert {"pupil", "time"}.issubset(pupil.columns)

    with pytest.raises(ValueError, match="AOI column required"):
        stats_mod.prepare_gazepoint_aoi_glmm_data(pd.DataFrame({"x": [1]}))

    aoi = stats_mod.prepare_gazepoint_aoi_glmm_data(
        pd.DataFrame(
            {
                "AOI": ["A", "A", "B"],
            }
        ),
        aoi_col="AOI",
    )

    assert aoi.attrs["target_aoi"] == "A"

    with pytest.raises(ValueError, match="time column required"):
        stats_mod.prepare_gazepoint_gca_data(pd.DataFrame({"x": [1.0]}))

    renamed = stats_mod.prepare_gazepoint_timecourse_test_data(
        pd.DataFrame(
            {
                "pid": ["S1"],
                "cond": ["A"],
                "t": [0.0],
                "v": [1.0],
            }
        ),
        subject_col="pid",
        condition_col="cond",
        time_col="t",
        value_col="v",
    )

    assert {
        "subject",
        "condition",
        "time",
        "value",
    }.issubset(renamed.columns)

    with pytest.raises(TypeError, match="Unexpected keyword"):
        stats_mod.prepare_gazepoint_timecourse_test_data(
            pd.DataFrame({"v": [1.0]}),
            value_col="v",
            nonsense=True,
        )


def test_stats_timecourse_aggregate_fallback():
    frame = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S1", "S1"],
            "condition": ["A", "B", "A", "B"],
            "time": [0, 0, 1, 1],
            "value": [1.0, 2.0, 1.5, 2.5],
        }
    )

    def r_style_mean(values, na_rm=None):
        if na_rm is None:
            raise TypeError("expects na_rm")
        return float(np.mean(values))

    out = stats_mod.prepare_gazepoint_timecourse_test_data(
        frame,
        subject_col="subject",
        condition_col="condition",
        time_col="time",
        outcome_col="value",
        aggregate_fun=r_style_mean,
        complete_only=True,
    )

    assert len(out) == 4


def test_stats_formula_family_and_wrapper_paths(monkeypatch):
    binary = pd.DataFrame(
        {
            "y": [0, 1, 0, 1, 0, 1, 0, 1],
            "x": [-2, -1, 0, 1, -1.5, -0.5, 0.5, 1.5],
        }
    )

    binomial = stats_mod._fit_formula(
        binary,
        formula="y ~ x",
        family="binomial",
    )

    assert hasattr(binomial, "params")

    poisson = stats_mod._fit_formula(
        pd.DataFrame(
            {
                "y": [0, 1, 2, 3, 1, 2, 4, 3],
                "x": list(range(8)),
            }
        ),
        formula="y ~ x",
        family="poisson",
    )

    assert hasattr(poisson, "params")

    called = []

    def fake_lmm(*args, **kwargs):
        called.append(("lmm", args, kwargs))
        return "lmm"

    monkeypatch.setattr(
        stats_mod,
        "fit_gazepoint_pupil_window_lmm",
        fake_lmm,
    )

    assert stats_mod.fit_gazepoint_face_window_lmm(pd.DataFrame({"x": [1]})) == "lmm"

    monkeypatch.setattr(
        stats_mod,
        "_fit_formula",
        lambda *args, **kwargs: ("formula", args, kwargs),
    )

    assert stats_mod.fit_gazepoint_aoi_window_glmm(pd.DataFrame({"x": [1]}))[0] == "formula"

    monkeypatch.setattr(
        stats_mod,
        "_fit_spline",
        lambda *args, **kwargs: ("spline", args, kwargs),
    )

    assert stats_mod.fit_gazepoint_pupil_gamm(pd.DataFrame({"x": [1]}))[0] == "spline"

    assert stats_mod.fit_gazepoint_aoi_gamm(pd.DataFrame({"x": [1]}))[0] == "spline"


def test_stats_optional_bambi_unavailable_and_fake_backend(
    monkeypatch,
):
    original = sys.modules.get("bambi")

    monkeypatch.setitem(
        sys.modules,
        "bambi",
        None,
    )

    unavailable = stats_mod.fit_gazepoint_brms_model(
        pd.DataFrame(
            {
                "y": [1.0, 2.0],
            }
        ),
        formula="y ~ 1",
    )

    assert unavailable["status"] == "optional-backend-unavailable"

    class FakeBambiModel:
        def __init__(self, formula, data, family):
            self.formula = formula
            self.data = data
            self.family = family

        def fit(self, **kwargs):
            return {
                "fit": True,
                "kwargs": kwargs,
            }

    fake = types.ModuleType("bambi")
    fake.Model = FakeBambiModel

    monkeypatch.setitem(
        sys.modules,
        "bambi",
        fake,
    )

    fitted = stats_mod.fit_gazepoint_brms_model(
        pd.DataFrame(
            {
                "y": [1.0, 2.0],
            }
        ),
        formula="y ~ 1",
        draws=10,
        nonsense=999,
    )

    assert fitted["backend"] == "bambi/pymc"
    assert fitted["idata"]["kwargs"]["draws"] == 10
    assert "nonsense" not in fitted["idata"]["kwargs"]

    if original is not None:
        monkeypatch.setitem(
            sys.modules,
            "bambi",
            original,
        )


def test_stats_model_helper_edge_paths(monkeypatch):
    assert stats_mod._gp3_model_collection({}) is None
    assert stats_mod._gp3_model_collection({"model": object()}) is None

    listed = stats_mod._gp3_model_collection([object(), object()])

    assert len(listed) == 2

    class Link:
        pass

    class Family:
        family = None
        link = Link()

    model = SimpleNamespace(
        family=Family(),
        formula="y ~ x",
        nobs="bad",
        df_resid="bad",
        aic="bad",
        bic=1,
        llf=2,
    )

    family, link = stats_mod._gp3_model_family_link(model)

    assert family == "Family"
    assert link == "Link"

    assert stats_mod._gp3_model_formula(model) == "y ~ x"

    info = stats_mod._gp3_model_info_table(
        model,
        "m",
    )

    assert pd.isna(info.loc[0, "n_observations"])

    monkeypatch.setitem(
        sys.modules,
        "arviz",
        None,
    )

    empty = stats_mod._gp3_model_legacy_summary(
        {
            "idata": object(),
        }
    )

    assert empty.empty

    class BadCI:
        params = np.array([1.0, 2.0])
        model = SimpleNamespace(
            exog_names=["x", "y"],
        )
        bse = np.array([0.1, 0.2])
        pvalues = np.array([0.01, 0.2])

        def conf_int(self):
            raise RuntimeError("no CI")

    legacy = stats_mod._gp3_model_legacy_summary(BadCI())

    assert legacy["term"].tolist() == ["x", "y"]

    assert stats_mod._gp3_model_pstars(np.nan) == ""
    assert stats_mod._gp3_model_pstars(0.0001) == "***"
    assert stats_mod._gp3_model_pstars(0.005) == "**"
    assert stats_mod._gp3_model_pstars(0.02) == "*"
    assert stats_mod._gp3_model_pstars(0.07) == "."
    assert stats_mod._gp3_model_pstars(0.2) == ""

    assert stats_mod._gp3_model_status(["unsupported_model_class"]) == "error"

    assert stats_mod._gp3_model_status(["singular_fit"]) == "diagnostic_warning"

    assert stats_mod._gp3_model_status(["ok"]) == "ok"

    assert stats_mod._gp3_model_status(["custom_status"]) == "custom_status"

    assert stats_mod._gp3_model_status([]) == "not_available"


def test_stats_fixed_effect_fallback_ci_and_validation():
    class FallbackModel:
        params = pd.Series(
            [1.0],
            index=["Intercept"],
        )
        bse = np.array([0.2])
        pvalues = np.array([0.04])
        zvalues = np.array([5.0])
        df_resid = "bad"

        def conf_int(self, alpha=None):
            raise RuntimeError("no ci")

    out = stats_mod._gp3_model_fixed_effects_r(
        FallbackModel(),
        "m",
        0.95,
        False,
        False,
    )

    assert len(out) == 1
    assert np.isfinite(out.loc[0, "conf_low"])

    dropped = stats_mod._gp3_model_fixed_effects_r(
        FallbackModel(),
        "m",
        0.95,
        False,
        True,
    )

    assert dropped.loc[0, "diagnostic_status"] == "not_available"

    with pytest.raises(TypeError, match="Unexpected argument"):
        stats_mod.summarise_gazepoint_fixed_effects(
            FallbackModel(),
            bad=True,
        )

    with pytest.raises(ValueError, match="conf_level"):
        stats_mod.summarise_gazepoint_fixed_effects(
            FallbackModel(),
            conf_level=2,
        )


def test_stats_emmeans_and_diagnostics_edge_paths():
    class Container:
        model = SimpleNamespace(
            data=SimpleNamespace(
                frame=pd.DataFrame(
                    {
                        "condition": ["A", "A", "B", "B"],
                        "pupil": [1.0, 2.0, 3.0, 4.0],
                    }
                )
            )
        )

    emmeans = stats_mod.summarise_gazepoint_emmeans(Container())

    assert len(emmeans) == 2

    assert stats_mod.summarise_gazepoint_emmeans(pd.DataFrame({"text": ["a"]})).empty

    with pytest.raises(ValueError, match="must not be None"):
        stats_mod._gp3_model_for_diagnostics(None)

    with pytest.raises(ValueError, match="must not be None"):
        stats_mod._gp3_model_for_diagnostics(
            {
                "model": None,
            }
        )

    with pytest.raises(ValueError, match="model_name"):
        stats_mod._gp3_model_for_diagnostics(
            object(),
            "",
        )

    mle = SimpleNamespace(
        mle_retvals={
            "converged": False,
        }
    )

    convergence = stats_mod.check_gazepoint_model_convergence(mle)

    assert convergence.loc[0, "diagnostic_status"] == ("convergence_warning")

    with pytest.raises(ValueError, match="tolerance"):
        stats_mod.check_gazepoint_model_singularity(
            object(),
            tolerance=0,
        )

    with pytest.raises(ValueError, match="ratio_threshold"):
        stats_mod.check_gazepoint_model_overdispersion(
            object(),
            ratio_threshold=0,
        )


class _ComparisonModel:
    def __init__(
        self,
        *,
        llf,
        n_params,
        aic=1.0,
        bic=2.0,
    ):
        self.llf = llf
        self.params = np.arange(n_params)
        self.aic = aic
        self.bic = bic
        self.nobs = 20


def test_stats_nested_model_comparison_status_paths():
    with pytest.raises(ValueError, match="non-empty"):
        stats_mod.compare_gazepoint_nested_models(
            [],
            comparison="sequential",
        )

    with pytest.raises(ValueError, match="comparison"):
        stats_mod.compare_gazepoint_nested_models(
            [_ComparisonModel(llf=1, n_params=1)],
            comparison="bad",
        )

    with pytest.raises(ValueError, match="name"):
        stats_mod.compare_gazepoint_nested_models(
            [_ComparisonModel(llf=1, n_params=1)],
            comparison="sequential",
            name="",
        )

    same_df = stats_mod.compare_gazepoint_nested_models(
        [
            _ComparisonModel(llf=1, n_params=1),
            _ComparisonModel(llf=2, n_params=1),
        ],
        comparison="sequential",
    )

    assert (
        same_df["lrt_table"].loc[
            0,
            "comparison_status",
        ]
        == "nonpositive_df_difference"
    )

    negative = stats_mod.compare_gazepoint_nested_models(
        [
            _ComparisonModel(llf=5, n_params=1),
            _ComparisonModel(llf=4, n_params=2),
        ],
        comparison="sequential",
    )

    assert (
        negative["lrt_table"].loc[
            0,
            "comparison_status",
        ]
        == "negative_lrt_statistic"
    )

    complete = stats_mod.compare_gazepoint_nested_models(
        [
            _ComparisonModel(llf=1, n_params=1),
            _ComparisonModel(llf=4, n_params=2),
        ],
        comparison="against_first",
    )

    assert (
        complete["lrt_table"].loc[
            0,
            "comparison_status",
        ]
        == "complete"
    )


def test_stats_recommendation_window_and_bootstrap_paths():
    unknown = stats_mod.recommend_gazepoint_model_family(
        pd.DataFrame(
            {
                "text": ["a", "b"],
            }
        )
    )

    assert unknown.loc[0, "family"] == "unknown"

    legacy = stats_mod.analyze_gazepoint_window(
        data=pd.DataFrame(
            {
                "value": [1.0, 2.0, 3.0],
            }
        ),
        value_col="value",
    )

    assert legacy["test"] is None

    with pytest.raises(ValueError, match="window_unit"):
        stats_mod.analyze_gazepoint_window(
            pd.DataFrame(
                {
                    "USER_ID": ["S1"],
                    "TIME": [0],
                    "pupil": [1.0],
                }
            ),
            value_cols=["pupil"],
            window_unit="bad",
        )

    with pytest.raises(ValueError, match="time_unit"):
        stats_mod.analyze_gazepoint_window(
            pd.DataFrame(
                {
                    "USER_ID": ["S1"],
                    "TIME": [0],
                    "pupil": [1.0],
                }
            ),
            value_cols=["pupil"],
            time_unit="bad",
        )

    window = stats_mod.analyze_gazepoint_window(
        pd.DataFrame(
            {
                "TIME": [0, 50, 100, 150],
                "pupil": [1.0, np.nan, 3.0, 4.0],
            }
        ),
        by=None,
        value_cols="pupil",
        window_size=100,
        step=50,
        summary_stats=[
            "mean",
            "sd",
            "median",
            "min",
            "max",
            "sum",
            "valid_prop",
        ],
        include_partial=True,
    )

    assert len(window) > 0

    with pytest.raises(ValueError, match="Unsupported"):
        stats_mod.analyze_gazepoint_window(
            pd.DataFrame(
                {
                    "TIME": [0, 1],
                    "pupil": [1.0, 2.0],
                }
            ),
            by=None,
            value_cols=["pupil"],
            summary_stats=["bad"],
        )

    with pytest.raises(ValueError, match="Two conditions"):
        stats_mod._bin_condition_difference(
            pd.DataFrame(
                {
                    "time_bin": [0, 1],
                    "condition": ["A", "A"],
                    "value": [1.0, 2.0],
                }
            )
        )

    boot = stats_mod.bootstrap_gazepoint_timecourse(
        pd.DataFrame(
            {
                "time_bin": [0, 0, 1, 1],
                "value": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        subject_col="missing",
        n_boot=2,
        random_state=1,
    )

    assert len(boot) == 2


def _valid_cluster_result():
    return {
        "timecourse": pd.DataFrame(
            {
                ".gp3_cluster_time_bin": [0.0, 10.0],
                "n_subjects": [10, 10],
                "mean_difference": [0.1, 0.2],
                "statistic": [1.0, 2.0],
                "cluster_id": [pd.NA, pd.NA],
                "point_candidate": [False, False],
            }
        ),
        "clusters": pd.DataFrame(),
        "permutation_distribution": pd.DataFrame(
            {
                "permutation": [1, 2],
                "max_cluster_statistic": [0.5, 1.0],
            }
        ),
        "settings": {
            "n_permutations": 2,
        },
        "model_status": "ok",
    }


def test_stats_cluster_summary_validation_and_empty_paths():
    with pytest.raises(ValueError, match="cluster-permutation"):
        stats_mod.summarise_gazepoint_clusters(
            pd.DataFrame(),
            alpha=0.05,
        )

    with pytest.raises(ValueError, match="missing required element"):
        stats_mod.summarise_gazepoint_clusters(
            {},
            alpha=0.05,
        )

    with pytest.raises(ValueError, match="include_timecourse"):
        stats_mod.summarise_gazepoint_clusters(
            _valid_cluster_result(),
            alpha=0.05,
            include_timecourse="yes",
        )

    bad_time = _valid_cluster_result()
    bad_time["timecourse"] = pd.DataFrame(
        {
            ".gp3_cluster_time_bin": [0.0],
        }
    )

    with pytest.raises(ValueError, match="timecourse"):
        stats_mod.summarise_gazepoint_clusters(
            bad_time,
            alpha=0.05,
        )

    bad_perm = _valid_cluster_result()
    bad_perm["permutation_distribution"] = pd.DataFrame(
        {
            "permutation": [1],
        }
    )

    with pytest.raises(ValueError, match="permutation_distribution"):
        stats_mod.summarise_gazepoint_clusters(
            bad_perm,
            alpha=0.05,
        )

    bad_cluster = _valid_cluster_result()
    bad_cluster["clusters"] = pd.DataFrame(
        {
            "cluster_id": [1],
        }
    )

    with pytest.raises(ValueError, match="clusters"):
        stats_mod.summarise_gazepoint_clusters(
            bad_cluster,
            alpha=0.05,
        )

    empty = stats_mod.summarise_gazepoint_clusters(
        _valid_cluster_result(),
        alpha=0.05,
        include_timecourse=True,
    )

    assert empty["clusters"].empty
    assert "timecourse" in empty

    no_timecourse = stats_mod.summarise_gazepoint_clusters(
        _valid_cluster_result(),
        alpha=0.05,
        include_timecourse=False,
    )

    assert "timecourse" not in no_timecourse

    with pytest.raises(ValueError, match="clusters element"):
        stats_mod.summarize_gazepoint_time_clusters(
            {},
            alpha=0.05,
        )

    blank = stats_mod.summarize_gazepoint_time_clusters(
        {
            "clusters": pd.DataFrame(),
        },
        alpha=0.05,
    )

    assert blank.empty


def test_stats_grid_multiverse_and_cluster_report_paths(
    monkeypatch,
):
    grid = stats_mod.audit_gazepoint_timecourse_grid(
        pd.DataFrame(
            {
                "time_bin": [0, 1],
            }
        ),
        subject_col="",
        condition_col="",
    )

    assert grid.loc[0, "expected_bins"] == 2

    import gp3tools.pupil as pupil

    monkeypatch.setattr(
        pupil,
        "preprocess_gazepoint_signals",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("deliberate")),
    )

    failed = stats_mod.run_gazepoint_pupil_multiverse(
        pd.DataFrame(
            {
                "pupil": [1.0],
            }
        ),
        registry=pd.DataFrame(
            {
                "physiological_min": [1.0],
            }
        ),
    )

    assert failed.loc[0, "status"] == "error"

    plain = stats_mod.summarise_gazepoint_multiverse_results(
        data=pd.DataFrame(
            {
                "status": ["ok", "error"],
            }
        )
    )

    assert plain.loc[0, "n_specifications"] == 2

    result = stats_mod.summarise_gazepoint_multiverse_results(
        results={
            "x": {
                "_gp3_class": "gp3_unknown",
                "overview": {},
            }
        }
    )

    assert result["overview"].iloc[-1]["multiverse_status"] == "not_run"

    no_clusters = stats_mod.report_gazepoint_cluster_permutation(
        {
            "clusters": pd.DataFrame(),
            "settings": {},
        },
        alpha=0.05,
    )

    assert "did not identify" in no_clusters["report_text"]

    nonsig = stats_mod.report_gazepoint_cluster_permutation(
        {
            "clusters": pd.DataFrame(
                {
                    "cluster_id": [1],
                    "start_time_bin": [0],
                    "end_time_bin": [1],
                    "p_value": [0.5],
                }
            ),
            "settings": {},
        },
        alpha=0.05,
    )

    assert "none reached" in nonsig["report_text"]
