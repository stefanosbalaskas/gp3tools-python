from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3

ORACLE_PATH = Path(__file__).parent / "oracles" / "r_v2_3_0_behavioral_r1.csv"


def _oracle_rows(function_name: str, case_name: str) -> dict[str, tuple[str, str]]:
    if not ORACLE_PATH.exists():
        pytest.fail(f"R v2.3.0 oracle is missing: {ORACLE_PATH}")
    rows: dict[str, tuple[str, str]] = {}
    with ORACLE_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["function_name"] == function_name and row["case_name"] == case_name:
                rows[row["key"]] = (row["value_type"], row["value"])
    if not rows:
        pytest.fail(f"No oracle rows for {function_name}/{case_name}")
    return rows


def _assert_value(actual, expected: tuple[str, str], *, atol: float = 1e-10) -> None:
    value_type, raw = expected
    if value_type == "na":
        assert pd.isna(actual)
    elif value_type == "bool":
        assert bool(actual) is (raw == "TRUE")
    elif value_type == "int":
        assert int(actual) == int(raw)
    elif value_type == "float":
        assert float(actual) == pytest.approx(float(raw), rel=1e-10, abs=atol)
    elif value_type == "str":
        assert str(actual) == raw
    else:
        raise AssertionError(f"Unknown oracle type: {value_type!r}")


def _assert_frame_header(out: pd.DataFrame, oracle: dict[str, tuple[str, str]]) -> None:
    _assert_value(len(out), oracle["nrow"])
    _assert_value("|".join(map(str, out.columns)), oracle["columns"])


def test_r230_behavioral_r1_analyze_gazepoint_window():
    data = pd.DataFrame(
        {
            "USER_ID": ["P01"] * 5,
            "TIME": [0, 50, 100, 150, 200],
            "mean_pupil": [1.0, 2.0, np.nan, 4.0, 5.0],
        }
    )
    out = gp3.analyze_gazepoint_window(
        data,
        window_size=100,
        step=50,
        summary_stats=["mean", "sd", "valid_prop"],
        by="USER_ID",
        value_cols="mean_pupil",
        ts_col="TIME",
        window_unit="milliseconds",
        time_unit="milliseconds",
        include_partial=False,
    )
    oracle = _oracle_rows("analyze_gazepoint_window", "canonical")
    _assert_frame_header(out, oracle)
    for i in range(len(out)):
        for column in (
            "window_start",
            "window_end",
            "n_samples",
            "mean_pupil_mean",
            "mean_pupil_sd",
            "mean_pupil_valid_prop",
        ):
            _assert_value(out.iloc[i][column], oracle[f"r{i + 1}.{column}"])


def test_r230_behavioral_r1_compute_sequence_metrics():
    data = pd.DataFrame(
        {
            "subject": ["S1"] * 6 + ["S2"] * 2,
            "trial": ["T1"] * 8,
            "time": [1, 2, 3, 4, 5, 6, 1, 2],
            "AOI": ["A", "A", "B", "A", "C", "C", pd.NA, ""],
        }
    )
    out = gp3.compute_gazepoint_aoi_sequence_metrics(
        data,
        aoi_col="AOI",
        group_cols=["subject", "trial"],
        time_col="time",
        include_missing=False,
        collapse_repeats=True,
    )
    oracle = _oracle_rows("compute_gazepoint_aoi_sequence_metrics", "canonical")
    _assert_frame_header(out, oracle)
    columns = (
        "subject",
        "sequence_length",
        "n_aoi_visits",
        "n_unique_aoi",
        "transition_count",
        "revisit_count",
        "revisit_prop",
        "dominant_aoi",
        "first_aoi",
        "last_aoi",
        "mean_run_length",
        "max_run_length",
        "sequence_status",
    )
    for i in range(len(out)):
        for column in columns:
            _assert_value(out.iloc[i][column], oracle[f"r{i + 1}.{column}"])


def test_r230_behavioral_r1_analysis_decision_audit():
    results = {
        "confirmatory": {
            "model_status": "ok",
            "model": {"dummy": 1},
            "diagnostics": {
                "checks": pd.DataFrame(
                    {
                        "diagnostic_status": ["ok", "warning"],
                        "message": ["", "small warning"],
                    }
                )
            },
        },
        "sensitivity": pd.DataFrame({"summary_status": ["ok"]}),
    }
    branch_roles = pd.DataFrame(
        {
            "branch_name": ["confirmatory", "sensitivity"],
            "decision_type": ["confirmatory", "sensitivity"],
            "analysis_family": ["model", "robustness"],
        }
    )
    out = gp3.create_gazepoint_analysis_decision_audit(
        results=results,
        branch_roles=branch_roles,
        required_confirmatory=["confirmatory"],
        diagnostics_required=True,
        require_clean_diagnostics=False,
    )
    oracle = _oracle_rows("create_gazepoint_analysis_decision_audit", "canonical")
    _assert_value("|".join(out.keys()), oracle["keys"])
    for column in (
        "n_branches",
        "n_confirmatory",
        "n_sensitivity",
        "n_diagnostic_warnings",
        "n_cautions",
        "readiness_status",
        "readiness_message",
    ):
        _assert_value(
            out["overview"].iloc[0][column],
            oracle[f"overview.{column}"],
        )
    diagnostic = (
        out["diagnostics_summary"]
        .loc[out["diagnostics_summary"]["branch_name"].eq("confirmatory")]
        .iloc[0]
    )
    _assert_value(
        diagnostic["diagnostic_status"],
        oracle["diagnostics.confirmatory.status"],
    )
    _assert_value(
        diagnostic["n_warning"],
        oracle["diagnostics.confirmatory.n_warning"],
    )
    _assert_value(
        out["readiness"].iloc[0]["readiness_status"],
        oracle["readiness.status"],
    )


def test_r230_behavioral_r1_event_review_template():
    data = pd.DataFrame(
        {
            "USER_ID": ["P01", "P01", "P02", "P02"],
            "trial": ["T1"] * 4,
            "TIME": [0, 100, 10, 110],
        }
    )
    out = gp3.create_gazepoint_event_review_template(
        data,
        trial_col="trial",
        rows_per_sequence=2,
        reviewer="AB",
    )
    oracle = _oracle_rows("create_gazepoint_event_review_template", "canonical")
    _assert_frame_header(out, oracle)
    for i in range(len(out)):
        for column in (
            "USER_ID",
            "review_event_id",
            "sequence_start",
            "sequence_end",
            "event_type",
            "review_status",
            "reviewer",
        ):
            _assert_value(out.iloc[i][column], oracle[f"r{i + 1}.{column}"])


def test_r230_behavioral_r1_cnn_uncertainty():
    data = pd.DataFrame(
        {
            "x": [0.1, 0.2, np.nan, 0.4],
            "y": [0.2, 0.3, 0.4, 0.5],
            "u": [0.2, 0.4, 0.3, 0.8],
        }
    )
    out = gp3.filter_gazepoint_cnn_uncertainty(
        data, x="x", y="y", uncertainty="u", max_uncertainty=0.5
    )
    oracle = _oracle_rows("filter_gazepoint_cnn_uncertainty", "canonical")
    _assert_frame_header(out, oracle)
    for i in range(len(out)):
        _assert_value(out.iloc[i]["cnn_uncertainty_weight"], oracle[f"r{i + 1}.weight"])
        _assert_value(out.iloc[i]["cnn_valid_frame"], oracle[f"r{i + 1}.valid"])


def test_r230_behavioral_r1_sequence_anomalies():
    data = pd.DataFrame(
        {
            "subject": ["S1"] * 4 + ["S2"] * 4 + ["S3"] * 4,
            "time": [1, 2, 3, 4] * 3,
            "AOI": ["A", "A", "B", pd.NA, "A", pd.NA, pd.NA, pd.NA, "A", "B", "C", "D"],
        }
    )
    out = gp3.flag_gazepoint_sequence_anomalies(
        data,
        aoi_col="AOI",
        group_cols="subject",
        time_col="time",
        min_length=2,
        max_length=3,
        max_missing_prop=0.5,
        z_threshold=1,
        min_unique_aoi=1,
    )
    oracle = _oracle_rows("flag_gazepoint_sequence_anomalies", "canonical")
    _assert_frame_header(out, oracle)
    columns = (
        "subject",
        "sequence_length",
        "missing_prop",
        "n_unique_aoi",
        "length_z",
        "flag_short",
        "flag_long",
        "flag_high_missing",
        "flag_length_outlier",
        "flag_low_unique",
        "anomaly_flag",
        "anomaly_reason",
        "anomaly_status",
    )
    for i in range(len(out)):
        for column in columns:
            _assert_value(out.iloc[i][column], oracle[f"r{i + 1}.{column}"])


def test_r230_behavioral_r1_tracking_quality():
    quality = pd.DataFrame(
        {
            "USER_FILE": ["P1", "P2"],
            "MEDIA_ID": ["M1", "M1"],
            "FPOGV_valid_pct": [85, 60],
            "LPV_valid_pct": [90, 80],
            "RPV_valid_pct": [88, 65],
        }
    )
    sampling = pd.DataFrame(
        {
            "USER_FILE": ["P1", "P2"],
            "MEDIA_ID": ["M1", "M1"],
            "estimated_hz": [60, 50],
            "duration_sec": [10, 2],
        }
    )
    out = gp3.flag_tracking_quality(
        quality,
        sampling,
        min_gaze_valid_pct=70,
        min_pupil_valid_pct=70,
        expected_hz=60,
        hz_tolerance=5,
        min_duration_sec=5,
    )
    oracle = _oracle_rows("flag_tracking_quality", "canonical")
    _assert_frame_header(out, oracle)
    columns = (
        "USER_FILE",
        "min_pupil_valid_pct_observed",
        "flag_low_gaze_validity",
        "flag_low_pupil_validity",
        "flag_sampling_rate",
        "flag_short_duration",
        "review_required",
    )
    for i in range(len(out)):
        for column in columns:
            _assert_value(out.iloc[i][column], oracle[f"r{i + 1}.{column}"])


def test_r230_behavioral_r1_pchip():
    data = pd.DataFrame(
        {
            "subject": ["P1"] * 4,
            "time": [0, 100, 200, 300],
            "pupil": [1.0, np.nan, 3.0, 4.0],
        }
    )
    out = gp3.interpolate_gazepoint_pupil_pchip(
        data,
        pupil_col="pupil",
        time_col="time",
        grouping_cols="subject",
        max_gap_ms=150,
        max_gap_samples=1,
        min_valid_points=3,
    )
    oracle = _oracle_rows("interpolate_gazepoint_pupil_pchip", "canonical")
    _assert_frame_header(out, oracle)
    columns = (
        "pupil_interpolated_pchip",
        "interpolated_pupil_pchip",
        "pchip_gap_id",
        "pchip_gap_n_samples",
        "pchip_gap_duration_ms",
        "pchip_gap_within_limit",
        "pchip_interpolation_status",
    )
    for i in range(len(out)):
        for column in columns:
            _assert_value(
                out.iloc[i][column],
                oracle[f"r{i + 1}.{column}"],
                atol=1e-9,
            )


def test_r230_behavioral_r1_inspect_columns():
    data = pd.DataFrame(
        {
            "TIME": [0.0, 1.0],
            "FPOGX": [0.1, np.nan],
            "LPMM": [3.0, 4.0],
            "foo": ["a", "b"],
        }
    )
    out = gp3.inspect_gazepoint_columns(data)
    oracle = _oracle_rows("inspect_gazepoint_columns", "canonical")
    _assert_value(len(out), oracle["nrow"])
    r_columns = oracle["columns"][1].split("|")
    assert list(map(str, out.columns[: len(r_columns)])) == r_columns
    for i in range(len(out)):
        for column in ("column", "semantic_group", "dtype", "n_missing", "pct_missing"):
            _assert_value(out.iloc[i][column], oracle[f"r{i + 1}.{column}"])


def test_r230_behavioral_r1_prepare_timecourse():
    data = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S1", "S1", "S1", "S2", "S2", "S2"],
            "condition": ["A", "A", "B", "A", "B", "A", "B", "A"],
            "time": [1, 1, 1, 2, 2, 1, 1, 2],
            "value": [1, 3, 2, 4, 6, 5, 7, 9],
        }
    )
    out = gp3.prepare_gazepoint_timecourse_test_data(
        data,
        subject_col="subject",
        condition_col="condition",
        time_col="time",
        outcome_col="value",
        condition_order=["A", "B"],
        aggregate_fun=np.mean,
        complete_only=True,
    )
    oracle = _oracle_rows("prepare_gazepoint_timecourse_test_data", "canonical")
    _assert_frame_header(out, oracle)
    columns = (
        ".gp3_cluster_subject",
        ".gp3_cluster_condition",
        ".gp3_cluster_time_bin",
        ".gp3_cluster_outcome",
        ".gp3_cluster_status",
    )
    for i in range(len(out)):
        for column in columns:
            _assert_value(out.iloc[i][column], oracle[f"r{i + 1}.{column}"])


def test_r230_behavioral_r1_exclusions():
    data = pd.DataFrame(
        {
            "participant": ["P1"] * 6 + ["P2"] * 6,
            "trial": ["T1"] * 3 + ["T2"] * 3 + ["T1"] * 3 + ["T2"] * 3,
            "condition": ["A"] * 3 + ["B"] * 3 + ["A"] * 3 + ["B"] * 3,
            "valid": [True, True, True, True, False, False, True, True, True, True, True, True],
            "x": [1, 1, 1, 1, np.nan, np.nan, 1, 1, 1, 1, 1, 1],
            "y": [1] * 12,
            "pupil": [3] * 12,
            "artifact": [False] * 9 + [True, True, False],
        }
    )
    out = gp3.recommend_gazepoint_exclusions(
        data,
        participant_col="participant",
        trial_col="trial",
        condition_col="condition",
        validity_col="valid",
        x_col="x",
        y_col="y",
        pupil_col="pupil",
        artifact_col="artifact",
        min_trial_samples=3,
        max_trial_missing_prop=0.5,
        max_trial_artifact_prop=0.5,
        min_participant_trials=2,
        min_participant_valid_trials=1,
        max_participant_missing_prop=0.5,
        max_participant_artifact_prop=0.5,
    )
    oracle = _oracle_rows("recommend_gazepoint_exclusions", "canonical")
    r_keys = oracle["keys"][1].split("|")
    assert list(out.keys())[: len(r_keys)] == r_keys
    assert out["exclusions"] is out["exclusion_table"]
    for column in (
        "n_input_rows",
        "n_participants",
        "n_trials",
        "n_recommended_participant_exclusions",
        "n_recommended_trial_exclusions",
        "min_trial_samples",
    ):
        _assert_value(out["overview"].iloc[0][column], oracle[f"overview.{column}"])

    for case, frame_name, columns in (
        (
            "trial_recommendations",
            "trial_recommendations",
            (
                "participant",
                "trial",
                "condition",
                "n_samples",
                "n_missing_or_unusable",
                "missing_or_unusable_prop",
                "n_artifact",
                "artifact_prop",
                "n_usable",
                "exclusion_reason",
                "recommend_exclude",
                "recommendation_status",
            ),
        ),
        (
            "participant_recommendations",
            "participant_recommendations",
            (
                "participant",
                "n_trials",
                "n_trial_exclusions",
                "n_retained_trials",
                "missing_or_unusable_prop",
                "artifact_prop",
                "exclusion_reason",
                "recommend_exclude",
                "recommendation_status",
            ),
        ),
    ):
        case_oracle = _oracle_rows("recommend_gazepoint_exclusions", case)
        frame = out[frame_name]
        _assert_frame_header(frame, case_oracle)
        for i in range(len(frame)):
            for column in columns:
                _assert_value(frame.iloc[i][column], case_oracle[f"r{i + 1}.{column}"])


def test_r230_behavioral_r1_adaptive_trial():
    data = pd.DataFrame(
        {
            "id": ["a", "b", "c"],
            "mu": [0.2, 0.5, 0.4],
            "sig": [0.1, 0.05, 0.3],
        }
    )
    out = gp3.select_gazepoint_adaptive_trial(
        data,
        mean="mu",
        sd="sig",
        acquisition="ucb",
        kappa=2,
        maximize=True,
    )
    oracle = _oracle_rows("select_gazepoint_adaptive_trial", "ucb")
    _assert_frame_header(out, oracle)
    _assert_value(out.iloc[0]["id"], oracle["id"])
    _assert_value(out.iloc[0]["acquisition_score"], oracle["acquisition_score"])
