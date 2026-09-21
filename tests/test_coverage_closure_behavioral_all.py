from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

r2 = importlib.import_module("gp3tools._behavioral_r2")
r3a = importlib.import_module("gp3tools._behavioral_r3a")
r3b = importlib.import_module("gp3tools._behavioral_r3b")


# ============================================================================
# R2 HELPERS / VALIDATION
# ============================================================================


def test_r2_scalar_helpers_and_status_contracts():
    assert pd.isna(r2._collapse(None))
    assert pd.isna(r2._collapse([]))
    assert r2._collapse(3) == "3"
    assert r2._collapse(["a", 2]) == "a, 2"

    assert r2._r_bool(pd.Series([True, False, pd.NA], dtype="boolean")).tolist() == [
        True,
        False,
        False,
    ]
    assert r2._r_bool(pd.Series([1, 0, np.nan])).tolist() == [True, False, False]
    assert r2._r_bool(pd.Series(["yes", "no", "valid", "bad"])).tolist() == [
        True,
        False,
        True,
        False,
    ]

    assert pd.isna(r2._r_sd([1.0]))
    assert r2._r_sd([1.0, 3.0]) == pytest.approx(np.sqrt(2.0))

    assert r2._r2_v5_chars(None) == []
    assert r2._r2_v5_chars("x") == ["x"]
    assert r2._r2_v5_chars([1, "x"]) == ["1", "x"]

    assert pd.isna(r2._r2_v5_num_text(None))
    assert r2._r2_v5_num_text("abc") == "abc"
    assert pd.isna(r2._r2_v5_num_text(np.inf))
    assert r2._r2_v5_num_text(2.0) == "2"
    assert r2._r2_v5_num_text(2.25) == "2.25"

    names = ["media_id", "subject", "time"]
    assert r2._r2_event_sync_resolve_col("MEDIA_ID", names, "x") == "media_id"
    assert r2._r2_event_sync_resolve_col("USER_FILE", names, "x") == "subject"
    with pytest.raises(ValueError, match="non-missing character"):
        r2._r2_event_sync_resolve_col("", names, "x")
    with pytest.raises(ValueError, match="present in"):
        r2._r2_event_sync_resolve_col("missing", names, "x")

    common = dict(
        n_samples=10,
        n_finite_time=10,
        has_event_col=True,
        n_events=2,
        n_missing_expected=0,
        onset_count=1,
        response_count=1,
        n_duplicate_time=0,
        has_large_gap=False,
        min_samples_per_unit=2,
    )
    expected = [
        ({"n_samples": 1}, "too_few_samples"),
        ({"n_finite_time": 0}, "missing_time"),
        ({"n_duplicate_time": 1}, "duplicate_time_values"),
        ({"has_large_gap": True}, "large_time_gap"),
        ({"has_event_col": False}, "event_column_not_available"),
        ({"n_events": 0}, "no_events_observed"),
        ({"n_missing_expected": 1}, "missing_expected_events"),
        ({"onset_count": 0}, "missing_onset_event"),
        ({"response_count": 0}, "missing_response_event"),
        ({}, "ok"),
    ]
    for override, status in expected:
        args = {**common, **override}
        assert r2._r2_event_sync_status(**args) == status

    assert pd.isna(r2._r2_detector_threshold("hmm", "hmm"))
    assert pd.isna(r2._r2_detector_threshold("custom", "velocity"))
    assert r2._r2_detector_threshold("velocity_10", "velocity") == 10.0
    assert pd.isna(r2._r2_detector_threshold("velocity_bad", "velocity"))

    empty = pd.DataFrame(columns=["start_time", "end_time"])
    assert r2._r2_detector_best_overlap(empty, empty).size == 0

    one = pd.DataFrame({"start_time": [0.0], "end_time": [1.0]})
    assert r2._r2_detector_best_overlap(one, empty).tolist() == [0.0]
    assert r2._r2_detector_best_overlap(one, one).tolist() == [1.0]

    assert r2._r2_detector_sequence_keys(pd.DataFrame(), ["subject"]) == []
    frame = pd.DataFrame({"subject": ["S1", pd.NA], "trial": [1, 2]})
    assert r2._r2_detector_sequence_keys(frame, []) == [".all", ".all"]
    assert r2._r2_detector_sequence_keys(frame, ["subject", "trial"]) == [
        "S1\r1",
        "<NA>\r2",
    ]


def test_r2_condition_quality_validation_and_ratio_paths():
    base = pd.DataFrame(
        {
            "MEDIA_ID": ["M1", "M1", "M2", "M2"],
            "USER_FILE": ["S1", "S2", "S3", "S4"],
            "condition": ["A", "A", "B", "B"],
            "zero_metric": [0.0, 0.0, 0.0, 0.0],
            "inf_ratio_metric": [0.0, 0.0, 1.0, 1.0],
            "negative_metric": [-1.0, -1.0, 1.0, 1.0],
            "missing_metric": [np.nan, np.nan, np.nan, np.nan],
        }
    )

    out = r2.audit_condition_quality_imbalance(
        base,
        quality_cols=[
            "zero_metric",
            "inf_ratio_metric",
            "negative_metric",
            "missing_metric",
        ],
        subject_col="subject",
        max_mean_difference=100,
        max_condition_ratio=2,
    )

    metrics = out["metric_summary"].set_index("quality_metric")
    assert metrics.loc["zero_metric", "condition_ratio"] == 1.0
    assert np.isinf(metrics.loc["inf_ratio_metric", "condition_ratio"])
    assert pd.isna(metrics.loc["negative_metric", "condition_ratio"])
    assert metrics.loc["missing_metric", "condition_quality_imbalance_status"] == (
        "insufficient_data"
    )

    with pytest.raises(ValueError, match="condition_col"):
        r2.audit_condition_quality_imbalance(base, condition_col="")
    with pytest.raises(ValueError, match="subject_col"):
        r2.audit_condition_quality_imbalance(base, subject_col="missing")
    with pytest.raises(ValueError, match="could not be detected"):
        r2.audit_condition_quality_imbalance(
            pd.DataFrame({"condition": ["A"]}),
        )
    with pytest.raises(ValueError, match="missing quality"):
        r2.audit_condition_quality_imbalance(
            base,
            quality_cols=["missing"],
        )
    with pytest.raises(ValueError, match="must be numeric"):
        r2.audit_condition_quality_imbalance(
            base.assign(text=["x"] * 4),
            quality_cols=["text"],
        )
    with pytest.raises(ValueError, match="min_units_per_condition"):
        r2.audit_condition_quality_imbalance(
            base,
            quality_cols=["zero_metric"],
            min_units_per_condition=0,
        )
    with pytest.raises(ValueError, match="max_mean_difference"):
        r2.audit_condition_quality_imbalance(
            base,
            quality_cols=["zero_metric"],
            max_mean_difference=-1,
        )
    with pytest.raises(ValueError, match="max_condition_ratio"):
        r2.audit_condition_quality_imbalance(
            base,
            quality_cols=["zero_metric"],
            max_condition_ratio=0,
        )
    with pytest.raises(ValueError, match="at least one usable"):
        r2.audit_condition_quality_imbalance(
            base.assign(condition=pd.NA),
            quality_cols=["zero_metric"],
        )


def test_r2_blink_interpolation_full_edge_matrix():
    master = pd.DataFrame(
        {
            "USER_ID": ["S1"] * 6,
            "TIME": [0, 1, 2, 3, 4, 5],
            "mean_pupil": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )

    blink = pd.DataFrame(
        {
            "USER_ID": ["S1"],
            "start_time": [2],
            "end_time": [3],
        }
    )

    auto = r2.interpolate_blinks(
        master,
        blink,
        pupil_cols=None,
        time_unit="auto",
        keep_mask=False,
    )
    assert "blink_masked" not in auto
    assert "blink_interpolated" not in auto
    assert auto["mean_pupil_blink_interp"].notna().all()

    spline = r2.interpolate_blinks(
        master,
        blink,
        pupil_cols="mean_pupil",
        time_unit="milliseconds",
        method="spline",
    )
    assert spline["blink_interpolated"].sum() == 2

    seconds = r2.interpolate_blinks(
        master.assign(TIME=np.arange(6) / 100.0),
        pd.DataFrame(
            {
                "USER_ID": ["S1"],
                "start_time": [0.02],
                "end_time": [0.03],
            }
        ),
        pupil_cols="mean_pupil",
        time_unit="seconds",
    )
    assert seconds["blink_masked"].sum() == 2

    no_match = r2.interpolate_blinks(
        master,
        pd.DataFrame(
            {
                "USER_ID": ["OTHER"],
                "start_time": [2],
                "end_time": [3],
            }
        ),
        pupil_cols="mean_pupil",
        time_unit="milliseconds",
    )
    assert no_match["blink_masked"].sum() == 0

    unbounded = r2.interpolate_blinks(
        master,
        pd.DataFrame(
            {
                "USER_ID": ["S1"],
                "start_time": [0],
                "end_time": [1],
            }
        ),
        pupil_cols="mean_pupil",
        time_unit="milliseconds",
    )
    assert unbounded["blink_masked"].sum() == 2
    assert unbounded["blink_interpolated"].sum() == 0

    with pytest.raises(ValueError, match="No pupil columns"):
        r2.interpolate_blinks(
            master.drop(columns=["mean_pupil"]),
            blink,
        )

    with pytest.raises(ValueError, match="master_df"):
        r2.interpolate_blinks(
            master.drop(columns=["TIME"]),
            blink,
            pupil_cols="mean_pupil",
        )

    with pytest.raises(ValueError, match="blink_df"):
        r2.interpolate_blinks(
            master,
            blink.drop(columns=["start_time"]),
            pupil_cols="mean_pupil",
        )

    with pytest.raises(ValueError, match="time_unit"):
        r2.interpolate_blinks(
            master,
            blink,
            pupil_cols="mean_pupil",
            time_unit="bad",
        )

    with pytest.raises(ValueError, match="method"):
        r2.interpolate_blinks(
            master,
            blink,
            pupil_cols="mean_pupil",
            time_unit="milliseconds",
            method="bad",
        )


def test_r2_multimodal_join_scaling_and_validation_paths():
    face = pd.DataFrame(
        {
            "pid": ["S1", "S2"],
            "constant": [2.0, 2.0],
            "signal": [1.0, 3.0],
            "text": ["a", "b"],
        }
    )
    gaze = pd.DataFrame(
        {
            "subject": ["S1", "S2"],
            "dwell": [0.2, 0.4],
        }
    )
    response = pd.DataFrame(
        {
            "subject": ["S1", "S2"],
            "rating": [5.0, np.nan],
        }
    )

    merged = r2.prepare_multimodal_data(
        face,
        gaze_data=gaze,
        response_data=response,
        by=["pid"],
        gaze_by={"pid": "subject"},
        response_by={"pid": "subject"},
        predictor_cols=["constant", "signal", "dwell"],
        outcome_cols=["rating"],
        scale_predictors=True,
        drop_missing_outcomes=True,
        keep_all=False,
    )
    assert len(merged) == 1
    assert merged["constant_z"].isna().all()
    assert "signal_z" in merged

    auto = r2.prepare_multimodal_data(
        pd.DataFrame(
            {
                "participant_id": ["S1", "S2"],
                "signal": [1.0, 2.0],
            }
        ),
        by=None,
        predictor_cols=None,
        scale_predictors=False,
    )
    assert len(auto) == 2

    with pytest.raises(ValueError, match="at least one join"):
        r2.prepare_multimodal_data(
            pd.DataFrame({"x": [1]}),
            by=None,
        )

    with pytest.raises(ValueError, match="left table"):
        r2.prepare_multimodal_data(
            face,
            gaze_data=gaze,
            by=["pid"],
            gaze_by={"missing": "subject"},
        )

    with pytest.raises(ValueError, match="right table"):
        r2.prepare_multimodal_data(
            face,
            gaze_data=gaze,
            by=["pid"],
            gaze_by={"pid": "missing"},
        )

    with pytest.raises(ValueError, match="Requested column"):
        r2.prepare_multimodal_data(
            face,
            by=["pid"],
            outcome_cols=["missing"],
        )

    with pytest.raises(ValueError, match="Predictor column not found"):
        r2.prepare_multimodal_data(
            face,
            by=["pid"],
            predictor_cols=["missing"],
        )

    with pytest.raises(ValueError, match="must be numeric"):
        r2.prepare_multimodal_data(
            face,
            by=["pid"],
            predictor_cols=["text"],
        )


def test_r2_recalibration_validation_paths():
    base = pd.DataFrame(
        {
            "x": [0.1, 0.2, 0.3],
            "y": [0.1, 0.2, 0.3],
            "tx": [0.2, 0.3, 0.4],
            "ty": [0.2, 0.3, 0.4],
            "time": [0.0, 1.0, 2.0],
            "group": ["A", "A", "A"],
            "cal": [True, True, True],
        }
    )

    with pytest.raises(ValueError, match="at least one row"):
        r2.recalibrate_gaze(
            base.iloc[0:0],
            "x",
            "y",
            "tx",
            "ty",
        )

    with pytest.raises(ValueError, match="method"):
        r2.recalibrate_gaze(
            base,
            "x",
            "y",
            "tx",
            "ty",
            method="bad",
        )

    with pytest.raises(ValueError, match="Column not found"):
        r2.recalibrate_gaze(
            base,
            "missing",
            "y",
            "tx",
            "ty",
        )

    with pytest.raises(ValueError, match="Grouping column"):
        r2.recalibrate_gaze(
            base,
            "x",
            "y",
            "tx",
            "ty",
            grouping_cols=["missing"],
        )

    with pytest.raises(ValueError, match="Calibration column"):
        r2.recalibrate_gaze(
            base,
            "x",
            "y",
            "tx",
            "ty",
            calibration_col="missing",
        )

    with pytest.raises(ValueError, match="unique"):
        r2.recalibrate_gaze(
            base,
            "x",
            "y",
            "tx",
            "ty",
            output_x_col="same",
            output_y_col="same",
        )

    existing = base.assign(gaze_x_recalibrated=0.0)
    with pytest.raises(ValueError, match="already exist"):
        r2.recalibrate_gaze(
            existing,
            "x",
            "y",
            "tx",
            "ty",
        )

    with pytest.raises(ValueError, match="time_col"):
        r2.recalibrate_gaze(
            base,
            "x",
            "y",
            "tx",
            "ty",
            time_col="missing",
        )

    with pytest.raises(ValueError, match="finite numeric"):
        r2.recalibrate_gaze(
            base.assign(time=[0.0, np.nan, 2.0]),
            "x",
            "y",
            "tx",
            "ty",
            time_col="time",
        )

    fitted = r2.recalibrate_gaze(
        base,
        "x",
        "y",
        "tx",
        "ty",
        time_col="time",
        grouping_cols=["group"],
        calibration_col="cal",
        method="mean_shift",
        min_valid_points=2,
    )
    assert fitted["gaze_recalibration_status"].eq("complete").all()


def test_r2_face_subset_empty_metrics_and_missing_group_label():
    frame = pd.DataFrame(
        {
            "subject": [pd.NA, pd.NA],
            "face_sync_status": ["matched", ""],
            "face_sync_within_tolerance": [True, False],
        }
    )

    out = r2._r2_face_subset(
        frame,
        frame.index,
        ["subject"],
        50,
        80,
        None,
    )

    assert out.loc[0, "subject"] == "missing"
    assert pd.isna(out.loc[0, "mean_abs_diff_sec"])
    assert pd.isna(out.loc[0, "n_abs_diff_above_limit"])


def test_r2_final_event_comparison_normalization(monkeypatch):
    events = pd.DataFrame(
        {
            "USER_ID": ["S1", "S1"],
            "trial": ["T1", "T1"],
            "detector": ["velocity_10", "custom"],
            "detector_status": ["ok", "ok"],
            "start_time": [0.0, 2.0],
            "end_time": [1.0, 3.0],
            "duration_ms": [1000.0, 1000.0],
        }
    )
    runs = pd.DataFrame(
        {
            "detector": ["velocity_10", "custom"],
            "detector_family": ["velocity", "custom"],
            "status": ["ok", "ok"],
        }
    )

    base = r2.RBundle(
        {
            "events": events,
            "runs": runs,
            "settings": {
                "id_col": "USER_ID",
                "trial_col": "trial",
                "methods": ["velocity"],
                "group_cols": None,
                "min_overlap": 0.5,
            },
        },
        r_class="gp3_event_detector_comparison|list",
    )

    monkeypatch.setattr(
        r2,
        "_compare_event_detectors_before_r2_v5",
        lambda *a, **k: base,
    )

    result = r2.compare_event_detectors(pd.DataFrame())

    assert result["settings"]["methods"] == "velocity"
    assert result["settings"]["group_cols"] == "<EMPTY>"
    assert result["events"]["family"].tolist() == ["velocity", "custom"]
    assert result["events"]["source_status"].eq("ok").all()
    assert "message" in result["runs"]
    assert "n_events" in result["runs"]

    string_sequence = r2.RBundle(
        {
            "events": events,
            "runs": runs.assign(message=np.nan, n_events=[1, 1]),
            "settings": {
                "sequence_cols": "USER_ID",
                "methods": ["velocity", "custom"],
                "group_cols": ["trial"],
                "min_overlap": 0.5,
            },
        },
        r_class="gp3_event_detector_comparison|list",
    )

    monkeypatch.setattr(
        r2,
        "_compare_event_detectors_before_r2_v5",
        lambda *a, **k: string_sequence,
    )
    second = r2.compare_event_detectors(pd.DataFrame())
    assert second["settings"]["sequence_cols"] == ["USER_ID"]
    assert second["settings"]["methods"] == ["velocity", "custom"]


# ============================================================================
# R3A
# ============================================================================


def test_r3a_basic_helpers_and_retired_contracts():
    with pytest.raises(RuntimeError, match="Superseded"):
        r3a._safe_cor(pd.Series([1]), pd.Series([1]))

    assert pd.isna(r3a._detect_sampling_rate(pd.Series([0.0])))
    assert pd.isna(r3a._detect_sampling_rate(pd.Series([1.0, 1.0])))
    assert r3a._detect_sampling_rate(pd.Series([0.0, 0.5, 1.0])) == pytest.approx(2.0)

    with pytest.raises(RuntimeError, match="Superseded"):
        r3a._binocular_policy(
            pd.DataFrame(),
            left_col="l",
            right_col="r",
            prefix="p",
            policy="complete_case",
            valid_min=None,
            valid_max=None,
        )

    with pytest.raises(RuntimeError, match="Superseded"):
        r3a._binocular_summary_block(
            pd.DataFrame(),
            pd.Series(dtype=float),
            "complete_case",
            {},
        )

    with pytest.raises(RuntimeError, match="Superseded"):
        r3a._r3a_binocular_policy_v2(
            pd.DataFrame(),
            left_col="l",
            right_col="r",
            prefix="p",
            policy="complete_case",
            valid_min=None,
            valid_max=None,
        )

    assert r3a._fmt_num(float("nan")) == "NA"
    assert r3a._r3a_group_key({}) == ""
    assert r3a._r3a_group_key({"g": pd.NA}) == "g=NA"


def test_r3a_event_iou_and_agreement_validation_paths():
    row = pd.Series({"start_time": 0.0, "end_time": 1.0})

    assert r3a._event_iou(row, pd.DataFrame()) == 0.0

    other = pd.DataFrame(
        {
            "start_time": [0.5, 2.0],
            "end_time": [1.5, 3.0],
        }
    )
    assert r3a._event_iou(row, other) == pytest.approx(1 / 3)

    with pytest.raises(ValueError, match="detector-comparison"):
        r3a._r3a_summarise_gazepoint_event_detector_agreement(
            x={"settings": {}},
        )

    events = pd.DataFrame(
        {
            "detector": ["a", "b"],
            "start_time": [0.0, 0.0],
            "end_time": [1.0, 1.0],
            "duration_ms": [1000.0, 1000.0],
            "subject": [pd.NA, pd.NA],
        }
    )

    with pytest.raises(ValueError, match="min_overlap"):
        r3a._r3a_summarise_gazepoint_event_detector_agreement(
            events,
            min_overlap=2,
        )

    out = r3a._r3a_summarise_gazepoint_event_detector_agreement(
        x={
            "events": events,
            "settings": {"sequence_cols": ["subject"]},
        },
        min_overlap=0.5,
    )
    assert out["settings"]["sequence_cols"] == ["subject"]
    assert len(out["pairwise_agreement"]) == 1


def test_r3a_face_summary_window_and_reactivity_edges():
    block = pd.DataFrame(
        {
            "valid": [True, False],
            "confidence": [0.9, 0.1],
            "AU": [1.0, 3.0],
        }
    )

    used = r3a._face_summary_row(
        block,
        group_values={"subject": "S1"},
        window_id="w",
        window_label="W",
        window_start=0,
        window_end=1,
        measure_cols=["AU"],
        validity_col="valid",
        confidence_col="confidence",
        require_valid=True,
    )
    assert used["n_used"] == 1
    assert used["AU_mean"] == 1.0

    all_rows = r3a._face_summary_row(
        block,
        group_values={},
        window_id="w",
        window_label="W",
        window_start=0,
        window_end=1,
        measure_cols=["AU"],
        validity_col=None,
        confidence_col=None,
        require_valid=False,
    )
    assert all_rows["n_used"] == 2
    assert pd.isna(all_rows["face_confidence_mean"])

    with pytest.raises(ValueError, match="time_col"):
        r3a._r3a_summarize_gazepoint_face_windows(
            pd.DataFrame({"AU": [1.0]}),
            windows=pd.DataFrame(
                {
                    "window_start_sec": [0.0],
                    "window_end_sec": [1.0],
                }
            ),
        )

    with pytest.raises(ValueError, match="statistic"):
        r3a._r3a_summarize_gazepoint_face_reactivity(
            pd.DataFrame({"window": ["a"], "AU_mean": [1.0]}),
            baseline_window="a",
            response_window="b",
            statistic="bad",
        )

    with pytest.raises(ValueError, match="window_col"):
        r3a._r3a_summarize_gazepoint_face_reactivity(
            pd.DataFrame({"AU_mean": [1.0]}),
            baseline_window="a",
            response_window="b",
        )

    with pytest.raises(ValueError, match="No reactivity"):
        r3a._r3a_summarize_gazepoint_face_reactivity(
            pd.DataFrame({"window": ["a"], "text_mean": ["x"]}),
            baseline_window="a",
            response_window="b",
            window_col="window",
            measure_cols=[],
        )


def _r3a_reconstructed_frame():
    return pd.DataFrame(
        {
            "subject": ["S1", "S1", "S2", "S2"],
            "condition": ["A", "B", "A", "B"],
            "left": [1.0, 2.0, np.nan, 4.0],
            "right": [1.2, np.nan, 3.0, 4.2],
            "gp3_binocular_left_final": [1.0, 2.0, 3.0, 4.0],
            "gp3_binocular_right_final": [1.2, 2.2, 3.0, 4.2],
            "gp3_binocular_left_reconstructed": [False, False, True, False],
            "gp3_binocular_right_reconstructed": [False, True, False, False],
        }
    )


def test_r3a_policy_values_all_contracts_and_sensitivity():
    frame = _r3a_reconstructed_frame()

    for policy in [
        "complete_case",
        "available_eye",
        "reconstructed_mean",
        "left_only",
        "right_only",
    ]:
        values, source = r3a._r3a_policy_values_v3(
            frame,
            left_col="left",
            right_col="right",
            prefix="gp3_binocular",
            policy=policy,
            valid_min=0,
            valid_max=10,
        )
        assert len(values) == len(frame)
        assert len(source) == len(frame)

    with pytest.raises(ValueError, match="Unsupported policy"):
        r3a._r3a_policy_values_v3(
            frame,
            left_col="left",
            right_col="right",
            prefix="gp3_binocular",
            policy="bad",
            valid_min=None,
            valid_max=None,
        )

    out = r3a._r3a_analyse_gazepoint_binocular_sensitivity_v3(
        frame,
        left_col="left",
        right_col="right",
        policies=[
            "complete_case",
            "available_eye",
            "reconstructed_mean",
            "left_only",
            "right_only",
        ],
        group_cols="subject",
        condition_col="condition",
        valid_min=0,
        valid_max=10,
    )

    assert len(out["summary"]) == 10
    assert not out["condition_summary"].empty
    assert not out["condition_contrasts"].empty

    with pytest.raises(ValueError, match="unsupported"):
        r3a._r3a_analyse_gazepoint_binocular_sensitivity_v3(
            frame,
            left_col="left",
            right_col="right",
            policies=["bad"],
        )


# ============================================================================
# R3B
# ============================================================================


def test_r3b_core_helpers_and_retired_contracts():
    with pytest.raises(TypeError, match="data frame"):
        r3b._frame([])

    assert r3b._listify(None) == []
    assert r3b._listify("x") == ["x"]
    assert r3b._listify(("a", "b")) == ["a", "b"]
    assert r3b._listify(3) == [3]

    with pytest.raises(ValueError, match="Missing required columns"):
        r3b._require(pd.DataFrame({"x": [1]}), ["x", "y"])

    assert r3b._collapse(None) == ""
    assert r3b._collapse(["a", 2]) == "a, 2"
    assert r3b._collapse(3) == "3"

    assert r3b._r_bool(True) == "TRUE"
    assert r3b._r_bool(False) == "FALSE"

    assert pd.isna(r3b._stat(pd.Series([np.nan]), "mean"))
    assert r3b._stat(pd.Series([1.0, 3.0]), "median") == 2.0
    assert r3b._stat(pd.Series([1.0, 3.0]), "mean") == 2.0

    retired = [
        lambda: r3b._quantile([1], 0.5),
        lambda: r3b._difference_curve(pd.DataFrame(), ["a", "b"], []),
        lambda: r3b._find_onset(pd.DataFrame(), 1, 0, 0, "two_sided"),
        lambda: r3b._bootstrap_participants(
            pd.DataFrame(),
            "subject",
            np.random.RandomState(1),
        ),
        lambda: r3b._multiverse_grid({}, "aoi"),
        lambda: r3b._select_branches(pd.DataFrame(), None),
        lambda: r3b._finish_multiverse(
            pd.DataFrame(),
            "aoi",
            {},
            None,
            None,
            False,
            False,
        ),
        lambda: r3b._read_csv_folder(Path("."), None),
        lambda: r3b._sampling(pd.DataFrame(), []),
        lambda: r3b._quality(pd.DataFrame(), []),
        lambda: r3b._aoi_summary(pd.DataFrame(), pd.DataFrame(), "USER", 60),
    ]
    for call in retired:
        with pytest.raises(RuntimeError, match="Retired unreachable"):
            call()


def test_r3b_html_table_and_candidate_directions():
    table = pd.DataFrame(
        {
            "text": ["<x>", "second"],
            "flag": [True, False],
            "missing": [np.nan, 2.0],
        }
    )

    html = r3b._html_table(table, 1)
    assert "&lt;x&gt;" in html
    assert "TRUE" in html
    assert "second" not in html

    values = pd.Series([-2.0, 0.0, 2.0, np.nan])
    assert r3b._candidate_mask(values, 0, 1, "positive").tolist() == [
        False,
        False,
        True,
        False,
    ]
    assert r3b._candidate_mask(values, 0, 1, "negative").tolist() == [
        True,
        False,
        False,
        False,
    ]
    assert r3b._candidate_mask(values, 0, 1, "two_sided").tolist() == [
        True,
        False,
        True,
        False,
    ]


def test_r3b_divergence_exact_null_is_no_divergence():
    data = pd.DataFrame(
        {
            "condition": ["A", "B"] * 3,
            "time": [0, 0, 1, 1, 2, 2],
            "outcome": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
        }
    )

    result = r3b.estimate_gazepoint_divergence_point(
        data,
        outcome_col="outcome",
        time_col="time",
        condition_col="condition",
        comparison=["A", "B"],
        bootstrap_unit="row",
        n_boot=2,
        ci=0.8,
        consecutive_points=1,
        null_value=1.0,
        min_abs_difference=0,
        direction="two_sided",
        seed=1,
        keep_bootstrap=True,
    )

    assert result["divergence_point"].loc[0, "detector_status"] == "no_divergence"
    assert result["divergence_point"].loc[0, "observed_direction"] is None


def test_r3b_workflow_single_group_scalar_keys(tmp_path):
    gaze = pd.DataFrame(
        {
            "USER": ["S1", "S1"],
            "USER_ID": ["S1", "S1"],
            "MEDIA_ID": ["M1", "M1"],
            "MEDIA_NAME": ["Stim", "Stim"],
            "TIME": [0.0, 0.1],
            "FPOGV": [np.nan, np.nan],
            "AOI": ["target", "target"],
        }
    )
    fix = pd.DataFrame(
        {
            "USER": ["S1"],
            "USER_ID": ["S1"],
            "MEDIA_ID": ["M1"],
            "MEDIA_NAME": ["Stim"],
            "AOI": ["target"],
            "FPOGD": [0.1],
            "FPOGS": [np.nan],
        }
    )

    gaze.to_csv(tmp_path / "case_all_gaze.csv", index=False)
    fix.to_csv(tmp_path / "case_fixations.csv", index=False)

    result = r3b.run_gazepoint_workflow(
        export_dir=tmp_path,
        all_gaze_pattern="all_gaze",
        fixation_pattern="fixations",
        check_file_pairs=False,
        group_cols=["USER"],
        user_col="USER",
        sample_rate=0,
        expected_hz=None,
        output_dir=None,
        save_plots=False,
        create_report=False,
    )

    assert len(result["sampling"]) == 1
    assert pd.isna(result["quality"].loc[0, "FPOGV_valid_pct"])
    assert pd.isna(result["aoi_table"].loc[0, "sample_time_viewed_sec"])
    assert pd.isna(result["aoi_table"].loc[0, "fixation_ttff_sec"])



def test_r3b_result_property_and_report_scalar_edges(tmp_path, monkeypatch):
    result = r3b._R3BResult({"x": 1}, r_class="coverage|list")
    assert result.gp3_r_class == "coverage|list"

    class Sentinel:
        def __str__(self):
            return "sentinel"

    sentinel = Sentinel()
    original_isna = pd.isna

    def guarded_isna(value):
        if value is sentinel:
            raise TypeError("synthetic missingness failure")
        return original_isna(value)

    monkeypatch.setattr(pd, "isna", guarded_isna)

    report_input = {
        "sampling": pd.DataFrame(
            {"value": [float("inf"), sentinel]}
        ),
        "quality": pd.DataFrame({"value": [float("-inf")]}),
        "flagged_quality": pd.DataFrame(
            {"review_required": [True], "value": [sentinel]}
        ),
        "aoi_table": pd.DataFrame({"value": [1.25]}),
    }

    output = tmp_path / "scalar_edges.html"
    out = r3b.create_gazepoint_report(
        report_input,
        output,
        save_plots=False,
    )
    html = output.read_text(encoding="utf-8")

    assert out.loc[0, "n_flagged"] == 1
    assert "inf" in html
    assert "-inf" in html
    assert "sentinel" in html


def test_r3b_workflow_all_missing_aoi_time_and_single_group(tmp_path):
    gaze = pd.DataFrame(
        {
            "USER": ["S1", "S1"],
            "MEDIA_ID": ["M1", "M1"],
            "MEDIA_NAME": ["Stim", "Stim"],
            "TIME": [np.nan, np.nan],
            "FPOGV": [np.nan, np.nan],
            "AOI": ["target", "target"],
        }
    )
    gaze.to_csv(tmp_path / "missing_all_gaze.csv", index=False)

    result = r3b.run_gazepoint_workflow(
        export_dir=tmp_path,
        all_gaze_pattern="all_gaze",
        fixation_pattern=None,
        check_file_pairs=False,
        group_cols=["USER"],
        user_col="USER",
        sample_rate=0,
        expected_hz=None,
        output_dir=None,
        save_plots=False,
        create_report=False,
    )

    assert pd.isna(result["aoi_table"].loc[0, "sample_ttff_sec"])
    assert pd.isna(result["aoi_table"].loc[0, "sample_time_viewed_sec"])
    assert pd.isna(result["quality"].loc[0, "FPOGV_valid_pct"])
