from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

import gp3tools

r2 = importlib.import_module("gp3tools._behavioral_r2")
r3a = importlib.import_module("gp3tools._behavioral_r3a")


def test_r2_collapse_and_small_helper_edges():
    assert pd.isna(r2._collapse(np.nan))
    assert r2._r2_fmt_number(None) == ""
    assert r2._r2_scalar_character_setting("x") == "x"
    assert pd.isna(r2._r2_gaze_collapse_nullable(pd.NA))
    assert r2._r2_gaze_collapse_nullable(7) == "7"

    empty = r2._r2_gaze_aggregate_summary(
        pd.DataFrame(),
        None,
        "status",
    )
    assert empty.empty


def _detector_input():
    return pd.DataFrame(
        {
            "USER_ID": ["S1", "S1", "S1"],
            "trial": ["T1", "T1", "T1"],
            "group": ["G", "G", "G"],
            "FPOGX": [0.1, 0.1, 0.1],
            "FPOGY": [0.2, 0.2, 0.2],
            "TIME": [0.0, 0.1, 0.2],
        }
    )


def test_r2_event_detector_validation_and_failure_paths(monkeypatch):
    base = _detector_input()

    with pytest.raises(ValueError, match="missing required column"):
        r2.compare_event_detectors(
            base.drop(columns=["USER_ID"]),
            methods=["velocity"],
        )

    with pytest.raises(ValueError, match="missing required column"):
        r2.compare_event_detectors(
            base.drop(columns=["FPOGX"]),
            methods=["velocity"],
        )

    with pytest.raises(ValueError, match="trial_col"):
        r2.compare_event_detectors(
            base,
            trial_col="missing",
            methods=["velocity"],
        )

    with pytest.raises(ValueError, match="group_cols"):
        r2.compare_event_detectors(
            base,
            group_cols=["missing"],
            methods=["velocity"],
        )

    events_mod = importlib.import_module("gp3tools.events")

    def broken(*args, **kwargs):
        raise RuntimeError("synthetic detector failure")

    monkeypatch.setattr(
        events_mod,
        "detect_gazepoint_fixations_velocity",
        broken,
    )

    with pytest.raises(ValueError, match="No event detector"):
        r2.compare_event_detectors(
            base,
            trial_col="trial",
            methods=["velocity"],
            velocity_thresholds=[10],
        )


def test_r2_event_detector_velocity_args_and_group_dedup(monkeypatch):
    events_mod = importlib.import_module("gp3tools.events")
    seen = {}

    def fake_detector(data, **kwargs):
        seen.update(kwargs)
        return pd.DataFrame(
            {
                "USER_ID": ["S1"],
                "trial": ["T1"],
                "group": ["G"],
                "start_time": [0.0],
                "end_time": [0.1],
                "duration_ms": [100.0],
                "mean_x": [0.1],
                "mean_y": [0.2],
                "n_samples": [2],
            }
        )

    monkeypatch.setattr(
        events_mod,
        "detect_gazepoint_fixations_velocity",
        fake_detector,
    )

    out = r2.compare_event_detectors(
        _detector_input(),
        trial_col="trial",
        group_cols=["group", "trial"],
        methods=["velocity"],
        velocity_thresholds=[10],
        velocity_args={"synthetic_option": True},
    )

    assert seen["synthetic_option"] is True
    assert out["settings"]["sequence_cols"] == [
        "USER_ID",
        "trial",
        "group",
    ]


def test_r2_cross_package_report_validation_and_write(tmp_path):
    with pytest.raises(ValueError, match="cross-package workflow"):
        r2.create_cross_package_report([])

    with pytest.raises(ValueError, match="contain"):
        r2.create_cross_package_report({})

    with pytest.raises(ValueError, match="non-empty"):
        r2.create_cross_package_report(
            {
                "audit": pd.DataFrame(),
                "report_text": "x",
            }
        )

    audit = pd.DataFrame(
        {
            "engine": ["nearest"],
            "gaze_rows": [10],
            "biometric_rows": [9],
            "matched_rows": [8],
            "unmatched_rows": [2],
            "matched_rate": [0.8],
            "median_absolute_difference_ms": [1.5],
            "maximum_absolute_difference_ms": [3.0],
        }
    )
    output = tmp_path / "cross.md"
    lines = r2.create_cross_package_report(
        {"audit": audit, "report_text": "Synthetic"},
        output,
    )
    assert output.exists()
    assert lines[0].startswith("# gp3tools")


def test_r2_blink_residual_gap_paths():
    one = pd.DataFrame(
        {
            "USER_ID": ["S1"],
            "TIME": [0.0],
            "mean_pupil": [1.0],
        }
    )
    no_blinks = pd.DataFrame(columns=["USER_ID", "start_time", "end_time"])
    out = r2.interpolate_blinks(
        one,
        no_blinks,
        pupil_cols="mean_pupil",
        time_unit="auto",
    )
    assert len(out) == 1

    sparse = pd.DataFrame(
        {
            "USER_ID": ["S1"] * 4,
            "TIME": [0.0, 1.0, 2.0, 3.0],
            "mean_pupil": [1.0, np.nan, np.nan, np.nan],
        }
    )
    blink = pd.DataFrame(
        {
            "USER_ID": ["S1"],
            "start_time": [1.0],
            "end_time": [2.0],
        }
    )
    sparse_out = r2.interpolate_blinks(
        sparse,
        blink,
        pupil_cols="mean_pupil",
        time_unit="milliseconds",
        max_gap_ms=10,
    )
    assert sparse_out["blink_interpolated"].sum() == 0

    no_sample_in_gap = pd.DataFrame(
        {
            "USER_ID": ["S1"] * 3,
            "TIME": [0.0, 10.0, 20.0],
            "mean_pupil": [1.0, 2.0, 3.0],
        }
    )
    empty_interval = pd.DataFrame(
        {
            "USER_ID": ["S1"],
            "start_time": [5.0],
            "end_time": [6.0],
        }
    )
    unchanged = r2.interpolate_blinks(
        no_sample_in_gap,
        empty_interval,
        pupil_cols="mean_pupil",
        time_unit="milliseconds",
        max_gap_ms=10,
    )
    assert unchanged["blink_interpolated"].sum() == 0


def _fixalign_frame():
    return pd.DataFrame(
        {
            "participant": ["P1", "P1", "P2", "P2"],
            "trial": ["T1", "T1", "T2", "T2"],
            "time": [0.0, 100.0, 0.0, 100.0],
            "aoi": ["other", "other", "other", "other"],
            "fixation": [False, False, False, False],
            "event": [False, True, False, False],
        }
    )


def test_r2_fixation_alignment_validation_custom_and_no_event():
    data = _fixalign_frame()

    with pytest.raises(ValueError, match="unsupported"):
        r2.prepare_fixation_aligned_data(
            data,
            time_col="time",
            alignment_event="bad",
        )

    with pytest.raises(ValueError, match="Column not found"):
        r2.prepare_fixation_aligned_data(
            data,
            time_col="missing",
        )

    with pytest.raises(ValueError, match="finite numeric"):
        r2.prepare_fixation_aligned_data(
            data.assign(time=[0.0, np.nan, 0.0, 1.0]),
            time_col="time",
        )

    custom = r2.prepare_fixation_aligned_data(
        data,
        time_col="time",
        participant_col="participant",
        trial_col="trial",
        event_col="event",
        event_value=None,
        alignment_event="custom",
        keep_unaligned=True,
    )
    assert custom["overview"].loc[0, "n_aligned_groups"] == 1

    none = r2.prepare_fixation_aligned_data(
        data,
        time_col="time",
        participant_col="participant",
        trial_col="trial",
        aoi_col="aoi",
        target_aoi="target",
        alignment_event="first_target_entry",
        keep_unaligned=False,
    )
    assert none["overview"].loc[0, "alignment_status"] == "no_alignment_events"
    assert none["aligned_data"].empty


def test_r2_fixation_alignment_wrapper_time_metadata_edges(monkeypatch):
    aligned = pd.DataFrame(
        {
            "time": [np.nan],
            "gp3_has_alignment_event": [False],
            "gp3_is_alignment_event_row": [False],
        }
    )
    event_table = pd.DataFrame(
        {
            "gp3_group_id": ["g"],
            "gp3_participant": ["p"],
            "gp3_trial": ["t"],
            "gp3_alignment_event": ["custom"],
            "gp3_has_alignment_event": [False],
            "gp3_alignment_time": [np.nan],
            "gp3_alignment_row_id": [np.nan],
            "gp3_event_aoi": [pd.NA],
            "gp3_event_is_target_aoi": [pd.NA],
            "gp3_event_is_fixation": [pd.NA],
            "gp3_event_is_saccade": [pd.NA],
            "gp3_pre_event_n": [0],
            "gp3_pre_event_target_n": [0],
            "gp3_post_event_n": [0],
            "gp3_target_present_before_event": [False],
            "gp3_fixation_to_target_before_event": [False],
            "gp3_already_on_target_at_trial_start": [False],
        }
    )
    trial_summary = pd.DataFrame(
        {
            "gp3_has_alignment_event": [False],
            "n_groups": [1],
            "n_with_pre_event_target": [0],
            "n_already_on_target_at_start": [0],
            "median_alignment_time": [np.nan],
        }
    )

    def fake_base(*args, **kwargs):
        return {
            "overview": pd.DataFrame(),
            "aligned_data": aligned.copy(),
            "event_table": event_table.copy(),
            "trial_summary": trial_summary.copy(),
            "settings": {},
        }

    monkeypatch.setattr(
        r2,
        "_prepare_fixation_aligned_data_before_r2_repair",
        fake_base,
    )

    explicit = r2.prepare_fixation_aligned_data(
        pd.DataFrame({"time": [1.0]}),
        time_col="time",
    )
    assert pd.isna(explicit["event_table"].loc[0, "gp3_start_time"])

    positional = r2.prepare_fixation_aligned_data(
        pd.DataFrame({"time": [1.0]}),
        "time",
    )
    assert pd.isna(positional["event_table"].loc[0, "gp3_end_time"])


def test_r2_multimodal_auto_join_intersection_paths():
    face = pd.DataFrame(
        {
            "participant_id": ["S1", "S2"],
            "face_signal": [1.0, 2.0],
        }
    )
    gaze = pd.DataFrame(
        {
            "participant_id": ["S1", "S2"],
            "dwell": [0.2, 0.3],
        }
    )
    response = pd.DataFrame(
        {
            "participant_id": ["S1", "S2"],
            "rating": [4.0, 5.0],
        }
    )
    out = r2.prepare_multimodal_data(
        face,
        gaze_data=gaze,
        response_data=response,
        by=None,
        predictor_cols=["face_signal", "dwell"],
        outcome_cols=["rating"],
    )
    assert len(out) == 2


def test_r2_gaze_quality_alias_and_detection_paths():
    data = pd.DataFrame(
        {
            "USER_FILE": ["S1", "S1"],
            "MEDIA_ID": ["M1", "M1"],
            "trial_global": ["T1", "T1"],
            "FPOGX": [0.1, 0.2],
            "FPOGY": [0.1, 0.2],
            "FPOGV": [1, 1],
            "pupil": [3.0, 3.1],
        }
    )
    out = r2.audit_gaze_signal_quality(
        data,
        subject_col="USER_FILE",
        group_cols=["USER_FILE", "MEDIA_ID", "trial_global"],
        validity_cols=["FPOGV"],
    )
    assert out["unit_summary"].loc[0, "gaze_signal_status"] == "ok"


def test_r2_event_sync_aliases_and_face_no_group_not_checked():
    sync = pd.DataFrame(
        {
            "USER_FILE": ["S1", "S1"],
            "MEDIA_ID": ["M1", "M1"],
            "time": [0.0, 1.0],
            "event": ["onset", "response"],
        }
    )
    out = r2.audit_event_sync(
        sync,
        time_col="time",
        event_col="event",
        group_cols=["USER_FILE", "MEDIA_ID"],
        min_samples_per_unit=1,
    )
    assert out["overview"].loc[0, "n_units"] == 1

    face = pd.DataFrame(
        {
            "face_sync_method": ["nearest"],
            "face_sync_status": ["matched"],
            "face_sync_within_tolerance": [True],
        }
    )
    face_out = r2.audit_face_sync(
        face,
        group_cols=None,
        max_abs_diff_sec=None,
    )
    row = face_out["issue_summary"].set_index("issue")
    assert row.loc["large_time_differences", "status"] == "not_checked"


def test_r2_final_detector_default_status_and_matched_skip(monkeypatch):
    events = pd.DataFrame(
        {
            "USER_ID": ["S1", "S1"],
            "detector": ["a", "b"],
            "start_time": [0.0, 0.0],
            "end_time": [1.0, 1.0],
            "duration_ms": [1000.0, 1000.0],
        }
    )
    runs = pd.DataFrame(
        {
            "detector": ["a", "b"],
            "family": ["a", "b"],
            "status": ["ok", "ok"],
            "n_events": [1, 1],
            "message": [np.nan, np.nan],
        }
    )
    base = r2.RBundle(
        {
            "events": events,
            "runs": runs,
            "settings": {
                "sequence_cols": ["USER_ID"],
                "methods": ["a", "b"],
                "group_cols": [],
                "min_overlap": 0.5,
            },
        },
        r_class="gp3_event_detector_comparison|list",
    )
    monkeypatch.setattr(
        r2,
        "_compare_event_detectors_before_r2_v5",
        lambda *args, **kwargs: base,
    )

    out = r2.compare_event_detectors(pd.DataFrame())
    assert out["events"]["source_status"].eq("ok").all()
    assert out["unmatched_events"].empty
    assert out["pairwise_agreement"].loc[0, "matched_a"] == 1


# ---------------------------------------------------------------------------
# R3-A residuals
# ---------------------------------------------------------------------------


def test_r3a_master_missing_optional_and_media_name_paths():
    with pytest.raises(ValueError, match="Missing required columns"):
        r3a._r3a_create_gazepoint_master_v3(pd.DataFrame({"USER_FILE": ["S1"]}))

    data = pd.DataFrame(
        {
            "USER_FILE": ["S1"],
            "USER": ["U1"],
            "MEDIA_ID": ["M1"],
            "TIME": [0.1],
            "condition": ["A"],
            "response": ["yes"],
        }
    )
    out = r3a._r3a_create_gazepoint_master_v3(data)
    assert out.loc[0, "MEDIA_NAME"] == "M1"
    assert out.loc[0, "condition"] == "A"
    assert out.loc[0, "response"] == "yes"


def test_r3a_ivt_no_group_empty_and_short_segment_paths():
    frame = pd.DataFrame(
        {
            "x": [0.0, 0.0, 10.0],
            "y": [0.0, 0.0, 0.0],
            "time": [0.0, 1.0, 2.0],
        }
    )
    no_group = r3a._r3a_detect_gazepoint_fixations_ivt(
        frame,
        group_cols=None,
        velocity_threshold=1,
        min_duration_ms=100,
    )
    assert isinstance(no_group, pd.DataFrame)

    empty = r3a._r3a_ivt_one_group_v2(
        frame.iloc[0:0],
        x_col="x",
        y_col="y",
        time_col="time",
        velocity_threshold=1,
        min_duration_ms=1,
        distance_scale=1,
        time_scale=1,
        group_cols=[],
    )
    assert empty.empty

    short = r3a._r3a_ivt_one_group_v2(
        pd.DataFrame(
            {
                "x": [0.0, 0.0],
                "y": [0.0, 0.0],
                "time": [0.0, 1.0],
            }
        ),
        x_col="x",
        y_col="y",
        time_col="time",
        velocity_threshold=1,
        min_duration_ms=10,
        distance_scale=1,
        time_scale=1,
        group_cols=[],
    )
    assert short.empty

    empty_grouped = r3a._r3a_detect_gazepoint_fixations_ivt(
        pd.DataFrame(columns=["x", "y", "time", "g"]),
        group_cols=["g"],
    )
    assert empty_grouped.empty


def test_r3a_event_agreement_without_sequence_columns():
    events = pd.DataFrame(
        {
            "detector": ["a", "b"],
            "start_time": [0.0, 0.0],
            "end_time": [1.0, 1.0],
            "duration_ms": [1000.0, 1000.0],
        }
    )
    out = r3a._r3a_summarise_gazepoint_event_detector_agreement(
        events,
        min_overlap=0.5,
    )
    assert len(out["pairwise_agreement"]) == 1


def test_r3a_face_windows_auto_columns_na_group_empty_skip_and_no_windows():
    data = pd.DataFrame(
        {
            "subject": [pd.NA, "S1"],
            "face_time_sec": [0.5, 0.5],
            "face_valid": [True, True],
            "face_confidence": [0.9, 0.8],
            "AU01": [1.0, 2.0],
        }
    )
    windows = pd.DataFrame(
        {
            "subject": [pd.NA, "S2"],
            "window_start_sec": [0.0, 0.0],
            "window_end_sec": [1.0, 1.0],
        }
    )

    out = r3a._r3a_summarize_gazepoint_face_windows(
        data,
        windows=windows,
        group_cols=["subject"],
        include_empty_windows=False,
    )
    assert len(out) == 1
    assert out.loc[0, "AU01_mean"] == 1.0

    with pytest.raises(ValueError, match="No numeric"):
        r3a._r3a_summarize_gazepoint_face_windows(
            pd.DataFrame(
                {
                    "time": [0.0],
                    "text": ["x"],
                }
            ),
            windows=pd.DataFrame(
                {
                    "window_start_sec": [0.0],
                    "window_end_sec": [1.0],
                }
            ),
        )

    with pytest.raises(ValueError, match="requires .*windows"):
        r3a._r3a_summarize_gazepoint_face_windows(
            pd.DataFrame(
                {
                    "time": [0.0],
                    "AU": [1.0],
                }
            ),
            windows=None,
        )


def test_r3a_face_reactivity_residual_matrix():
    frame = pd.DataFrame(
        {
            "subject": [pd.NA, pd.NA, "S1", "S1"],
            "window": ["base", "resp", "base", "resp"],
            "AU_mean": [0.0, 1.0, 2.0, 4.0],
            "face_confidence_mean": [0.9, 0.9, 0.8, 0.8],
        }
    )

    out = r3a._r3a_summarize_gazepoint_face_reactivity(
        frame,
        baseline_window="base",
        response_window="resp",
        group_cols=["subject"],
        window_col=None,
        measure_cols=None,
        statistic=["mean"],
    )
    assert set(out["measure"]) == {"AU"}
    assert out.loc[out["subject"].isna(), "percent_reactivity"].isna().all()

    pooled = r3a._r3a_summarize_gazepoint_face_reactivity(
        frame.loc[frame["subject"].eq("S1")],
        baseline_window="base",
        response_window="resp",
        group_cols=None,
        window_col="window",
        measure_cols=["AU"],
        statistic="mean",
    )
    assert len(pooled) == 1

    with pytest.raises(ValueError, match="No baseline"):
        r3a._r3a_summarize_gazepoint_face_reactivity(
            frame,
            baseline_window="missing",
            response_window="resp",
            window_col="window",
            measure_cols=["AU"],
        )

    with pytest.raises(ValueError, match="No response"):
        r3a._r3a_summarize_gazepoint_face_reactivity(
            frame,
            baseline_window="base",
            response_window="missing",
            window_col="window",
            measure_cols=["AU"],
        )


def test_r3a_reporting_validation_and_no_eligible_column(monkeypatch):
    with pytest.raises(ValueError, match="reconstruct_gazepoint_binocular_pupil"):
        r3a._r3a_summarise_gazepoint_binocular_reporting(pd.DataFrame({"x": [1.0]}))

    fake_result = {
        "summary": pd.DataFrame(
            {
                "n_rows": [1],
                "n_reconstructed": [0],
                "reconstruction_fraction": [0.0],
                "bilateral_observed_fraction": [1.0],
                "monocular_unreconstructed_fraction": [0.0],
            }
        ),
        "models": pd.DataFrame({"n_pairs": [3], "r_squared": [0.9]}),
        "validation": pd.DataFrame(),
        "text": "",
        "limitations": [],
    }

    monkeypatch.setattr(
        r3a,
        "_r3a_reporting_pass1",
        lambda **kwargs: {
            key: value.copy() if hasattr(value, "copy") else value
            for key, value in fake_result.items()
        },
    )

    data = pd.DataFrame({"x": [1.0]})
    data.attrs["gp3_binocular_reconstruction"] = {"method": "synthetic"}
    out = r3a._r3a_summarise_gazepoint_binocular_reporting_v2(data)
    assert "Eligible cross-eye" not in out["text"]


def _sensitivity_frame(condition):
    return pd.DataFrame(
        {
            "condition": condition,
            "left": [1.0, np.nan],
            "right": [np.nan, 2.0],
            "gp3_binocular_left_final": [1.0, 2.0],
            "gp3_binocular_right_final": [1.0, 2.0],
            "gp3_binocular_left_reconstructed": [False, True],
            "gp3_binocular_right_reconstructed": [True, False],
        }
    )


def test_r3a_sensitivity_no_groups_no_complete_pairs_and_nan_conditions():
    base = _sensitivity_frame(["A", "B"])

    no_groups = r3a._r3a_analyse_gazepoint_binocular_sensitivity_v3(
        base,
        left_col="left",
        right_col="right",
        policies=["left_only", "right_only"],
        group_cols=None,
    )
    assert len(no_groups["summary"]) == 2
    assert no_groups["correlations"]["n_complete"].eq(0).all()
    assert no_groups["correlations"]["mean_difference"].isna().all()

    nan_reference = r3a._r3a_analyse_gazepoint_binocular_sensitivity_v3(
        _sensitivity_frame([np.nan, "A"]),
        left_col="left",
        right_col="right",
        policies=["left_only"],
        condition_col="condition",
    )
    assert nan_reference["condition_contrasts"].empty

    nan_comparison = r3a._r3a_analyse_gazepoint_binocular_sensitivity_v3(
        _sensitivity_frame(["A", np.nan]),
        left_col="left",
        right_col="right",
        policies=["left_only"],
        condition_col="condition",
    )
    assert nan_comparison["condition_contrasts"].empty


def test_r3a_public_dispatch_still_has_final_sensitivity():
    assert (
        r3a.R3A_IMPLEMENTATIONS["analyse_gazepoint_binocular_sensitivity"]
        is r3a._r3a_analyse_gazepoint_binocular_sensitivity_v3
    )
    assert len(gp3tools.R_EXPORTS) == 278
