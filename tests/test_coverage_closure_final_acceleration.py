from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

aoi = importlib.import_module("gp3tools.aoi")
pupil = importlib.import_module("gp3tools.pupil")
qc = importlib.import_module("gp3tools.qc")


# ============================================================================
# AOI RECTANGLE ASSIGNMENT
# ============================================================================


def _rectangles():
    return pd.DataFrame(
        {
            "name": ["A", "B"],
            "left": [0.0, 0.25],
            "right": [1.0, 1.25],
            "top": [0.0, 0.25],
            "bottom": [1.0, 1.25],
        }
    )


def test_final_aoi_rectangle_assignment_contracts():
    data = pd.DataFrame(
        {
            "x": [0.1, 0.5, 1.1, np.nan],
            "y": [0.1, 0.5, 1.1, 0.5],
        }
    )

    legacy = aoi.add_gazepoint_aoi(
        data,
        x_col="x",
        y_col="y",
        aoi_geometry=_rectangles().iloc[[0]],
    )

    assert legacy["aoi_current"].iloc[0] == "A"
    assert legacy["aoi_current"].iloc[2] == "outside"
    assert pd.isna(legacy["aoi_current"].iloc[3])

    both = aoi.add_gazepoint_aoi(
        data,
        x_col="x",
        y_col="y",
        aoi_geometry=_rectangles(),
        output="both",
        prefix="hit_",
        overlap="first",
        include_overlap_count=True,
    )

    assert {"hit_A", "hit_B", "aoi_current", "aoi_overlap_count"}.issubset(both.columns)
    assert both.loc[1, "aoi_overlap_count"] == 2
    assert both.loc[1, "aoi_current"] == "A"

    last = aoi.add_gazepoint_aoi(
        data,
        x_col="x",
        y_col="y",
        aoi_geometry=_rectangles(),
        output="label",
        overlap="last",
    )

    assert last.loc[1, "aoi_current"] == "B"

    logical = aoi.add_gazepoint_aoi(
        data,
        x_col="x",
        y_col="y",
        aoi_geometry=_rectangles(),
        aoi_name=["A"],
        output="logical",
    )

    assert "aoi_A" in logical.columns

    with pytest.raises(ValueError, match="at least one AOI"):
        aoi.add_gazepoint_aoi(
            data,
            x_col="x",
            y_col="y",
            aoi_geometry=pd.DataFrame(),
        )

    with pytest.raises(ValueError, match="resolve AOI definition"):
        aoi.add_gazepoint_aoi(
            data,
            x_col="x",
            y_col="y",
            aoi_geometry=pd.DataFrame({"name": ["A"]}),
        )

    with pytest.raises(ValueError, match="did not match"):
        aoi.add_gazepoint_aoi(
            data,
            x_col="x",
            y_col="y",
            aoi_geometry=_rectangles(),
            aoi_name="missing",
            output="label",
        )

    duplicated = pd.concat(
        [_rectangles().iloc[[0]], _rectangles().iloc[[0]]],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="unique"):
        aoi.add_gazepoint_aoi(
            data,
            x_col="x",
            y_col="y",
            aoi_geometry=duplicated,
        )

    nonfinite = _rectangles().iloc[[0]].copy()
    nonfinite.loc[0, "left"] = np.inf

    with pytest.raises(ValueError, match="finite"):
        aoi.add_gazepoint_aoi(
            data,
            x_col="x",
            y_col="y",
            aoi_geometry=nonfinite,
        )

    with pytest.raises(ValueError, match="output"):
        aoi.add_gazepoint_aoi(
            data,
            x_col="x",
            y_col="y",
            aoi_geometry=_rectangles(),
            output="bad",
        )

    with pytest.raises(ValueError, match="overlap"):
        aoi.add_gazepoint_aoi(
            data,
            x_col="x",
            y_col="y",
            aoi_geometry=_rectangles(),
            overlap="bad",
        )

    with pytest.raises(ValueError, match="overlapping"):
        aoi.add_gazepoint_aoi(
            data,
            x_col="x",
            y_col="y",
            aoi_geometry=_rectangles(),
            overlap="error",
        )


# ============================================================================
# AOI TRANSITIONS
# ============================================================================


def _transition_sequence():
    return pd.DataFrame(
        {
            "aoi_state": [
                "target",
                "target",
                "background",
                "distractor",
                "other",
            ],
            "transition_from": [
                "target",
                "target",
                "background",
                "distractor",
                "other",
            ],
            "transition_to": [
                "target",
                "background",
                "distractor",
                "other",
                pd.NA,
            ],
            "entry_duration_ms": [10, 20, 30, 40, 50],
            "dwell_before_transition_ms": [10, 20, 30, 40, np.nan],
            "is_non_aoi": [False, False, True, False, False],
            "is_terminal_state": [False, False, False, False, True],
        }
    )


def test_final_aoi_transition_status_matrix():
    seq = _transition_sequence()

    both = aoi.summarise_gazepoint_aoi_transitions(
        seq,
        group_cols=[],
        include_non_aoi=True,
        target_aoi_values=["target"],
        distractor_aoi_values=["distractor"],
        non_aoi_values=["background"],
    )

    assert both.loc[0, "transition_feature_status"] == "ok"
    assert both.loc[0, "target_to_target"] == 1
    assert both.loc[0, "target_to_background"] == 1
    assert both.loc[0, "background_to_distractor"] == 1
    assert both.loc[0, "other_transitions"] >= 1

    target_only = aoi.summarise_gazepoint_aoi_transitions(
        seq,
        group_cols=[],
        include_non_aoi=True,
        target_aoi_values=["target"],
        distractor_aoi_values=[],
        non_aoi_values=["background"],
    )

    assert target_only.loc[0, "transition_feature_status"] == "no_distractor_defined"

    distractor_only = aoi.summarise_gazepoint_aoi_transitions(
        seq,
        group_cols=[],
        include_non_aoi=True,
        target_aoi_values=[],
        distractor_aoi_values=["distractor"],
        non_aoi_values=["background"],
    )

    assert distractor_only.loc[0, "transition_feature_status"] == "no_target_defined"

    neither = aoi.summarise_gazepoint_aoi_transitions(
        seq,
        group_cols=[],
        include_non_aoi=True,
        target_aoi_values=[],
        distractor_aoi_values=[],
        non_aoi_values=["background"],
    )

    assert neither.loc[0, "transition_feature_status"] == ("no_target_or_distractor_defined")

    terminal = seq.iloc[[-1]].copy()

    no_transitions = aoi.summarise_gazepoint_aoi_transitions(
        terminal,
        group_cols=[],
        include_non_aoi=True,
        target_aoi_values=["target"],
        distractor_aoi_values=["distractor"],
    )

    assert no_transitions.loc[0, "transition_feature_status"] == "no_transitions"
    assert no_transitions.loc[0, "total_transitions"] == 0


# ============================================================================
# AOI WINDOWS
# ============================================================================


def _window_frame(states):
    return pd.DataFrame(
        {
            "subject": ["S1"] * len(states),
            "time": np.arange(len(states), dtype=float) * 50,
            "aoi": states,
        }
    )


def test_final_aoi_window_statuses_and_validation():
    ok = aoi.summarise_gazepoint_aoi_windows(
        _window_frame(["target", "other", "target", "background"]),
        aoi_col="aoi",
        time_col="time",
        windows=[0, 100, 200],
        group_cols=["subject"],
        condition_col=None,
        target_aoi_values=["target"],
        distractor_aoi_values=["distractor"],
        non_aoi_values=["background"],
    )

    assert len(ok) == 2
    assert "ok" in set(ok["aoi_window_status"])

    no_target_defined = aoi.summarise_gazepoint_aoi_windows(
        _window_frame(["other", "other"]),
        aoi_col="aoi",
        time_col="time",
        windows=[0, 100],
        group_cols=["subject"],
        condition_col=None,
        target_aoi_values=[],
    )

    assert no_target_defined.loc[0, "aoi_window_status"] == "no_target_aoi_defined"

    target_missing = aoi.summarise_gazepoint_aoi_windows(
        _window_frame(["other", "other"]),
        aoi_col="aoi",
        time_col="time",
        windows=[0, 100],
        group_cols=["subject"],
        condition_col=None,
        target_aoi_values=["target"],
    )

    assert target_missing.loc[0, "aoi_window_status"] == "target_not_observed"

    target_only = aoi.summarise_gazepoint_aoi_windows(
        _window_frame(["target", "target"]),
        aoi_col="aoi",
        time_col="time",
        windows=[0, 100],
        group_cols=["subject"],
        condition_col=None,
        target_aoi_values=["target"],
    )

    assert target_only.loc[0, "aoi_window_status"] == "target_only"

    missing_only = aoi.summarise_gazepoint_aoi_windows(
        _window_frame([pd.NA, pd.NA]),
        aoi_col="aoi",
        time_col="time",
        windows=[0, 100],
        group_cols=["subject"],
        condition_col=None,
        target_aoi_values=["target"],
    )

    assert missing_only.loc[0, "aoi_window_status"] == "zero_valid_denominator"

    with pytest.raises(ValueError, match="windows"):
        aoi.summarise_gazepoint_aoi_windows(
            _window_frame(["target"]),
            aoi_col="aoi",
            time_col="time",
            windows=None,
            group_cols=["subject"],
        )

    with pytest.raises(ValueError, match="include_right_endpoint"):
        aoi.summarise_gazepoint_aoi_windows(
            _window_frame(["target"]),
            aoi_col="aoi",
            time_col="time",
            windows=[0, 100],
            group_cols=["subject"],
            include_right_endpoint="yes",
        )

    with pytest.raises(ValueError, match="at least two finite"):
        aoi.summarise_gazepoint_aoi_windows(
            _window_frame(["target"]),
            aoi_col="aoi",
            time_col="time",
            windows=[0],
            group_cols=["subject"],
        )

    with pytest.raises(ValueError, match="distinct"):
        aoi.summarise_gazepoint_aoi_windows(
            _window_frame(["target"]),
            aoi_col="aoi",
            time_col="time",
            windows=[0, 0],
            group_cols=["subject"],
        )

    with pytest.raises(ValueError, match="No rows fall"):
        aoi.summarise_gazepoint_aoi_windows(
            _window_frame(["target"]),
            aoi_col="aoi",
            time_col="time",
            windows=[1000, 2000],
            group_cols=["subject"],
        )


# ============================================================================
# PUPIL BASELINE CORRECTION
# ============================================================================


def _baseline_frame():
    return pd.DataFrame(
        {
            "subject": ["S1"] * 4 + ["S2"] * 4,
            "media_id": ["M1"] * 8,
            "time_ms": [-100, 0, 100, 200] * 2,
            "pupil": [2.0, 4.0, 6.0, 8.0, 0.0, 0.0, 2.0, np.nan],
            "baseline_flag": [True, True, False, False] * 2,
        }
    )


def test_final_pupil_baseline_legacy_modes():
    frame = _baseline_frame()

    for mode in ["subtract", "divide", "percent", "percent_change"]:
        out = pupil.baseline_correct_gazepoint_pupil(
            frame,
            pupil_col="pupil",
            time_col="time_ms",
            baseline=(-100, 0),
            group_cols=["subject"],
            mode=mode,
        )

        assert "pupil_baseline" in out
        assert "pupil_baseline_corrected" in out

    with pytest.raises(ValueError, match="Unknown baseline mode"):
        pupil.baseline_correct_gazepoint_pupil(
            frame,
            pupil_col="pupil",
            time_col="time_ms",
            baseline=(-100, 0),
            mode="bad",
        )


def test_final_pupil_baseline_r_mode_and_validation():
    frame = _baseline_frame()

    out = pupil.baseline_correct_gazepoint_pupil(
        frame,
        pupil_col="pupil",
        time_col="time_ms",
        baseline_time_col="time_ms",
        baseline_window=(-100, 0),
        group_cols=["subject", "media_id"],
        baseline_method="median",
        min_baseline_samples=2,
    )

    assert {"corrected", "no_baseline", "missing_pupil"}.intersection(
        set(out["pupil_baseline_status"])
    )

    flagged = pupil.baseline_correct_gazepoint_pupil(
        frame,
        pupil_col="pupil",
        time_col="time_ms",
        baseline_time_col="time_ms",
        baseline_window=None,
        baseline_flag_col="baseline_flag",
        group_cols=[],
        min_baseline_samples=1,
    )

    assert flagged["pupil_baseline_used"].sum() == 4

    with pytest.raises(ValueError, match="baseline_method"):
        pupil.baseline_correct_gazepoint_pupil(
            frame,
            pupil_col="pupil",
            time_col="time_ms",
            baseline_method="bad",
        )

    with pytest.raises(ValueError, match="min_baseline_samples"):
        pupil.baseline_correct_gazepoint_pupil(
            frame,
            pupil_col="pupil",
            time_col="time_ms",
            min_baseline_samples=0,
        )

    with pytest.raises(ValueError, match="baseline_window"):
        pupil.baseline_correct_gazepoint_pupil(
            frame,
            pupil_col="pupil",
            time_col="time_ms",
            baseline_window=(1,),
        )

    with pytest.raises(ValueError, match="greater than or equal"):
        pupil.baseline_correct_gazepoint_pupil(
            frame,
            pupil_col="pupil",
            time_col="time_ms",
            baseline_window=(1, 0),
        )

    with pytest.raises(ValueError, match="baseline_flag_col"):
        pupil.baseline_correct_gazepoint_pupil(
            frame,
            pupil_col="pupil",
            time_col="time_ms",
            baseline_flag_col="missing",
        )

    with pytest.raises(ValueError, match="No pupil"):
        pupil.baseline_correct_gazepoint_pupil(
            frame.drop(columns=["pupil"]),
            time_col="time_ms",
        )

    with pytest.raises(ValueError, match="grouping column"):
        pupil.baseline_correct_gazepoint_pupil(
            frame,
            pupil_col="pupil",
            time_col="time_ms",
            group_cols=["missing"],
        )


# ============================================================================
# PUPIL DRIFT
# ============================================================================


def _drift_frame():
    return pd.DataFrame(
        {
            "subject": ["S1"] * 4 + ["S2"] * 4,
            "condition": ["A", "A", "B", "B"] * 2,
            "trial": [1, 2, 3, 4] * 2,
            "time": [0, 100, 200, 300] * 2,
            "pupil": [1, 2, 3, 4, 4, 3, 2, 1],
            "excluded_trial": [
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
        }
    )


def test_final_pupil_drift_legacy_and_r_mode():
    frame = _drift_frame()

    legacy = pupil.audit_gazepoint_pupil_drift(
        frame,
        pupil_col="pupil",
        time_col="time",
        group_cols=["subject"],
    )

    assert len(legacy) == 2
    assert legacy["slope"].notna().all()

    rmode = pupil.audit_gazepoint_pupil_drift(
        frame,
        pupil_col="pupil",
        time_col="time",
        group_cols=["subject"],
        order_col="trial",
        condition_col="condition",
        exclude_col="excluded_trial",
        include_excluded=False,
        min_valid_samples=2,
        max_abs_slope_per_min=0.01,
        max_condition_time_mean_diff_ms=1,
        max_condition_order_mean_diff=0.1,
    )

    assert "summary" in rmode
    assert "condition_balance" in rmode
    assert not rmode["by_group"].empty

    text_exclusion = frame.copy()
    text_exclusion["excluded_trial"] = [
        "false",
        "no",
        "yes",
        "false",
        "0",
        "0",
        "0",
        "0",
    ]

    text_result = pupil.audit_gazepoint_pupil_drift(
        text_exclusion,
        pupil_col="pupil",
        time_col="time",
        group_cols=["subject"],
        order_col="trial",
        condition_col="condition",
        exclude_col="excluded_trial",
        min_valid_samples=2,
    )

    assert len(text_result["by_group"]) == 2


# ============================================================================
# QC MASTER CONSTRUCTION
# ============================================================================


def test_final_qc_master_legacy_and_r_mode():
    legacy = qc.as_gazepoint_master(
        pd.DataFrame(
            {
                "USER_FILE": ["S1"],
                "MEDIA_ID": ["M1"],
                "TIME": [0.0],
                "FPOGX": [0.5],
                "FPOGY": [0.5],
            }
        )
    )

    assert legacy.attrs["gp3_class"] == "gazepoint_master"

    raw = pd.DataFrame(
        {
            "USER_FILE": ["S1_all_gaze.csv", "S1_all_gaze.csv"],
            "MEDIA_ID": ["M1", "M1"],
            "MEDIA_NAME": ["Stim", "Stim"],
            "TIME": [0.0, 0.1],
            "CNT": [1, 2],
            "BPOGX": [0.5, 1.2],
            "BPOGY": [0.5, 0.5],
            "BPOGV": [1, 1],
            "LPMM": [3.0, np.nan],
            "RPMM": [3.2, np.nan],
            "LPMMV": [1, 0],
            "RPMMV": [1, 0],
            "USER": ["TRIAL_START", "TRIAL_END"],
            "AOI": ["target", ""],
            "FPOGX": [0.5, 1.2],
            "FPOGY": [0.5, 0.5],
        }
    )

    master = qc.as_gazepoint_master(
        raw,
        screen_width_px=1000,
        screen_height_px=800,
        coordinate_unit="normalised",
        event_latency_offset_ms=25,
    )

    assert master.loc[0, "x"] == pytest.approx(500)
    assert master.loc[0, "y"] == pytest.approx(400)
    assert master.loc[0, "time_ms"] == pytest.approx(25)
    assert master.loc[0, "event_type"] == "trial_start"
    assert master.loc[1, "gaze_offscreen"]
    assert master.loc[0, "pupil_unit"] == "diameter_mm"

    pixels = qc.as_gazepoint_master(
        raw,
        coordinate_unit="pixels",
        event_latency_offset_ms=0,
    )

    assert pixels.loc[0, "x"] == pytest.approx(0.5)

    with pytest.raises(ValueError, match="coordinate_unit"):
        qc.as_gazepoint_master(
            raw,
            coordinate_unit="bad",
        )

    with pytest.raises(ValueError, match="screen_width_px"):
        qc.as_gazepoint_master(
            raw,
            coordinate_unit="pixels",
            screen_width_px=np.inf,
        )

    with pytest.raises(ValueError, match="event_latency_offset_ms"):
        qc.as_gazepoint_master(
            raw,
            coordinate_unit="pixels",
            event_latency_offset_ms=np.inf,
        )

    with pytest.raises(KeyError, match="TIME"):
        qc.as_gazepoint_master(
            raw.drop(columns=["TIME"]),
            coordinate_unit="pixels",
        )


# ============================================================================
# QC WRAPPERS / PHASES
# ============================================================================


def test_final_qc_condition_quality_wrapper_paths():
    frame = pd.DataFrame(
        {
            "condition": ["A", "A", "B", "B"],
            "valid": [1, 0, 1, 1],
            "x": [0.1, np.nan, 0.3, 0.4],
            "y": [0.1, 0.2, 0.3, 0.4],
        }
    )

    legacy = qc.audit_gazepoint_condition_quality_imbalance(
        frame,
        condition_col="condition",
        validity_col="valid",
        x_col="x",
        y_col="y",
    )

    assert not legacy.empty

    with pytest.raises(TypeError, match="Unexpected keyword"):
        qc.audit_gazepoint_condition_quality_imbalance(
            frame,
            condition_col="condition",
            impossible=True,
        )


def test_final_qc_phase_segmentation_and_coverage():
    frame = pd.DataFrame(
        {
            "subject": ["S1"] * 4,
            "time": [0.0, 50.0, 100.0, np.nan],
            "value": [1.0, np.nan, 3.0, 4.0],
        }
    )

    windows = pd.DataFrame(
        {
            "phase": ["early", "late"],
            "start": [0.0, 50.0],
            "end": [50.0, 100.0],
        }
    )

    segmented = qc.segment_gazepoint_task_phases(
        frame,
        time_col="time",
        phase_windows=windows,
        phase_col="task_phase",
        include_lower=True,
        include_upper=True,
        keep_window_metadata=True,
    )

    assert "task_phase" in segmented
    assert ".gp3_phase_window_start" in segmented
    assert ".gp3_phase_window_end" in segmented

    coverage = qc.summarise_gazepoint_phase_coverage(
        segmented,
        phase_col="task_phase",
        group_cols=["subject"],
        time_col="time",
        value_cols=["value"],
    )

    assert not coverage.empty
    assert "complete_value_rate" in coverage

    no_time = qc.summarise_gazepoint_phase_coverage(
        segmented,
        phase_col="task_phase",
        group_cols=None,
        time_col=None,
        value_cols=["value"],
    )

    assert no_time["n_finite_time"].isna().all()

    bad_windows = pd.DataFrame(
        {
            "phase": ["x"],
            "start": ["bad"],
            "end": [1],
        }
    )

    with pytest.raises(ValueError, match="numeric"):
        qc.segment_gazepoint_task_phases(
            frame,
            time_col="time",
            phase_windows=bad_windows,
        )

    backwards = pd.DataFrame(
        {
            "phase": ["x"],
            "start": [2],
            "end": [1],
        }
    )

    with pytest.raises(ValueError, match="precede"):
        qc.segment_gazepoint_task_phases(
            frame,
            time_col="time",
            phase_windows=backwards,
        )

    with pytest.raises(ValueError, match="required column"):
        qc.summarise_gazepoint_phase_coverage(
            segmented,
            phase_col="task_phase",
            group_cols=["missing"],
            time_col="time",
        )
