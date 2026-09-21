from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

aoi = importlib.import_module("gp3tools.aoi")
pupil = importlib.import_module("gp3tools.pupil")
qc = importlib.import_module("gp3tools.qc")


# ============================================================================
# AOI — DYNAMIC RECTANGLE / POLYGON
# ============================================================================


def _dynamic_samples():
    return pd.DataFrame(
        {
            "subject": ["S1", "S1", "S1", "S2"],
            "TIME": [0.0, 5.0, 20.0, 5.0],
            "FPOGX": [0.5, 0.5, 0.8, 0.5],
            "FPOGY": [0.5, 0.5, 0.8, 0.5],
        }
    )


def _dynamic_rectangles():
    return pd.DataFrame(
        {
            "subject": ["S1", "S1", "S1"],
            "aoi_time": [0.0, 0.0, 10.0],
            "aoi_name": ["A", "B", "A"],
            "left": [0.0, 0.25, 0.5],
            "right": [1.0, 0.75, 1.0],
            "top": [0.0, 0.25, 0.5],
            "bottom": [1.0, 0.75, 1.0],
        }
    )


def test_h7_dynamic_rectangle_routes():
    out = aoi.add_gazepoint_dynamic_aoi(
        _dynamic_samples(),
        _dynamic_rectangles(),
        shape="rectangle",
        group_cols=["subject"],
        match="nearest",
        max_time_gap=6,
        output="both",
        overlap="last",
        include_overlap_count=True,
    )

    assert "aoi_current" in out
    assert "aoi_A" in out
    assert "aoi_B" in out
    assert "aoi_overlap_count" in out
    assert "aoi_definition_time" in out
    assert "aoi_time_gap" in out

    # S2 has no matching AOI definitions.
    assert pd.isna(out.loc[3, "aoi_definition_time"])

    previous = aoi.add_gazepoint_dynamic_aoi(
        _dynamic_samples().iloc[:3],
        _dynamic_rectangles(),
        shape="rectangle",
        group_cols=["subject"],
        match="previous",
        output="label",
        overlap="first",
    )

    assert "aoi_current" in previous

    nxt = aoi.add_gazepoint_dynamic_aoi(
        _dynamic_samples().iloc[:2],
        _dynamic_rectangles(),
        shape="rectangle",
        group_cols=["subject"],
        match="next",
        output="logical",
    )

    assert "aoi_A" in nxt


def test_h7_dynamic_polygon_route():
    vertices = pd.DataFrame(
        {
            "subject": ["S1"] * 4,
            "aoi_time": [0.0] * 4,
            "aoi_name": ["P"] * 4,
            "vertex_x": [0.0, 1.0, 1.0, 0.0],
            "vertex_y": [0.0, 0.0, 1.0, 1.0],
            "vertex_order": [1, 2, 3, 4],
        }
    )

    samples = pd.DataFrame(
        {
            "subject": ["S1", "S1"],
            "TIME": [0.0, 0.0],
            "FPOGX": [0.5, 2.0],
            "FPOGY": [0.5, 2.0],
        }
    )

    out = aoi.add_gazepoint_dynamic_aoi(
        samples,
        vertices,
        shape="polygon",
        group_cols=["subject"],
        vertex_order_col="vertex_order",
        output="both",
        boundary="inside",
    )

    assert out.loc[0, "aoi_current"] == "P"
    assert out.loc[1, "aoi_current"] == "outside"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shape": "bad"},
        {"shape": "rectangle", "match": "bad"},
        {"shape": "rectangle", "output": "bad"},
        {"shape": "rectangle", "overlap": "bad"},
        {"shape": "rectangle", "boundary": "bad"},
        {"shape": "rectangle", "max_time_gap": -1},
    ],
)
def test_h7_dynamic_aoi_validation(kwargs):
    with pytest.raises(ValueError):
        aoi.add_gazepoint_dynamic_aoi(
            _dynamic_samples(),
            _dynamic_rectangles(),
            group_cols=["subject"],
            **kwargs,
        )


def test_h7_dynamic_aoi_shape_and_name_failures():
    bad_shape = pd.DataFrame(
        {
            "subject": ["S1"],
            "aoi_time": [0.0],
            "aoi_name": ["A"],
            "foo": [1],
        }
    )

    with pytest.raises(ValueError):
        aoi.add_gazepoint_dynamic_aoi(
            _dynamic_samples(),
            bad_shape,
            shape="auto",
            group_cols=["subject"],
        )

    bad_name = _dynamic_rectangles()
    bad_name.loc[0, "aoi_name"] = ""

    with pytest.raises(ValueError):
        aoi.add_gazepoint_dynamic_aoi(
            _dynamic_samples(),
            bad_name,
            shape="rectangle",
            group_cols=["subject"],
        )

    bad_bounds = _dynamic_rectangles()
    bad_bounds.loc[0, "left"] = np.nan

    with pytest.raises(ValueError):
        aoi.add_gazepoint_dynamic_aoi(
            _dynamic_samples(),
            bad_bounds,
            shape="rectangle",
            group_cols=["subject"],
        )


# ============================================================================
# AOI — GEOMETRY AUDIT
# ============================================================================


def test_h7_geometry_audit_bounds_and_origin_size():
    bounds = pd.DataFrame(
        {
            "AOI": [
                "good",
                "tiny",
                "outside",
                "missing",
                "dup1",
                "dup2",
            ],
            "x_min": [0.1, 0.1, -0.2, np.nan, 0.2, 0.2],
            "y_min": [0.1, 0.1, 0.1, 0.1, 0.2, 0.2],
            "x_max": [0.4, 0.11, 0.2, 0.4, 0.4, 0.4],
            "y_max": [0.4, 0.11, 0.2, 0.4, 0.4, 0.4],
        }
    )

    result = aoi._gp3_aoi_geometry_r_audit(
        bounds,
        min_width=0.05,
        min_height=0.05,
        min_area=0.002,
        max_area_prop=0.5,
        require_within_screen=True,
    )

    assert result["overview"].loc[0, "coordinate_format"] == "bounds"
    assert result["overview"].loc[0, "aoi_geometry_status"] == "review"
    assert not result["flagged_aois"].empty
    assert not result["duplicate_geometry"].empty

    origin = pd.DataFrame(
        {
            "AOI": ["A", "B"],
            "x": [0.1, 0.2],
            "y": [0.1, 0.2],
            "width": [0.3, -0.1],
            "height": [0.3, 0.2],
        }
    )

    result2 = aoi._gp3_aoi_geometry_r_audit(
        origin,
        require_within_screen=False,
    )

    assert result2["overview"].loc[0, "coordinate_format"] == "origin_size"
    assert not result2["flagged_aois"].empty


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_width": -1},
        {"min_height": -1},
        {"min_area": -1},
        {"max_area_prop": 2},
        {"require_within_screen": "yes"},
    ],
)
def test_h7_geometry_validation(kwargs):
    frame = pd.DataFrame(
        {
            "AOI": ["A"],
            "x_min": [0.1],
            "y_min": [0.1],
            "x_max": [0.5],
            "y_max": [0.5],
        }
    )

    with pytest.raises(ValueError):
        aoi._gp3_aoi_geometry_r_audit(
            frame,
            **kwargs,
        )


# ============================================================================
# AOI — TRANSITIONS
# ============================================================================


def _sequence_table():
    return pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S1",
                "S1",
                "S1",
            ],
            "aoi_state": [
                "target",
                "background",
                "distractor",
                "other",
                "target",
            ],
            "transition_from": [
                "target",
                "background",
                "distractor",
                "other",
                "target",
            ],
            "transition_to": [
                "background",
                "distractor",
                "target",
                "target",
                pd.NA,
            ],
            "entry_duration_ms": [
                10.0,
                20.0,
                30.0,
                np.nan,
                50.0,
            ],
            "dwell_before_transition_ms": [
                10.0,
                20.0,
                np.nan,
                40.0,
                np.nan,
            ],
            "is_non_aoi": [
                False,
                True,
                False,
                False,
                False,
            ],
            "is_terminal_state": [
                False,
                False,
                False,
                False,
                True,
            ],
        }
    )


def test_h7_transition_full_classification():
    result = aoi.summarise_gazepoint_aoi_transitions(
        _sequence_table(),
        group_cols=["subject"],
        include_non_aoi=True,
        target_aoi_values=["target"],
        distractor_aoi_values=["distractor"],
        non_aoi_values=["background"],
    )

    row = result.iloc[0]

    assert row["transition_feature_status"] == "ok"
    assert row["target_to_background"] == 1
    assert row["background_to_distractor"] == 1
    assert row["distractor_to_target"] == 1
    assert row["other_transitions"] >= 1
    assert row["total_state_dwell_ms"] > 0


def test_h7_transition_terminal_only():
    terminal = _sequence_table().iloc[[-1]].copy()

    result = aoi.summarise_gazepoint_aoi_transitions(
        terminal,
        group_cols=["subject"],
        include_non_aoi=True,
        target_aoi_values=["target"],
        distractor_aoi_values=["distractor"],
    )

    assert result.loc[0, "total_transitions"] == 0
    assert result.loc[0, "transition_feature_status"] == "no_transitions"


# ============================================================================
# AOI — SEQUENCE ANOMALIES
# ============================================================================


def _anomaly_rows():
    rows = []

    for i, states in enumerate(
        [
            ["A"],
            ["A", "B", "C"],
            ["A", None, None, None],
            ["A", "A", "A", "A", "A", "A"],
        ],
        start=1,
    ):
        for j, state in enumerate(states):
            rows.append(
                {
                    "subject": f"S{i}",
                    "time": j,
                    "aoi": state,
                }
            )

    return pd.DataFrame(rows)


def test_h7_sequence_anomaly_statuses():
    out = aoi.flag_gazepoint_sequence_anomalies(
        _anomaly_rows(),
        aoi_col="aoi",
        group_cols=["subject"],
        time_col="time",
        min_length=2,
        max_length=4,
        max_missing_prop=0.25,
        z_threshold=0.5,
        min_unique_aoi=2,
    )

    assert out["anomaly_flag"].any()
    assert out["flag_short"].any()
    assert out["flag_long"].any()
    assert out["flag_high_missing"].any()
    assert out["flag_low_unique"].any()
    assert out["anomaly_reason"].ne("none").any()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_length": -1},
        {"max_length": "bad"},
        {"max_missing_prop": -1},
        {"max_missing_prop": 2},
        {"z_threshold": 0},
    ],
)
def test_h7_sequence_anomaly_validation(kwargs):
    with pytest.raises(ValueError):
        aoi.flag_gazepoint_sequence_anomalies(
            _anomaly_rows(),
            aoi_col="aoi",
            group_cols=["subject"],
            **kwargs,
        )


def test_h7_sequence_anomaly_misc_validation():
    with pytest.raises(TypeError):
        aoi.flag_gazepoint_sequence_anomalies(
            _anomaly_rows(),
            aoi_col="aoi",
            group_cols=["subject"],
            impossible=True,
        )

    with pytest.raises(ValueError):
        aoi.flag_gazepoint_sequence_anomalies(
            _anomaly_rows(),
            aoi_col="aoi",
            group_cols=[],
        )

    with pytest.raises(ValueError):
        aoi.flag_gazepoint_sequence_anomalies(
            _anomaly_rows(),
            aoi_col="aoi",
            group_cols=["subject"],
            time_col="",
        )


# ============================================================================
# AOI — WINDOWS
# ============================================================================


def _window_rows():
    return pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S2",
                "S2",
                "S3",
                "S3",
                "S4",
                "S4",
            ],
            "time": [
                0,
                50,
                0,
                50,
                0,
                50,
                0,
                50,
            ],
            "aoi_current": [
                "target",
                "target",
                "other",
                "other",
                pd.NA,
                pd.NA,
                "target",
                "other",
            ],
        }
    )


def test_h7_aoi_window_status_matrix():
    result = aoi.summarise_gazepoint_aoi_windows(
        _window_rows(),
        windows=[0, 100],
        group_cols=["subject"],
        condition_col=None,
        target_aoi_values=["target"],
        distractor_aoi_values=["distractor"],
    )

    statuses = set(result["aoi_window_status"])

    assert "target_only" in statuses
    assert "target_not_observed" in statuses
    assert "zero_valid_denominator" in statuses
    assert "ok" in statuses

    no_target = aoi.summarise_gazepoint_aoi_windows(
        _window_rows(),
        windows=[0, 100],
        group_cols=["subject"],
        condition_col=None,
        target_aoi_values=None,
    )

    assert set(no_target["aoi_window_status"]) == {"no_target_aoi_defined"}


def test_h7_aoi_window_dataframe_and_validation():
    windows = pd.DataFrame(
        {
            "window_label": ["early"],
            "window_start_ms": [0.0],
            "window_end_ms": [100.0],
        }
    )

    result = aoi.summarise_gazepoint_aoi_windows(
        _window_rows(),
        windows=windows,
        group_cols=["subject"],
        condition_col=None,
        target_aoi_values=["target"],
        include_right_endpoint=True,
    )

    assert len(result) == 4

    with pytest.raises(ValueError):
        aoi.summarise_gazepoint_aoi_windows(
            _window_rows(),
            windows=[0],
            group_cols=["subject"],
            condition_col=None,
        )

    with pytest.raises(ValueError):
        aoi.summarise_gazepoint_aoi_windows(
            _window_rows(),
            windows=[0, 0],
            group_cols=["subject"],
            condition_col=None,
        )

    with pytest.raises(ValueError):
        aoi.summarise_gazepoint_aoi_windows(
            _window_rows(),
            windows=[1000, 2000],
            group_cols=["subject"],
            condition_col=None,
        )

    with pytest.raises(ValueError):
        aoi.summarise_gazepoint_aoi_windows(
            _window_rows(),
            windows=windows,
            group_cols=["subject"],
            condition_col=None,
            include_right_endpoint="yes",
        )


# ============================================================================
# PUPIL — BASELINE QUALITY REASONS
# ============================================================================


def _baseline_quality_rows():
    rows = []

    specs = {
        "no": {
            "n": 0,
            "status": "no_baseline",
            "available": False,
            "pupils": [3.0, 3.0],
            "interp": [False, False],
            "artifact": [False, False],
        },
        "missing_n": {
            "n": np.nan,
            "status": "ok",
            "available": True,
            "pupils": [3.0, 3.0],
            "interp": [False, False],
            "artifact": [False, False],
        },
        "missing": {
            "n": 2,
            "status": "ok",
            "available": True,
            "pupils": [3.0, np.nan],
            "interp": [False, False],
            "artifact": [False, False],
        },
        "interp": {
            "n": 2,
            "status": "ok",
            "available": True,
            "pupils": [3.0, 3.1],
            "interp": [True, True],
            "artifact": [False, False],
        },
        "artifact": {
            "n": 2,
            "status": "ok",
            "available": True,
            "pupils": [3.0, 3.1],
            "interp": [False, False],
            "artifact": [True, True],
        },
        "ok": {
            "n": 2,
            "status": "ok",
            "available": True,
            "pupils": [3.0, 3.1],
            "interp": [False, False],
            "artifact": [False, False],
        },
    }

    for subject, spec in specs.items():
        for i in range(2):
            rows.append(
                {
                    "subject": subject,
                    "media_id": "M1",
                    "time": -100 + (50 * i),
                    "pupil_interpolated": spec["pupils"][i],
                    "pupil_baseline_n": spec["n"],
                    "pupil_baseline_status": spec["status"],
                    "pupil_baseline_available": spec["available"],
                    "pupil_baseline_used": spec["available"],
                    "pupil_baseline_window_start": -200,
                    "pupil_baseline_window_end": 0,
                    "pupil_was_interpolated": spec["interp"][i],
                    "artifact_flag": spec["artifact"][i],
                }
            )

    return pd.DataFrame(rows)


def test_h7_baseline_quality_reason_matrix():
    result = pupil.audit_gazepoint_pupil_baseline(
        _baseline_quality_rows(),
        group_cols=["subject", "media_id"],
        baseline_n_col="pupil_baseline_n",
        artifact_col="artifact_flag",
        min_baseline_samples=2,
        max_missing_pct=20,
        max_interpolated_pct=20,
        max_artifact_pct=20,
    )

    reasons = set(result["baseline_quality_reason"])

    assert "no_baseline" in reasons
    assert "missing_baseline_n" in reasons
    assert "high_baseline_missing_pct" in reasons
    assert "high_baseline_interpolated_pct" in reasons
    assert "high_baseline_artifact_pct" in reasons
    assert "ok" in reasons


def test_h7_baseline_flag_and_artifact_reason_paths():
    frame = _baseline_quality_rows().loc[lambda z: z["subject"].eq("ok")].copy()

    frame["baseline_flag"] = ["yes", "yes"]
    frame["artifact_reason"] = ["valid", "blink"]

    result = pupil.audit_gazepoint_pupil_baseline(
        frame,
        group_cols=["subject"],
        baseline_n_col="pupil_baseline_n",
        baseline_flag_col="baseline_flag",
        artifact_col=None,
        artifact_reason_col="artifact_reason",
    )

    assert len(result) == 1


# ============================================================================
# PUPIL — DRIFT
# ============================================================================


def _drift_rows():
    rows = []

    for subject, slope in (
        ("S1", 0.01),
        ("S2", 0.0),
        ("S3", -0.01),
    ):
        for trial in range(1, 5):
            rows.append(
                {
                    "subject": subject,
                    "condition": "A" if trial <= 2 else "B",
                    "trial": trial,
                    "time": trial * 1000.0,
                    "pupil": 3.0 + slope * trial * 1000.0,
                    "excluded_trial": trial == 4,
                }
            )

    rows.append(
        {
            "subject": "S4",
            "condition": "A",
            "trial": 1,
            "time": 1000.0,
            "pupil": 3.0,
            "excluded_trial": False,
        }
    )

    return pd.DataFrame(rows)


def test_h7_pupil_drift_statuses():
    result = pupil.audit_gazepoint_pupil_drift(
        _drift_rows(),
        pupil_col="pupil",
        time_col="time",
        group_cols=["subject"],
        order_col="trial",
        condition_col="condition",
        exclude_col="excluded_trial",
        include_excluded=False,
        min_valid_samples=2,
        max_abs_slope_per_min=1,
        max_condition_time_mean_diff_ms=100,
        max_condition_order_mean_diff=0.1,
    )

    statuses = set(result["by_group"]["drift_status"])

    assert "possible_drift" in statuses
    assert "insufficient_valid_samples" in statuses
    assert not result["summary"].empty


@pytest.mark.parametrize(
    "exclude_values",
    [
        [0, 0, 0, 1] * 3 + [0],
        ["no", "no", "no", "yes"] * 3 + ["no"],
    ],
)
def test_h7_pupil_drift_exclusion_types(exclude_values):
    frame = _drift_rows()
    frame["excluded_trial"] = exclude_values

    result = pupil.audit_gazepoint_pupil_drift(
        frame,
        pupil_col="pupil",
        time_col="time",
        group_cols=["subject"],
        order_col="trial",
        condition_col="condition",
        exclude_col="excluded_trial",
        min_valid_samples=2,
    )

    assert not result["by_group"].empty


@pytest.mark.parametrize(
    "kwargs",
    [
        {"group_cols": []},
        {"min_valid_samples": 0},
        {"max_abs_slope_per_min": np.inf},
        {"max_condition_time_mean_diff_ms": np.inf},
        {"max_condition_order_mean_diff": np.inf},
    ],
)
def test_h7_pupil_drift_validation(kwargs):
    with pytest.raises(ValueError):
        pupil.audit_gazepoint_pupil_drift(
            _drift_rows(),
            pupil_col="pupil",
            time_col="time",
            order_col="trial",
            condition_col="condition",
            **kwargs,
        )


# ============================================================================
# PUPIL — RELIABILITY
# ============================================================================


def _reliability_rows():
    rows = []

    for participant in range(1, 6):
        for trial in range(1, 5):
            rows.append(
                {
                    "subject": f"S{participant}",
                    "trial": trial,
                    "condition": "A",
                    "pupil": (participant * 10 + trial),
                    "constant": 1.0,
                    "half": ("first" if trial <= 2 else "second"),
                }
            )

    return pd.DataFrame(rows)


def test_h7_reliability_modes():
    frame = _reliability_rows()

    odd_even = pupil.audit_gazepoint_pupil_reliability(
        frame,
        outcome_cols=["pupil", "constant"],
        participant_col="subject",
        trial_col="trial",
        by_cols=["condition"],
        split_method="odd_even",
        min_trials_per_split=2,
    )

    assert not odd_even["reliability_summary"].empty

    statuses = set(odd_even["reliability_summary"]["reliability_status"])

    assert "ready" in statuses
    assert "constant_split_values" in statuses

    first_second = pupil.audit_gazepoint_pupil_reliability(
        frame,
        outcome_cols="pupil",
        participant_col="subject",
        trial_col="trial",
        split_method="first_second",
        aggregate_function="median",
        correlation_method="spearman",
        min_trials_per_split=2,
    )

    assert (
        first_second["overview"].loc[
            0,
            "split_method",
        ]
        == "first_second"
    )

    predefined = pupil.audit_gazepoint_pupil_reliability(
        frame,
        outcome_cols="pupil",
        participant_col="subject",
        trial_col="trial",
        split_col="half",
        min_trials_per_split=2,
    )

    assert (
        predefined["overview"].loc[
            0,
            "split_method",
        ]
        == "predefined_split_col"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"split_method": "bad"},
        {"aggregate_function": "bad"},
        {"correlation_method": "bad"},
        {"min_trials_per_split": 0},
        {"name": ""},
    ],
)
def test_h7_reliability_validation(kwargs):
    with pytest.raises(ValueError):
        pupil.audit_gazepoint_pupil_reliability(
            _reliability_rows(),
            outcome_cols="pupil",
            participant_col="subject",
            trial_col="trial",
            **kwargs,
        )


def test_h7_reliability_column_validation():
    frame = _reliability_rows()

    with pytest.raises(KeyError):
        pupil.audit_gazepoint_pupil_reliability(
            frame,
            outcome_cols="pupil",
            participant_col="missing",
        )

    with pytest.raises(KeyError):
        pupil.audit_gazepoint_pupil_reliability(
            frame,
            outcome_cols="pupil",
            participant_col="subject",
            trial_col="missing",
        )

    with pytest.raises(KeyError):
        pupil.audit_gazepoint_pupil_reliability(
            frame,
            outcome_cols="missing",
            participant_col="subject",
        )

    bad_split = frame.copy()
    bad_split["half"] = "one"

    with pytest.raises(ValueError):
        pupil.audit_gazepoint_pupil_reliability(
            bad_split,
            outcome_cols="pupil",
            participant_col="subject",
            split_col="half",
        )


# ============================================================================
# PUPIL — TRIAL FEATURES
# ============================================================================


def _feature_rows():
    rows = []

    for subject in ("S1", "S2"):
        for time, value in (
            (0, 1.0),
            (250, 2.0),
            (750, 3.0),
            (1750, 4.0),
            (2500, np.nan if subject == "S2" else 5.0),
        ):
            rows.append(
                {
                    "subject": subject,
                    "trial_global": "T1",
                    "time": time,
                    "pupil": value,
                    "pupil_was_interpolated": (time == 250),
                    "artifact_reason": ("blink" if time == 750 else "valid"),
                }
            )

    return pd.DataFrame(rows)


def test_h7_trial_feature_full_paths():
    result = pupil.summarise_gazepoint_pupil_trial_features(
        _feature_rows(),
        pupil_col="pupil",
        time_col="time",
        group_cols=["subject", "trial_global"],
        artifact_reason_col="artifact_reason",
        early_window=(0, 500),
        middle_window=(500, 1500),
        late_window=(1500, 3000),
        min_valid_samples=5,
    )

    assert len(result) == 2
    assert "pupil_auc" in result
    assert "early_mean_pupil" in result
    assert "late_mean_pupil" in result
    assert "insufficient_valid_samples" in set(result["pupil_feature_status"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"early_window": (1, 1)},
        {"middle_window": (2, 1)},
        {"late_window": (0,)},
    ],
)
def test_h7_trial_feature_window_validation(kwargs):
    with pytest.raises(ValueError):
        pupil.summarise_gazepoint_pupil_trial_features(
            _feature_rows(),
            pupil_col="pupil",
            time_col="time",
            group_cols=["subject", "trial_global"],
            **kwargs,
        )


# ============================================================================
# QC — NAMING
# ============================================================================


def test_h7_naming_all_branches(tmp_path):
    empty = qc.audit_gazepoint_naming_consistency([])

    assert empty["summary"].loc[0, "status"] == "pass"

    mixed = qc.audit_gazepoint_naming_consistency(
        [
            None,
            "",
            "summarise_alpha",
            "summarize_alpha",
            "summarise_beta",
            "summarize_gamma",
            "summarise_alpha",
        ]
    )

    statuses = set(mixed["pairs"]["status"])

    assert "paired" in statuses
    assert "canonical_only" in statuses
    assert "missing_british_alias" in statuses

    default = qc.audit_gazepoint_naming_consistency()

    assert "summary" in default

    obj = SimpleNamespace(pairs=mixed["pairs"])

    path = qc.write_gazepoint_naming_audit(
        x=obj,
        output_file=tmp_path / "nested" / "audit.csv",
    )

    assert path.exists()


# ============================================================================
# QC — EXCLUSIONS, SINGLE-COORDINATE AND NULL METADATA
# ============================================================================


def test_h7_exclusion_single_coordinate_paths():
    frame = pd.DataFrame(
        {
            "participant": [
                "S1",
                "S1",
                "S2",
                "S2",
            ],
            "x": [
                0.1,
                np.nan,
                0.2,
                0.3,
            ],
            "validity": [
                "yes",
                "no",
                "yes",
                "yes",
            ],
        }
    )

    result = qc.recommend_gazepoint_exclusions(
        frame,
        participant_col="participant",
        validity_col="validity",
        x_col="x",
        require_both_gaze_coordinates=False,
        min_trial_samples=1,
        min_participant_trials=1,
        min_participant_valid_trials=1,
    )

    assert result["overview"].loc[0, "trial_col"] is pd.NA or pd.isna(
        result["overview"].loc[0, "trial_col"]
    )

    assert not result["participant_recommendations"].empty


def test_h7_exclusion_y_only_and_artifact():
    frame = pd.DataFrame(
        {
            "participant": ["S1", "S1"],
            "condition": ["A", "B"],
            "y": [0.1, np.nan],
            "artifact": [0, 1],
        }
    )

    result = qc.recommend_gazepoint_exclusions(
        frame,
        participant_col="participant",
        condition_col="condition",
        y_col="y",
        artifact_col="artifact",
        require_both_gaze_coordinates=False,
        min_trial_samples=1,
        min_participant_trials=1,
        min_participant_valid_trials=1,
    )

    assert not result["exclusion_table"].empty
