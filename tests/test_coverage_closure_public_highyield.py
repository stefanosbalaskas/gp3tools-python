from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

aoi = importlib.import_module("gp3tools.aoi")
pupil = importlib.import_module("gp3tools.pupil")
qc = importlib.import_module("gp3tools.qc")


def test_qc_real_data_readiness_explicit_success_and_audit_objects():
    data = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S2", "S2"],
            "trial": ["T1", "T2", "T1", "T2"],
            "time": [0.0, 1.0, 0.0, 1.0],
            "condition": ["A", "B", "A", "B"],
            "stimulus": ["I1", "I2", "I1", "I2"],
            "aoi": ["target", "other", "target", "other"],
            "pupil": [3.0, 3.1, 3.2, 3.3],
            "x": [0.1, 0.2, 0.3, 0.4],
            "y": [0.1, 0.2, 0.3, 0.4],
            "valid": [True, True, True, True],
            "extra": [1, 2, 3, 4],
        }
    )

    result = qc.check_gazepoint_real_data_readiness(
        data,
        analysis_type="combined",
        participant_col="subject",
        trial_col="trial",
        time_col="time",
        condition_col="condition",
        stimulus_col="stimulus",
        aoi_col="aoi",
        pupil_col="pupil",
        gaze_x_col="x",
        gaze_y_col="y",
        tracking_valid_col="valid",
        required_cols=["extra"],
        audit_objects=[
            {"overview": pd.DataFrame({"status": ["pass"]})},
            {"overview": pd.DataFrame({"status": ["warn"]})},
            {"overview": pd.DataFrame({"status": ["fail"]})},
            "unknown",
        ],
        min_rows=1,
        min_participants=1,
        min_trials=1,
        max_missing_pupil_prop=0.5,
        max_missing_gaze_prop=0.5,
        max_condition_imbalance_ratio=3,
        name="coverage_gate",
    )

    assert result["overview"].loc[0, "object_name"] == "coverage_gate"
    assert {"fail", "warn", "pass", "info"}.issuperset(
        set(result["checks"]["status"])
    )
    assert len(result["detected_columns"]) == 10

    with pytest.raises(ValueError, match="analysis_type"):
        qc.check_gazepoint_real_data_readiness(
            data,
            analysis_type="bad",
        )

    with pytest.raises(KeyError, match="Missing required column"):
        qc.check_gazepoint_real_data_readiness(
            data,
            analysis_type="general",
            participant_col="missing",
        )


def test_qc_real_data_readiness_warning_branches():
    data = pd.DataFrame(
        {
            "subject": ["S1", "S1"],
            "trial": ["T1", "T1"],
            "time": [0.0, 0.0],
            "condition": ["A", "A"],
            "pupil": [np.nan, 3.0],
            "x": [0.1, np.nan],
            "valid": [True, False],
        }
    )

    result = qc.check_gazepoint_real_data_readiness(
        data,
        analysis_type="general",
        participant_col="subject",
        trial_col="trial",
        time_col="time",
        condition_col="condition",
        pupil_col="pupil",
        gaze_x_col="x",
        tracking_valid_col="valid",
        required_cols=["missing_user"],
        min_rows=3,
        min_participants=2,
        min_trials=2,
        max_missing_pupil_prop=0.1,
        max_missing_gaze_prop=0.1,
    )

    checks = result["checks"].set_index("check_id")
    assert checks.loc["user_required_columns", "status"] == "fail"
    assert checks.loc["minimum_rows", "status"] == "fail"
    assert checks.loc["paired_gaze_coordinates", "status"] == "warn"
    assert checks.loc["condition_count", "status"] == "warn"


def test_aoi_geometry_legacy_and_r_bounds_origin_paths():
    legacy = pd.DataFrame(
        {
            "xmin": [0.0, 0.5],
            "xmax": [0.4, 0.4],
            "ymin": [0.0, 0.0],
            "ymax": [0.4, 0.5],
        }
    )
    legacy_result = aoi.audit_gazepoint_aoi_geometry(legacy)
    assert not legacy_result["valid"]

    missing = aoi.audit_gazepoint_aoi_geometry(
        pd.DataFrame({"xmin": [0.0]})
    )
    assert not missing["valid"]

    bounds = pd.DataFrame(
        {
            "aoi": ["A", "B", "C", "D", "E"],
            "stimulus": ["I1"] * 5,
            "x_min": [0.0, 0.0, 0.2, -0.1, np.nan],
            "y_min": [0.0, 0.0, 0.2, 0.0, 0.0],
            "x_max": [0.2, 0.2, 0.2, 0.2, 0.3],
            "y_max": [0.2, 0.2, 0.3, 0.2, 0.3],
        }
    )
    result = aoi.audit_gazepoint_aoi_geometry(
        data=bounds,
        aoi_col="aoi",
        stimulus_col="stimulus",
        x_min_col="x_min",
        y_min_col="y_min",
        x_max_col="x_max",
        y_max_col="y_max",
        min_width=0.05,
        min_height=0.05,
        min_area=0.001,
        max_area_prop=0.5,
        require_within_screen=True,
    )

    statuses = set(result["geometry_summary"]["aoi_geometry_status"])
    assert "invalid_dimension" in statuses
    assert "outside_screen" in statuses
    assert "invalid_coordinate" in statuses
    assert len(result["duplicate_geometry"]) >= 1

    origin = pd.DataFrame(
        {
            "aoi": ["A"],
            "x": [0.1],
            "y": [0.1],
            "width": [0.2],
            "height": [0.3],
        }
    )
    origin_result = aoi.audit_gazepoint_aoi_geometry(
        data=origin,
        aoi_col="aoi",
        x_col="x",
        y_col="y",
        width_col="width",
        height_col="height",
    )
    assert origin_result["overview"].loc[0, "coordinate_format"] == "origin_size"


def test_aoi_geometry_validation_matrix():
    with pytest.raises(ValueError, match="data frame"):
        aoi._gp3_aoi_geometry_r_audit([])

    with pytest.raises(ValueError, match="at least one row"):
        aoi._gp3_aoi_geometry_r_audit(pd.DataFrame())

    no_geometry = pd.DataFrame({"aoi": ["A"]})
    with pytest.raises(ValueError, match="AOI geometry requires"):
        aoi._gp3_aoi_geometry_r_audit(no_geometry, aoi_col="aoi")

    geometry = pd.DataFrame(
        {
            "aoi": ["A"],
            "x_min": [0.0],
            "y_min": [0.0],
            "x_max": [1.0],
            "y_max": [1.0],
        }
    )

    with pytest.raises(ValueError, match="min_width"):
        aoi._gp3_aoi_geometry_r_audit(
            geometry,
            aoi_col="aoi",
            min_width=-1,
        )

    with pytest.raises(ValueError, match="max_area_prop"):
        aoi._gp3_aoi_geometry_r_audit(
            geometry,
            aoi_col="aoi",
            max_area_prop=2,
        )

    with pytest.raises(ValueError, match="require_within_screen"):
        aoi._gp3_aoi_geometry_r_audit(
            geometry,
            aoi_col="aoi",
            require_within_screen="yes",
        )

    with pytest.raises(TypeError, match="either aoi_geometry or data"):
        aoi.audit_gazepoint_aoi_geometry(
            geometry,
            data=geometry,
        )


def test_pupil_interpolate_legacy_methods_and_r_statuses():
    legacy = pd.DataFrame(
        {
            "subject": ["S1"] * 5,
            "time": [0.0, 0.1, 0.2, 0.3, 0.4],
            "pupil": [1.0, np.nan, 3.0, np.nan, 5.0],
        }
    )

    out = pupil.interpolate_gazepoint_pupil(
        legacy,
        pupil_col="pupil",
        time_col="time",
        group_cols=["subject"],
        max_gap_ms=150,
        output_col="filled",
        method="linear",
    )
    assert out["filled_interpolated"].sum() >= 1

    r_data = pd.DataFrame(
        {
            "subject": ["S1"] * 8,
            "media_id": ["M1"] * 8,
            "time_ms": [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, np.nan, 700.0],
            "pupil": [np.nan, 1.0, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0],
        }
    )

    result = pupil.interpolate_gazepoint_pupil(
        r_data,
        pupil_col="pupil",
        time_col="time_ms",
        group_cols=["subject", "media_id"],
        max_gap_ms=250,
        max_gap_samples=1,
        min_valid_points=2,
    )

    statuses = set(result["pupil_interpolation_status"])
    assert "missing_edge_gap" in statuses
    assert "interpolated" in statuses
    assert "missing_long_gap" in statuses
    assert "missing_no_time" in statuses


def test_pupil_interpolate_r_validation_and_insufficient_points():
    data = pd.DataFrame(
        {
            "subject": ["S1", "S1"],
            "media_id": ["M1", "M1"],
            "time_ms": [0.0, 1.0],
            "pupil": [1.0, np.nan],
        }
    )

    result = pupil.interpolate_gazepoint_pupil(
        data,
        group_cols=["subject", "media_id"],
        min_valid_points=2,
    )
    assert "missing_insufficient_valid" in set(result["pupil_interpolation_status"])

    for kwargs, match in [
        ({"pupil_col": ""}, "pupil_col"),
        ({"time_col": ""}, "time_col"),
        ({"group_cols": [1]}, "group_cols"),
        ({"max_gap_ms": -1}, "max_gap_ms"),
        ({"max_gap_samples": -1}, "max_gap_samples"),
        ({"min_valid_points": 1}, "min_valid_points"),
    ]:
        with pytest.raises(ValueError, match=match):
            pupil.interpolate_gazepoint_pupil(data, **kwargs)

    with pytest.raises(ValueError, match="No pupil"):
        pupil.interpolate_gazepoint_pupil(
            data.drop(columns=["pupil"]),
        )

    with pytest.raises(ValueError, match="No time"):
        pupil.interpolate_gazepoint_pupil(
            data.drop(columns=["time_ms"]),
        )

    with pytest.raises(ValueError, match="group_cols can only"):
        pupil.interpolate_gazepoint_pupil(
            data,
            group_cols=["bad"],
        )

    with pytest.raises(ValueError, match="requested but not found"):
        pupil.interpolate_gazepoint_pupil(
            data.drop(columns=["media_id"]),
            group_cols=["subject", "media_id"],
        )


def _reliability_data():
    rows = []
    for participant_index, participant in enumerate(["S1", "S2", "S3", "S4"], start=1):
        for trial in range(1, 5):
            rows.append(
                {
                    "subject": participant,
                    "trial": trial,
                    "condition": "A" if participant_index <= 2 else "B",
                    "pupil": participant_index + trial / 10,
                    "split": "left" if trial <= 2 else "right",
                }
            )
    return pd.DataFrame(rows)


def test_pupil_reliability_legacy_and_r_contracts():
    data = _reliability_data()

    pooled = pupil.audit_gazepoint_pupil_reliability(
        data[["pupil"]],
        pupil_col="pupil",
    )
    assert "mean_even" in pooled

    grouped = pupil.audit_gazepoint_pupil_reliability(
        data,
        pupil_col="pupil",
        subject_col="subject",
    )
    assert grouped.loc[0, "n_subjects"] == 4

    result = pupil.audit_gazepoint_pupil_reliability(
        data,
        outcome_cols=["pupil"],
        participant_col="subject",
        trial_col="trial",
        by_cols=["condition"],
        split_method="odd_even",
        aggregate_function="median",
        correlation_method="spearman",
        min_trials_per_split=2,
        name="coverage_reliability",
    )

    assert result["overview"].loc[0, "n_participants"] == 4
    assert not result["split_summary"].empty

    predefined = pupil.audit_gazepoint_pupil_reliability(
        data,
        outcome_cols=["pupil"],
        participant_col="subject",
        trial_col="trial",
        split_col="split",
        aggregate_function="mean",
        correlation_method="pearson",
        min_trials_per_split=2,
    )
    assert predefined["overview"].loc[0, "split_method"] == "predefined_split_col"


def test_pupil_reliability_validation_matrix():
    data = _reliability_data()

    with pytest.raises(ValueError, match="at least one row"):
        pupil.audit_gazepoint_pupil_reliability(
            data.iloc[0:0],
            outcome_cols=["pupil"],
        )

    for kwargs, match in [
        ({"split_method": "bad"}, "split_method"),
        ({"aggregate_function": "bad"}, "aggregate_function"),
        ({"correlation_method": "bad"}, "correlation_method"),
        ({"min_trials_per_split": 0}, "min_trials_per_split"),
        ({"name": ""}, "name"),
    ]:
        with pytest.raises(ValueError, match=match):
            pupil.audit_gazepoint_pupil_reliability(
                data,
                outcome_cols=["pupil"],
                **kwargs,
            )

    with pytest.raises(KeyError, match="participant_col"):
        pupil.audit_gazepoint_pupil_reliability(
            data.drop(columns=["subject"]),
            outcome_cols=["pupil"],
        )

    with pytest.raises(KeyError, match="trial_col"):
        pupil.audit_gazepoint_pupil_reliability(
            data,
            outcome_cols=["pupil"],
            participant_col="subject",
            trial_col="missing",
        )

    with pytest.raises(KeyError, match="split_col"):
        pupil.audit_gazepoint_pupil_reliability(
            data,
            outcome_cols=["pupil"],
            participant_col="subject",
            split_col="missing",
        )

    with pytest.raises(KeyError, match="Missing by_cols"):
        pupil.audit_gazepoint_pupil_reliability(
            data,
            outcome_cols=["pupil"],
            participant_col="subject",
            by_cols=["missing"],
        )

    with pytest.raises(KeyError, match="Missing outcome_cols"):
        pupil.audit_gazepoint_pupil_reliability(
            data,
            outcome_cols=["missing"],
            participant_col="subject",
        )

    with pytest.raises(ValueError, match="at least one numeric"):
        pupil.audit_gazepoint_pupil_reliability(
            data.assign(text="x"),
            outcome_cols=["text"],
            participant_col="subject",
        )

    with pytest.raises(ValueError, match="exactly two"):
        pupil.audit_gazepoint_pupil_reliability(
            data.assign(split="only"),
            outcome_cols=["pupil"],
            participant_col="subject",
            split_col="split",
        )
