from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

aoi = importlib.import_module("gp3tools.aoi")
pupil = importlib.import_module("gp3tools.pupil")
qc = importlib.import_module("gp3tools.qc")


# ============================================================================
# QC — close all executable H13 residuals
# ============================================================================


def test_h14_qc_exclusion_flow_without_condition():
    result = qc.audit_gazepoint_exclusion_flow(
        pd.DataFrame(
            {
                "subject": [
                    "S1",
                    "S2",
                ],
                "include": [
                    True,
                    False,
                ],
            }
        ),
        subject_col="subject",
        condition_col=None,
        unit_cols=None,
        include_col="include",
    )

    assert result["condition_summary"].empty

    assert len(result["subject_summary"]) == 2


def test_h14_qc_recommendation_string_setting_values():
    result = qc.recommend_gazepoint_exclusions(
        pd.DataFrame(
            {
                "subject": [
                    "S1",
                    "S1",
                    "S2",
                    "S2",
                ],
                "trial": [
                    1,
                    1,
                    1,
                    1,
                ],
                "condition": [
                    "A",
                    "A",
                    "B",
                    "B",
                ],
                "valid": [
                    True,
                    False,
                    True,
                    True,
                ],
                "x": [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                ],
                "y": [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                ],
                "pupil": [
                    3.0,
                    3.1,
                    3.2,
                    3.3,
                ],
            }
        ),
        participant_col="subject",
        trial_col="trial",
        condition_col="condition",
        validity_col="valid",
        x_col="x",
        y_col="y",
        pupil_col="pupil",
        min_trial_samples=1,
        min_participant_trials=1,
        min_participant_valid_trials=1,
    )

    overview = result["overview"].iloc[0]

    assert overview["participant_col"] == "subject"

    assert overview["trial_col"] == "trial"

    assert overview["condition_col"] == "condition"


def test_h14_qc_naming_audit_whitespace_path():
    with pytest.raises(ValueError):
        qc.write_gazepoint_naming_audit(
            x={
                "pairs": pd.DataFrame(
                    {
                        "raw_name": ["x"],
                        "standard_name": ["y"],
                    }
                )
            },
            output_file="   ",
        )


# ============================================================================
# AOI — high-confidence executable residuals
# ============================================================================


def test_h14_polygon_legacy_missing_geometry_columns():
    with pytest.raises(ValueError):
        aoi.add_gazepoint_polygon_aoi(
            pd.DataFrame(
                {
                    "x": [0.2],
                    "y": [0.2],
                }
            ),
            pd.DataFrame({"bad": [1]}),
            x_col="x",
            y_col="y",
        )


def _dynamic_gaze():
    return pd.DataFrame(
        {
            "TIME": [0.0],
            "FPOGX": [0.2],
            "FPOGY": [0.2],
        }
    )


def test_h14_dynamic_auto_polygon():
    definitions = pd.DataFrame(
        {
            "aoi_time": [
                0.0,
                0.0,
                0.0,
            ],
            "aoi_name": [
                "A",
                "A",
                "A",
            ],
            "vertex_x": [
                0.0,
                1.0,
                0.0,
            ],
            "vertex_y": [
                0.0,
                0.0,
                1.0,
            ],
        }
    )

    result = aoi.add_gazepoint_dynamic_aoi(
        _dynamic_gaze(),
        definitions,
        shape="auto",
    )

    assert (
        result.loc[
            0,
            "aoi_current",
        ]
        == "A"
    )


def test_h14_dynamic_auto_shape_failure():
    with pytest.raises(
        ValueError,
        match="infer dynamic AOI shape",
    ):
        aoi.add_gazepoint_dynamic_aoi(
            _dynamic_gaze(),
            pd.DataFrame(
                {
                    "aoi_time": [0.0],
                    "aoi_name": ["A"],
                }
            ),
            shape="auto",
        )


def test_h14_geometry_all_nonfinite_size_summary():
    result = aoi.audit_gazepoint_aoi_geometry(
        pd.DataFrame(
            {
                "aoi": ["A"],
                "x_min": [np.nan],
                "x_max": [np.nan],
                "y_min": [np.nan],
                "y_max": [np.nan],
            }
        ),
        aoi_col="aoi",
    )

    assert np.isnan(
        result["size_summary"].loc[
            0,
            "min_width",
        ]
    )


def test_h14_geometry_data_conflict():
    geometry = pd.DataFrame(
        {
            "aoi": [
                "A",
            ],
            "x_min": [
                0.0,
            ],
            "x_max": [
                1.0,
            ],
            "y_min": [
                0.0,
            ],
            "y_max": [
                1.0,
            ],
        }
    )

    result = aoi.audit_gazepoint_aoi_geometry(
        data=geometry,
        aoi_col="aoi",
    )

    assert result["_gp3_class"] == "gp3_aoi_geometry_audit"

    assert (
        result["geometry_summary"].loc[
            0,
            "aoi",
        ]
        == "A"
    )


def test_h14_overlap_data_conflict():
    geometry = pd.DataFrame(
        {
            "aoi": ["A"],
            "xmin": [0.0],
            "xmax": [1.0],
            "ymin": [0.0],
            "ymax": [1.0],
        }
    )

    with pytest.raises(TypeError):
        aoi.audit_gazepoint_aoi_overlap(
            geometry,
            data=geometry,
            aoi_col="aoi",
        )


def test_h14_dynamic_coverage_missing_columns():
    with pytest.raises(ValueError):
        aoi.audit_gazepoint_dynamic_aoi_coverage(
            pd.DataFrame({"aoi_current": ["A"]}),
            label_col="aoi_current",
            definition_time_col="aoi_definition_time",
            time_gap_col="aoi_time_gap",
        )


def test_h14_dynamic_coverage_bad_gap():
    with pytest.raises(ValueError):
        aoi.audit_gazepoint_dynamic_aoi_coverage(
            pd.DataFrame(
                {
                    "aoi_current": ["A"],
                    "aoi_definition_time": [0.0],
                    "aoi_time_gap": [0.0],
                }
            ),
            label_col="aoi_current",
            definition_time_col="aoi_definition_time",
            time_gap_col="aoi_time_gap",
            max_time_gap=-1,
        )


def test_h14_coding_resolve_alias_paths():
    columns = pd.Index(
        [
            "MEDIA_ID",
            "media_id",
            "AOI",
            "aoi",
        ]
    )

    assert (
        aoi._gp3_aoi_coding_r_resolve(
            "MEDIA_ID",
            columns,
            "column",
        )
        == "media_id"
    )

    assert (
        aoi._gp3_aoi_coding_r_resolve(
            "AOI",
            columns,
            "column",
        )
        == "aoi"
    )

    assert (
        aoi._gp3_aoi_coding_r_resolve(
            None,
            columns,
            "column",
            candidates=("MEDIA_ID",),
        )
        == "media_id"
    )

    assert (
        aoi._gp3_aoi_coding_r_resolve(
            None,
            columns,
            "column",
            candidates=("AOI",),
        )
        == "aoi"
    )


def test_h14_coding_range_conversion_failure():
    with pytest.raises(ValueError):
        aoi._gp3_aoi_coding_r_range(
            object(),
            "screen",
        )


def test_h14_coding_character_scalar():
    assert aoi._gp3_aoi_coding_r_character_vector(
        "target",
        "values",
    ) == ["target"]


def test_h14_coding_geometry_empty():
    with pytest.raises(ValueError):
        aoi._gp3_aoi_coding_r_geometry(
            pd.DataFrame(),
            aoi_col=None,
            stimulus_col=None,
            x_min_col=None,
            y_min_col=None,
            x_max_col=None,
            y_max_col=None,
            x_col=None,
            y_col=None,
            width_col=None,
            height_col=None,
            screen_x_range=(
                0,
                1,
            ),
            screen_y_range=(
                0,
                1,
            ),
        )


def test_h14_coding_assign_no_matching_stimulus():
    result = aoi._gp3_aoi_coding_r_assign(
        pd.DataFrame(
            {
                "x": [0.5],
                "y": [0.5],
                "stimulus": ["S2"],
            }
        ),
        pd.DataFrame(
            {
                "aoi": ["A"],
                "stimulus": ["S1"],
                "x_min": [0.0],
                "x_max": [1.0],
                "y_min": [0.0],
                "y_max": [1.0],
            }
        ),
        gaze_x_col="x",
        gaze_y_col="y",
        gaze_stimulus_col="stimulus",
        geometry_aoi_col="aoi",
        geometry_stimulus_col="stimulus",
        tie_method="ambiguous",
        outside_label="outside",
        ambiguous_label="ambiguous",
        missing_label="missing",
    )

    assert (
        result.loc[
            0,
            "derived_aoi",
        ]
        == "outside"
    )


def test_h14_coding_matrix_empty_gaze():
    with pytest.raises(ValueError):
        aoi.audit_gazepoint_aoi_coding_matrix(
            gaze_data=pd.DataFrame(),
            aoi_geometry=pd.DataFrame(
                {
                    "aoi": ["A"],
                    "x_min": [0.0],
                    "x_max": [1.0],
                    "y_min": [0.0],
                    "y_max": [1.0],
                }
            ),
        )


def test_h14_aoi_summary_requires_fixations():
    with pytest.raises(TypeError):
        aoi.summarise_gazepoint_aoi(
            gaze_data=pd.DataFrame(
                {
                    "USER_FILE": ["u1.csv"],
                    "MEDIA_ID": ["M1"],
                    "MEDIA_NAME": ["stimulus"],
                    "AOI": ["A"],
                    "TIME": [0.0],
                }
            )
        )


def test_h14_r_entries_validation_and_no_time():
    frame = pd.DataFrame(
        {
            "subject": ["S1"],
            "time": [0.0],
            "aoi": ["A"],
        }
    )

    with pytest.raises(ValueError):
        aoi._gp3_aoi_r_entries(
            frame,
            aoi_col="aoi",
            time_col="time",
            group_cols=["subject"],
            include_non_aoi="yes",
        )

    frame = frame.copy()

    frame["time"] = np.nan

    with pytest.raises(
        ValueError,
        match="No non-missing time",
    ):
        aoi._gp3_aoi_r_entries(
            frame,
            aoi_col="aoi",
            time_col="time",
            group_cols=["subject"],
        )


def test_h14_recurrence_validation_paths():
    with pytest.raises(ValueError):
        aoi.compute_gazepoint_sequence_recurrence(
            data=None,
            sequence=None,
            min_line=3,
        )

    with pytest.raises(ValueError):
        aoi.compute_gazepoint_sequence_recurrence(
            data=pd.DataFrame({"aoi": ["A"]}),
            aoi_col="",
            min_line=3,
        )

    with pytest.raises(ValueError):
        aoi.compute_gazepoint_sequence_recurrence(
            data=pd.DataFrame({"x": ["A"]}),
            aoi_col="aoi",
            min_line=3,
        )


def test_h14_scanpath_sequence_validation_paths():
    frame = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S2",
            ],
            "time": [
                0,
                0,
            ],
            "aoi": [
                "A",
                "B",
            ],
        }
    )

    with pytest.raises(ValueError):
        aoi._gp3_scanpath_r_sequences(
            frame,
            aoi_col="aoi",
            group_cols=["subject"],
            time_col="",
            include_missing=False,
            missing_label="missing",
            collapse_repeats=False,
            max_sequences=10,
        )

    with pytest.raises(ValueError):
        aoi._gp3_scanpath_r_sequences(
            frame,
            aoi_col="aoi",
            group_cols=["subject"],
            time_col="time",
            include_missing=False,
            missing_label="missing",
            collapse_repeats=False,
            max_sequences="not-a-number",
        )


def test_h14_network_both_inputs_rejected():
    frame = pd.DataFrame(
        {
            "aoi": [
                "A",
                "B",
            ]
        }
    )

    with pytest.raises(TypeError):
        aoi.compute_gazepoint_transition_network_metrics(
            matrix=pd.DataFrame(
                [
                    [
                        0,
                        1,
                    ],
                    [
                        1,
                        0,
                    ],
                ]
            ),
            data=frame,
            aoi_col="aoi",
        )


def test_h14_network_missing_transition_columns():
    with pytest.raises(ValueError):
        aoi.compute_gazepoint_transition_network_metrics(
            data=pd.DataFrame({"from": ["A"]}),
            from_col="from",
            to_col="to",
        )


def test_h14_sequence_wrapper_aliases(monkeypatch):
    sentinel = pd.DataFrame({"ok": [1]})

    monkeypatch.setattr(
        aoi,
        "prepare_gazepoint_aoi_sequences",
        lambda data, **kwargs: sentinel,
    )

    assert aoi.prepare_gazepoint_semimarkov_data(pd.DataFrame()) is sentinel

    assert aoi.prepare_gazepoint_traminer_data(pd.DataFrame()) is sentinel


def test_h14_anomaly_front_contract_validation():
    with pytest.raises(ValueError):
        aoi.flag_gazepoint_sequence_anomalies(
            [
                "A",
                "B",
            ],
            aoi_col="aoi",
            group_cols=["subject"],
        )

    with pytest.raises(ValueError):
        aoi.flag_gazepoint_sequence_anomalies(
            pd.DataFrame(
                {
                    "subject": ["S1"],
                    "aoi": ["A"],
                }
            ),
            aoi_col="",
            group_cols=["subject"],
        )

    with pytest.raises(ValueError):
        aoi.flag_gazepoint_sequence_anomalies(
            pd.DataFrame(
                {
                    "subject": ["S1"],
                    "aoi": ["A"],
                }
            ),
            aoi_col="aoi",
            group_cols=None,
        )


def test_h14_anomaly_nonzero_length_spread():
    result = aoi.flag_gazepoint_sequence_anomalies(
        pd.DataFrame(
            {
                "subject": [
                    "S1",
                    "S2",
                    "S2",
                    "S2",
                ],
                "aoi": [
                    "A",
                    "A",
                    "B",
                    "C",
                ],
            }
        ),
        aoi_col="aoi",
        group_cols=["subject"],
        min_length=0,
    )

    assert result["length_z"].abs().max() > 0


# ============================================================================
# PUPIL — high-confidence executable residuals
# ============================================================================


def test_h14_blink_missing_required_column():
    with pytest.raises(ValueError):
        pupil.detect_gazepoint_blinks(
            pd.DataFrame(
                {
                    "USER_ID": ["S1"],
                    "pupil": [3.0],
                }
            ),
            pupil_col="pupil",
            id_col="USER_ID",
            time_unit="seconds",
        )


def test_h14_blink_short_event_is_filtered():
    result = pupil.detect_gazepoint_blinks(
        pd.DataFrame(
            {
                "USER_ID": [
                    "S1",
                    "S1",
                    "S1",
                ],
                "TIME": [
                    0.0,
                    0.01,
                    0.02,
                ],
                "pupil": [
                    3.0,
                    np.nan,
                    3.0,
                ],
            }
        ),
        pupil_col="pupil",
        id_col="USER_ID",
        time_unit="seconds",
        min_duration_ms=1000,
        include_rapid_changes=False,
        return_mode="events",
    )

    assert result.empty


def test_h14_smooth_pupil_bad_legacy_method():
    with pytest.raises(ValueError):
        pupil.smooth_gazepoint_pupil(
            pd.DataFrame(
                {
                    "pupil": [
                        1.0,
                        2.0,
                        3.0,
                    ]
                }
            ),
            pupil_col="pupil",
            method="unknown",
            output_col="smoothed",
        )


def test_h14_smooth_pupil_validation():
    frame = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S1",
            ],
            "media_id": [
                "M1",
                "M1",
                "M1",
            ],
            "time": [
                0,
                1,
                2,
            ],
            "pupil": [
                1.0,
                2.0,
                3.0,
            ],
        }
    )

    with pytest.raises(ValueError):
        pupil.smooth_gazepoint_pupil(
            frame,
            pupil_col="pupil",
            time_col="time",
            min_points=0,
        )

    with pytest.raises(ValueError):
        pupil.smooth_gazepoint_pupil(
            frame,
            pupil_col="pupil",
            time_col="time",
            window_samples=2,
            min_points=3,
        )


def test_h14_smooth_coordinate_auto_column():
    result = pupil.smooth_gazepoint_coordinate(
        pd.DataFrame(
            {
                "x": [
                    1.0,
                    2.0,
                    3.0,
                ]
            }
        ),
        method="moving_average",
        window=3,
    )

    assert "x_smoothed" in result

    assert result["x_smoothed"].notna().all()


def test_h14_smooth_coordinate_both_inputs_rejected():
    frame = pd.DataFrame(
        {
            "FPOGX": [0.1],
            "FPOGY": [0.2],
        }
    )

    with pytest.raises(TypeError):
        pupil.smooth_gazepoint_coordinate(
            data=frame,
            all_gaze=frame,
        )


def test_h14_pupil_windows_default_r_definitions():
    result = pupil.summarise_gazepoint_pupil_windows(
        pd.DataFrame(
            {
                "time": [
                    0.0,
                    100.0,
                ],
                "pupil": [
                    3.0,
                    3.1,
                ],
            }
        ),
        pupil_col="pupil",
        time_col="time",
        windows=None,
        group_cols=[],
        include_window_end=True,
    )

    assert not result.empty


def test_h14_response_feature_alias(monkeypatch):
    sentinel = pd.DataFrame({"ok": [1]})

    monkeypatch.setattr(
        pupil,
        "summarise_gazepoint_pupil_trial_features",
        lambda data, **kwargs: sentinel,
    )

    assert pupil.summarize_gazepoint_pupil_response_features(pd.DataFrame()) is sentinel


def test_h14_regression_rejects_two_inputs():
    frame = pd.DataFrame(
        {
            "LPupil": [
                1.0,
                2.0,
                3.0,
            ],
            "RPupil": [
                1.1,
                2.1,
                3.1,
            ],
        }
    )

    with pytest.raises(TypeError):
        pupil.regress_gazepoint_pupils(
            data=frame,
            master_df=frame,
        )


def test_h14_reconstruct_unknown_legacy_method():
    with pytest.raises(ValueError):
        pupil.reconstruct_gazepoint_binocular_pupil(
            pd.DataFrame(
                {
                    "left_pupil": [1.0],
                    "right_pupil": [1.0],
                }
            ),
            method="not-a-method",
        )


def test_h14_reconstruct_finite_gap_requires_time():
    with pytest.raises(
        ValueError,
        match="time_col is required",
    ):
        pupil.reconstruct_gazepoint_binocular_pupil(
            pd.DataFrame(
                {
                    "left_pupil": [
                        1.0,
                        2.0,
                    ],
                    "right_pupil": [
                        1.1,
                        2.1,
                    ],
                }
            ),
            method="none",
            max_gap_ms=100,
        )


def test_h14_audit_reconstruction_skips_absent_reconstruction_eye():
    result = pupil.audit_gazepoint_binocular_reconstruction(
        pd.DataFrame(
            {
                "left_pupil": [
                    1.0,
                    np.nan,
                ],
                "right_pupil": [
                    1.1,
                    1.2,
                ],
                "left_pupil_reconstructed": [
                    1.0,
                    1.1,
                ],
            }
        )
    )

    assert len(result) == 1

    assert (
        result.loc[
            0,
            "eye",
        ]
        == "left"
    )


def test_h14_gp_invalid_length_scale():
    with pytest.raises(ValueError):
        pupil.impute_gazepoint_pupil_gp(
            pd.DataFrame(
                {
                    "time": [
                        0.0,
                        1.0,
                        2.0,
                    ],
                    "pupil": [
                        1.0,
                        np.nan,
                        2.0,
                    ],
                }
            ),
            pupil_col="pupil",
            time_col="time",
            length_scale=0,
        )
