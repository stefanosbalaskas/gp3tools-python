from __future__ import annotations

import importlib
import inspect

import numpy as np
import pandas as pd
import pytest

aoi = importlib.import_module("gp3tools.aoi")
pupil = importlib.import_module("gp3tools.pupil")
qc = importlib.import_module("gp3tools.qc")


# ============================================================================
# AOI — POLYGON HELPERS
# ============================================================================


def _triangle():
    return pd.DataFrame(
        {
            "aoi": [
                "A",
                "A",
                "A",
            ],
            "x": [
                0.0,
                1.0,
                0.0,
            ],
            "y": [
                0.0,
                0.0,
                1.0,
            ],
            "order": [
                1,
                2,
                3,
            ],
        }
    )


def test_h13_polygon_prepare_valid():
    result = aoi._gp3_polygon_r_prepare(
        _triangle(),
        aoi_col="aoi",
        vertex_x_col="x",
        vertex_y_col="y",
        vertex_order_col="order",
    )

    assert len(result) == 1
    assert result[0]["name"] == "A"

    assert result[0]["x"].tolist() == [
        0.0,
        1.0,
        0.0,
    ]


def test_h13_polygon_prepare_missing_column():
    with pytest.raises(ValueError):
        aoi._gp3_polygon_r_prepare(
            _triangle().drop(columns=["x"]),
            aoi_col="aoi",
            vertex_x_col="x",
            vertex_y_col="y",
            vertex_order_col="order",
        )


def test_h13_polygon_prepare_blank_name():
    frame = _triangle()
    frame["aoi"] = ""

    with pytest.raises(ValueError):
        aoi._gp3_polygon_r_prepare(
            frame,
            aoi_col="aoi",
            vertex_x_col="x",
            vertex_y_col="y",
            vertex_order_col="order",
        )


def test_h13_polygon_prepare_nonfinite():
    frame = _triangle()
    frame.loc[
        1,
        "x",
    ] = np.nan

    with pytest.raises(ValueError):
        aoi._gp3_polygon_r_prepare(
            frame,
            aoi_col="aoi",
            vertex_x_col="x",
            vertex_y_col="y",
            vertex_order_col="order",
        )


def test_h13_polygon_prepare_too_few_unique():
    frame = _triangle()

    frame.loc[
        2,
        [
            "x",
            "y",
        ],
    ] = [
        1.0,
        0.0,
    ]

    with pytest.raises(ValueError):
        aoi._gp3_polygon_r_prepare(
            frame,
            aoi_col="aoi",
            vertex_x_col="x",
            vertex_y_col="y",
            vertex_order_col="order",
        )


def test_h13_polygon_all_invalid_points():
    result = aoi._gp3_polygon_r_points_in_polygon(
        np.array(
            [
                np.nan,
                np.inf,
            ]
        ),
        np.array(
            [
                np.nan,
                0.5,
            ]
        ),
        np.array(
            [
                0.0,
                1.0,
                0.0,
            ]
        ),
        np.array(
            [
                0.0,
                0.0,
                1.0,
            ]
        ),
        boundary="inside",
    )

    assert result.tolist() == [
        False,
        False,
    ]


def test_h13_polygon_boundary_inside_outside():
    px = np.array(
        [
            0.0,
            0.25,
            2.0,
        ]
    )

    py = np.array(
        [
            0.0,
            0.25,
            2.0,
        ]
    )

    polygon_x = np.array(
        [
            0.0,
            1.0,
            0.0,
        ]
    )

    polygon_y = np.array(
        [
            0.0,
            0.0,
            1.0,
        ]
    )

    inside = aoi._gp3_polygon_r_points_in_polygon(
        px,
        py,
        polygon_x,
        polygon_y,
        boundary="inside",
    )

    outside = aoi._gp3_polygon_r_points_in_polygon(
        px,
        py,
        polygon_x,
        polygon_y,
        boundary="outside",
    )

    assert inside.tolist() == [
        True,
        True,
        False,
    ]

    assert outside.tolist() == [
        False,
        True,
        False,
    ]


def test_h13_polygon_make_names():
    result = aoi._gp3_polygon_r_make_names(
        [
            "1 target",
            ".2bad",
            "class",
            "hello world",
            "hello world",
        ]
    )

    assert result[0].startswith("X")

    assert result[1].startswith("X")

    assert result[2] == "class."

    assert result[3] == "hello.world"
    assert result[4] == "hello.world.1"


# ============================================================================
# AOI — GEOMETRY ORIGIN/SIZE CONTRACT
# ============================================================================


def test_h13_geometry_origin_size():
    frame = pd.DataFrame(
        {
            "aoi_name": [
                "A",
                "B",
            ],
            "x": [
                0.1,
                0.8,
            ],
            "y": [
                0.1,
                0.2,
            ],
            "width": [
                0.2,
                0.4,
            ],
            "height": [
                0.3,
                0.4,
            ],
        }
    )

    result = aoi.audit_gazepoint_aoi_geometry(
        frame,
        aoi_col="aoi_name",
        x_col="x",
        y_col="y",
        width_col="width",
        height_col="height",
    )

    assert not result["geometry_summary"].empty

    assert (
        result["overview"].loc[
            0,
            "coordinate_format",
        ]
        == "origin_size"
    )

    assert result["_gp3_class"] == "gp3_aoi_geometry_audit"


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "min_width": True,
        },
        {
            "min_height": -1,
        },
        {
            "min_area": np.inf,
        },
        {
            "max_area_prop": -1,
        },
        {
            "max_area_prop": 2,
        },
        {
            "require_within_screen": "yes",
        },
    ],
)
def test_h13_geometry_r_validation(kwargs):
    frame = pd.DataFrame(
        {
            "aoi_name": ["A"],
            "x_min": [0.0],
            "x_max": [1.0],
            "y_min": [0.0],
            "y_max": [1.0],
        }
    )

    with pytest.raises(ValueError):
        aoi.audit_gazepoint_aoi_geometry(
            frame,
            aoi_col="aoi_name",
            **kwargs,
        )


def test_h13_geometry_invalid_screen_ranges():
    frame = pd.DataFrame(
        {
            "aoi_name": ["A"],
            "x_min": [0.0],
            "x_max": [1.0],
            "y_min": [0.0],
            "y_max": [1.0],
        }
    )

    with pytest.raises(ValueError):
        aoi.audit_gazepoint_aoi_geometry(
            frame,
            aoi_col="aoi_name",
            screen_x_range=[
                1,
                0,
            ],
        )


# ============================================================================
# AOI — TIME-VARYING TRANSITION EDGE CONTRACTS
# ============================================================================


def _tv_transition():
    return pd.DataFrame(
        {
            "from_aoi": [
                "A",
                "A",
                "B",
            ],
            "to_aoi": [
                "A",
                "B",
                "A",
            ],
            "time": [
                0.0,
                10.0,
                20.0,
            ],
            "count": [
                1.0,
                2.0,
                1.0,
            ],
        }
    )


def test_h13_time_transition_drop_self():
    result = aoi.compute_gazepoint_time_varying_transition_matrix(
        _tv_transition(),
        window_size_ms=10,
        count_col="count",
        drop_self_transitions=True,
        complete_states=False,
        normalise="global",
    )

    assert not (result["matrix_long"][".gp3_from"] == result["matrix_long"][".gp3_to"]).any()


def test_h13_time_transition_negative_count():
    frame = _tv_transition()

    frame.loc[
        0,
        "count",
    ] = -1

    with pytest.raises(ValueError):
        aoi.compute_gazepoint_time_varying_transition_matrix(
            frame,
            window_size_ms=10,
            count_col="count",
        )


def test_h13_time_transition_all_missing_states():
    frame = _tv_transition()

    frame["from_aoi"] = pd.NA

    with pytest.raises(ValueError):
        aoi.compute_gazepoint_time_varying_transition_matrix(
            frame,
            window_size_ms=10,
        )


def test_h13_time_transition_drop_self_removes_all():
    frame = pd.DataFrame(
        {
            "from_aoi": ["A"],
            "to_aoi": ["A"],
            "time": [0],
        }
    )

    with pytest.raises(ValueError):
        aoi.compute_gazepoint_time_varying_transition_matrix(
            frame,
            window_size_ms=100,
            drop_self_transitions=True,
        )


# ============================================================================
# AOI — SEQUENCE ANOMALY VALIDATION / STATUS
# ============================================================================


def _sequence_anomaly():
    return pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S1",
                "S2",
                "S2",
            ],
            "aoi": [
                "A",
                "B",
                "A",
                "A",
                pd.NA,
            ],
        }
    )


def test_h13_sequence_anomaly_include_missing():
    result = aoi.flag_gazepoint_sequence_anomalies(
        _sequence_anomaly(),
        aoi_col="aoi",
        group_cols=["subject"],
        min_length=1,
        max_missing_prop=0.4,
    )

    assert len(result) == 2

    s2 = result.loc[result["subject"].eq("S2")].iloc[0]

    assert s2["missing_prop"] == pytest.approx(0.5)

    assert bool(s2["flag_high_missing"])

    assert bool(s2["anomaly_flag"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "min_length": True,
        },
        {
            "min_length": -1,
        },
        {
            "max_length": True,
        },
        {
            "max_missing_prop": -1,
        },
        {
            "max_missing_prop": 2,
        },
        {
            "z_threshold": 0,
        },
    ],
)
def test_h13_sequence_anomaly_validation(kwargs):
    with pytest.raises(ValueError):
        aoi.flag_gazepoint_sequence_anomalies(
            _sequence_anomaly(),
            aoi_col="aoi",
            group_cols=["subject"],
            **kwargs,
        )


# ============================================================================
# AOI — SCANPATH CLUSTER BOOTSTRAP
# ============================================================================


def _distance4():
    return np.array(
        [
            [
                0.0,
                1.0,
                2.0,
                3.0,
            ],
            [
                1.0,
                0.0,
                1.0,
                2.0,
            ],
            [
                2.0,
                1.0,
                0.0,
                1.0,
            ],
            [
                3.0,
                2.0,
                1.0,
                0.0,
            ],
        ]
    )


def test_h13_bootstrap_scanpath_clusters():
    result = aoi.bootstrap_gazepoint_scanpath_clusters(
        x=_distance4(),
        k=2,
        n_boot=3,
        sample_fraction=1.0,
        seed=123,
    )

    assert result["_gp3_class"] == "gp3_scanpath_cluster_bootstrap"


def test_h13_bootstrap_more_validation():
    with pytest.raises(ValueError):
        aoi.bootstrap_gazepoint_scanpath_clusters(
            x=_distance4(),
            k=2,
            sample_fraction=1.0,
            n_boot=0,
        )

    bool_boot = aoi.bootstrap_gazepoint_scanpath_clusters(
        x=_distance4(),
        k=2,
        sample_fraction=1.0,
        n_boot=True,
        seed=True,
    )

    assert bool_boot["_gp3_class"] == "gp3_scanpath_cluster_bootstrap"


# ============================================================================
# PUPIL — BLINK RAPID-CHANGE ROUTE
# ============================================================================


def test_h13_blinks_rapid_change():
    frame = pd.DataFrame(
        {
            "USER_ID": ["S1"] * 7,
            "TIME": np.arange(
                7,
                dtype=float,
            )
            / 100,
            "pupil": [
                3.0,
                3.0,
                0.1,
                3.0,
                3.0,
                3.0,
                3.0,
            ],
        }
    )

    result = pupil.detect_gazepoint_blinks(
        frame,
        pupil_col="pupil",
        id_col="USER_ID",
        time_unit="seconds",
        min_duration_ms=0,
        include_rapid_changes=True,
        return_mode="samples",
    )

    assert len(result) == len(frame)


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "time_unit": "bad",
        },
        {
            "return_mode": "bad",
        },
        {
            "min_duration_ms": -1,
        },
        {
            "merge_gap_ms": -1,
        },
    ],
)
def test_h13_blink_validation(kwargs):
    frame = pd.DataFrame(
        {
            "USER_ID": ["S1"],
            "TIME": [0.0],
            "pupil": [3.0],
        }
    )

    with pytest.raises(ValueError):
        pupil.detect_gazepoint_blinks(
            frame,
            pupil_col="pupil",
            id_col="USER_ID",
            **kwargs,
        )


# ============================================================================
# PUPIL — INTERPOLATION
# ============================================================================


def _interp_frame():
    return pd.DataFrame(
        {
            "subject": ["S1"] * 5,
            "time": [
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            "pupil": [
                1.0,
                np.nan,
                np.nan,
                4.0,
                5.0,
            ],
        }
    )


def test_h13_linear_interpolation():
    result = pupil.interpolate_gazepoint_pupil(
        _interp_frame(),
        pupil_col="pupil",
        time_col="time",
        group_cols="subject",
        max_gap_ms=2,
    )

    assert len(result) == 5

    assert "pupil_interpolated" in result.columns


def test_h13_pchip_interpolation():
    result = pupil.interpolate_gazepoint_pupil_pchip(
        _interp_frame(),
        pupil_col="pupil",
        time_col="time",
        group_cols="subject",
        max_gap_ms=2,
    )

    assert len(result) == 5

    assert "pupil_interpolated_pchip" in result.columns


def test_h13_interpolate_blinks():
    frame = _interp_frame()

    frame["blink"] = [
        False,
        True,
        True,
        False,
        False,
    ]

    result = pupil.interpolate_gazepoint_blinks(
        frame,
        pupil_col="pupil",
        blink_col="blink",
        time_col="time",
        output_col="pupil_blink_interp",
    )

    assert len(result) == 5

    assert "pupil_blink_interp" in result.columns


# ============================================================================
# PUPIL — SMOOTHING
# ============================================================================


def _smooth_frame():
    return pd.DataFrame(
        {
            "subject": ["S1"] * 5,
            "time": np.arange(
                5,
                dtype=float,
            ),
            "pupil": [
                1.0,
                2.0,
                np.nan,
                4.0,
                5.0,
            ],
        }
    )


@pytest.mark.parametrize(
    "method",
    [
        "moving_average",
        "median",
    ],
)
def test_h13_smooth_pupil_methods(method):
    result = pupil.smooth_gazepoint_pupil(
        _smooth_frame(),
        pupil_col="pupil",
        group_cols="subject",
        method=method,
        window=3,
    )

    assert len(result) == 5


def test_h13_smooth_coordinate():
    result = pupil.smooth_gazepoint_coordinate(
        pd.DataFrame(
            {
                "x": [
                    1,
                    2,
                    np.nan,
                    4,
                    5,
                ],
            }
        ),
        column="x",
        window=3,
    )

    assert len(result) == 5

    assert "x_smoothed" in result.columns


# ============================================================================
# PUPIL — GAP AUDIT MORE STATUS ROUTES
# ============================================================================


def test_h13_gap_audit_edge_and_long():
    frame = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S1",
                "S1",
            ],
            "pupil_interpolation_status": [
                "missing_edge_gap",
                "missing_long_gap",
                "interpolated",
                "observed",
            ],
            "pupil_gap_id": [
                1,
                2,
                3,
                pd.NA,
            ],
            "pupil_gap_n_samples": [
                1,
                2,
                1,
                np.nan,
            ],
            "pupil_gap_duration_ms": [
                10,
                1000,
                10,
                np.nan,
            ],
            "pupil_was_interpolated": [
                False,
                False,
                True,
                False,
            ],
            "pupil_interpolated": [
                np.nan,
                np.nan,
                3.0,
                3.0,
            ],
        }
    )

    result = pupil.audit_gazepoint_pupil_gaps(
        frame,
        group_cols="subject",
    )

    assert (
        result.loc[
            0,
            "n_missing_edge_gap_samples",
        ]
        == 1
    )

    assert (
        result.loc[
            0,
            "n_missing_long_gap_samples",
        ]
        == 1
    )


# ============================================================================
# PUPIL — RELIABILITY VALIDATION + CONSTANT VALUES
# ============================================================================


def _reliability_frame():
    rows = []

    for participant in (
        "S1",
        "S2",
        "S3",
    ):
        for trial in (
            1,
            2,
            3,
            4,
        ):
            rows.append(
                {
                    "subject": participant,
                    "trial": trial,
                    "pupil": 1.0,
                }
            )

    return pd.DataFrame(rows)


def test_h13_reliability_constant_values():
    result = pupil.audit_gazepoint_pupil_reliability(
        _reliability_frame(),
        participant_col="subject",
        trial_col="trial",
        outcome_cols="pupil",
        min_trials_per_split=1,
    )

    assert "constant_split_values" in set(result["reliability_summary"]["reliability_status"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "split_method": "bad",
        },
        {
            "aggregate_function": "bad",
        },
        {
            "correlation_method": "bad",
        },
        {
            "min_trials_per_split": 0,
        },
        {
            "name": "",
        },
    ],
)
def test_h13_reliability_validation(kwargs):
    with pytest.raises(ValueError):
        pupil.audit_gazepoint_pupil_reliability(
            _reliability_frame(),
            participant_col="subject",
            trial_col="trial",
            outcome_cols="pupil",
            **kwargs,
        )


def test_h13_reliability_split_col_levels():
    frame = _reliability_frame()

    frame["half"] = "only_one"

    with pytest.raises(ValueError):
        pupil.audit_gazepoint_pupil_reliability(
            frame,
            participant_col="subject",
            trial_col="trial",
            outcome_cols="pupil",
            split_col="half",
        )


# ============================================================================
# PUPIL — WINDOWS SPECIAL CASES
# ============================================================================


def _window_frame():
    return pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
            ],
            "time": [
                0.0,
                100.0,
            ],
            "pupil": [
                np.nan,
                np.nan,
            ],
        }
    )


def test_h13_windows_no_valid_pupil():
    result = pupil.summarise_gazepoint_pupil_windows(
        _window_frame(),
        windows=[
            0,
            100,
        ],
        group_cols="subject",
        include_window_end=True,
    )

    assert (
        result.loc[
            0,
            "n_valid_pupil",
        ]
        == 0
    )

    assert (
        result.loc[
            0,
            "n_missing_pupil",
        ]
        == 2
    )


def test_h13_windows_single_valid_auc_unavailable():
    frame = _window_frame()

    frame.loc[
        0,
        "pupil",
    ] = 3.0

    result = pupil.summarise_gazepoint_pupil_windows(
        frame,
        windows=[
            0,
            100,
        ],
        group_cols="subject",
        include_window_end=True,
    )

    assert np.isnan(
        result.loc[
            0,
            "pupil_auc",
        ]
    )


# ============================================================================
# PUPIL — BINOCULAR CALIBRATION / RECONSTRUCTION
# ============================================================================


def _binoc_fit_frame():
    return pd.DataFrame(
        {
            "left_pupil": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
            ],
            "right_pupil": [
                2.0,
                4.0,
                6.0,
                8.0,
                10.0,
            ],
        }
    )


def test_h13_binoc_fit_eligible_without_r2_threshold():
    result = pupil._gp3_binoc_r_fit_one(
        _binoc_fit_frame()["right_pupil"],
        _binoc_fit_frame()["left_pupil"],
        2,
        2,
        None,
        True,
        None,
    )

    assert result["eligible"]


def test_h13_reconstruct_none_r_mode():
    frame = pd.DataFrame(
        {
            "time": [
                0.0,
                1.0,
            ],
            "left_pupil": [
                1.0,
                np.nan,
            ],
            "right_pupil": [
                1.1,
                2.0,
            ],
        }
    )

    result = pupil.reconstruct_gazepoint_binocular_pupil(
        frame,
        method="none",
        time_col="time",
    )

    assert len(result) == 2

    assert (
        result.loc[
            1,
            "gp3_binocular_status",
        ]
        == "right_only_observed"
    )


def test_h13_reconstruct_available_eye_r_mode():
    frame = pd.DataFrame(
        {
            "time": [
                0.0,
                1.0,
            ],
            "left_pupil": [
                1.0,
                np.nan,
            ],
            "right_pupil": [
                np.nan,
                2.0,
            ],
        }
    )

    result = pupil.reconstruct_gazepoint_binocular_pupil(
        frame,
        method="available_eye",
        time_col="time",
    )

    assert len(result) == 2


# ============================================================================
# PUPIL — VALIDATION CONTIGUOUS
# ============================================================================


def test_h13_validate_contiguous():
    frame = pd.DataFrame(
        {
            "time": np.arange(
                12,
                dtype=float,
            ),
            "left_pupil": np.linspace(
                1,
                2,
                12,
            ),
            "right_pupil": np.linspace(
                1.1,
                2.1,
                12,
            ),
        }
    )

    result = pupil.validate_gazepoint_binocular_reconstruction(
        frame,
        time_col="time",
        direction="both",
        mask_mode="contiguous",
        mask_prop=0.25,
        block_size=2,
        repeats=1,
        seed=123,
        min_pairs=2,
        min_unique=2,
    )

    assert not result["metrics"].empty


# ============================================================================
# QC — VALIDATE MASTER ALL-UNKNOWN QUALITY
# ============================================================================


def test_h13_validate_master_all_unknown_quality():
    frame = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S2",
            ],
            "time": [
                np.nan,
                np.nan,
            ],
            "x": [
                np.nan,
                np.nan,
            ],
            "y": [
                np.nan,
                np.nan,
            ],
            "valid_sample": pd.Series(
                [
                    pd.NA,
                    pd.NA,
                ],
                dtype="boolean",
            ),
            "missing_gaze": pd.Series(
                [
                    pd.NA,
                    pd.NA,
                ],
                dtype="boolean",
            ),
        }
    )

    result = qc.validate_gazepoint_master(
        frame,
        min_valid_sample_pct=75,
    )

    assert "summary" in result


# ============================================================================
# QC — POST-EXCLUSION WITHOUT CONDITION
# ============================================================================


def test_h13_post_exclusion_without_condition():
    frame = pd.DataFrame(
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
    )

    with pytest.raises(ValueError):
        qc.audit_gazepoint_post_exclusion_balance(
            frame,
            subject_col="subject",
            condition_col=None,
            include_col="include",
            unit_cols=[],
        )


# ============================================================================
# QC — EXCLUSION UNIT FALLBACK REASONS
# ============================================================================


def test_h13_exclusion_unit_default_reason():
    frame = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S2",
            ]
        }
    )

    flags = pd.Series(
        pd.array(
            [
                True,
                False,
            ],
            dtype="boolean",
        )
    )

    reasons = pd.Series(
        [
            pd.NA,
            "",
        ],
        dtype="string",
    )

    result = qc._gp3_exclusion_r_units(
        frame,
        flags,
        "subject",
        None,
        [],
        reason=reasons,
    )

    assert set(result["exclusion_reason"]) == {
        "retained",
        "excluded_unspecified",
    }


# ============================================================================
# QC — NAMING WRITER LEGACY
# ============================================================================


def test_h13_naming_writer_python_interface(tmp_path):
    out = qc.write_gazepoint_naming_audit(
        path=(tmp_path / "naming.csv"),
        names=[
            "A",
            "A",
            "B",
        ],
    )

    assert out.exists()


def test_h13_naming_writer_interface_validation(tmp_path):
    pairs = pd.DataFrame(
        {
            "stem": ["x"],
            "status": ["paired"],
        }
    )

    with pytest.raises(ValueError):
        qc.write_gazepoint_naming_audit(
            x={"pairs": pairs},
        )

    with pytest.raises(TypeError):
        qc.write_gazepoint_naming_audit(
            path=(tmp_path / "x.csv"),
            x={"pairs": pairs},
            output_file=(tmp_path / "y.csv"),
        )

    with pytest.raises(TypeError):
        qc.write_gazepoint_naming_audit(
            output_file=(tmp_path / "x.csv"),
            x={},
        )


# ============================================================================
# QC — RESIDUAL SOURCE GUARD ACCOUNTING
# ============================================================================


def test_h13_residual_qc_helpers_have_expected_contracts():
    # These assertions intentionally document the signatures before the
    # final invariant/dead-branch audit. They prevent accidental API drift.
    assert "expected_hz" in inspect.signature(qc.check_sampling_rate).parameters

    assert "output_file" in inspect.signature(qc.write_gazepoint_naming_audit).parameters
