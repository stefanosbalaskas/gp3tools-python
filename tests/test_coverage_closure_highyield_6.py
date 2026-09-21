from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

aoi = importlib.import_module("gp3tools.aoi")
pupil = importlib.import_module("gp3tools.pupil")
qc = importlib.import_module("gp3tools.qc")


# =====================================================================
# AOI TIME-VARYING TRANSITION MATRIX
# =====================================================================


def _transition_data():
    return pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S1",
                "S1",
                "S1",
                "S2",
                "S2",
                "S2",
            ],
            "from_state": [
                "A",
                "A",
                "B",
                "B",
                "A",
                "A",
                "B",
                "B",
            ],
            "to_state": [
                "A",
                "B",
                "A",
                "B",
                "B",
                "B",
                "A",
                "B",
            ],
            "time": [
                0.0,
                50.0,
                120.0,
                160.0,
                240.0,
                0.0,
                120.0,
                240.0,
            ],
            "window": [
                "w1",
                "w1",
                "w2",
                "",
                "w3",
                "w1",
                "w2",
                "w3",
            ],
            "count": [
                1.0,
                2.0,
                1.0,
                3.0,
                1.0,
                1.0,
                2.0,
                1.0,
            ],
        }
    )


def test_h6_transition_matrix_row_global_none():
    data = _transition_data()

    row = aoi.compute_gazepoint_time_varying_transition_matrix(
        data,
        from_col="from_state",
        to_col="to_state",
        time_col="time",
        window_size_ms=100,
        by_cols=["subject"],
        count_col="count",
        states=["A", "B"],
        complete_states=True,
        drop_self_transitions=False,
        normalise="row",
    )

    assert row["_gp3_class"] == ("gp3_time_varying_transition_matrix")

    assert row["overview"].loc[0, "normalise"] == "row"

    assert row["matrix_long"]["transition_probability"].notna().any()

    glob = aoi.compute_gazepoint_time_varying_transition_matrix(
        data,
        from_col="from_state",
        to_col="to_state",
        window_col="window",
        by_cols=["subject"],
        states=["A", "B"],
        complete_states=False,
        drop_self_transitions=True,
        normalise="global",
    )

    assert glob["overview"].loc[0, "normalise"] == "global"

    assert (glob["matrix_long"][".gp3_from"] != glob["matrix_long"][".gp3_to"]).all()

    raw = aoi.compute_gazepoint_time_varying_transition_matrix(
        data,
        from_col="from_state",
        to_col="to_state",
        window_col="window",
        states=["A", "B"],
        complete_states=True,
        normalise="none",
    )

    assert raw["matrix_long"]["transition_probability"].isna().all()

    assert raw["matrix_long"]["transition_denominator"].isna().all()


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "window_size_ms": 0,
        },
        {
            "window_size_ms": np.inf,
        },
        {
            "complete_states": "yes",
        },
        {
            "drop_self_transitions": "yes",
        },
        {
            "normalise": "bad",
        },
    ],
)
def test_h6_transition_matrix_validation(kwargs):
    with pytest.raises(ValueError):
        aoi.compute_gazepoint_time_varying_transition_matrix(
            _transition_data(),
            from_col="from_state",
            to_col="to_state",
            time_col="time",
            **kwargs,
        )


def test_h6_transition_matrix_invalid_data():
    with pytest.raises(ValueError):
        aoi.compute_gazepoint_time_varying_transition_matrix(
            pd.DataFrame(),
            from_col="from_state",
            to_col="to_state",
            time_col="time",
            window_size_ms=100,
        )

    with pytest.raises(ValueError):
        aoi.compute_gazepoint_time_varying_transition_matrix(
            _transition_data(),
            from_col="missing",
            to_col="to_state",
            time_col="time",
            window_size_ms=100,
        )

    bad_count = _transition_data()
    bad_count.loc[0, "count"] = -1

    with pytest.raises(ValueError):
        aoi.compute_gazepoint_time_varying_transition_matrix(
            bad_count,
            from_col="from_state",
            to_col="to_state",
            time_col="time",
            window_size_ms=100,
            count_col="count",
        )

    no_states = _transition_data()
    no_states["from_state"] = pd.NA

    with pytest.raises(ValueError):
        aoi.compute_gazepoint_time_varying_transition_matrix(
            no_states,
            from_col="from_state",
            to_col="to_state",
            time_col="time",
            window_size_ms=100,
        )

    self_only = pd.DataFrame(
        {
            "from_state": [
                "A",
                "A",
            ],
            "to_state": [
                "A",
                "A",
            ],
            "time": [
                0.0,
                1.0,
            ],
        }
    )

    with pytest.raises(ValueError):
        aoi.compute_gazepoint_time_varying_transition_matrix(
            self_only,
            from_col="from_state",
            to_col="to_state",
            time_col="time",
            window_size_ms=100,
            drop_self_transitions=True,
        )


# =====================================================================
# AOI DENOMINATOR AUDIT
# =====================================================================


def _denominator_data():
    return pd.DataFrame(
        {
            "subject": [
                "S1",
                "S2",
                "S3",
                "S4",
                "S5",
                "S6",
                "S7",
                "S8",
                "S9",
                "S10",
            ],
            "condition": [
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
                "A",
                "B",
            ],
            "window_label": [
                "w1",
                "w1",
                "w1",
                "w1",
                "w1",
                "w2",
                "w2",
                "w2",
                "w2",
                "w2",
            ],
            "window_start_ms": [
                0,
                0,
                0,
                0,
                0,
                100,
                100,
                100,
                100,
                100,
            ],
            "window_end_ms": [
                100,
                100,
                100,
                100,
                100,
                200,
                200,
                200,
                200,
                200,
            ],
            "n_window_samples": [
                10,
                np.nan,
                10,
                0,
                10,
                10,
                10,
                10,
                20,
                20,
            ],
            "n_valid_denominator_samples": [
                np.nan,
                5,
                -1,
                2,
                10,
                0,
                2,
                5,
                5,
                20,
            ],
            "n_target_samples": [
                1,
                1,
                0,
                1,
                np.nan,
                0,
                3,
                -1,
                1,
                20,
            ],
        }
    )


def test_h6_denominator_status_matrix():
    data = _denominator_data()

    # Dedicated low-denominator row:
    # denominator=4 < threshold=5,
    # total=4 => valid proportion=1.0,
    # target=1 => no target-exceeds flag.
    data.loc[
        6,
        [
            "n_window_samples",
            "n_valid_denominator_samples",
            "n_target_samples",
        ],
    ] = [4, 4, 1]

    # Dedicated target > denominator row.
    target_exceeds_row = pd.DataFrame(
        {
            "subject": ["S11"],
            "condition": ["A"],
            "window_label": ["w2"],
            "window_start_ms": [100],
            "window_end_ms": [200],
            "n_window_samples": [10],
            "n_valid_denominator_samples": [2],
            "n_target_samples": [3],
        }
    )

    data = pd.concat(
        [data, target_exceeds_row],
        ignore_index=True,
    )

    result = aoi.audit_gazepoint_aoi_window_denominators(
        data,
        group_cols=["subject"],
        min_denominator_samples=5,
        min_valid_denominator_prop=0.7,
        max_denominator_cv=0.1,
        max_condition_ratio=1.1,
    )

    statuses = set(result["row_audit"]["denominator_audit_status"])

    assert "missing_denominator" in statuses
    assert "missing_total" in statuses
    assert "negative_denominator" in statuses
    assert "non_positive_total" in statuses
    assert "missing_target" in statuses
    assert "zero_denominator" in statuses
    assert "low_denominator" in statuses
    assert "target_exceeds_denominator" in statuses
    assert "negative_target" in statuses

    assert not result["window_summary"].empty

    assert not result["condition_window_summary"].empty


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "min_denominator_samples": 0,
        },
        {
            "min_valid_denominator_prop": 2,
        },
        {
            "max_denominator_cv": 0,
        },
        {
            "max_condition_ratio": 0,
        },
    ],
)
def test_h6_denominator_validation(kwargs):
    with pytest.raises(ValueError):
        aoi.audit_gazepoint_aoi_window_denominators(
            _denominator_data(),
            group_cols=["subject"],
            **kwargs,
        )


# =====================================================================
# PUPIL BLINKS
# =====================================================================


def _blink_data():
    return pd.DataFrame(
        {
            "USER_ID": ["S1"] * 10,
            "TIME": (np.arange(10) * 20.0),
            "pupil": [
                3.0,
                3.1,
                np.nan,
                np.nan,
                3.0,
                0.0,
                3.1,
                1.0,
                3.2,
                3.1,
            ],
        }
    )


def test_h6_blink_modes():
    legacy = pupil.detect_gazepoint_blinks(
        _blink_data(),
        pupil_col="pupil",
        time_col="TIME",
        min_duration_ms=0,
        max_duration_ms=1000,
    )

    assert "blink" in legacy

    result = pupil.detect_gazepoint_blinks(
        _blink_data(),
        pupil_col="pupil",
        time_col="TIME",
        id_col="USER_ID",
        z_thresh=1,
        zero_threshold=0,
        merge_gap_ms=30,
        time_unit="milliseconds",
        include_rapid_changes=True,
        min_duration_ms=0,
        max_duration_ms=1000,
        return_mode="both",
    )

    assert "events" in result
    assert "samples" in result

    samples = pupil.detect_gazepoint_blinks(
        _blink_data(),
        pupil_col="pupil",
        time_col="TIME",
        id_col="USER_ID",
        include_rapid_changes=False,
        min_duration_ms=0,
        return_mode="samples",
    )

    assert len(samples) == 10


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
            "z_thresh": -1,
        },
        {
            "merge_gap_ms": -1,
        },
        {
            "zero_threshold": np.inf,
        },
        {
            "include_rapid_changes": "bad",
        },
    ],
)
def test_h6_blink_validation(kwargs):
    with pytest.raises(ValueError):
        pupil.detect_gazepoint_blinks(
            _blink_data(),
            pupil_col="pupil",
            time_col="TIME",
            id_col="USER_ID",
            **kwargs,
        )


def test_h6_blink_keyword_validation():
    with pytest.raises(TypeError):
        pupil.detect_gazepoint_blinks(
            _blink_data(),
            pupil_col="pupil",
            id_col="USER_ID",
            return_mode="events",
            **{
                "return": "samples",
            },
        )

    with pytest.raises(TypeError):
        pupil.detect_gazepoint_blinks(
            _blink_data(),
            pupil_col="pupil",
            impossible=True,
        )


# =====================================================================
# PUPIL SMOOTHING
# =====================================================================


def _smooth_data():
    return pd.DataFrame(
        {
            "subject": [
                "S1",
            ]
            * 7,
            "media_id": [
                "M1",
            ]
            * 7,
            "time": np.arange(
                7,
                dtype=float,
            ),
            "pupil": [
                1.0,
                2.0,
                np.nan,
                4.0,
                10.0,
                6.0,
                7.0,
            ],
        }
    )


def test_h6_smoothing_paths():
    for method in (
        "moving_average",
        "median",
        "savgol",
    ):
        output = f"out_{method}"

        result = pupil.smooth_gazepoint_pupil(
            _smooth_data(),
            pupil_col="pupil",
            method=method,
            window=5,
            output_col=output,
        )

        assert output in result

    for align in (
        "center",
        "left",
        "right",
    ):
        result = pupil.smooth_gazepoint_pupil(
            _smooth_data(),
            pupil_col="pupil",
            time_col="time",
            group_cols=[
                "subject",
                "media_id",
            ],
            method="median",
            align=align,
            window_samples=3,
            min_points=2,
            preserve_missing=False,
        )

        assert "pupil_smoothed" in result


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "method": "bad",
        },
        {
            "align": "bad",
        },
        {
            "preserve_missing": "bad",
        },
        {
            "window_samples": 0,
        },
        {
            "window_samples": 3,
            "min_points": 4,
        },
    ],
)
def test_h6_smoothing_validation(kwargs):
    with pytest.raises(ValueError):
        pupil.smooth_gazepoint_pupil(
            _smooth_data(),
            **kwargs,
        )


# =====================================================================
# PUPIL GAP AUDIT
# =====================================================================


def _gap_data():
    return pd.DataFrame(
        {
            "subject": [
                "S1",
            ]
            * 7,
            "media_id": [
                "M1",
            ]
            * 7,
            "pupil_interpolation_status": [
                "observed",
                "interpolated",
                "missing_edge_gap",
                "missing_long_gap",
                "missing_no_time",
                "missing_insufficient_valid_samples",
                "missing_unfilled",
            ],
            "pupil_gap_id": [
                pd.NA,
                1,
                2,
                3,
                4,
                5,
                6,
            ],
            "pupil_gap_n_samples": [
                np.nan,
                1,
                2,
                3,
                1,
                2,
                1,
            ],
            "pupil_gap_duration_ms": [
                np.nan,
                10,
                20,
                30,
                10,
                20,
                10,
            ],
            "pupil_was_interpolated": [
                False,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
            "pupil_interpolated": [
                3.0,
                3.0,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
            ],
        }
    )


def test_h6_gap_audit_types():
    result = pupil.audit_gazepoint_pupil_gaps(
        _gap_data(),
        group_cols=[
            "subject",
            "media_id",
        ],
    )

    assert (
        result.loc[
            0,
            "n_gaps_total",
        ]
        == 6
    )

    assert (
        result.loc[
            0,
            "n_gaps_interpolated",
        ]
        == 1
    )

    numeric = _gap_data()

    numeric["pupil_was_interpolated"] = [
        0,
        1,
        0,
        0,
        0,
        0,
        0,
    ]

    result_numeric = pupil.audit_gazepoint_pupil_gaps(
        numeric,
        group_cols=["subject"],
    )

    assert (
        result_numeric.loc[
            0,
            "n_gaps_interpolated",
        ]
        == 1
    )

    text = _gap_data()

    text["pupil_was_interpolated"] = [
        "false",
        "yes",
        "false",
        "false",
        "false",
        "false",
        "false",
    ]

    result_text = pupil.audit_gazepoint_pupil_gaps(
        text,
        group_cols=["subject"],
    )

    assert (
        result_text.loc[
            0,
            "n_gaps_interpolated",
        ]
        == 1
    )

    with pytest.raises(ValueError):
        pupil.audit_gazepoint_pupil_gaps(
            _gap_data(),
            group_cols=[
                "subject",
                "subject",
            ],
        )


# =====================================================================
# PUPIL BASELINE AUDIT
# =====================================================================


def _baseline_data():
    return pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S1",
                "S2",
                "S2",
                "S2",
            ],
            "media_id": [
                "M1",
            ]
            * 6,
            "time": [
                -100,
                -50,
                100,
                -100,
                -50,
                100,
            ],
            "pupil_interpolated": [
                3.0,
                np.nan,
                4.0,
                3.0,
                3.1,
                4.0,
            ],
            "pupil_baseline_n": [
                2,
                2,
                2,
                0,
                0,
                0,
            ],
            "pupil_baseline_status": [
                "ok",
                "ok",
                "ok",
                "no_baseline",
                "no_baseline",
                "no_baseline",
            ],
            "pupil_baseline_available": [
                True,
                True,
                True,
                False,
                False,
                False,
            ],
            "pupil_baseline_used": [
                True,
                True,
                False,
                False,
                False,
                False,
            ],
            "pupil_baseline_window_start": [
                -100,
            ]
            * 6,
            "pupil_baseline_window_end": [
                0,
            ]
            * 6,
            "pupil_was_interpolated": [
                False,
                True,
                False,
                False,
                False,
                False,
            ],
            "artifact_flag": [
                False,
                True,
                False,
                False,
                False,
                False,
            ],
        }
    )


def test_h6_baseline_audit_paths():
    result = pupil.audit_gazepoint_pupil_baseline(
        _baseline_data(),
        group_cols=[
            "subject",
            "media_id",
        ],
        baseline_n_col=("pupil_baseline_n"),
        artifact_col=("artifact_flag"),
        min_baseline_samples=2,
        max_missing_pct=20,
        max_interpolated_pct=20,
        max_artifact_pct=20,
    )

    assert len(result) == 2

    assert result["low_quality_baseline_flag"].any()

    with pytest.raises(ValueError):
        pupil.audit_gazepoint_pupil_baseline(
            _baseline_data(),
            group_cols=[
                "subject",
                "subject",
            ],
            baseline_n_col=("pupil_baseline_n"),
        )


# =====================================================================
# QC MISSINGNESS
# =====================================================================


def _qc_data():
    return pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S2",
                "S2",
            ],
            "x": [
                0.0,
                0.5,
                1.2,
                np.nan,
            ],
            "y": [
                0.0,
                0.5,
                0.5,
                np.nan,
            ],
        }
    )


def test_h6_missingness():
    result = qc.summarise_gazepoint_missingness(
        _qc_data(),
        group_cols=["subject"],
        cols=["x", "y"],
        include_group_cols=True,
    )

    assert len(result) == 4

    result2 = qc.summarise_gazepoint_missingness(
        _qc_data(),
        group_cols=["subject"],
        include_group_cols=False,
    )

    assert "subject" not in set(result2["variable"])

    with pytest.raises(TypeError):
        qc.summarise_gazepoint_missingness(
            _qc_data(),
            cols=["x"],
            columns=["x"],
        )

    with pytest.raises(ValueError):
        qc.summarise_gazepoint_missingness(
            _qc_data(),
            cols=[],
        )

    with pytest.raises(ValueError):
        qc.summarise_gazepoint_missingness(
            _qc_data(),
            cols=["missing"],
        )


# =====================================================================
# QC SCREEN BOUNDS
# =====================================================================


def test_h6_screen_bounds():
    result = qc.audit_gazepoint_screen_bounds(
        _qc_data(),
        x_col="x",
        y_col="y",
        group_cols=["subject"],
        width=1,
        height=1,
        margin=0.1,
        treat_zero_zero_as_out_of_bounds=True,
    )

    assert (
        result["overall_summary"].loc[
            0,
            "n_invalid_coordinate",
        ]
        >= 2
    )

    result2 = qc.audit_gazepoint_screen_bounds(
        _qc_data(),
        x_col="x",
        y_col="y",
        group_cols=[],
        width=1,
        height=1,
        margin=0,
        treat_zero_zero_as_out_of_bounds=False,
    )

    assert (
        result2["overall_summary"].loc[
            0,
            "n_zero_zero",
        ]
        == 1
    )

    for kwargs in (
        {
            "width": 0,
        },
        {
            "height": 0,
        },
        {
            "margin": -1,
        },
    ):
        with pytest.raises(ValueError):
            qc.audit_gazepoint_screen_bounds(
                _qc_data(),
                x_col="x",
                y_col="y",
                group_cols=[],
                **kwargs,
            )


# =====================================================================
# QC COORDINATE COVERAGE
# =====================================================================


def test_h6_coordinate_coverage():
    result = qc.summarise_gazepoint_coordinate_coverage(
        _qc_data(),
        x_col="x",
        y_col="y",
        group_cols=["subject"],
        screen_width=1,
        screen_height=1,
        grid_n_x=2,
        grid_n_y=2,
        include_out_of_bounds=True,
    )

    assert len(result) == 2

    outside = qc.summarise_gazepoint_coordinate_coverage(
        pd.DataFrame(
            {
                "x": [
                    2.0,
                    3.0,
                ],
                "y": [
                    2.0,
                    3.0,
                ],
            }
        ),
        x_col="x",
        y_col="y",
        screen_width=1,
        screen_height=1,
        grid_n_x=2,
        grid_n_y=2,
    )

    assert (
        outside.loc[
            0,
            "occupied_grid_cells",
        ]
        == 0
    )

    with pytest.raises(ValueError):
        qc.summarise_gazepoint_coordinate_coverage(
            _qc_data(),
            x_col="x",
            y_col="y",
            screen_width=1,
        )

    with pytest.raises(ValueError):
        qc.summarise_gazepoint_coordinate_coverage(
            _qc_data(),
            x_col="x",
            y_col="y",
            screen_width=-1,
            screen_height=1,
        )

    with pytest.raises(ValueError):
        qc.summarise_gazepoint_coordinate_coverage(
            _qc_data(),
            x_col="x",
            y_col="y",
            screen_width=1,
            screen_height=1,
            grid_n_x=0,
        )


# =====================================================================
# QC DESIGN BALANCE
# =====================================================================


def _design_data():
    return pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S1",
                "S2",
                "S2",
            ],
            "condition": [
                "A",
                "A",
                "B",
                "A",
                "A",
            ],
            "media_id": [
                "M1",
                "M2",
                "M3",
                "M1",
                "M2",
            ],
            "trial_global": [
                1,
                2,
                3,
                1,
                2,
            ],
        }
    )


def test_h6_design_balance():
    result = qc.audit_gazepoint_design_balance(
        _design_data(),
        expected_conditions=[
            "A",
            "B",
        ],
        min_units_per_condition=2,
        max_condition_ratio=1.2,
        require_all_conditions_per_subject=True,
    )

    assert (
        result["overview"].loc[
            0,
            "design_balance_status",
        ]
        == "review"
    )

    statuses = set(result["subject_summary"]["design_balance_status"])

    assert "missing_condition" in statuses

    grouped = qc.audit_gazepoint_design_balance(
        _design_data(),
        group_cols=[
            "subject",
            "condition",
        ],
    )

    assert "n" in grouped

    with pytest.raises(ValueError):
        qc.audit_gazepoint_design_balance(pd.DataFrame())

    with pytest.raises(ValueError):
        qc.audit_gazepoint_design_balance(
            _design_data(),
            min_units_per_condition=0,
        )

    with pytest.raises(ValueError):
        qc.audit_gazepoint_design_balance(
            _design_data(),
            max_condition_ratio=0,
        )

    with pytest.raises(ValueError):
        qc.audit_gazepoint_design_balance(
            _design_data(),
            require_all_conditions_per_subject="bad",
        )

    with pytest.raises(ValueError):
        qc.audit_gazepoint_design_balance(
            _design_data(),
            expected_conditions=[],
        )
