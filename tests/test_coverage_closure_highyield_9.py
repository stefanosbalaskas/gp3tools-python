from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

aoi = importlib.import_module("gp3tools.aoi")
pupil = importlib.import_module("gp3tools.pupil")
qc = importlib.import_module("gp3tools.qc")


# ============================================================================
# AOI — MARGIN SENSITIVITY
# ============================================================================


def _margin_geometry():
    return pd.DataFrame(
        {
            "AOI": ["A", "B"],
            "x_min": [0.0, 0.6],
            "y_min": [0.0, 0.0],
            "x_max": [0.4, 1.0],
            "y_max": [1.0, 1.0],
        }
    )


def _margin_gaze():
    return pd.DataFrame(
        {
            "sample": [1, 2, 3],
            "x": [0.2, 0.5, np.nan],
            "y": [0.5, 0.5, 0.5],
        }
    )


def _margin_kwargs():
    return {
        "gaze_x_col": "x",
        "gaze_y_col": "y",
        "sample_id_cols": ["sample"],
        "geometry_aoi_col": "AOI",
        "x_min_col": "x_min",
        "y_min_col": "y_min",
        "x_max_col": "x_max",
        "y_max_col": "y_max",
    }


def test_h9_margin_sensitive_statuses():
    ambiguous = aoi.audit_gazepoint_aoi_margin_sensitivity(
        _margin_gaze(),
        _margin_geometry(),
        margins=[0.11],
        max_margin_change_prop=1.0,
        max_ambiguous_prop=0.0,
        **_margin_kwargs(),
    )

    statuses = set(ambiguous["margin_summary"]["margin_sensitivity_status"])

    assert "ambiguous_margin" in statuses

    sample_statuses = set(ambiguous["sample_sensitivity"]["margin_assignment_status"])

    assert "missing_coordinate" in sample_statuses
    assert "ambiguous_aoi" in sample_statuses
    assert "single_aoi" in sample_statuses

    stable = aoi.audit_gazepoint_aoi_margin_sensitivity(
        _margin_gaze(),
        _margin_geometry(),
        margins=[0.01],
        max_margin_change_prop=1.0,
        max_ambiguous_prop=1.0,
        **_margin_kwargs(),
    )

    assert "ok" in set(stable["margin_summary"]["margin_sensitivity_status"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "margins": [np.inf],
        },
        {
            "max_margin_change_prop": -1,
        },
        {
            "max_ambiguous_prop": 2,
        },
        {
            "tie_method": "bad",
        },
        {
            "ignore_invalid_geometry": "yes",
        },
    ],
)
def test_h9_margin_validation(kwargs):
    call = _margin_kwargs()
    call.update(kwargs)

    with pytest.raises(ValueError):
        aoi.audit_gazepoint_aoi_margin_sensitivity(
            _margin_gaze(),
            _margin_geometry(),
            **call,
        )


def test_h9_margin_empty_validation():
    with pytest.raises(ValueError):
        aoi.audit_gazepoint_aoi_margin_sensitivity(
            _margin_gaze().iloc[0:0],
            _margin_geometry(),
            **_margin_kwargs(),
        )

    with pytest.raises(ValueError):
        aoi.audit_gazepoint_aoi_margin_sensitivity(
            _margin_gaze(),
            _margin_geometry().iloc[0:0],
            **_margin_kwargs(),
        )


# ============================================================================
# AOI — INTERNAL ENTRY -> SEQUENCE CONTRACT
# ============================================================================


def _entry_frame(with_subject=True):
    data = {
        "aoi_state": ["A", "outside", "B"],
        "entry_order": [1, 2, 3],
        "entry_start_time": [0.0, 10.0, 20.0],
        "entry_end_time": [10.0, 20.0, 30.0],
        "entry_duration_ms": [10.0, 10.0, 10.0],
        "n_samples": [2, 2, 2],
    }

    if with_subject:
        data["subject"] = ["S1", "S1", "S1"]

    return pd.DataFrame(data)


def test_h9_r_sequences_entry_id_branches():
    ungrouped = aoi._gp3_aoi_r_sequences(
        _entry_frame(with_subject=False),
        group_cols=[],
        non_aoi_values=["outside"],
    )

    assert ungrouped["entry_id"].tolist() == [
        1,
        2,
        3,
    ]

    assert "is_non_aoi" in ungrouped

    grouped = aoi._gp3_aoi_r_sequences(
        _entry_frame(),
        group_cols=["subject"],
        non_aoi_values=["outside"],
    )

    assert grouped["entry_id"].tolist() == [
        1,
        2,
        3,
    ]


def test_h9_r_sequences_validation():
    with pytest.raises(ValueError):
        aoi._gp3_aoi_r_sequences(
            _entry_frame(with_subject=False),
            group_cols=["subject"],
        )

    empty = _entry_frame(with_subject=False).iloc[0:0]

    with pytest.raises(ValueError):
        aoi._gp3_aoi_r_sequences(
            empty,
            group_cols=[],
        )

    background = _entry_frame(with_subject=False)

    background["aoi_state"] = "outside"

    with pytest.raises(ValueError):
        aoi._gp3_aoi_r_sequences(
            background,
            group_cols=[],
            include_non_aoi=False,
            non_aoi_values=["outside"],
        )


# ============================================================================
# AOI — SEQUENCE DISTANCE
# ============================================================================


def test_h9_sequence_distance_legacy_and_r_contract():
    assert aoi.compute_gazepoint_sequence_distance(
        ["A", "B"],
        ["A", "C"],
        method="jaccard",
    ) == pytest.approx(2 / 3)

    with pytest.raises(ValueError):
        aoi.compute_gazepoint_sequence_distance(
            ["A"],
            ["B"],
            method="bad",
        )

    result = aoi.compute_gazepoint_sequence_distance(
        [
            "A",
            None,
            "A",
            "A",
        ],
        [
            "A",
            "",
            "B",
        ],
        ignore_missing=False,
        collapse_repeats=True,
        missing_label="<M>",
        substitution_cost=2,
        insertion_cost=1,
        deletion_cost=1,
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )

    assert len(result) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "ignore_missing": "yes",
        },
        {
            "ignore_missing": True,
            "collapse_repeats": "yes",
        },
        {
            "ignore_missing": True,
            "missing_label": "",
        },
        {
            "ignore_missing": True,
            "substitution_cost": -1,
        },
        {
            "ignore_missing": True,
            "insertion_cost": np.inf,
        },
    ],
)
def test_h9_sequence_distance_validation(kwargs):
    with pytest.raises(ValueError):
        aoi.compute_gazepoint_sequence_distance(
            ["A"],
            ["B"],
            **kwargs,
        )


# ============================================================================
# AOI — REPRESENTATIVE SCANPATHS
# ============================================================================


def test_h9_representative_scanpath_legacy():
    frame = pd.DataFrame(
        {
            "cluster": [
                1,
                1,
                2,
            ],
            "sequence": [
                ["A", "B"],
                ["A", "C"],
                ["D"],
            ],
        }
    )

    result = aoi.extract_gazepoint_representative_scanpaths(frame)

    assert len(result) == 2
    assert set(result["cluster"]) == {
        1,
        2,
    }


def _distance_frame():
    return pd.DataFrame(
        [
            [0.0, 1.0],
            [1.0, 0.0],
        ],
        index=[
            "s1",
            "s2",
        ],
        columns=[
            "s1",
            "s2",
        ],
    )


def test_h9_representative_scanpath_validation():
    valid_assignments = pd.DataFrame(
        {
            "sequence_id": [
                "s1",
                "s2",
            ],
            "cluster": [
                1,
                1,
            ],
        }
    )

    with pytest.raises(ValueError):
        aoi.extract_gazepoint_representative_scanpaths(
            {
                "distance": _distance_frame(),
                "assignments": valid_assignments,
            },
            n_per_cluster=0,
        )

    with pytest.raises(ValueError):
        aoi.extract_gazepoint_representative_scanpaths({})

    with pytest.raises(ValueError):
        aoi.extract_gazepoint_representative_scanpaths(
            {
                "distance": pd.DataFrame(
                    np.zeros(
                        (
                            2,
                            3,
                        )
                    )
                ),
                "assignments": valid_assignments,
            }
        )

    with pytest.raises(ValueError):
        aoi.extract_gazepoint_representative_scanpaths(
            {
                "distance": _distance_frame(),
                "assignments": pd.DataFrame(
                    {
                        "sequence_id": [
                            "s1",
                            "s2",
                        ]
                    }
                ),
            }
        )

    duplicates = pd.DataFrame(
        {
            "sequence_id": [
                "s1",
                "s1",
            ],
            "cluster": [
                1,
                1,
            ],
        }
    )

    with pytest.raises(ValueError):
        aoi.extract_gazepoint_representative_scanpaths(
            {
                "distance": _distance_frame(),
                "assignments": duplicates,
            }
        )

    missing = pd.DataFrame(
        {
            "sequence_id": [
                "s1",
            ],
            "cluster": [
                1,
            ],
        }
    )

    with pytest.raises(ValueError):
        aoi.extract_gazepoint_representative_scanpaths(
            {
                "distance": _distance_frame(),
                "assignments": missing,
            }
        )


# ============================================================================
# PUPIL — BASIC BINOCULAR CONTRACTS
# ============================================================================


def _eye_frame():
    return pd.DataFrame(
        {
            "left_pupil": [
                0.5,
                2.0,
                np.nan,
                4.0,
            ],
            "right_pupil": [
                3.0,
                10.0,
                5.0,
                np.nan,
            ],
        }
    )


def test_h9_mean_pupil_min_eye_validation():
    with pytest.raises(ValueError):
        pupil.mean_gazepoint_pupil(
            _eye_frame(),
            min_eyes=3,
        )


def test_h9_combine_eye_bounds_and_methods():
    mean = pupil.combine_gazepoint_eyes(
        _eye_frame(),
        method="mean",
        valid_min=1,
        valid_max=9,
    )

    assert (
        pd.isna(
            mean.loc[
                0,
                "pupil_combined",
            ]
        )
        is False
    )

    assert mean.loc[
        1,
        "pupil_combined",
    ] == pytest.approx(2.0)

    best_left = pupil.combine_gazepoint_eyes(
        pd.DataFrame(
            {
                "left_pupil": [
                    1.0,
                    2.0,
                    3.0,
                ],
                "right_pupil": [
                    1.0,
                    np.nan,
                    3.0,
                ],
            }
        ),
        method="best",
    )

    assert best_left.loc[
        1,
        "pupil_combined",
    ] == pytest.approx(2.0)

    best_right = pupil.combine_gazepoint_eyes(
        pd.DataFrame(
            {
                "left_pupil": [
                    np.nan,
                    2.0,
                    np.nan,
                ],
                "right_pupil": [
                    1.0,
                    2.0,
                    3.0,
                ],
            }
        ),
        method="best",
    )

    assert best_right.loc[
        0,
        "pupil_combined",
    ] == pytest.approx(1.0)


# ============================================================================
# PUPIL — FLAGGING
# ============================================================================


def _flag_frame(values):
    return pd.DataFrame(
        {
            "subject": ["S1"] * len(values),
            "media_id": ["M1"] * len(values),
            "time": np.arange(
                len(values),
                dtype=float,
            ),
            "pupil": values,
        }
    )


def test_h9_flag_pupil_small_and_zero_iqr_groups():
    small = pupil.flag_gazepoint_pupil(
        _flag_frame(
            [
                3.0,
                3.0,
                3.0,
            ]
        ),
        pupil_col="pupil",
        time_col="time",
        group_cols="subject",
    )

    assert not small["pupil_flag_iqr_outlier"].any()

    zero_iqr = pupil.flag_gazepoint_pupil(
        _flag_frame(
            [
                3.0,
                3.0,
                3.0,
                3.0,
            ]
        ),
        pupil_col="pupil",
        time_col="time",
        group_cols="subject",
    )

    assert not zero_iqr["pupil_flag_iqr_outlier"].any()


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "physiological_min": 5,
            "physiological_max": 4,
            "group_cols": "subject",
        },
        {
            "outlier_k": -1,
            "group_cols": "subject",
        },
        {
            "flag_iqr_outliers": "yes",
            "group_cols": "subject",
        },
        {
            "missing_pupil_col": "missing",
            "group_cols": "subject",
        },
    ],
)
def test_h9_flag_pupil_validation(kwargs):
    with pytest.raises(
        (
            ValueError,
            KeyError,
        )
    ):
        pupil.flag_gazepoint_pupil(
            _flag_frame(
                [
                    3,
                    3,
                    3,
                    3,
                ]
            ),
            pupil_col="pupil",
            time_col="time",
            **kwargs,
        )


# ============================================================================
# PUPIL — BASELINE CORRECTION
# ============================================================================


def _baseline_correct_frame():
    return pd.DataFrame(
        {
            "subject": ["S1"] * 4,
            "media_id": ["M1"] * 4,
            "custom": ["G1"] * 4,
            "time": [
                -100.0,
                -50.0,
                100.0,
                200.0,
            ],
            "pupil": [
                3.0,
                3.2,
                4.0,
                4.2,
            ],
            "is_baseline": [
                True,
                True,
                False,
                False,
            ],
        }
    )


def test_h9_baseline_correct_custom_group():
    result = pupil.baseline_correct_gazepoint_pupil(
        _baseline_correct_frame(),
        pupil_col="pupil",
        time_col="time",
        baseline_time_col="time",
        baseline_flag_col="is_baseline",
        baseline_window=None,
        group_cols="custom",
    )

    assert len(result) == 4


def test_h9_baseline_correct_validation():
    frame = _baseline_correct_frame()

    with pytest.raises(ValueError):
        pupil.baseline_correct_gazepoint_pupil(
            frame,
            pupil_col="pupil",
            time_col="time",
            baseline_flag_col="is_baseline",
            baseline_window=[0],
        )

    with pytest.raises(ValueError):
        pupil.baseline_correct_gazepoint_pupil(
            frame.drop(columns=["time"]),
            pupil_col="pupil",
            group_cols=["subject"],
        )

    with pytest.raises(ValueError):
        pupil.baseline_correct_gazepoint_pupil(
            frame,
            pupil_col="pupil",
            time_col="time",
            baseline_time_col="missing",
            group_cols=["subject"],
        )

    with pytest.raises(ValueError):
        pupil.baseline_correct_gazepoint_pupil(
            frame.drop(columns=["media_id"]),
            pupil_col="pupil",
            time_col="time",
            group_cols=["media_id"],
        )


# ============================================================================
# PUPIL — GAP AUDIT
# ============================================================================


def test_h9_gap_legacy_single_group():
    frame = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S1",
            ],
            "time": [
                0,
                1,
                2,
            ],
            "pupil": [
                3.0,
                np.nan,
                3.1,
            ],
        }
    )

    result = pupil.audit_gazepoint_pupil_gaps(
        frame,
        pupil_col="pupil",
        group_cols="subject",
        time_col="time",
    )

    assert len(result) == 1
    assert (
        result.loc[
            0,
            "n_gaps",
        ]
        == 1
    )


def _gap_r_frame():
    return pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
            ],
            "pupil_interpolation_status": [
                "observed",
                "interpolated",
            ],
            "pupil_gap_id": [
                pd.NA,
                1,
            ],
            "pupil_gap_n_samples": [
                np.nan,
                1,
            ],
            "pupil_gap_duration_ms": [
                np.nan,
                20,
            ],
            "pupil_was_interpolated": [
                False,
                True,
            ],
            "pupil_interpolated": [
                3.0,
                3.1,
            ],
        }
    )


def test_h9_gap_r_single_group():
    result = pupil.audit_gazepoint_pupil_gaps(
        _gap_r_frame(),
        group_cols="subject",
    )

    assert len(result) == 1
    assert (
        result.loc[
            0,
            "n_gaps_interpolated",
        ]
        == 1
    )


def test_h9_gap_validation():
    with pytest.raises(ValueError):
        pupil.audit_gazepoint_pupil_gaps(
            _gap_r_frame(),
            group_cols="subject",
            gap_id_col="",
        )

    with pytest.raises(ValueError):
        pupil.audit_gazepoint_pupil_gaps(
            _gap_r_frame().drop(columns=["pupil_gap_id"]),
            group_cols="subject",
        )


# ============================================================================
# PUPIL — BASELINE AUDIT
# ============================================================================


def test_h9_baseline_audit_legacy_group():
    frame = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S2",
                "S2",
            ],
            "time": [
                -100,
                -50,
                -100,
                -50,
            ],
            "pupil": [
                3.0,
                3.1,
                4.0,
                4.1,
            ],
        }
    )

    result = pupil.audit_gazepoint_pupil_baseline(
        frame,
        pupil_col="pupil",
        time_col="time",
        group_cols="subject",
    )

    assert len(result) == 2


def _baseline_audit_r():
    return pd.DataFrame(
        {
            "subject": ["S1", "S1"],
            "time": [-100, -50],
            "pupil_interpolated": [3.0, 3.1],
            "pupil_baseline_n": [2, 2],
            "pupil_baseline_status": ["ok", "ok"],
            "pupil_baseline_available": [True, True],
            "pupil_baseline_used": [True, True],
            "pupil_baseline_window_start": [-200, -200],
            "pupil_baseline_window_end": [0, 0],
            "pupil_was_interpolated": [False, False],
        }
    )


def test_h9_baseline_audit_no_artifact_and_reason():
    plain = pupil.audit_gazepoint_pupil_baseline(
        _baseline_audit_r(),
        group_cols=["subject"],
        baseline_n_col="pupil_baseline_n",
    )

    assert (
        plain.loc[
            0,
            "baseline_quality_reason",
        ]
        == "ok"
    )

    reason_frame = _baseline_audit_r()
    reason_frame["artifact_reason"] = [
        "valid",
        "blink",
    ]

    reason = pupil.audit_gazepoint_pupil_baseline(
        reason_frame,
        group_cols=["subject"],
        baseline_n_col="pupil_baseline_n",
        artifact_reason_col="artifact_reason",
        max_artifact_pct=10,
    )

    assert (
        reason.loc[
            0,
            "baseline_quality_reason",
        ]
        == "high_baseline_artifact_pct"
    )


def test_h9_baseline_audit_missing_required():
    with pytest.raises(KeyError):
        pupil.audit_gazepoint_pupil_baseline(
            pd.DataFrame(
                {
                    "subject": ["S1"],
                    "pupil_baseline_n": [1],
                }
            ),
            group_cols=["subject"],
            baseline_n_col="pupil_baseline_n",
        )


# ============================================================================
# PUPIL — DRIFT
# ============================================================================


def test_h9_drift_legacy_single_group():
    frame = pd.DataFrame(
        {
            "subject": ["S1"] * 3,
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

    result = pupil.audit_gazepoint_pupil_drift(
        frame,
        pupil_col="pupil",
        time_col="time",
        group_cols="subject",
    )

    assert len(result) == 1


def _drift_r_frame():
    return pd.DataFrame(
        {
            "subject": ["S1"] * 4,
            "condition": [
                "A",
                "A",
                "B",
                "B",
            ],
            "trial": [
                1,
                2,
                3,
                4,
            ],
            "time": [
                0,
                1000,
                2000,
                3000,
            ],
            "pupil": [
                3.0,
                3.1,
                3.2,
                3.3,
            ],
        }
    )


def test_h9_drift_r_string_group():
    result = pupil.audit_gazepoint_pupil_drift(
        _drift_r_frame(),
        pupil_col="pupil",
        time_col="time",
        group_cols="subject",
        order_col="trial",
        condition_col="condition",
    )

    assert not result["by_group"].empty


def test_h9_drift_auto_pupil_missing():
    frame = _drift_r_frame().drop(columns=["pupil"])

    with pytest.raises(KeyError):
        pupil.audit_gazepoint_pupil_drift(
            frame,
            group_cols="subject",
            order_col="trial",
            condition_col="condition",
        )


# ============================================================================
# PUPIL — IMBALANCE
# ============================================================================


def _imbalance_frame():
    return pd.DataFrame(
        {
            "condition": [
                "A",
                "A",
                "B",
            ],
            "pupil_interpolated": [
                3.0,
                3.1,
                np.nan,
            ],
            "pupil_was_interpolated": [
                False,
                False,
                True,
            ],
            "pupil_interpolation_status": [
                "observed",
                "observed",
                "missing_long_gap",
            ],
        }
    )


def test_h9_imbalance_no_artifact_branch():
    result = pupil.audit_gazepoint_pupil_imbalance(
        _imbalance_frame(),
        group_cols="condition",
        min_group_n=2,
        max_valid_pct_diff=0,
        max_artifact_pct_diff=0,
        max_missing_pct_diff=0,
        max_interpolated_pct_diff=0,
    )

    assert result["preprocessing_imbalance_warning"].all()

    reasons = result.loc[
        0,
        "preprocessing_imbalance_reason",
    ]

    assert "interpolated_pct_diff" in reasons
    assert "small_group_n" in reasons


def test_h9_imbalance_artifact_reason_branch():
    frame = _imbalance_frame()

    frame["artifact_reason"] = [
        "valid",
        "blink",
        "valid",
    ]

    result = pupil.audit_gazepoint_pupil_imbalance(
        frame,
        group_cols="condition",
        artifact_reason_col="artifact_reason",
        max_artifact_pct_diff=0,
    )

    assert result["artifact_sample_pct_range"].iloc[0] > 0


def test_h9_imbalance_validation():
    with pytest.raises(ValueError):
        pupil.audit_gazepoint_pupil_imbalance(
            _imbalance_frame(),
            group_cols=[
                "condition",
                "condition",
            ],
        )

    with pytest.raises(KeyError):
        pupil.audit_gazepoint_pupil_imbalance(
            _imbalance_frame().drop(columns=["pupil_was_interpolated"]),
            group_cols="condition",
        )


# ============================================================================
# QC — CREATE MASTER INTERNAL FALLBACKS
# ============================================================================


def test_h9_create_master_source_subject_and_trial(monkeypatch):
    source = pd.DataFrame(
        {
            "USER_FILE": ["participant12.csv"],
            "MEDIA_ID": ["M1"],
        }
    )

    monkeypatch.setattr(
        qc,
        "as_gazepoint_master",
        lambda data: source.copy(),
    )

    result = qc.create_gazepoint_master(source)

    assert (
        result.loc[
            0,
            "subject",
        ]
        == "12"
    )

    assert (
        result.loc[
            0,
            "trial_global",
        ]
        == "12::M1"
    )

    assert (
        result.loc[
            0,
            "sample_index",
        ]
        == 0
    )


def test_h9_create_master_fallback_trial(monkeypatch):
    source = pd.DataFrame(
        {
            "value": [1],
        }
    )

    monkeypatch.setattr(
        qc,
        "as_gazepoint_master",
        lambda data: source.copy(),
    )

    result = qc.create_gazepoint_master(source)

    assert (
        result.loc[
            0,
            "trial_global",
        ]
        == 1
    )


# ============================================================================
# QC — SAMPLING / TRACKING
# ============================================================================


def test_h9_sampling_rate_single_group():
    result = qc.check_sampling_rate(
        pd.DataFrame(
            {
                "subject": [
                    "S1",
                    "S1",
                    "S1",
                ],
                "time": [
                    0.0,
                    1 / 60,
                    2 / 60,
                ],
            }
        ),
        time_col="time",
        group_cols="subject",
    )

    assert len(result) == 1

    assert result.loc[
        0,
        "sampling_hz",
    ] == pytest.approx(
        60,
        rel=1e-6,
    )


def test_h9_flag_tracking_raw_default_branch():
    result = qc.flag_tracking_quality(
        pd.DataFrame(
            {
                "x": [
                    0.1,
                    np.nan,
                ],
                "y": [
                    0.1,
                    0.2,
                ],
            }
        )
    )

    assert "quality_flag" in result


def test_h9_gaze_signal_quality_legacy():
    result = qc.audit_gazepoint_gaze_signal_quality(
        pd.DataFrame(
            {
                "x": [
                    0.1,
                    np.nan,
                ],
                "y": [
                    0.1,
                    0.2,
                ],
            }
        )
    )

    assert {
        "tracking",
        "missingness",
    }.issubset(result)


# ============================================================================
# QC — DESIGN BALANCE VALIDATION
# ============================================================================


def test_h9_design_balance_group_fallback():
    result = qc.audit_gazepoint_design_balance(
        pd.DataFrame(
            {
                "x": [
                    1,
                    2,
                ],
            }
        ),
        group_cols=["missing"],
    )

    assert (
        result.loc[
            0,
            "n_rows",
        ]
        == 2
    )


def test_h9_design_balance_empty_conditions():
    with pytest.raises(ValueError):
        qc.audit_gazepoint_design_balance(
            pd.DataFrame(
                {
                    "subject": ["S1"],
                    "condition": [pd.NA],
                }
            ),
            unit_cols=None,
        )


def test_h9_design_balance_no_usable_rows():
    with pytest.raises(ValueError):
        qc.audit_gazepoint_design_balance(
            pd.DataFrame(
                {
                    "subject": [""],
                    "condition": ["A"],
                }
            ),
            unit_cols=None,
        )


# ============================================================================
# QC — STATUS / MESSAGE REDUCERS
# ============================================================================


def test_h9_qc_status_reducer_matrix():
    assert (
        qc._gp3_qc_status_from_overview(
            pd.DataFrame({"x": [1]}),
            [],
        )
        == "unknown"
    )

    assert (
        qc._gp3_qc_status_from_overview(
            pd.DataFrame({"review_flag": [True]}),
            ["review_flag"],
        )
        == "warn"
    )

    assert (
        qc._gp3_qc_status_from_overview(
            pd.DataFrame({"ready": [False]}),
            ["ready"],
        )
        == "fail"
    )

    for value, expected in (
        (
            "failed",
            "fail",
        ),
        (
            "review",
            "warn",
        ),
        (
            "missing",
            "info",
        ),
        (
            "ok",
            "pass",
        ),
        (
            "mystery",
            "info",
        ),
    ):
        assert (
            qc._gp3_qc_status_from_overview(
                pd.DataFrame({"status": [value]}),
                ["status"],
            )
            == expected
        )


def test_h9_qc_message_reducer():
    overview = pd.DataFrame(
        {
            "reason": [
                pd.NA,
                "",
                "first",
                "first",
                "second",
                "third",
                "fourth",
            ]
        }
    )

    assert (
        qc._gp3_qc_message_from_overview(
            overview,
            [],
            "warn",
        )
        == "QC status interpreted as 'warn'."
    )

    value = qc._gp3_qc_message_from_overview(
        overview,
        ["reason"],
        "warn",
    )

    assert value == ("first | second | third")

    empty_values = pd.DataFrame(
        {
            "reason": [
                pd.NA,
                "",
            ]
        }
    )

    assert qc._gp3_qc_message_from_overview(
        empty_values,
        ["reason"],
        "info",
    ).startswith("QC status interpreted")


# ============================================================================
# QC — SUMMARY BUNDLE
# ============================================================================


def test_h9_collect_qc_summary_statuses():
    objects = {
        "pass_obj": {
            "overview": pd.DataFrame(
                {
                    "status": ["ok"],
                    "message": ["clean"],
                }
            )
        },
        "warn_obj": {
            "overview": pd.DataFrame(
                {
                    "review_flag": [True],
                    "reason": ["review"],
                }
            )
        },
        "fail_obj": {
            "overview": pd.DataFrame(
                {
                    "ready": [False],
                    "message": ["blocked"],
                }
            )
        },
        "unknown_obj": object(),
    }

    result = qc.collect_gazepoint_qc_summaries(
        objects=objects,
    )

    assert (
        result["overview"].loc[
            0,
            "qc_bundle_status",
        ]
        == "fail"
    )

    assert set(result["object_summary"]["qc_status"]) == {
        "pass",
        "warn",
        "fail",
        "unknown",
    }

    no_rows = qc.collect_gazepoint_qc_summaries(
        objects=[pd.DataFrame({"status": ["ok"]})],
        object_names=["named"],
        include_overview_rows=False,
    )

    assert no_rows["overview_rows"].empty


def test_h9_collect_qc_validation():
    with pytest.raises(TypeError):
        qc.collect_gazepoint_qc_summaries()

    with pytest.raises(TypeError):
        qc.collect_gazepoint_qc_summaries(
            pd.DataFrame({"x": [1]}),
            objects=[pd.DataFrame({"status": ["ok"]})],
        )

    with pytest.raises(ValueError):
        qc.collect_gazepoint_qc_summaries(
            objects=[],
        )

    with pytest.raises(ValueError):
        qc.collect_gazepoint_qc_summaries(
            objects=[pd.DataFrame({"status": ["ok"]})],
            name="",
        )

    with pytest.raises(ValueError):
        qc.collect_gazepoint_qc_summaries(
            objects=[pd.DataFrame({"status": ["ok"]})],
            include_overview_rows="yes",
        )

    with pytest.raises(ValueError):
        qc.collect_gazepoint_qc_summaries(
            objects=[pd.DataFrame({"status": ["ok"]})],
            object_names=[],
        )


# ============================================================================
# QC — READINESS
# ============================================================================


def test_h9_readiness_legacy_paths():
    ready = qc.check_gazepoint_real_data_readiness(
        pd.DataFrame(
            {
                "subject": ["S1"],
                "time": [0.0],
                "x": [0.1],
                "y": [0.1],
                "pupil": [3.0],
            }
        )
    )

    assert "checks" in ready

    not_ready = qc.check_gazepoint_real_data_readiness(
        pd.DataFrame(
            {
                "subject": ["S1"],
                "time": [0.0],
            }
        )
    )

    assert not not_ready["ready"]


def test_h9_readiness_r_validation():
    frame = pd.DataFrame(
        {
            "subject": ["S1"],
            "trial": [1],
            "time": [0],
        }
    )

    with pytest.raises(ValueError):
        qc.check_gazepoint_real_data_readiness(
            frame,
            analysis_type="bad",
        )

    with pytest.raises(KeyError):
        qc.check_gazepoint_real_data_readiness(
            frame,
            analysis_type="general",
            participant_col="missing",
        )


# ============================================================================
# QC — LEGACY TRACKLOSS EXCLUSIONS
# ============================================================================


def test_h9_exclusion_legacy_trackloss_branch():
    frame = pd.DataFrame(
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
            "trackloss": [
                False,
                True,
                False,
                False,
            ],
        }
    )

    result = qc.recommend_gazepoint_exclusions(
        frame,
        validity_col="trackloss",
        min_trial_samples=1,
        min_participant_trials=1,
        min_participant_valid_trials=1,
    )

    assert not result["trial_recommendations"].empty

    assert not result["participant_recommendations"].empty
