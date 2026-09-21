from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

aoi = importlib.import_module("gp3tools.aoi")
pupil = importlib.import_module("gp3tools.pupil")
qc = importlib.import_module("gp3tools.qc")


# ============================================================================
# AOI — SAMPLE SUMMARY
# ============================================================================


def test_h10_summarise_aoi_samples_timing_ungrouped():
    frame = pd.DataFrame(
        {
            "AOI": ["A", "A", "B"],
            "TIME": [0.0, 0.1, 0.2],
        }
    )

    result = aoi.summarise_aoi_samples(
        frame,
        aoi_col="AOI",
        group_cols=[],
        time_col="TIME",
    )

    assert len(result) == 2
    assert {
        "time_to_first_view_sec",
        "aoi_sample_count",
        "approx_time_viewed_sec",
    }.issubset(result.columns)


def test_h10_summarise_aoi_samples_timing_missing_column():
    with pytest.raises(ValueError):
        aoi.summarise_aoi_samples(
            pd.DataFrame(
                {
                    "AOI": ["A"],
                }
            ),
            aoi_col="AOI",
            group_cols=[],
            time_col="TIME",
        )


# ============================================================================
# AOI — R SUMMARY CONTRACT
# ============================================================================


def _summary_gaze():
    return pd.DataFrame(
        {
            "USER_FILE": [
                "participant12.csv",
                "participant12.csv",
            ],
            "MEDIA_ID": ["M1", "M1"],
            "MEDIA_NAME": ["stim", "stim"],
            "AOI": ["A", "A"],
            "TIME": [0.1, np.nan],
        }
    )


def _summary_fix():
    return pd.DataFrame(
        {
            "USER_FILE": [
                "participant12.csv",
                "participant12.csv",
            ],
            "MEDIA_ID": ["M1", "M1"],
            "MEDIA_NAME": ["stim", "stim"],
            "AOI": ["A", "A"],
            "FPOGD": [0.1, 0.2],
            "FPOGS": [0.05, np.nan],
        }
    )


def test_h10_summarise_gazepoint_aoi_positional_r_mode():
    result = aoi.summarise_gazepoint_aoi(
        _summary_gaze(),
        _summary_fix(),
    )

    assert len(result) == 1
    assert (
        result.loc[
            0,
            "USER_ID",
        ]
        == 12
    )

    assert (
        result.loc[
            0,
            "sample_count",
        ]
        == 2
    )

    assert (
        result.loc[
            0,
            "fixation_count",
        ]
        == 2
    )


def test_h10_summarise_gazepoint_aoi_empty_aoi_rows():
    gaze = _summary_gaze()
    fixation = _summary_fix()

    gaze["AOI"] = pd.NA
    fixation["AOI"] = pd.NA

    result = aoi.summarise_gazepoint_aoi(
        gaze_data=gaze,
        fixation_data=fixation,
    )

    assert result.empty


def test_h10_summarise_gazepoint_aoi_errors():
    with pytest.raises(TypeError):
        aoi.summarise_gazepoint_aoi()

    with pytest.raises(TypeError):
        aoi.summarise_gazepoint_aoi(
            fixation_data=_summary_fix(),
        )

    with pytest.raises(TypeError):
        aoi.summarise_gazepoint_aoi(
            gaze_data=_summary_gaze(),
        )

    with pytest.raises(ValueError):
        aoi.summarise_gazepoint_aoi(
            gaze_data=_summary_gaze(),
            fixation_data=_summary_fix(),
            user_col="",
        )

    with pytest.raises(ValueError):
        aoi.summarise_gazepoint_aoi(
            gaze_data=_summary_gaze().drop(columns=["TIME"]),
            fixation_data=_summary_fix(),
        )

    with pytest.raises(ValueError):
        aoi.summarise_gazepoint_aoi(
            gaze_data=_summary_gaze(),
            fixation_data=_summary_fix().drop(columns=["FPOGD"]),
        )


# ============================================================================
# AOI — TRANSITION TABLE
# ============================================================================


def _transition_frame():
    return pd.DataFrame(
        {
            "AOI": [
                "A",
                "A",
                "B",
                "A",
            ],
            "TIME": [
                0.0,
                1.0,
                2.0,
                3.0,
            ],
        }
    )


def test_h10_compute_transition_matrix_dataframe_ungrouped():
    result = aoi.compute_transition_matrix(
        data=_transition_frame(),
        group_cols=[],
        aoi_col="AOI",
        time_col="TIME",
        collapse_repeats=True,
    )

    assert not result.empty
    assert set(result.columns) == {
        "from",
        "to",
        "n",
        "prob",
    }


def test_h10_compute_transition_matrix_empty_visits():
    frame = _transition_frame()
    frame["AOI"] = pd.NA

    result = aoi.compute_transition_matrix(
        data=frame,
        group_cols=[],
        aoi_col="AOI",
        time_col="TIME",
    )

    assert result.empty


def test_h10_compute_transition_matrix_validation():
    with pytest.raises(TypeError):
        aoi.compute_transition_matrix(
            ["A", "B"],
            data=_transition_frame(),
        )

    with pytest.raises(ValueError):
        aoi.compute_transition_matrix(
            data=_transition_frame(),
            group_cols=1,
            aoi_col="AOI",
            time_col="TIME",
        )

    with pytest.raises(ValueError):
        aoi.compute_transition_matrix(
            data=_transition_frame(),
            group_cols=[1],
            aoi_col="AOI",
            time_col="TIME",
        )

    with pytest.raises(ValueError):
        aoi.compute_transition_matrix(
            data=_transition_frame(),
            group_cols=[],
            aoi_col="missing",
            time_col="TIME",
        )


# ============================================================================
# AOI — TIME-VARYING TRANSITIONS
# ============================================================================


def _time_transition_frame():
    return pd.DataFrame(
        {
            "from_aoi": [
                "A",
                "A",
                "B",
            ],
            "to_aoi": [
                "B",
                "A",
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
            "window": [
                "w1",
                "",
                "w2",
            ],
        }
    )


def test_h10_time_varying_auto_resolve():
    result = aoi.compute_gazepoint_time_varying_transition_matrix(
        _time_transition_frame(),
        window_size_ms=10,
        normalise="row",
    )

    assert not result["matrix_long"].empty


def test_h10_time_varying_missing_window_label():
    result = aoi.compute_gazepoint_time_varying_transition_matrix(
        _time_transition_frame(),
        window_col="window",
        complete_states=False,
        normalise="none",
    )

    assert "missing_window" in set(result["matrix_long"][".gp3_time_window"].astype(str))


def test_h10_time_varying_more_validation():
    frame = _time_transition_frame()

    with pytest.raises(ValueError):
        aoi.compute_gazepoint_time_varying_transition_matrix(
            frame,
            window_size_ms=10,
            normalise="row",
            by_cols=["missing"],
        )

    with pytest.raises(ValueError):
        aoi.compute_gazepoint_time_varying_transition_matrix(
            frame,
            window_col="window",
            count_col="missing",
        )

    with pytest.raises(ValueError):
        aoi.compute_gazepoint_time_varying_transition_matrix(
            frame,
            window_size_ms=10,
            states=["Z"],
        )

    bad_time = frame.copy()
    bad_time.loc[
        0,
        "time",
    ] = np.nan

    with pytest.raises(ValueError):
        aoi.compute_gazepoint_time_varying_transition_matrix(
            bad_time,
            window_size_ms=10,
        )

    with pytest.raises(ValueError):
        aoi.compute_gazepoint_time_varying_transition_matrix(
            frame,
            window_size_ms=10,
            name="",
        )


# ============================================================================
# AOI — SEQUENCE ANOMALY EDGE ROUTES
# ============================================================================


def test_h10_sequence_anomaly_empty_group_result():
    frame = pd.DataFrame(
        {
            "subject": [
                pd.NA,
                pd.NA,
            ],
            "aoi": [
                "A",
                "B",
            ],
        }
    )

    result = aoi.flag_gazepoint_sequence_anomalies(
        frame,
        aoi_col="aoi",
        group_cols=["subject"],
    )

    assert result.empty


def test_h10_sequence_anomaly_standalone():
    result = aoi.flag_gazepoint_sequence_anomalies(
        sequence=[
            "A",
            "B",
            "A",
        ],
    )

    assert "anomaly_score" in result
    assert "anomaly" in result


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "min_length": True,
        },
        {
            "max_length": True,
        },
        {
            "max_missing_prop": True,
        },
        {
            "z_threshold": True,
        },
    ],
)
def test_h10_sequence_anomaly_bool_validation(kwargs):
    with pytest.raises(ValueError):
        aoi.flag_gazepoint_sequence_anomalies(
            pd.DataFrame(
                {
                    "subject": ["S1"],
                    "aoi": ["A"],
                }
            ),
            aoi_col="aoi",
            group_cols=["subject"],
            **kwargs,
        )


# ============================================================================
# AOI — CLUSTER / BOOTSTRAP VALIDATION
# ============================================================================


def test_h10_cluster_r_matrix():
    matrix = np.array(
        [
            [
                0.0,
                1.0,
                2.0,
            ],
            [
                1.0,
                0.0,
                1.0,
            ],
            [
                2.0,
                1.0,
                0.0,
            ],
        ]
    )

    result = aoi.cluster_gazepoint_scanpaths(
        x=matrix,
        k=2,
        method="hierarchical",
    )

    assert result["_gp3_class"] == "gp3_scanpath_clusters"

    assert len(result["assignments"]) == 3

    assert set(result["assignments"]["cluster"]).issubset(
        {
            1,
            2,
        }
    )


def test_h10_cluster_r_validation():
    with pytest.raises(TypeError):
        aoi.cluster_gazepoint_scanpaths(
            x=None,
            k=2,
        )

    with pytest.raises(ValueError):
        aoi.cluster_gazepoint_scanpaths(
            x=pd.DataFrame(
                {
                    "aoi": [
                        "A",
                        "B",
                    ]
                }
            ),
            k=2,
        )


def test_h10_bootstrap_validation():
    matrix = np.array(
        [
            [
                0.0,
                1.0,
            ],
            [
                1.0,
                0.0,
            ],
        ]
    )

    with pytest.raises(TypeError):
        aoi.bootstrap_gazepoint_scanpath_clusters(
            sample_fraction=0.8,
        )

    with pytest.raises(TypeError):
        aoi.bootstrap_gazepoint_scanpath_clusters(
            x=matrix,
            k=2,
            sample_fraction=0.8,
            impossible=True,
        )

    with pytest.raises(ValueError):
        aoi.bootstrap_gazepoint_scanpath_clusters(
            x=matrix,
            k=2,
            sample_fraction=0,
        )

    with pytest.raises(ValueError):
        aoi.bootstrap_gazepoint_scanpath_clusters(
            x=matrix,
            k=2,
            sample_fraction=1,
            n_boot=0,
        )


# ============================================================================
# PUPIL — BLINK ROUTES
# ============================================================================


def test_h10_blink_legacy_without_time_column():
    result = pupil.detect_gazepoint_blinks(
        pd.DataFrame(
            {
                "pupil": [
                    3.0,
                    np.nan,
                    3.1,
                ]
            }
        ),
        pupil_col="pupil",
        min_duration_ms=0,
    )

    assert "blink" in result


def test_h10_blink_r_no_candidates():
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
                    0.1,
                    0.2,
                ],
                "pupil": [
                    3.0,
                    3.0,
                    3.0,
                ],
            }
        ),
        pupil_col="pupil",
        id_col="USER_ID",
        time_unit="seconds",
        include_rapid_changes=False,
        min_duration_ms=0,
        return_mode="events",
    )

    assert result.empty


def test_h10_blink_merge_runs():
    result = pupil.detect_gazepoint_blinks(
        pd.DataFrame(
            {
                "USER_ID": [
                    "S1",
                    "S1",
                    "S1",
                ],
                "TIME": [
                    0.00,
                    0.01,
                    0.02,
                ],
                "pupil": [
                    np.nan,
                    3.0,
                    np.nan,
                ],
            }
        ),
        pupil_col="pupil",
        id_col="USER_ID",
        time_unit="seconds",
        include_rapid_changes=False,
        merge_gap_ms=20,
        min_duration_ms=0,
        return_mode="events",
    )

    assert len(result) == 1
    assert (
        "missing"
        in result.loc[
            0,
            "reason",
        ]
    )


# ============================================================================
# PUPIL — FINAL HELPER CONTRACTS
# ============================================================================


def test_h10_final_list_helper():
    assert pupil._gp3_final_list(None) == []

    assert pupil._gp3_final_list("x") == ["x"]

    assert pupil._gp3_final_list(
        (
            "x",
            "y",
        )
    ) == [
        "x",
        "y",
    ]


def test_h10_final_detect_helper():
    frame = pd.DataFrame({"x": [1]})

    assert (
        pupil._gp3_final_detect(
            frame,
            None,
            ["x"],
            "arg",
        )
        == "x"
    )

    with pytest.raises(ValueError):
        pupil._gp3_final_detect(
            frame,
            "",
            [],
            "arg",
        )

    with pytest.raises(ValueError):
        pupil._gp3_final_detect(
            frame,
            "missing",
            [],
            "arg",
        )

    with pytest.raises(ValueError):
        pupil._gp3_final_detect(
            frame,
            None,
            ["missing"],
            "arg",
            required=True,
        )


def test_h10_final_bool_helper():
    index = pd.RangeIndex(3)

    assert pupil._gp3_final_bool(
        [
            True,
            False,
            None,
        ],
        index,
    ).tolist() == [
        True,
        False,
        False,
    ]

    assert pupil._gp3_final_bool(
        [
            1,
            0,
            np.nan,
        ],
        index,
    ).tolist() == [
        True,
        False,
        False,
    ]

    assert pupil._gp3_final_bool(
        [
            "yes",
            "no",
            "valid",
        ],
        index,
    ).tolist() == [
        True,
        False,
        True,
    ]


def test_h10_registry_helper():
    assert (
        pupil._gp3_final_registry_value(
            None,
            "x",
            7,
        )
        == 7
    )

    registry = pd.DataFrame(
        {
            "parameter": ["x"],
            "value": [3],
        }
    )

    assert (
        pupil._gp3_final_registry_value(
            registry,
            "x",
            7,
        )
        == 3
    )

    with pytest.raises(ValueError):
        pupil._gp3_final_registry_value(
            pd.DataFrame({"x": [1]}),
            "x",
            0,
        )

    with pytest.raises(ValueError):
        pupil._gp3_final_registry_value(
            registry,
            "missing",
            0,
        )


def test_h10_mad_and_group_positions():
    assert np.isnan(pupil._gp3_final_mad([np.nan]))

    assert pupil._gp3_final_mad(
        [
            1,
            2,
            3,
        ]
    ) == pytest.approx(1)

    frame = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
            ]
        }
    )

    ungrouped = pupil._gp3_final_group_positions(
        frame,
        [],
    )

    assert len(ungrouped) == 1

    grouped = pupil._gp3_final_group_positions(
        frame,
        ["subject"],
    )

    assert len(grouped) == 1


def test_h10_merge_args_helper():
    assert pupil._gp3_final_merge_args(
        {"x": 1},
        None,
    ) == {"x": 1}

    with pytest.raises(ValueError):
        pupil._gp3_final_merge_args(
            {},
            [],
        )

    with pytest.raises(ValueError):
        pupil._gp3_final_merge_args(
            {"x": 1},
            {"x": 2},
            protected=("x",),
        )


# ============================================================================
# PUPIL — LUMINANCE PATH / READER
# ============================================================================


def test_h10_luminance_path_resolution(tmp_path):
    assert (
        pupil._gp3_final_luminance_path(
            None,
            None,
            True,
        )
        is None
    )

    direct = tmp_path / "direct.png"

    direct.write_bytes(b"x")

    resolved = pupil._gp3_final_luminance_path(
        str(direct),
        None,
        True,
    )

    assert Path(resolved).exists()

    root = tmp_path / "images"

    nested = root / "nested"

    nested.mkdir(parents=True)

    target = nested / "stim.png"

    target.write_bytes(b"x")

    recursive = pupil._gp3_final_luminance_path(
        "stim.png",
        str(root),
        True,
    )

    assert Path(recursive).resolve() == target.resolve()


def test_h10_luminance_reader_missing_and_available(tmp_path):
    missing_name = pupil._gp3_final_read_luminance(
        "s1",
        "",
        None,
        True,
    )

    assert missing_name["luminance_status"] == "missing_file_name"

    missing_file = pupil._gp3_final_read_luminance(
        "s1",
        "missing.png",
        str(tmp_path),
        True,
    )

    assert missing_file["luminance_status"] == "file_missing"

    from matplotlib import image as mpimg

    image_path = tmp_path / "real.png"

    image = np.array(
        [
            [
                [
                    0.0,
                    0.0,
                    0.0,
                ],
                [
                    1.0,
                    1.0,
                    1.0,
                ],
            ],
            [
                [
                    0.5,
                    0.5,
                    0.5,
                ],
                [
                    0.25,
                    0.25,
                    0.25,
                ],
            ],
        ],
        dtype=float,
    )

    mpimg.imsave(
        image_path,
        image,
    )

    available = pupil._gp3_final_read_luminance(
        "s2",
        str(image_path),
        None,
        True,
    )

    assert available["luminance_available"]

    assert available["luminance_status"] == "available"


def test_h10_luminance_audit_statuses(tmp_path):
    from matplotlib import image as mpimg

    image_path = tmp_path / "real.png"

    mpimg.imsave(
        image_path,
        np.ones(
            (
                2,
                2,
                3,
            )
        )
        * 0.5,
    )

    result = pupil.audit_gazepoint_stimulus_luminance(
        pd.DataFrame(
            {
                "stimulus_file": [
                    str(image_path),
                    "missing.png",
                ],
                "stimulus_id": [
                    "A",
                    "B",
                ],
                "condition": [
                    "C1",
                    "C2",
                ],
            }
        ),
        stimulus_file_col="stimulus_file",
        stimulus_id_col="stimulus_id",
        condition_col="condition",
        image_dir=str(tmp_path),
    )

    assert (
        result["overview"].loc[
            0,
            "audit_status",
        ]
        == "partial_luminance_available"
    )

    assert (
        result["balance_summary"].loc[
            0,
            "luminance_balance_status",
        ]
        == "partial_condition_luminance_available"
    )


def test_h10_luminance_audit_validation():
    frame = pd.DataFrame({"stimulus_file": ["x.png"]})

    with pytest.raises(ValueError):
        pupil.audit_gazepoint_stimulus_luminance(
            frame.iloc[0:0],
            stimulus_file_col="stimulus_file",
        )

    with pytest.raises(ValueError):
        pupil.audit_gazepoint_stimulus_luminance(
            frame,
            stimulus_file_col="stimulus_file",
            recursive="yes",
        )

    with pytest.raises(ValueError):
        pupil.audit_gazepoint_stimulus_luminance(
            frame,
            stimulus_file_col="stimulus_file",
            name="",
        )

    with pytest.raises(ValueError):
        pupil.audit_gazepoint_stimulus_luminance(
            frame,
            stimulus_file_col="stimulus_file",
            image_dir="",
        )


# ============================================================================
# PUPIL — RELIABILITY
# ============================================================================


def test_h10_reliability_legacy_without_subject():
    result = pupil.audit_gazepoint_pupil_reliability(
        pd.DataFrame(
            {
                "pupil": [
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                ]
            }
        ),
        pupil_col="pupil",
    )

    assert len(result) == 1

    assert "mean_even" in result


def test_h10_reliability_legacy_subject():
    frame = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S2",
                "S2",
            ],
            "pupil": [
                1.0,
                2.0,
                2.0,
                4.0,
            ],
        }
    )

    result = pupil.audit_gazepoint_pupil_reliability(
        frame,
        pupil_col="pupil",
        subject_col="subject",
    )

    assert len(result) == 1

    assert "spearman_brown" in result


def test_h10_reliability_empty_r_mode():
    with pytest.raises(ValueError):
        pupil.audit_gazepoint_pupil_reliability(
            pd.DataFrame(),
            participant_col="subject",
            outcome_cols="pupil",
        )


def test_h10_reliability_auto_numeric_outcome():
    rows = []

    for participant in (
        "S1",
        "S2",
        "S3",
    ):
        for trial, label in enumerate(
            [
                "alpha",
                "beta",
                "gamma",
                "delta",
            ],
            start=1,
        ):
            rows.append(
                {
                    "subject": participant,
                    "trial_name": label,
                    "score": (int(participant[1:]) * 10 + trial),
                }
            )

    result = pupil.audit_gazepoint_pupil_reliability(
        pd.DataFrame(rows),
        participant_col="subject",
        trial_col="trial_name",
        min_trials_per_split=1,
    )

    assert "score" in set(result["reliability_summary"]["outcome"])


# ============================================================================
# PUPIL — BINOCULAR FIT INTERNAL CONTRACT
# ============================================================================


def test_h10_binoc_fit_reasons():
    fit = pupil._gp3_binoc_r_fit_one

    insufficient = fit(
        [
            1,
        ],
        [
            1,
        ],
        2,
        2,
        None,
        False,
        None,
    )

    assert insufficient["reason"] == "insufficient_paired_samples"

    unique = fit(
        [
            1,
            1,
            1,
        ],
        [
            1,
            2,
            3,
        ],
        2,
        2,
        None,
        False,
        None,
    )

    assert unique["reason"] == "insufficient_unique_values"

    negative = fit(
        [
            1,
            2,
            3,
        ],
        [
            3,
            2,
            1,
        ],
        2,
        2,
        None,
        False,
        None,
    )

    assert negative["reason"] == "non_positive_slope"

    slope_limit = fit(
        [
            1,
            2,
            3,
        ],
        [
            10,
            20,
            30,
        ],
        2,
        2,
        None,
        True,
        2,
    )

    assert slope_limit["reason"] == "slope_outside_limit"

    low_r2 = fit(
        [
            1,
            2,
            3,
            4,
        ],
        [
            1,
            4,
            2,
            5,
        ],
        2,
        2,
        0.99,
        True,
        None,
    )

    assert low_r2["reason"] == "r_squared_below_threshold"

    eligible = fit(
        [
            1,
            2,
            3,
            4,
        ],
        [
            2,
            4,
            6,
            8,
        ],
        2,
        2,
        0.9,
        False,
        None,
    )

    assert eligible["eligible"]

    assert eligible["status"] == "eligible"


def test_h10_binoc_level_specs():
    assert pupil._gp3_binoc_r_level_specs(
        None,
        None,
    ) == [
        [],
    ]

    specs = pupil._gp3_binoc_r_level_specs(
        ["subject"],
        [
            ["condition"],
            [],
        ],
    )

    assert ["subject"] in specs

    assert ["condition"] in specs

    assert [] in specs


# ============================================================================
# PUPIL — RECONSTRUCTION VALIDATION
# ============================================================================


def _binoc_frame():
    return pd.DataFrame(
        {
            "time": [
                0.0,
                1.0,
                2.0,
            ],
            "left_pupil": [
                3.0,
                np.nan,
                3.2,
            ],
            "right_pupil": [
                3.1,
                3.2,
                np.nan,
            ],
        }
    )


def test_h10_reconstruction_validation():
    frame = _binoc_frame()

    with pytest.raises(ValueError):
        pupil.reconstruct_gazepoint_binocular_pupil(
            frame,
            method="bad",
            time_col="time",
        )

    with pytest.raises(ValueError):
        pupil.reconstruct_gazepoint_binocular_pupil(
            frame,
            method="none",
            time_col="time",
            time_unit="bad",
        )

    with pytest.raises(ValueError):
        pupil.reconstruct_gazepoint_binocular_pupil(
            frame,
            method="none",
            max_gap_ms=10,
        )

    conflict = frame.copy()

    conflict["gp3_binocular_left_observed"] = 1

    with pytest.raises(ValueError):
        pupil.reconstruct_gazepoint_binocular_pupil(
            conflict,
            method="none",
            time_col="time",
        )

    bad_flag = frame.copy()

    bad_flag["bad_flag"] = [
        "yes",
        "no",
        "no",
    ]

    with pytest.raises(TypeError):
        pupil.reconstruct_gazepoint_binocular_pupil(
            bad_flag,
            method="none",
            time_col="time",
            exclude_flag_cols=["bad_flag"],
        )


def test_h10_audit_reconstruction_requires_metadata():
    with pytest.raises(ValueError):
        pupil.audit_gazepoint_binocular_reconstruction(
            _binoc_frame(),
            by=[],
        )


# ============================================================================
# QC — MASTER PUPIL UNITS
# ============================================================================


def _unit_master(
    left_name,
    right_name,
):
    return pd.DataFrame(
        {
            "TIME": [0.0],
            "BPOGX": [100.0],
            "BPOGY": [100.0],
            left_name: [3.0],
            right_name: [3.1],
        }
    )


def test_h10_master_pupil_units():
    meters = qc.as_gazepoint_master(
        _unit_master(
            "LPUPILD",
            "RPUPILD",
        ),
        coordinate_unit="pixels",
    )

    assert (
        meters.loc[
            0,
            "pupil_unit",
        ]
        == "diameter_meters"
    )

    tracker = qc.as_gazepoint_master(
        _unit_master(
            "LPD",
            "RPD",
        ),
        coordinate_unit="pixels",
    )

    assert (
        tracker.loc[
            0,
            "pupil_unit",
        ]
        == "tracker_units"
    )

    absent = qc.as_gazepoint_master(
        pd.DataFrame({"TIME": [0.0]}),
        coordinate_unit="pixels",
    )

    assert pd.isna(
        absent.loc[
            0,
            "pupil_unit",
        ]
    )


# ============================================================================
# QC — MISSINGNESS / GAZE QUALITY
# ============================================================================


def test_h10_missingness_string_selection():
    result = qc.summarise_gazepoint_missingness(
        pd.DataFrame(
            {
                "subject": ["S1"],
                "x": [np.nan],
            }
        ),
        group_cols="subject",
        cols="x",
    )

    assert len(result) == 1

    with pytest.raises(ValueError):
        qc.summarise_gazepoint_missingness(
            pd.DataFrame({"x": [1]}),
            group_cols=["subject"],
        )


def test_h10_gaze_signal_quality_unknown_kwarg():
    with pytest.raises(TypeError):
        qc.audit_gazepoint_gaze_signal_quality(
            pd.DataFrame({"x": [1]}),
            impossible=True,
        )


# ============================================================================
# QC — HARMONISATION VALIDATION
# ============================================================================


def test_h10_harmonise_partial_dimensions():
    frame = pd.DataFrame(
        {
            "x": [10.0],
            "y": [20.0],
        }
    )

    with pytest.raises(ValueError):
        qc.harmonize_gazepoint_screen_coordinates(
            frame,
            from_width=100,
        )

    with pytest.raises(ValueError):
        qc.harmonize_gazepoint_screen_coordinates(
            frame,
            from_width=100,
            from_height=100,
            to_width=0,
            to_height=100,
        )


# ============================================================================
# QC — PHASE COVERAGE
# ============================================================================


def test_h10_phase_coverage_values_without_time():
    result = qc.summarise_gazepoint_phase_coverage(
        pd.DataFrame(
            {
                "phase": [
                    "a",
                    "a",
                ],
                "x": [
                    1.0,
                    np.nan,
                ],
            }
        ),
        value_cols=["x"],
    )

    assert len(result) == 1

    assert np.isnan(
        result.loc[
            0,
            "n_finite_time",
        ]
    )

    assert (
        result.loc[
            0,
            "n_complete_value_rows",
        ]
        == 1
    )


def test_h10_phase_coverage_nonfinite_time():
    result = qc.summarise_gazepoint_phase_coverage(
        pd.DataFrame(
            {
                "phase": ["a"],
                "time": [np.nan],
            }
        ),
        time_col="time",
    )

    assert (
        result.loc[
            0,
            "n_finite_time",
        ]
        == 0
    )


# ============================================================================
# QC — NORMALISE OBJECT / STATUS HELPERS
# ============================================================================


def test_h10_qc_normalise_scalar_object():
    marker = object()

    result = qc._gp3_qc_normalise_objects(marker)

    assert result == [
        (
            "",
            marker,
        )
    ]


def test_h10_qc_status_object_bool_column():
    overview = pd.DataFrame(
        {
            "review_flag": pd.Series(
                [
                    True,
                    False,
                ],
                dtype=object,
            )
        }
    )

    assert (
        qc._gp3_qc_status_from_overview(
            overview,
            ["review_flag"],
        )
        == "warn"
    )


# ============================================================================
# QC — POST-EXCLUSION VALIDATION
# ============================================================================


def _post_exclusion_frame():
    return pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
            ],
            "condition": [
                "A",
                "B",
            ],
        }
    )


def test_h10_post_exclusion_threshold_validation():
    with pytest.raises(ValueError):
        qc.audit_gazepoint_post_exclusion_balance(
            _post_exclusion_frame(),
            min_retained_units_per_condition=0,
        )

    with pytest.raises(ValueError):
        qc.audit_gazepoint_post_exclusion_balance(
            _post_exclusion_frame(),
            max_condition_count_ratio=0,
        )


def test_h10_post_exclusion_no_usable_rows():
    frame = pd.DataFrame(
        {
            "subject": [""],
            "condition": ["A"],
        }
    )

    with pytest.raises(ValueError):
        qc.audit_gazepoint_post_exclusion_balance(
            frame,
            min_retained_units_per_condition=1,
        )


# ============================================================================
# QC — NAMING WRITER OBJECT FORM
# ============================================================================


def test_h10_naming_writer_attribute_object(tmp_path):
    class Audit:
        pass

    obj = Audit()

    obj.pairs = pd.DataFrame(
        {
            "stem": ["x"],
            "status": ["paired"],
        }
    )

    out = qc.write_gazepoint_naming_audit(
        x=obj,
        output_file=(tmp_path / "audit.csv"),
    )

    assert out.exists()
