from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from gp3tools import _behavioral_r2 as r2

# ===========================================================================
# LOW-LEVEL R SEMANTIC HELPERS
# ===========================================================================


def test_r2_helper_character_and_numeric_normalization():
    assert r2._r2_gaze_standardise_name("MEDIA_ID") == "media_id"

    assert r2._r2_gaze_standardise_name("USER_FILE") == "subject"

    assert r2._r2_gaze_standardise_name("trial") == "trial"

    assert r2._r2_gaze_cols(None) == []

    assert r2._r2_gaze_cols("MEDIA_ID") == ["media_id"]

    assert r2._r2_gaze_cols(
        [
            "USER_FILE",
            "trial",
        ]
    ) == [
        "subject",
        "trial",
    ]

    assert math.isnan(r2._r2_v5_num_text(None))

    assert r2._r2_v5_num_text("not-a-number") == "not-a-number"

    assert math.isnan(r2._r2_v5_num_text(np.inf))

    assert r2._r2_v5_num_text(3) == "3"

    assert r2._r2_v5_num_text(3.25) == "3.25"

    assert r2._r2_fmt_number(5) == "5"

    assert r2._r2_fmt_number(2.5) == "2.5"


def test_r2_nullable_and_scalar_setting_semantics():
    assert math.isnan(r2._r2_gaze_collapse_nullable(None))

    assert math.isnan(r2._r2_gaze_collapse_nullable(""))

    assert r2._r2_gaze_collapse_nullable("x") == "x"

    assert math.isnan(r2._r2_gaze_collapse_nullable([]))

    assert (
        r2._r2_gaze_collapse_nullable(
            [
                "x",
                "y",
            ]
        )
        == "x, y"
    )

    assert r2._r2_scalar_character_setting(None) == "<EMPTY>"

    assert r2._r2_scalar_character_setting([]) == "<EMPTY>"

    assert r2._r2_scalar_character_setting(["x"]) == "x"

    assert r2._r2_scalar_character_setting(
        [
            "x",
            "y",
        ]
    ) == [
        "x",
        "y",
    ]

    assert r2._r2_setting_text_v3(True) == "TRUE"

    assert r2._r2_setting_text_v3(False) == "FALSE"

    assert r2._r2_setting_text_v3("True") == "TRUE"

    assert r2._r2_setting_text_v3("false") == "FALSE"

    assert r2._r2_setting_text_v3("other") == "other"


def test_r2_validity_vector_boolean_numeric_and_text_modes():
    boolean = r2._r2_gaze_validity_vector(
        pd.Series(
            [
                True,
                False,
                pd.NA,
            ],
            dtype="boolean",
        )
    )

    assert boolean.astype("string").tolist() == [
        "True",
        "False",
        pd.NA,
    ]

    numeric = r2._r2_gaze_validity_vector(
        pd.Series(
            [
                1,
                0,
                -1,
                np.nan,
            ]
        )
    )

    assert numeric.astype("string").tolist() == [
        "True",
        "False",
        "False",
        pd.NA,
    ]

    text = r2._r2_gaze_validity_vector(
        pd.Series(
            [
                "VALID",
                "ok",
                "yes",
                "1",
                "invalid",
                "missing",
                "0",
                "unknown",
            ]
        )
    )

    assert text.astype("string").tolist() == [
        "True",
        "True",
        "True",
        "True",
        "False",
        "False",
        "False",
        pd.NA,
    ]


def test_r2_event_sync_status_precedence_all_outcomes():
    base = dict(
        n_samples=3,
        n_finite_time=3,
        has_event_col=True,
        n_events=2,
        n_missing_expected=0,
        onset_count=1,
        response_count=1,
        n_duplicate_time=0,
        has_large_gap=False,
        min_samples_per_unit=1,
    )

    cases = [
        (
            {
                "n_samples": 0,
                "min_samples_per_unit": 1,
            },
            "too_few_samples",
        ),
        (
            {
                "n_finite_time": 0,
            },
            "missing_time",
        ),
        (
            {
                "n_duplicate_time": 1,
            },
            "duplicate_time_values",
        ),
        (
            {
                "has_large_gap": True,
            },
            "large_time_gap",
        ),
        (
            {
                "has_event_col": False,
            },
            "event_column_not_available",
        ),
        (
            {
                "n_events": 0,
            },
            "no_events_observed",
        ),
        (
            {
                "n_missing_expected": 1,
            },
            "missing_expected_events",
        ),
        (
            {
                "onset_count": 0,
            },
            "missing_onset_event",
        ),
        (
            {
                "response_count": 0,
            },
            "missing_response_event",
        ),
        (
            {},
            "ok",
        ),
    ]

    for changes, expected in cases:
        args = dict(base)

        args.update(changes)

        assert r2._r2_event_sync_status(**args) == expected


def test_r2_gaze_status_precedence_all_outcomes():
    base = dict(
        has_xy=True,
        has_validity=True,
        has_pupil=True,
        gaze_valid_prop=1.0,
        missing_gaze_prop=0.0,
        offscreen_prop=0.0,
        pupil_valid_prop=1.0,
        min_gaze_valid_prop=0.7,
        max_missing_gaze_prop=0.3,
        max_offscreen_prop=0.3,
        min_pupil_valid_prop=0.7,
    )

    cases = [
        (
            {
                "has_xy": False,
                "has_validity": False,
            },
            "gaze_columns_not_available",
        ),
        (
            {
                "gaze_valid_prop": 0.2,
            },
            "low_gaze_validity",
        ),
        (
            {
                "missing_gaze_prop": 0.8,
            },
            "high_missing_gaze",
        ),
        (
            {
                "offscreen_prop": 0.8,
            },
            "high_offscreen_gaze",
        ),
        (
            {
                "pupil_valid_prop": 0.2,
            },
            "low_pupil_validity",
        ),
        (
            {},
            "ok",
        ),
    ]

    for changes, expected in cases:
        args = dict(base)

        args.update(changes)

        assert r2._r2_gaze_status(**args) == expected


def test_r2_face_helper_status_messages_and_percentages():
    assert math.isnan(
        r2._r2_face_percent(
            1,
            0,
        )
    )

    assert (
        r2._r2_face_percent(
            1,
            4,
        )
        == 25.0
    )

    assert (
        r2._r2_face_status(
            n_rows=0,
            matched_percent=100,
            max_abs_diff_sec_observed=0,
            min_matched_percent=70,
            warning_matched_percent=85,
            max_abs_diff_sec=None,
        )
        == "fail"
    )

    assert (
        r2._r2_face_status(
            n_rows=2,
            matched_percent=np.nan,
            max_abs_diff_sec_observed=0,
            min_matched_percent=70,
            warning_matched_percent=85,
            max_abs_diff_sec=None,
        )
        == "unknown"
    )

    assert (
        r2._r2_face_status(
            n_rows=2,
            matched_percent=50,
            max_abs_diff_sec_observed=0,
            min_matched_percent=70,
            warning_matched_percent=85,
            max_abs_diff_sec=None,
        )
        == "fail"
    )

    assert (
        r2._r2_face_status(
            n_rows=2,
            matched_percent=80,
            max_abs_diff_sec_observed=0,
            min_matched_percent=70,
            warning_matched_percent=85,
            max_abs_diff_sec=None,
        )
        == "warn"
    )

    assert (
        r2._r2_face_status(
            n_rows=2,
            matched_percent=100,
            max_abs_diff_sec_observed=0.2,
            min_matched_percent=70,
            warning_matched_percent=85,
            max_abs_diff_sec=0.1,
        )
        == "warn"
    )

    assert (
        r2._r2_face_status(
            n_rows=2,
            matched_percent=100,
            max_abs_diff_sec_observed=0.01,
            min_matched_percent=70,
            warning_matched_percent=85,
            max_abs_diff_sec=0.1,
        )
        == "pass"
    )

    assert "could not be evaluated" in (
        r2._r2_face_message(
            "unknown",
            np.nan,
            70,
            85,
        )
    )

    assert "minimum threshold" in (
        r2._r2_face_message(
            "fail",
            50,
            70,
            85,
        )
    )

    assert "should be reviewed" in (
        r2._r2_face_message(
            "warn",
            80,
            70,
            85,
        )
    )

    assert "passed" in (
        r2._r2_face_message(
            "pass",
            100,
            70,
            85,
        )
    )


def test_r2_detector_helpers_cover_threshold_overlap_and_sequences():
    assert math.isnan(
        r2._r2_detector_threshold(
            "hmm",
            "hmm",
        )
    )

    assert math.isnan(
        r2._r2_detector_threshold(
            "velocity_bad",
            "velocity",
        )
    )

    assert (
        r2._r2_detector_threshold(
            "velocity_5",
            "velocity",
        )
        == 5.0
    )

    empty = pd.DataFrame(
        columns=[
            "start_time",
            "end_time",
        ]
    )

    one = pd.DataFrame(
        {
            "start_time": [0.0],
            "end_time": [1.0],
        }
    )

    assert (
        r2._r2_detector_best_overlap(
            empty,
            one,
        ).size
        == 0
    )

    assert np.allclose(
        r2._r2_detector_best_overlap(
            one,
            empty,
        ),
        [0.0],
    )

    half = pd.DataFrame(
        {
            "start_time": [0.5],
            "end_time": [1.5],
        }
    )

    overlap = r2._r2_detector_best_overlap(
        one,
        half,
    )

    assert overlap[0] == pytest.approx(1.0 / 3.0)

    assert (
        r2._r2_detector_sequence_keys(
            empty,
            [],
        )
        == []
    )

    sequence_frame = pd.DataFrame(
        {
            "subject": [
                "S1",
                pd.NA,
            ],
            "trial": [
                "T1",
                "T2",
            ],
        }
    )

    assert r2._r2_detector_sequence_keys(
        sequence_frame,
        [],
    ) == [
        ".all",
        ".all",
    ]

    keys = r2._r2_detector_sequence_keys(
        sequence_frame,
        [
            "subject",
            "trial",
        ],
    )

    assert keys == [
        "S1\rT1",
        "<NA>\rT2",
    ]


# ===========================================================================
# EVENT-SYNC EDGE CONTRACT
# ===========================================================================


def _event_frame(
    *,
    time,
    event,
):
    return pd.DataFrame(
        {
            "subject": ["S1"] * len(time),
            "time": time,
            "event": event,
        }
    )


@pytest.mark.parametrize(
    ("data", "kwargs"),
    [
        (
            "not-a-frame",
            {},
        ),
        (
            pd.DataFrame(),
            {},
        ),
        (
            pd.DataFrame(
                {
                    "subject": ["S1"],
                    "event": ["onset"],
                }
            ),
            {},
        ),
        (
            pd.DataFrame(
                {
                    "subject": ["S1"],
                    "time": [0],
                }
            ),
            {
                "condition_col": "condition",
            },
        ),
        (
            pd.DataFrame(
                {
                    "subject": ["S1"],
                    "time": [0],
                }
            ),
            {
                "group_cols": ("missing_group",),
            },
        ),
        (
            pd.DataFrame(
                {
                    "subject": ["S1"],
                    "time": [0],
                }
            ),
            {
                "event_col": "missing_event",
            },
        ),
        (
            pd.DataFrame(
                {
                    "subject": ["S1"],
                    "time": [0],
                }
            ),
            {
                "min_samples_per_unit": 0,
            },
        ),
        (
            pd.DataFrame(
                {
                    "subject": ["S1"],
                    "time": [0],
                }
            ),
            {
                "max_time_gap_ms": 0,
            },
        ),
        (
            pd.DataFrame(
                {
                    "subject": ["S1"],
                    "time": [0],
                }
            ),
            {
                "expected_event_labels": [],
            },
        ),
        (
            pd.DataFrame(
                {
                    "subject": ["S1"],
                    "time": [0],
                }
            ),
            {
                "onset_event_label": "",
            },
        ),
    ],
)
def test_r2_event_sync_validation_contract(
    data,
    kwargs,
):
    with pytest.raises(ValueError):
        r2.audit_event_sync(
            data,
            **kwargs,
        )


@pytest.mark.parametrize(
    ("frame", "kwargs", "expected"),
    [
        (
            _event_frame(
                time=[0],
                event=["onset"],
            ),
            {
                "min_samples_per_unit": 2,
            },
            "too_few_samples",
        ),
        (
            _event_frame(
                time=[np.nan],
                event=["onset"],
            ),
            {},
            "missing_time",
        ),
        (
            _event_frame(
                time=[
                    0,
                    0,
                ],
                event=[
                    "onset",
                    "response",
                ],
            ),
            {},
            "duplicate_time_values",
        ),
        (
            _event_frame(
                time=[
                    0,
                    100,
                ],
                event=[
                    "onset",
                    "response",
                ],
            ),
            {
                "max_time_gap_ms": 50,
            },
            "large_time_gap",
        ),
        (
            _event_frame(
                time=[
                    0,
                    1,
                ],
                event=[
                    "",
                    "",
                ],
            ),
            {},
            "no_events_observed",
        ),
        (
            _event_frame(
                time=[
                    0,
                    1,
                ],
                event=[
                    "onset",
                    "onset",
                ],
            ),
            {
                "expected_event_labels": [
                    "onset",
                    "response",
                ],
            },
            "missing_expected_events",
        ),
        (
            _event_frame(
                time=[
                    0,
                    1,
                ],
                event=[
                    "response",
                    "response",
                ],
            ),
            {
                "onset_event_label": "onset",
            },
            "missing_onset_event",
        ),
        (
            _event_frame(
                time=[
                    0,
                    1,
                ],
                event=[
                    "onset",
                    "onset",
                ],
            ),
            {
                "response_event_label": "response",
            },
            "missing_response_event",
        ),
        (
            _event_frame(
                time=[
                    0,
                    1,
                ],
                event=[
                    "onset",
                    "response",
                ],
            ),
            {
                "onset_event_label": "onset",
                "response_event_label": "response",
            },
            "ok",
        ),
    ],
)
def test_r2_event_sync_status_contract(
    frame,
    kwargs,
    expected,
):
    out = r2.audit_event_sync(
        frame,
        group_cols=("subject",),
        event_col="event",
        **kwargs,
    )

    assert out["unit_summary"].iloc[0]["event_sync_status"] == expected


def test_r2_event_sync_without_event_column():
    frame = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
            ],
            "time": [
                0,
                1,
            ],
        }
    )

    out = r2.audit_event_sync(
        frame,
        group_cols=("subject",),
    )

    assert out["overview"].iloc[0]["audit_status"] == "event_column_not_available"

    assert out["event_summary"].empty


# ===========================================================================
# GAZE-SIGNAL QUALITY EDGE CONTRACT
# ===========================================================================


@pytest.mark.parametrize(
    ("data", "kwargs"),
    [
        (
            "not-a-frame",
            {},
        ),
        (
            pd.DataFrame(),
            {},
        ),
        (
            pd.DataFrame(
                {
                    "other": [1],
                }
            ),
            {},
        ),
        (
            pd.DataFrame(
                {
                    "subject": ["S1"],
                }
            ),
            {
                "condition_col": "condition",
            },
        ),
        (
            pd.DataFrame(
                {
                    "subject": ["S1"],
                }
            ),
            {
                "group_cols": ("absent",),
            },
        ),
        (
            pd.DataFrame(
                {
                    "subject": ["S1"],
                }
            ),
            {
                "group_cols": ("subject",),
                "x_col": "absent",
            },
        ),
        (
            pd.DataFrame(
                {
                    "subject": ["S1"],
                }
            ),
            {
                "group_cols": ("subject",),
                "validity_cols": ["absent"],
            },
        ),
        (
            pd.DataFrame(
                {
                    "subject": ["S1"],
                }
            ),
            {
                "group_cols": ("subject",),
                "screen_x_range": (
                    1,
                    0,
                ),
            },
        ),
        (
            pd.DataFrame(
                {
                    "subject": ["S1"],
                }
            ),
            {
                "group_cols": ("subject",),
                "screen_y_range": (
                    0,
                    np.inf,
                ),
            },
        ),
        (
            pd.DataFrame(
                {
                    "subject": ["S1"],
                }
            ),
            {
                "group_cols": ("subject",),
                "min_gaze_valid_prop": 1.5,
            },
        ),
    ],
)
def test_r2_gaze_signal_validation_contract(
    data,
    kwargs,
):
    with pytest.raises(ValueError):
        r2.audit_gaze_signal_quality(
            data,
            **kwargs,
        )


def test_r2_gaze_signal_no_gaze_columns_status():
    frame = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
            ],
        }
    )

    out = r2.audit_gaze_signal_quality(
        frame,
        group_cols=("subject",),
    )

    assert out["overview"].iloc[0]["gaze_signal_quality_status"] == "gaze_columns_not_available"

    assert out["unit_summary"].iloc[0]["gaze_signal_status"] == "gaze_columns_not_available"


def test_r2_gaze_signal_numeric_validity_without_coordinates():
    frame = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S1",
            ],
            "valid": [
                1,
                0,
                np.nan,
            ],
        }
    )

    out = r2.audit_gaze_signal_quality(
        frame,
        group_cols=("subject",),
        validity_cols=["valid"],
    )

    row = out["unit_summary"].iloc[0]

    assert row["n_valid_gaze"] == 1

    assert row["gaze_valid_prop"] == pytest.approx(1 / 3)


def test_r2_gaze_signal_text_validity_coordinates_and_pupil():
    frame = pd.DataFrame(
        {
            "subject": ["S1"] * 4,
            "x": [
                0.5,
                1.2,
                np.nan,
                0.4,
            ],
            "y": [
                0.5,
                0.5,
                0.5,
                0.4,
            ],
            "valid": [
                "valid",
                "valid",
                "valid",
                "invalid",
            ],
            "pupil": [
                3.0,
                -1.0,
                np.nan,
                2.0,
            ],
        }
    )

    out = r2.audit_gaze_signal_quality(
        frame,
        group_cols=("subject",),
        x_col="x",
        y_col="y",
        validity_cols=["valid"],
        pupil_col="pupil",
        min_gaze_valid_prop=0.5,
        max_missing_gaze_prop=0.2,
        max_offscreen_prop=0.2,
        min_pupil_valid_prop=0.75,
    )

    row = out["unit_summary"].iloc[0]

    assert row["n_missing_gaze"] == 1

    assert row["n_offscreen_gaze"] == 1

    assert row["n_valid_pupil"] == 2

    assert row["gaze_signal_status"] in {
        "low_gaze_validity",
        "high_missing_gaze",
        "high_offscreen_gaze",
        "low_pupil_validity",
    }


# ===========================================================================
# FACE-SYNC EDGE CONTRACT
# ===========================================================================


def test_r2_face_sync_validation_and_overall_mode():
    with pytest.raises(ValueError):
        r2.audit_face_sync("not-a-frame")

    with pytest.raises(ValueError):
        r2.audit_face_sync(pd.DataFrame({"face_sync_status": ["matched"]}))

    frame = pd.DataFrame(
        {
            "face_sync_method": [
                "nearest_time",
                "nearest_time",
                "nearest_time",
            ],
            "face_sync_status": [
                "matched",
                "",
                "unmatched",
            ],
            "face_sync_within_tolerance": [
                True,
                pd.NA,
                False,
            ],
            "face_sync_abs_diff_sec": [
                0.01,
                np.nan,
                0.2,
            ],
        }
    )

    out = r2.audit_face_sync(
        frame,
        group_cols=["absent"],
        min_matched_percent=20,
        warning_matched_percent=80,
        max_abs_diff_sec=0.1,
    )

    assert out["group_summary"].iloc[0]["face_sync_group"] == "overall"

    assert out["group_summary"].iloc[0]["n_unknown_status"] == 1

    assert len(out["issue_summary"]) == 7


# ===========================================================================
# TIMECOURSE-GRID EDGE CONTRACT
# ===========================================================================


@pytest.mark.parametrize(
    ("data", "kwargs"),
    [
        (
            "not-a-frame",
            {},
        ),
        (
            pd.DataFrame(
                {
                    "s": ["S1"],
                    "c": ["A"],
                    "t": [0],
                    "y": [1],
                }
            ),
            {
                "subject_col": "",
                "condition_col": "c",
                "time_col": "t",
                "outcome_col": "y",
            },
        ),
        (
            pd.DataFrame(
                {
                    "s": ["S1"],
                }
            ),
            {
                "subject_col": "s",
                "condition_col": "c",
                "time_col": "t",
                "outcome_col": "y",
            },
        ),
    ],
)
def test_r2_timecourse_grid_validation_contract(
    data,
    kwargs,
):
    with pytest.raises(ValueError):
        r2.audit_timecourse_grid(
            data,
            **kwargs,
        )


def test_r2_timecourse_grid_ready_and_duplicate_variants():
    ready = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S1",
                "S1",
            ],
            "condition": [
                "A",
                "B",
                "A",
                "B",
            ],
            "time_bin": [
                0,
                0,
                1,
                1,
            ],
            "value": [
                1,
                2,
                3,
                4,
            ],
        }
    )

    out = r2.audit_timecourse_grid(
        ready,
        subject_col="subject",
        condition_col="condition",
        time_col="time_bin",
        outcome_col="value",
    )

    assert out["audit_status"] == "ready"

    assert out["duplicate_cell_count"] == []

    duplicate = pd.concat(
        [
            ready,
            ready.iloc[
                [
                    0,
                    1,
                ]
            ],
        ],
        ignore_index=True,
    )

    out_dup = r2.audit_timecourse_grid(
        duplicate,
        subject_col="subject",
        condition_col="condition",
        time_col="time_bin",
        outcome_col="value",
    )

    assert out_dup["grid_summary"].iloc[0]["n_duplicate_cells"] == 2

    assert isinstance(
        out_dup["duplicate_cell_count"],
        list,
    )


# ===========================================================================
# PREPROCESSING MULTIVERSE EDGE CONTRACT
# ===========================================================================


def test_r2_preprocessing_multiverse_single_family_and_validation():
    empty = r2.create_preprocessing_multiverse(
        include_pupil=False,
        include_aoi=False,
    )

    assert empty["pupil_grid"].empty

    assert empty["aoi_grid"].empty

    assert empty["combined_grid"].empty

    pupil_only = r2.create_preprocessing_multiverse(
        pupil_max_gap_ms=[75],
        pupil_smoothing_window_samples=[3],
        pupil_baseline_windows=[
            (
                -200,
                0,
            )
        ],
        pupil_artifact_padding_ms=[0],
        include_pupil=True,
        include_aoi=False,
        label_prefix="edge",
    )

    assert len(pupil_only["pupil_grid"]) == 1

    assert pupil_only["aoi_grid"].empty

    assert pupil_only["combined_grid"].empty

    aoi_only = r2.create_preprocessing_multiverse(
        aoi_denominators=["valid"],
        aoi_min_denominator_samples=[1],
        include_pupil=False,
        include_aoi=True,
        label_prefix="edge",
    )

    assert aoi_only["pupil_grid"].empty

    assert len(aoi_only["aoi_grid"]) == 1


# ===========================================================================
# FIXATION-ALIGNMENT ALTERNATE EVENT MODES
# ===========================================================================


def _fixalign_frame():
    return pd.DataFrame(
        {
            "participant": ["P1"] * 4,
            "trial": ["T1"] * 4,
            "time": [
                0,
                100,
                200,
                300,
            ],
            "aoi": [
                "other",
                "target",
                "target",
                "other",
            ],
            "fixation": [
                False,
                True,
                True,
                False,
            ],
            "saccade": [
                False,
                True,
                False,
                False,
            ],
            "event": [
                "",
                "",
                "go",
                "",
            ],
        }
    )


@pytest.mark.parametrize(
    ("alignment_event", "extra"),
    [
        (
            "first_target_entry",
            {
                "target_aoi": "target",
            },
        ),
        (
            "first_saccade_to_aoi",
            {
                "target_aoi": "target",
                "saccade_col": "saccade",
            },
        ),
        (
            "first_fixation",
            {},
        ),
        (
            "custom",
            {
                "event_col": "event",
                "event_value": "go",
            },
        ),
    ],
)
def test_r2_fixation_alignment_alternate_event_modes(
    alignment_event,
    extra,
):
    kwargs = dict(
        time_col="time",
        participant_col="participant",
        trial_col="trial",
        aoi_col="aoi",
        fixation_col="fixation",
        alignment_event=alignment_event,
        keep_unaligned=True,
        name="edge_fixalign",
    )

    kwargs.update(extra)

    out = r2.prepare_fixation_aligned_data(
        _fixalign_frame(),
        **kwargs,
    )

    assert out["overview"].iloc[0]["alignment_event"] == alignment_event

    assert out["event_table"].iloc[0]["gp3_has_alignment_event"]


# ===========================================================================
# MULTIMODAL ALTERNATE MERGE / SCALING MODES
# ===========================================================================


def test_r2_multimodal_unscaled_explicit_join_keys():
    face = pd.DataFrame(
        {
            "participant_id": [
                "P1",
                "P2",
            ],
            "trial_id": [
                1,
                1,
            ],
            "face_metric": [
                0.2,
                0.4,
            ],
            "cov": [
                1,
                2,
            ],
        }
    )

    gaze = pd.DataFrame(
        {
            "participant_id": [
                "P1",
                "P2",
            ],
            "trial_id": [
                1,
                1,
            ],
            "dwell": [
                0.3,
                0.5,
            ],
        }
    )

    response = pd.DataFrame(
        {
            "participant_id": [
                "P1",
                "P2",
            ],
            "trial_id": [
                1,
                1,
            ],
            "rating": [
                4,
                np.nan,
            ],
        }
    )

    out = r2.prepare_multimodal_data(
        face,
        gaze_data=gaze,
        response_data=response,
        by=[
            "participant_id",
            "trial_id",
        ],
        gaze_by=[
            "participant_id",
            "trial_id",
        ],
        response_by=[
            "participant_id",
            "trial_id",
        ],
        predictor_cols=[
            "face_metric",
            "dwell",
        ],
        outcome_cols=["rating"],
        covariate_cols=["cov"],
        scale_predictors=False,
        drop_missing_outcomes=False,
        keep_all=False,
    )

    assert len(out) == 2

    assert "face_metric_z" not in out.columns

    assert "dwell_z" not in out.columns


# ===========================================================================
# RECALIBRATION ALTERNATE METHOD / FAILURE STATUS
# ===========================================================================


def test_r2_recalibration_mean_shift_without_groups():
    frame = pd.DataFrame(
        {
            "x": [
                0.1,
                0.2,
                0.3,
            ],
            "y": [
                0.1,
                0.2,
                0.3,
            ],
            "target_x": [
                0.2,
                0.3,
                0.4,
            ],
            "target_y": [
                0.2,
                0.3,
                0.4,
            ],
        }
    )

    out = r2.recalibrate_gaze(
        frame,
        x_col="x",
        y_col="y",
        target_x_col="target_x",
        target_y_col="target_y",
        grouping_cols=None,
        method="mean_shift",
        min_valid_points=1,
        max_shift=None,
        name="edge_recalibration",
    )

    overview = out.attrs["gp3_gaze_recalibration_overview"]

    assert overview.iloc[0]["recalibration_method"] == "mean_shift"

    assert out["gaze_recalibration_status"].eq("complete").all()


def test_r2_recalibration_insufficient_targets():
    frame = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
            ],
            "x": [
                0.1,
                np.nan,
            ],
            "y": [
                0.1,
                np.nan,
            ],
            "target_x": [
                0.2,
                0.2,
            ],
            "target_y": [
                0.2,
                0.2,
            ],
        }
    )

    out = r2.recalibrate_gaze(
        frame,
        x_col="x",
        y_col="y",
        target_x_col="target_x",
        target_y_col="target_y",
        grouping_cols=["subject"],
        min_valid_points=2,
        name="edge_insufficient",
    )

    assert out["gaze_recalibration_status"].eq("insufficient_valid_targets").all()


# === R2 CONDITION QUALITY FINAL COVERAGE EDGES ===


def _r2_condition_quality_edge_frame():
    return pd.DataFrame(
        {
            "condition": [
                "A",
                "A",
                "B",
                "B",
            ],
            "subject": [
                "S1",
                "S2",
                "S3",
                "S4",
            ],
            "missing_gaze_prop": [
                0.10,
                0.20,
                0.40,
                0.50,
            ],
            "quality_score": [
                0.90,
                0.80,
                0.70,
                0.60,
            ],
            "text_quality": [
                "good",
                "good",
                "bad",
                "bad",
            ],
        }
    )


@pytest.mark.parametrize(
    ("data", "kwargs"),
    [
        (
            "not-a-data-frame",
            {},
        ),
        (
            pd.DataFrame(),
            {},
        ),
        (
            _r2_condition_quality_edge_frame(),
            {
                "condition_col": "missing_condition",
            },
        ),
        (
            _r2_condition_quality_edge_frame(),
            {
                "subject_col": "missing_subject",
            },
        ),
        (
            _r2_condition_quality_edge_frame(),
            {
                "quality_cols": [],
            },
        ),
        (
            _r2_condition_quality_edge_frame(),
            {
                "quality_cols": ["missing_quality"],
            },
        ),
        (
            _r2_condition_quality_edge_frame(),
            {
                "quality_cols": ["text_quality"],
            },
        ),
        (
            _r2_condition_quality_edge_frame(),
            {
                "min_units_per_condition": 0,
            },
        ),
        (
            _r2_condition_quality_edge_frame(),
            {
                "max_mean_difference": -0.01,
            },
        ),
        (
            _r2_condition_quality_edge_frame(),
            {
                "max_condition_ratio": 0,
            },
        ),
        (
            pd.DataFrame(
                {
                    "condition": [
                        "",
                        None,
                    ],
                    "missing_gaze_prop": [
                        0.1,
                        0.2,
                    ],
                }
            ),
            {
                "quality_cols": ["missing_gaze_prop"],
            },
        ),
        (
            pd.DataFrame(
                {
                    "condition": [
                        "A",
                        "B",
                    ],
                    "label": [
                        "x",
                        "y",
                    ],
                }
            ),
            {},
        ),
    ],
)
def test_r2_condition_quality_validation_edges(
    data,
    kwargs,
):
    with pytest.raises(
        (
            TypeError,
            ValueError,
        )
    ):
        r2.audit_condition_quality_imbalance(
            data,
            **kwargs,
        )


def test_r2_condition_quality_explicit_subject_and_metrics():
    frame = _r2_condition_quality_edge_frame()

    out = r2.audit_condition_quality_imbalance(
        frame,
        condition_col="condition",
        quality_cols=[
            "missing_gaze_prop",
            "quality_score",
        ],
        subject_col="subject",
        min_units_per_condition=2,
        max_mean_difference=0.05,
        max_condition_ratio=1.25,
        lower_is_better=["missing_gaze_prop"],
    )

    condition_summary = out["condition_summary"]

    # Canonical result is wide:
    # one row per condition with metric-specific
    # summary columns.
    assert len(condition_summary) == 2

    assert set(condition_summary["condition"]) == {
        "A",
        "B",
    }

    assert condition_summary["n_units"].eq(2).all()

    columns = set(condition_summary.columns)

    assert any(column.startswith("missing_gaze_prop_") for column in columns)

    assert any(column.startswith("quality_score_") for column in columns)

    overview = out["overview"].iloc[0]

    assert overview["n_conditions"] == 2

    assert overview["n_quality_metrics"] == 2

    assert not out["metric_summary"].empty

    assert isinstance(
        out["flagged_metrics"],
        pd.DataFrame,
    )

    assert isinstance(
        out["settings"],
        pd.DataFrame,
    )


def test_r2_condition_quality_alias_and_auto_detection():
    frame = pd.DataFrame(
        {
            "USER_FILE": [
                "S1",
                "S2",
                "S3",
                "S4",
            ],
            "condition": [
                "A",
                "A",
                "B",
                "B",
            ],
            "missing_gaze_prop": [
                0.1,
                0.2,
                0.1,
                0.2,
            ],
        }
    )

    out = r2.audit_condition_quality_imbalance(
        frame,
        condition_col="condition",
        subject_col="USER_FILE",
    )

    assert not out["condition_summary"].empty

    assert out["overview"].iloc[0]["n_conditions"] == 2
