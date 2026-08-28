from __future__ import annotations

import pandas as pd
import pytest

from gp3tools import _behavioral_r3a as r3a


def test_r3a_dual_contract_argument_and_column_helpers() -> None:
    """Routing helpers preserve kwargs and inspect tabular columns safely."""
    args = r3a._r3a_call_arguments(
        {
            "data": "sentinel",
            "kwargs": {
                "x_col": "x",
                "time_col": "time",
            },
        }
    )

    assert args["data"] == "sentinel"
    assert args["x_col"] == "x"
    assert args["time_col"] == "time"

    non_mapping = r3a._r3a_call_arguments(
        {
            "kwargs": 42,
        }
    )

    assert non_mapping["kwargs"] == 42

    frame = pd.DataFrame(
        {
            "x": [0.1],
            "y": [0.2],
        }
    )

    assert r3a._r3a_columns(frame) == {
        "x",
        "y",
    }

    assert r3a._r3a_columns(object()) == set()


def test_r3a_master_dual_contract_routing_edges() -> None:
    """Raw Gazepoint exports use R parity; standardized Python inputs do not."""
    canonical = pd.DataFrame(
        {
            "USER_FILE": ["S1.csv"],
            "MEDIA_ID": ["stimA"],
            "TIME": [0.0],
        }
    )

    assert r3a._should_use_r3a(
        "create_gazepoint_master",
        {
            "data": canonical,
        },
    )

    assert r3a._should_use_r3a(
        "create_gazepoint_master",
        {
            "data": pd.DataFrame(),
            "gaze_data": canonical,
        },
    )

    assert not r3a._should_use_r3a(
        "create_gazepoint_master",
        {
            "data": canonical.drop(columns=["USER_FILE"]),
        },
    )

    assert not r3a._should_use_r3a(
        "create_gazepoint_master",
        {
            "data": canonical.drop(columns=["MEDIA_ID"]),
        },
    )


def test_r3a_ivt_dual_contract_routing_edges() -> None:
    """Explicit canonical column declarations select the R I-VT implementation."""
    samples = pd.DataFrame(
        {
            "x": [0.1, 0.1],
            "y": [0.2, 0.2],
            "time": [0.0, 0.02],
        }
    )

    canonical = {
        "data": samples,
        "x_col": "x",
        "y_col": "y",
        "time_col": "time",
    }

    assert r3a._should_use_r3a(
        "detect_gazepoint_fixations_ivt",
        canonical,
    )

    assert not r3a._should_use_r3a(
        "detect_gazepoint_fixations_ivt",
        {
            **canonical,
            "x_col": None,
        },
    )

    assert not r3a._should_use_r3a(
        "detect_gazepoint_fixations_ivt",
        {
            **canonical,
            "x_col": "missing_x",
        },
    )

    assert not r3a._should_use_r3a(
        "detect_gazepoint_fixations_ivt",
        {
            "data": samples,
        },
    )


def test_r3a_face_window_dual_contract_routing_edges() -> None:
    """Explicit window tables select R semantics; convenience calls stay Python."""
    data = pd.DataFrame(
        {
            "time": [0.0, 0.1],
            "condition": ["A", "A"],
        }
    )

    windows = pd.DataFrame(
        {
            "window_start": [0.0],
            "window_end": [0.1],
        }
    )

    assert r3a._should_use_r3a(
        "summarize_gazepoint_face_windows",
        {
            "data": data,
            "windows": windows,
        },
    )

    assert not r3a._should_use_r3a(
        "summarize_gazepoint_face_windows",
        {
            "data": data,
            "windows": None,
        },
    )


def test_r3a_binocular_sensitivity_prefix_routing_edges() -> None:
    """Reconstructed binocular columns are required for canonical R sensitivity."""
    reconstructed = pd.DataFrame(
        {
            "gp3_binocular_left_final": [3.0],
            "gp3_binocular_right_final": [3.1],
            "gp3_binocular_left_reconstructed": [False],
            "gp3_binocular_right_reconstructed": [False],
        }
    )

    assert r3a._should_use_r3a(
        "analyse_gazepoint_binocular_sensitivity",
        {
            "data": reconstructed,
            "prefix": "gp3_binocular",
        },
    )

    assert not r3a._should_use_r3a(
        "analyse_gazepoint_binocular_sensitivity",
        {
            "data": reconstructed.drop(
                columns=[
                    "gp3_binocular_right_reconstructed",
                ]
            ),
            "prefix": "gp3_binocular",
        },
    )

    custom = reconstructed.rename(
        columns={
            name: name.replace(
                "gp3_binocular",
                "custom",
            )
            for name in reconstructed.columns
        }
    )

    assert r3a._should_use_r3a(
        "analyse_gazepoint_binocular_sensitivity",
        {
            "data": custom,
            "prefix": "custom",
        },
    )


def test_r3a_event_agreement_dataframe_and_bundle_routing_edges() -> None:
    """Both canonical event frames and comparison bundles route to R semantics."""
    events = pd.DataFrame(
        {
            "detector": ["ivt"],
            "start_time": [0.0],
            "end_time": [0.1],
            "duration_ms": [100.0],
        }
    )

    assert r3a._should_use_r3a(
        "summarise_gazepoint_event_detector_agreement",
        {
            "data": events,
        },
    )

    assert r3a._should_use_r3a(
        "summarise_gazepoint_event_detector_agreement",
        {
            "x": {
                "events": events,
            },
        },
    )

    assert not r3a._should_use_r3a(
        "summarise_gazepoint_event_detector_agreement",
        {
            "data": events.drop(columns=["duration_ms"]),
        },
    )

    assert not r3a._should_use_r3a(
        "summarise_gazepoint_event_detector_agreement",
        {
            "x": {
                "events": events.drop(columns=["detector"]),
            },
        },
    )

    assert not r3a._should_use_r3a(
        "summarise_gazepoint_event_detector_agreement",
        {
            "x": "legacy-result",
        },
    )


def test_r3a_face_reactivity_window_routing_edges() -> None:
    """Canonical reactivity requires both baseline and response windows."""
    data = pd.DataFrame(
        {
            "time": [0.0],
        }
    )

    assert r3a._should_use_r3a(
        "summarize_gazepoint_face_reactivity",
        {
            "data": data,
            "baseline_window": (-0.2, 0.0),
            "response_window": (0.0, 0.5),
        },
    )

    assert not r3a._should_use_r3a(
        "summarize_gazepoint_face_reactivity",
        {
            "data": data,
            "baseline_window": (-0.2, 0.0),
            "response_window": None,
        },
    )

    assert not r3a._should_use_r3a(
        "summarize_gazepoint_face_reactivity",
        {
            "data": data,
            "baseline_window": None,
            "response_window": (0.0, 0.5),
        },
    )


def test_r3a_binocular_reporting_metadata_routing_edges() -> None:
    """Only reconstructed-data metadata selects canonical R reporting."""
    reconstructed = pd.DataFrame(
        {
            "left": [3.0],
            "right": [3.1],
        }
    )

    reconstructed.attrs["gp3_binocular_reconstruction"] = {
        "prefix": "gp3_binocular",
    }

    assert r3a._should_use_r3a(
        "summarise_gazepoint_binocular_reporting",
        {
            "data": reconstructed,
        },
    )

    plain = reconstructed.copy()
    plain.attrs.clear()

    assert not r3a._should_use_r3a(
        "summarise_gazepoint_binocular_reporting",
        {
            "data": plain,
        },
    )

    malformed = reconstructed.copy()
    malformed.attrs["gp3_binocular_reconstruction"] = "legacy"

    assert not r3a._should_use_r3a(
        "summarise_gazepoint_binocular_reporting",
        {
            "data": malformed,
        },
    )


def test_r3a_selector_default_and_raw_mad_semantics() -> None:
    """Unspecialized R3-A exports default to canonical routing; MAD is raw."""
    assert r3a._should_use_r3a(
        "summarize_gazepoint_face_quality",
        {
            "data": pd.DataFrame(),
        },
    )

    values = pd.Series(
        [
            1.0,
            2.0,
            100.0,
        ]
    )

    assert r3a._r3a_mad(values) == pytest.approx(1.0)

    assert pd.isna(
        r3a._r3a_mad(
            pd.Series(
                [
                    float("nan"),
                    float("nan"),
                ]
            )
        )
    )


def test_r3a_column_validation_edge() -> None:
    """The canonical helper reports all absent required columns."""
    frame = pd.DataFrame(
        {
            "x": [1.0],
        }
    )

    r3a._check_cols(
        frame,
        [
            "x",
        ],
    )

    with pytest.raises(
        ValueError,
        match="Missing columns: y, time",
    ):
        r3a._check_cols(
            frame,
            [
                "x",
                "y",
                "time",
            ],
        )


# === R3A FINAL RESIDUAL EDGES ===


def test_r3a_frame_and_list_coercion_edges() -> None:
    """R3-A coercion helpers preserve frames and normalize scalar/list inputs."""
    frame = pd.DataFrame(
        {
            "x": [
                1.0,
            ]
        }
    )

    copied = r3a._as_frame(frame)

    assert copied.equals(frame)

    assert copied is not frame

    with pytest.raises(
        TypeError,
        match="must be a data frame",
    ):
        r3a._as_frame(
            object(),
            arg="samples",
        )

    assert r3a._as_list(None) == []

    assert r3a._as_list("one") == ["one"]

    assert r3a._as_list(
        (
            "a",
            "b",
        )
    ) == [
        "a",
        "b",
    ]

    assert r3a._as_list(42) == [42]

    assert (
        r3a._first_not_none(
            None,
            "selected",
            "later",
        )
        == "selected"
    )

    assert (
        r3a._first_not_none(
            None,
            None,
        )
        is None
    )


def test_r3a_empty_numeric_summary_semantics() -> None:
    """R-style summary helpers return NA where R would have no usable values."""
    empty = pd.Series(
        [],
        dtype=float,
    )

    singleton = pd.Series(
        [
            5.0,
        ],
        dtype=float,
    )

    assert pd.isna(r3a._r_sd(singleton))

    assert pd.isna(r3a._median(empty))

    assert pd.isna(r3a._mean(empty))

    assert pd.isna(r3a._min(empty))

    assert pd.isna(r3a._max(empty))


def test_r3a_event_iou_empty_disjoint_and_degenerate_edges() -> None:
    """Event overlap handles absent, disjoint, and zero-width comparison events."""
    row = pd.Series(
        {
            "start_time": 0.0,
            "end_time": 1.0,
        }
    )

    empty = pd.DataFrame(
        columns=[
            "start_time",
            "end_time",
        ]
    )

    assert r3a._event_iou(
        row,
        empty,
    ) == pytest.approx(0.0)

    disjoint = pd.DataFrame(
        {
            "start_time": [
                2.0,
            ],
            "end_time": [
                3.0,
            ],
        }
    )

    assert r3a._event_iou(
        row,
        disjoint,
    ) == pytest.approx(0.0)

    degenerate_row = pd.Series(
        {
            "start_time": 1.0,
            "end_time": 1.0,
        }
    )

    degenerate_other = pd.DataFrame(
        {
            "start_time": [
                1.0,
            ],
            "end_time": [
                1.0,
            ],
        }
    )

    assert r3a._event_iou(
        degenerate_row,
        degenerate_other,
    ) == pytest.approx(0.0)


def test_r3a_sampling_rate_detection_short_duplicate_and_regular_edges() -> None:
    """Sampling-rate inference rejects insufficient/duplicate time grids."""
    regular = pd.Series(
        [
            0.00,
            0.01,
            0.02,
            0.03,
        ]
    )

    assert r3a._detect_sampling_rate(regular) == pytest.approx(100.0)

    assert pd.isna(
        r3a._detect_sampling_rate(
            pd.Series(
                [
                    0.0,
                ]
            )
        )
    )

    assert pd.isna(
        r3a._detect_sampling_rate(
            pd.Series(
                [
                    0.0,
                    0.0,
                    0.0,
                ]
            )
        )
    )


def test_r3a_group_key_empty_missing_and_mixed_edges() -> None:
    """Canonical group keys render empty groups and missing values deterministically."""
    assert r3a._r3a_group_key({}) == ""

    assert (
        r3a._r3a_group_key(
            {
                "condition": pd.NA,
                "subject": "S1",
            }
        )
        == "condition=NA||subject=S1"
    )


def test_r3a_face_measure_column_explicit_and_inferred_edges() -> None:
    """Face-measure selection supports explicit columns and numeric discovery."""
    frame = pd.DataFrame(
        {
            "time": [
                0.0,
                0.1,
            ],
            "condition": [
                "A",
                "A",
            ],
            "face_valid": [
                True,
                True,
            ],
            "face_confidence": [
                0.95,
                0.90,
            ],
            "AU01": [
                0.1,
                0.2,
            ],
            "AU12": [
                0.3,
                0.4,
            ],
            "label": [
                "neutral",
                "smile",
            ],
        }
    )

    explicit = r3a._face_measure_columns(
        frame,
        [
            "AU12",
        ],
        set(),
    )

    assert explicit == ["AU12"]

    inferred = r3a._face_measure_columns(
        frame,
        None,
        {
            "time",
            "condition",
            "face_valid",
            "face_confidence",
        },
    )

    assert inferred == [
        "AU01",
        "AU12",
    ]

    with pytest.raises(
        ValueError,
        match="Missing columns",
    ):
        r3a._face_measure_columns(
            frame,
            [
                "not_a_measure",
            ],
            set(),
        )
