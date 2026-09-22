from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

aoi = importlib.import_module("gp3tools.aoi")
pupil = importlib.import_module("gp3tools.pupil")
qc = importlib.import_module("gp3tools.qc")


# ============================================================================
# AOI — LEGACY DYNAMIC AOI
# ============================================================================


def test_h11_dynamic_aoi_legacy_success():
    gaze = pd.DataFrame(
        {
            "time": [0.0, 1.0],
            "x": [0.5, 0.9],
            "y": [0.5, 0.9],
        }
    )

    geometry = pd.DataFrame(
        {
            "aoi_time": [0.0],
            "aoi": ["target"],
            "xmin": [0.0],
            "xmax": [0.75],
            "ymin": [0.0],
            "ymax": [0.75],
        }
    )

    result = aoi.add_gazepoint_dynamic_aoi(
        gaze,
        geometry,
        time_col="time",
        x_col="x",
        y_col="y",
        aoi_time_col="aoi_time",
        tolerance=2,
    )

    assert result["aoi_current"].tolist() == [
        "target",
        "outside",
    ]


def test_h11_dynamic_aoi_legacy_bad_geometry():
    gaze = pd.DataFrame(
        {
            "time": [0.0],
            "x": [0.5],
            "y": [0.5],
        }
    )

    geometry = pd.DataFrame(
        {
            "aoi_time": [0.0],
            "xmin": [0.0],
            "xmax": [1.0],
            "ymin": [0.0],
            "ymax": [1.0],
        }
    )

    with pytest.raises(ValueError):
        aoi.add_gazepoint_dynamic_aoi(
            gaze,
            geometry,
            time_col="time",
            x_col="x",
            y_col="y",
            aoi_time_col="aoi_time",
        )


def test_h11_dynamic_aoi_nan_gap_validation():
    frame = pd.DataFrame(
        {
            "TIME": [0.0],
            "FPOGX": [0.5],
            "FPOGY": [0.5],
        }
    )

    defs = pd.DataFrame(
        {
            "aoi_time": [0.0],
            "aoi_name": ["A"],
            "left": [0.0],
            "right": [1.0],
            "top": [0.0],
            "bottom": [1.0],
        }
    )

    with pytest.raises(ValueError):
        aoi.add_gazepoint_dynamic_aoi(
            frame,
            defs,
            shape="rectangle",
            max_time_gap=np.nan,
        )


# ============================================================================
# AOI — LEGACY GEOMETRY + OVERLAP
# ============================================================================


def test_h11_geometry_legacy_missing_columns():
    result = aoi.audit_gazepoint_aoi_geometry(
        pd.DataFrame(
            {
                "xmin": [0.0],
            }
        )
    )

    assert not result["valid"]
    assert result["issues"]


def test_h11_geometry_legacy_invalid_rectangle():
    result = aoi.audit_gazepoint_aoi_geometry(
        pd.DataFrame(
            {
                "xmin": [1.0],
                "xmax": [0.0],
                "ymin": [0.0],
                "ymax": [1.0],
            }
        )
    )

    assert not result["valid"]
    assert result["summary"].loc[0, "n_issues"] == 1


def test_h11_geometry_data_alias_and_conflict():
    geometry = pd.DataFrame(
        {
            "xmin": [0.0],
            "xmax": [1.0],
            "ymin": [0.0],
            "ymax": [1.0],
        }
    )

    result = aoi.audit_gazepoint_aoi_geometry(
        data=geometry,
    )

    assert result["valid"]

    with pytest.raises(TypeError):
        aoi.audit_gazepoint_aoi_geometry(
            geometry,
            data=geometry,
        )


def test_h11_overlap_legacy_index_names():
    geometry = pd.DataFrame(
        {
            "xmin": [0.0, 0.5],
            "xmax": [1.0, 1.5],
            "ymin": [0.0, 0.5],
            "ymax": [1.0, 1.5],
        }
    )

    result = aoi.audit_gazepoint_aoi_overlap(geometry)

    assert len(result) == 1
    assert result.loc[0, "aoi1"] == 0
    assert result.loc[0, "aoi2"] == 1
    assert result.loc[0, "overlap_area"] > 0


# ============================================================================
# AOI — PRECOMPUTED TRIAL FEATURES
# ============================================================================


def _precomputed_entries():
    return pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S1",
            ],
            # Precomputed entries still feed the transition-feature
            # calculation, whose canonical default time column is `time`.
            # Entry starts are the deterministic equivalent timestamps.
            "time": [
                0.0,
                10.0,
                20.0,
            ],
            "aoi_state": [
                "target",
                "outside",
                "distractor",
            ],
            "entry_start_time": [
                0.0,
                10.0,
                20.0,
            ],
            "entry_end_time": [
                10.0,
                20.0,
                30.0,
            ],
            "entry_duration_ms": [
                10.0,
                10.0,
                10.0,
            ],
            "n_samples": [
                2,
                2,
                2,
            ],
            "is_non_aoi": [
                False,
                pd.NA,
                False,
            ],
        }
    )


def test_h11_trial_features_precomputed_entries():
    result = aoi.summarise_gazepoint_aoi_trial_features(
        _precomputed_entries(),
        group_cols=["subject"],
        target_aoi_values=["target"],
        distractor_aoi_values=["distractor"],
        non_aoi_values=["outside"],
    )

    assert len(result) == 1
    assert (
        result.loc[
            0,
            "aoi_trial_feature_status",
        ]
        == "ok"
    )


def test_h11_trial_features_precomputed_missing_group():
    with pytest.raises(ValueError):
        aoi.summarise_gazepoint_aoi_trial_features(
            _precomputed_entries().drop(columns=["subject"]),
            group_cols=["subject"],
        )


def test_h11_trial_features_start_time_filter():
    frame = _precomputed_entries()
    frame["entry_start_time"] = np.nan

    with pytest.raises(ValueError):
        aoi.summarise_gazepoint_aoi_trial_features(
            frame,
            group_cols=["subject"],
        )


# ============================================================================
# AOI — WINDOW NORMALIZATION
# ============================================================================


def test_h11_windows_missing_subject_condition_and_aoi_labels():
    frame = pd.DataFrame(
        {
            "subject": ["", "", "S2"],
            "condition": ["", pd.NA, "C2"],
            "time": [0.0, 10.0, 20.0],
            "aoi_current": [
                pd.NA,
                "target",
                "other",
            ],
        }
    )

    result = aoi.summarise_gazepoint_aoi_windows(
        frame,
        windows=[0, 30],
        target_aoi_values=["target"],
        distractor_aoi_values=["distractor"],
    )

    assert "unknown_subject" in set(result["subject"].astype(str))

    assert "all_data" in set(result["condition"].astype(str))


def test_h11_windows_right_endpoint():
    frame = pd.DataFrame(
        {
            "subject": ["S1", "S1"],
            "time": [0.0, 10.0],
            "aoi_current": ["target", "target"],
        }
    )

    result = aoi.summarise_gazepoint_aoi_windows(
        frame,
        windows=[0, 10],
        group_cols=["subject"],
        condition_col=None,
        target_aoi_values=["target"],
        include_right_endpoint=True,
    )

    assert (
        result.loc[
            0,
            "n_window_samples",
        ]
        == 2
    )


# ============================================================================
# AOI — EMPIRICAL LOGIT
# ============================================================================


def test_h11_empirical_logit_status_matrix():
    frame = pd.DataFrame(
        {
            "num": [
                1.0,
                np.nan,
                1.0,
                0.0,
                -1.0,
                1.0,
            ],
            "den": [
                2.0,
                2.0,
                np.nan,
                0.0,
                0.0,
                0.0,
            ],
        }
    )

    result = aoi.transform_gazepoint_aoi_empirical_logit(
        frame,
        numerator_col="num",
        denominator_col="den",
    )

    statuses = set(result["aoi_empirical_logit_status"])

    assert "complete" in statuses
    assert "missing_or_nonfinite_numerator" in statuses
    assert "missing_or_nonfinite_denominator" in statuses
    assert "invalid_denominator" in statuses
    assert "invalid_numerator" in statuses
    assert "numerator_exceeds_denominator" in statuses


def test_h11_empirical_logit_proportion_denominator_modes():
    frame = pd.DataFrame(
        {
            "prop": [0.25, 0.75],
            "den": [20, 20],
        }
    )

    observed = aoi.transform_gazepoint_aoi_empirical_logit(
        frame,
        proportion_col="prop",
        denominator_col="den",
    )

    assert (
        observed.attrs["gp3_empirical_logit_overview"].loc[
            0,
            "denominator_source",
        ]
        == "observed_denominator_from_proportion"
    )

    pseudo = aoi.transform_gazepoint_aoi_empirical_logit(
        frame[["prop"]],
        proportion_col="prop",
        pseudo_denominator=10,
    )

    assert (
        pseudo.attrs["gp3_empirical_logit_overview"].loc[
            0,
            "denominator_source",
        ]
        == "pseudo_denominator_from_proportion"
    )


def test_h11_empirical_logit_output_validation():
    frame = pd.DataFrame(
        {
            "prop": [0.5],
        }
    )

    with pytest.raises(ValueError):
        aoi.transform_gazepoint_aoi_empirical_logit(
            frame,
            proportion_col="prop",
            output_col="",
        )

    with pytest.raises(ValueError):
        aoi.transform_gazepoint_aoi_empirical_logit(
            frame,
            proportion_col="prop",
            output_col="dup",
            adjusted_proportion_col="dup",
        )

    existing = frame.copy()
    existing["aoi_empirical_logit"] = 1

    with pytest.raises(ValueError):
        aoi.transform_gazepoint_aoi_empirical_logit(
            existing,
            proportion_col="prop",
        )


# ============================================================================
# AOI — RECURRENCE
# ============================================================================


def test_h11_recurrence_r_short_and_repeated():
    short = aoi.compute_gazepoint_sequence_recurrence(
        sequence=["A"],
        min_line=3,
    )

    assert (
        short.loc[
            0,
            "recurrence_status",
        ]
        == "too_short"
    )

    repeated = aoi.compute_gazepoint_sequence_recurrence(
        sequence=[
            "A",
            "B",
            "A",
            "B",
        ],
        min_line=2,
        include_missing=True,
    )

    assert (
        repeated.loc[
            0,
            "recurrence_points",
        ]
        > 0
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "min_line": 0,
        },
        {
            "include_missing": "yes",
        },
        {
            "missing_label": "",
        },
    ],
)
def test_h11_recurrence_validation(kwargs):
    with pytest.raises(ValueError):
        aoi.compute_gazepoint_sequence_recurrence(
            sequence=["A", "B"],
            data=pd.DataFrame({"aoi": ["A", "B"]}),
            aoi_col="aoi",
            **kwargs,
        )


# ============================================================================
# AOI — SCANPATH INTERNAL CONTRACTS
# ============================================================================


def test_h11_scanpath_sequences_missing_and_collapse():
    frame = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S1",
                "S2",
            ],
            "time": [0, 1, 2, 0],
            "aoi": [
                "A",
                "A",
                pd.NA,
                "B",
            ],
        }
    )

    ids, sequences = aoi._gp3_scanpath_r_sequences(
        frame,
        aoi_col="aoi",
        group_cols=["subject"],
        time_col="time",
        include_missing=True,
        missing_label="<M>",
        collapse_repeats=True,
        max_sequences=2,
    )

    assert len(ids) == 2
    assert sequences[0] == [
        "A",
        "<M>",
    ]


def test_h11_scanpath_sequences_validation():
    frame = pd.DataFrame(
        {
            "subject": ["S1"],
            "aoi": ["A"],
        }
    )

    with pytest.raises(ValueError):
        aoi._gp3_scanpath_r_sequences(
            frame,
            aoi_col="",
            group_cols=["subject"],
            time_col=None,
            include_missing=False,
            missing_label="missing",
            collapse_repeats=False,
            max_sequences=2,
        )

    with pytest.raises(ValueError):
        aoi._gp3_scanpath_r_sequences(
            frame,
            aoi_col="aoi",
            group_cols=[],
            time_col=None,
            include_missing=False,
            missing_label="missing",
            collapse_repeats=False,
            max_sequences=2,
        )

    with pytest.raises(ValueError):
        aoi._gp3_scanpath_r_sequences(
            frame,
            aoi_col="aoi",
            group_cols=["subject"],
            time_col=None,
            include_missing=False,
            missing_label="missing",
            collapse_repeats=False,
            max_sequences=1,
        )


def _distance3():
    return pd.DataFrame(
        [
            [0.0, 1.0, 2.0],
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 0.0],
        ],
        index=["s1", "s2", "s3"],
        columns=["s1", "s2", "s3"],
    )


def test_h11_distance_matrix_validation():
    with pytest.raises(ValueError):
        aoi._gp3_scanpath_r_validate_distance_matrix([[0, 1, 2], [1, 0, 1]])

    with pytest.raises(ValueError):
        aoi._gp3_scanpath_r_validate_distance_matrix([[0, np.nan], [np.nan, 0]])

    with pytest.raises(ValueError):
        aoi._gp3_scanpath_r_validate_distance_matrix([[0, -1], [-1, 0]])

    with pytest.raises(ValueError):
        aoi._gp3_scanpath_r_validate_distance_matrix([[0, 1], [2, 0]])

    with pytest.raises(ValueError):
        aoi._gp3_scanpath_r_validate_distance_matrix([[1, 0], [0, 1]])

    with pytest.raises(ValueError):
        aoi._gp3_scanpath_r_validate_distance_matrix(
            [[0, 1], [1, 0]],
            labels=["x", "x"],
        )


def test_h11_cluster_matrix_pam():
    labels, model, medoids = aoi._gp3_scanpath_r_cluster_matrix(
        _distance3(),
        k=2,
        method="pam",
        linkage="average",
    )

    assert len(labels) == 3
    assert model["method"] == "pam"
    assert len(medoids) == 2


def test_h11_cluster_matrix_validation():
    with pytest.raises(ValueError):
        aoi._gp3_scanpath_r_cluster_matrix(
            pd.DataFrame(
                [
                    [0, 1],
                    [1, 0],
                ]
            ),
            k=2,
            method="pam",
            linkage="average",
        )

    with pytest.raises(ValueError):
        aoi._gp3_scanpath_r_cluster_matrix(
            _distance3(),
            k=1,
            method="pam",
            linkage="average",
        )

    with pytest.raises(ValueError):
        aoi._gp3_scanpath_r_cluster_matrix(
            _distance3(),
            k=2,
            method="hierarchical",
            linkage="bad",
        )

    with pytest.raises(ValueError):
        aoi._gp3_scanpath_r_cluster_matrix(
            _distance3(),
            k=2,
            method="bad",
            linkage="average",
        )


def test_h11_pairwise_distance_incomplete():
    pairs = pd.DataFrame(
        {
            "sequence_a": ["s1"],
            "sequence_b": ["s2"],
            "normalized_distance": [1.0],
        }
    )

    # Two labels have exactly one pair, so this is complete.
    matrix = aoi._gp3_scanpath_r_pairs_to_matrix(
        pairs,
        "normalized_distance",
    )

    assert matrix.shape == (
        2,
        2,
    )

    incomplete = pd.DataFrame(
        {
            "sequence_a": [
                "s1",
                "s1",
            ],
            "sequence_b": [
                "s2",
                "s3",
            ],
            "normalized_distance": [
                1.0,
                2.0,
            ],
        }
    )

    with pytest.raises(ValueError):
        aoi._gp3_scanpath_r_pairs_to_matrix(
            incomplete,
            "normalized_distance",
        )


# ============================================================================
# PUPIL — HAMPEL EDGE CONTRACT
# ============================================================================


def test_h11_hampel_empty_and_ungrouped():
    with pytest.raises(ValueError):
        pupil.flag_gazepoint_pupil_hampel(
            pd.DataFrame({"pupil": pd.Series(dtype=float)}),
            pupil_col="pupil",
        )

    result = pupil.flag_gazepoint_pupil_hampel(
        pd.DataFrame(
            {
                "pupil": [
                    1.0,
                    1.0,
                    1.0,
                ]
            }
        ),
        pupil_col="pupil",
        window_size_samples=3,
        min_valid_samples=2,
    )

    assert len(result) == 3


# ============================================================================
# PUPIL — BLINK ALIASES + RETURN MODES
# ============================================================================


def _blink_frame():
    return pd.DataFrame(
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
    )


def test_h11_blink_return_alias_samples():
    result = pupil.detect_gazepoint_blinks(
        _blink_frame(),
        pupil_col="pupil",
        min_duration_ms=0,
        time_unit="seconds",
        include_rapid_changes=False,
        **{"return": "samples"},
    )

    assert len(result) == 3


def test_h11_blink_argument_conflicts():
    with pytest.raises(TypeError):
        pupil.detect_gazepoint_blinks(
            _blink_frame(),
            pupil_col="pupil",
            return_mode="events",
            **{"return": "samples"},
        )

    with pytest.raises(TypeError):
        pupil.detect_gazepoint_blinks(
            _blink_frame(),
            pupil_col="pupil",
            impossible=True,
        )


def test_h11_blink_auto_pupil_detection():
    result = pupil.detect_gazepoint_blinks(
        _blink_frame(),
        pupil_col=None,
        min_duration_ms=0,
        time_unit="seconds",
        include_rapid_changes=False,
        return_mode="samples",
    )

    assert len(result) == 3


# ============================================================================
# PUPIL — LUMINANCE READER BRANCHES
# ============================================================================


def test_h11_luminance_grayscale(monkeypatch, tmp_path):
    import matplotlib.image as mpimg

    path = tmp_path / "gray.png"
    path.write_bytes(b"x")

    monkeypatch.setattr(
        mpimg,
        "imread",
        lambda _: np.array(
            [
                [0.0, 0.5],
                [1.0, 0.25],
            ],
            dtype=float,
        ),
    )

    result = pupil._gp3_final_read_luminance(
        "gray",
        str(path),
        None,
        True,
    )

    assert result["luminance_available"]
    assert result["luminance_status"] == "available"


def test_h11_luminance_integer_rgb(monkeypatch, tmp_path):
    import matplotlib.image as mpimg

    path = tmp_path / "rgb.png"
    path.write_bytes(b"x")

    monkeypatch.setattr(
        mpimg,
        "imread",
        lambda _: np.array(
            [
                [
                    [0, 0, 0],
                    [255, 255, 255],
                ]
            ],
            dtype=np.uint8,
        ),
    )

    result = pupil._gp3_final_read_luminance(
        "rgb",
        str(path),
        None,
        True,
    )

    assert result["luminance_available"]


def test_h11_luminance_read_error(monkeypatch, tmp_path):
    import matplotlib.image as mpimg

    path = tmp_path / "bad.png"
    path.write_bytes(b"x")

    monkeypatch.setattr(
        mpimg,
        "imread",
        lambda _: np.array(
            [1, 2, 3],
            dtype=float,
        ),
    )

    result = pupil._gp3_final_read_luminance(
        "bad",
        str(path),
        None,
        True,
    )

    assert result["luminance_status"] == "read_error"

    assert result["error_message"]


# ============================================================================
# PUPIL — GAP FLAG COERCION
# ============================================================================


def _gap_frame(flag_values):
    return pd.DataFrame(
        {
            "subject": ["S1", "S1"],
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
            "pupil_was_interpolated": flag_values,
            "pupil_interpolated": [
                3.0,
                3.1,
            ],
        }
    )


@pytest.mark.parametrize(
    "flags",
    [
        [0, 1],
        ["no", "yes"],
    ],
)
def test_h11_gap_flag_coercions(flags):
    result = pupil.audit_gazepoint_pupil_gaps(
        _gap_frame(flags),
        group_cols="subject",
    )

    assert (
        result.loc[
            0,
            "n_gaps_interpolated",
        ]
        == 1
    )


# ============================================================================
# PUPIL — WINDOWS
# ============================================================================


def _pupil_window_frame():
    return pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S1",
            ],
            "time": [
                0.0,
                500.0,
                1000.0,
            ],
            "pupil": [
                1.0,
                np.nan,
                3.0,
            ],
        }
    )


def test_h11_pupil_windows_numeric():
    result = pupil.summarise_gazepoint_pupil_windows(
        _pupil_window_frame(),
        windows=[
            0,
            500,
            1000,
        ],
        group_cols="subject",
        min_valid_samples=2,
    )

    assert not result.empty
    assert "pupil_window_status" in result


def test_h11_pupil_windows_include_endpoint():
    result = pupil.summarise_gazepoint_pupil_windows(
        _pupil_window_frame(),
        windows=[
            0,
            1000,
        ],
        group_cols="subject",
        include_window_end=True,
    )

    assert (
        result.loc[
            0,
            "n_samples",
        ]
        == 3
    )


def test_h11_pupil_windows_no_rows_in_windows():
    result = pupil.summarise_gazepoint_pupil_windows(
        _pupil_window_frame(),
        windows=[
            5000,
            6000,
        ],
        group_cols="subject",
    )

    assert result.empty


def test_h11_pupil_window_validation():
    frame = _pupil_window_frame()

    with pytest.raises(ValueError):
        pupil.summarise_gazepoint_pupil_windows(
            frame,
            windows=[
                0,
                0,
            ],
            group_cols="subject",
        )

    with pytest.raises(KeyError):
        pupil.summarise_gazepoint_pupil_windows(
            frame,
            windows=pd.DataFrame({"label": ["x"]}),
            group_cols="subject",
        )

    with pytest.raises(ValueError):
        pupil.summarise_gazepoint_pupil_windows(
            frame,
            windows=pd.DataFrame(
                {
                    "start": [0],
                    "end": [np.nan],
                }
            ),
            group_cols="subject",
        )


# ============================================================================
# PUPIL — R GROUP HELPERS
# ============================================================================


def test_h11_pupil_group_helpers():
    assert (
        pupil._gp3_pupil_r_group_cols(
            None,
            None,
        )
        == []
    )

    assert pupil._gp3_pupil_r_group_cols(
        "USER_ID",
        [
            "condition",
            "USER_ID",
        ],
    ) == [
        "USER_ID",
        "condition",
    ]

    empty = pupil._gp3_pupil_r_group_positions(
        pd.DataFrame(),
        [],
    )

    assert empty == []

    ungrouped = pupil._gp3_pupil_r_group_positions(
        pd.DataFrame(
            {
                "x": [
                    1,
                    2,
                ]
            }
        ),
        [],
    )

    assert len(ungrouped) == 1
    assert ungrouped[0].tolist() == [
        0,
        1,
    ]


# ============================================================================
# PUPIL — R DOWNSAMPLING
# ============================================================================


def _downsample_frame():
    return pd.DataFrame(
        {
            "USER_ID": [
                "S1",
                "S1",
                "S1",
                "S1",
            ],
            "TIME": [
                0.0,
                0.01,
                0.02,
                0.03,
            ],
            "pupil": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
        }
    )


def test_h11_downsample_first_mode():
    result = pupil.downsample_gazepoint_pupil(
        _downsample_frame(),
        factor=2,
        pupil_cols=["pupil"],
        method="first",
        keep_bin=True,
    )

    assert len(result) == 2


def test_h11_downsample_validation():
    frame = _downsample_frame()

    with pytest.raises(TypeError):
        pupil.downsample_gazepoint_pupil(
            frame,
            master_df=frame,
            factor=2,
        )

    with pytest.raises(ValueError):
        pupil.downsample_gazepoint_pupil(
            frame,
            factor=True,
        )

    with pytest.raises(ValueError):
        pupil.downsample_gazepoint_pupil(
            frame,
            factor=2,
            method="bad",
        )

    with pytest.raises(ValueError):
        pupil.downsample_gazepoint_pupil(
            frame,
            factor=2,
            pupil_cols=["missing"],
        )


# ============================================================================
# PUPIL — BINOCULAR FIT EDGE REASONS
# ============================================================================


def test_h11_binoc_zero_predictor_variance():
    result = pupil._gp3_binoc_r_fit_one(
        [
            1.0,
            1.0 + 1e-10,
            1.0 + 2e-10,
        ],
        [
            1.0,
            2.0,
            3.0,
        ],
        2,
        2,
        None,
        True,
        None,
    )

    assert result["reason"] == "zero_predictor_variance"


def test_h11_binoc_zero_outcome_variance():
    result = pupil._gp3_binoc_r_fit_one(
        [
            1.0,
            2.0,
            3.0,
        ],
        [
            1.0,
            1.0 + 1e-10,
            1.0 + 2e-10,
        ],
        2,
        2,
        None,
        True,
        None,
    )

    assert result["reason"] == "zero_outcome_variance"


def test_h11_binoc_level_specs_primary_fallback():
    result = pupil._gp3_binoc_r_level_specs(
        ["subject"],
        None,
    )

    assert result == [
        ["subject"],
        [],
    ]


def test_h11_binoc_assign_empty_models():
    frame = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S2",
            ]
        }
    )

    calibration = {
        "models": pd.DataFrame(),
        "levels": [],
    }

    result = pupil._gp3_binoc_r_assign_models(
        frame,
        calibration,
        "left_from_right",
    )

    assert result.tolist() == [
        -1,
        -1,
    ]


# ============================================================================
# QC — SAMPLING + TRACKLOSS
# ============================================================================


def test_h11_sampling_no_positive_diffs():
    result = qc.check_sampling_rate(
        pd.DataFrame({"time": [0.0]}),
        time_col="time",
    )

    assert np.isnan(
        result.loc[
            0,
            "sampling_hz",
        ]
    )

    assert not bool(
        result.loc[
            0,
            "within_tolerance",
        ]
    )


def test_h11_clean_trackloss_missing_validity():
    frame = pd.DataFrame(
        {
            "x": [
                1,
                2,
            ]
        }
    )

    result = qc.clean_gazepoint_by_trackloss(
        frame,
        validity_col="missing",
    )

    pd.testing.assert_frame_equal(
        result,
        frame,
    )


# ============================================================================
# QC — SCREEN BOUNDS
# ============================================================================


def test_h11_screen_bounds_missing_group():
    with pytest.raises(ValueError):
        qc.audit_gazepoint_screen_bounds(
            pd.DataFrame(
                {
                    "x": [0.5],
                    "y": [0.5],
                }
            ),
            group_cols=["missing"],
        )


def test_h11_screen_bounds_zero_zero_allowed():
    result = qc.audit_gazepoint_screen_bounds(
        pd.DataFrame(
            {
                "x": [0.0],
                "y": [0.0],
            }
        ),
        group_cols=[],
        treat_zero_zero_as_out_of_bounds=False,
    )

    assert not bool(
        result["row_flags"].loc[
            0,
            "invalid_coordinate",
        ]
    )


# ============================================================================
# QC — EXCLUSION INTERNAL HELPERS
# ============================================================================


def test_h11_exclusion_units_status_matrix():
    frame = pd.DataFrame(
        {
            "subject": [
                "unknown",
                "retained",
                "excluded",
                "conflict",
                "conflict",
            ]
        }
    )

    flags = pd.Series(
        pd.array(
            [
                pd.NA,
                True,
                False,
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
            "",
            "keep",
            "drop",
        ]
    )

    result = qc._gp3_exclusion_r_units(
        frame,
        flags,
        "subject",
        None,
        [],
        reason=reasons,
    )

    status = dict(
        zip(
            result["subject"],
            result["exclusion_flow_status"],
            strict=True,
        )
    )

    assert status["unknown"] == "unclear_status"
    assert status["retained"] == "retained"
    assert status["excluded"] == "excluded"
    assert status["conflict"] == "conflicting_flags"


def test_h11_exclusion_ratio_branches():
    assert np.isnan(qc._gp3_exclusion_r_ratio([1]))

    assert np.isnan(
        qc._gp3_exclusion_r_ratio(
            [
                0,
                0,
            ]
        )
    )

    assert qc._gp3_exclusion_r_ratio(
        [
            0,
            0,
        ],
        zero_returns_one=True,
    ) == pytest.approx(1.0)

    assert np.isinf(
        qc._gp3_exclusion_r_ratio(
            [
                0,
                2,
            ]
        )
    )

    assert qc._gp3_exclusion_r_ratio(
        [
            2,
            4,
        ]
    ) == pytest.approx(2.0)


# ============================================================================
# QC — PHASE WINDOWS
# ============================================================================


def test_h11_phase_window_metadata():
    frame = pd.DataFrame(
        {
            "time": [
                0.0,
                5.0,
                10.0,
            ]
        }
    )

    windows = pd.DataFrame(
        {
            "phase": [
                "early",
                "late",
            ],
            "start": [
                0.0,
                5.0,
            ],
            "end": [
                5.0,
                10.0,
            ],
        }
    )

    result = qc.segment_gazepoint_task_phases(
        frame,
        time_col="time",
        phase_windows=windows,
        include_upper=True,
        keep_window_metadata=True,
    )

    assert "task_phase" in result
    assert ".gp3_phase_window_start" in result
    assert ".gp3_phase_window_end" in result


def test_h11_phase_window_validation():
    frame = pd.DataFrame({"time": [0.0]})

    with pytest.raises(ValueError):
        qc.segment_gazepoint_task_phases(
            frame,
            time_col="time",
            phase_windows=pd.DataFrame({"phase": ["x"]}),
        )

    with pytest.raises(ValueError):
        qc.segment_gazepoint_task_phases(
            frame,
            time_col="time",
            phase_windows=pd.DataFrame(
                {
                    "phase": ["x"],
                    "start": [2],
                    "end": [1],
                }
            ),
        )


# ============================================================================
# QC — STATUS REDUCER
# ============================================================================


def test_h11_qc_status_empty_character_values():
    result = qc._gp3_qc_status_from_overview(
        pd.DataFrame(
            {
                "status": [
                    "",
                    pd.NA,
                ]
            }
        ),
        ["status"],
    )

    assert result == "pass"


# ============================================================================
# QC — READINESS GATE BRANCHES
# ============================================================================


def _readiness_frame():
    return pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
            ],
            "trial": [
                1,
                1,
            ],
            "time": [
                0.0,
                0.0,
            ],
            "condition": [
                "A",
                "A",
            ],
            "x": [
                0.5,
                np.nan,
            ],
            "pupil": [
                3.0,
                np.nan,
            ],
            "valid": [
                True,
                False,
            ],
        }
    )


def test_h11_readiness_signal_and_audit_branches():
    audit_pass = {"overview": pd.DataFrame({"status": ["ok"]})}

    audit_warn = {"overview": pd.DataFrame({"status": ["review"]})}

    result = qc.check_gazepoint_real_data_readiness(
        _readiness_frame(),
        analysis_type="general",
        participant_col="subject",
        trial_col="trial",
        time_col="time",
        condition_col="condition",
        gaze_x_col="x",
        tracking_valid_col="valid",
        audit_objects=[
            audit_pass,
            audit_warn,
        ],
        min_rows=1,
        min_participants=1,
        min_trials=1,
    )

    check_ids = set(result["checks"]["check_id"])

    assert "paired_gaze_coordinates" in check_ids
    assert "tracking_validity" in check_ids


def test_h11_readiness_required_user_column_failure():
    result = qc.check_gazepoint_real_data_readiness(
        _readiness_frame(),
        analysis_type="general",
        required_cols=["definitely_missing"],
    )

    row = result["checks"].loc[lambda x: x["check_id"].eq("user_required_columns")].iloc[0]

    assert row["status"] == "fail"
