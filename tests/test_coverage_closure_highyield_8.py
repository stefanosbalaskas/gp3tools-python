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
# AOI — TRANSITION MATRICES
# ============================================================================


def test_h8_transition_matrix_legacy_paths():
    matrix = aoi.compute_gazepoint_aoi_transition_matrix(
        sequence=["A", "A", "B", "A"],
        normalize=False,
    )

    assert matrix.loc["A", "A"] == 1
    assert matrix.loc["A", "B"] == 1
    assert matrix.loc["B", "A"] == 1

    raw = pd.DataFrame(
        {
            "subject": ["S1"] * 4,
            "time": [0, 1, 2, 3],
            "aoi": ["A", "A", "B", "A"],
        }
    )

    pooled = aoi.compute_gazepoint_aoi_transition_matrix(
        raw,
        aoi_col="aoi",
        time_col="time",
        include_self=False,
    )

    assert not pooled.empty

    grouped = aoi.compute_gazepoint_aoi_transition_matrix(
        raw,
        aoi_col="aoi",
        group_cols=["subject"],
        time_col="time",
        normalize=True,
        include_self=False,
    )

    assert not grouped.empty


def _matrix_raw():
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
            "condition": [
                "C1",
                "C1",
                "C1",
                pd.NA,
                pd.NA,
                pd.NA,
            ],
            "time": [
                0.0,
                1.0,
                2.0,
                0.0,
                1.0,
                2.0,
            ],
            "aoi_current": [
                "A",
                "B",
                "A",
                "A",
                "A",
                "A",
            ],
        }
    )


def test_h8_transition_matrix_r_paths():
    result = aoi.compute_gazepoint_aoi_transition_matrix(
        _matrix_raw(),
        aoi_col="aoi_current",
        group_cols=["subject"],
        time_col="time",
        by_cols=["condition"],
        include_non_aoi=True,
        include_self_transitions=False,
        states=["A", "B", "C"],
        time_window=[0, 2],
    )

    assert result["count_matrix"] is None
    assert result["count_matrices"]
    assert any("condition=NA" in key for key in result["count_matrices"])

    self_only = pd.DataFrame(
        {
            "subject": ["S1", "S1"],
            "time": [0.0, 1.0],
            "aoi_current": ["A", "A"],
        }
    )

    zero = aoi.compute_gazepoint_aoi_transition_matrix(
        self_only,
        aoi_col="aoi_current",
        group_cols=["subject"],
        time_col="time",
        by_cols=[],
        include_self_transitions=False,
        states=["A"],
    )

    assert zero["long_table"].empty
    assert zero["count_matrix"].loc["A", "A"] == 0


def test_h8_transition_matrix_r_validation():
    with pytest.raises(ValueError):
        aoi.compute_gazepoint_aoi_transition_matrix(
            data=None,
            by_cols=[],
        )

    with pytest.raises(ValueError):
        aoi.compute_gazepoint_aoi_transition_matrix(
            _matrix_raw(),
            aoi_col="aoi_current",
            group_cols=["subject"],
            time_col="time",
            by_cols=[],
            include_self_transitions="yes",
        )

    with pytest.raises(ValueError):
        aoi.compute_gazepoint_aoi_transition_matrix(
            _matrix_raw(),
            aoi_col="aoi_current",
            group_cols=["subject"],
            time_col="time",
            by_cols=[],
            time_window=[0],
        )

    with pytest.raises(ValueError):
        aoi.compute_gazepoint_aoi_transition_matrix(
            _matrix_raw(),
            aoi_col="aoi_current",
            group_cols=["subject"],
            time_col="time",
            by_cols=[],
            time_window=["x", "y"],
        )

    with pytest.raises(ValueError):
        aoi.compute_gazepoint_aoi_transition_matrix(
            _matrix_raw(),
            aoi_col="aoi_current",
            group_cols=["subject"],
            time_col="time",
            by_cols=[],
            states=[],
        )

    with pytest.raises(ValueError):
        aoi.compute_gazepoint_aoi_transition_matrix(
            _matrix_raw(),
            aoi_col="aoi_current",
            group_cols=["subject"],
            time_col="time",
            by_cols=[],
            time_window=[100, 200],
        )


# ============================================================================
# AOI — TRIAL FEATURES
# ============================================================================


def _aoi_samples(states):
    return pd.DataFrame(
        {
            "subject": ["S1"] * len(states),
            "time": np.arange(
                len(states),
                dtype=float,
            ),
            "aoi_current": states,
        }
    )


def test_h8_aoi_trial_feature_statuses():
    no_aoi = aoi.summarise_gazepoint_aoi_trial_features(
        _aoi_samples(["outside", "outside"]),
        aoi_col="aoi_current",
        group_cols=["subject"],
        non_aoi_values=["outside"],
    )

    assert (
        no_aoi.loc[
            0,
            "aoi_trial_feature_status",
        ]
        == "no_aoi_entries"
    )

    undefined = aoi.summarise_gazepoint_aoi_trial_features(
        _aoi_samples(["other", "other"]),
        aoi_col="aoi_current",
        group_cols=["subject"],
        non_aoi_values=["outside"],
    )

    assert (
        undefined.loc[
            0,
            "aoi_trial_feature_status",
        ]
        == "no_target_or_distractor_defined"
    )

    target_absent = aoi.summarise_gazepoint_aoi_trial_features(
        _aoi_samples(["other", "other"]),
        aoi_col="aoi_current",
        group_cols=["subject"],
        target_aoi_values=["target"],
        non_aoi_values=["outside"],
    )

    assert (
        target_absent.loc[
            0,
            "aoi_trial_feature_status",
        ]
        == "target_not_observed"
    )

    distractor_absent = aoi.summarise_gazepoint_aoi_trial_features(
        _aoi_samples(["target", "target"]),
        aoi_col="aoi_current",
        group_cols=["subject"],
        target_aoi_values=["target"],
        distractor_aoi_values=["distractor"],
        non_aoi_values=["outside"],
    )

    assert (
        distractor_absent.loc[
            0,
            "aoi_trial_feature_status",
        ]
        == "distractor_not_observed"
    )

    ready = aoi.summarise_gazepoint_aoi_trial_features(
        _aoi_samples(
            [
                "target",
                "distractor",
                "target",
                "other",
            ]
        ),
        aoi_col="aoi_current",
        group_cols=["subject"],
        target_aoi_values=["target"],
        distractor_aoi_values=["distractor"],
        non_aoi_values=["outside"],
    )

    assert (
        ready.loc[
            0,
            "aoi_trial_feature_status",
        ]
        == "ok"
    )

    assert ready.loc[0, "target_revisits"] == 1


def test_h8_aoi_trial_feature_legacy_and_filtering():
    frame = pd.DataFrame(
        {
            "subject": ["S1", "S1"],
            "trial": [1, 1],
            "aoi_current": ["A", "B"],
        }
    )

    legacy = aoi.summarise_gazepoint_aoi_trial_features(
        frame,
        aoi_col="aoi_current",
        trial_col="trial",
        group_cols=["subject"],
    )

    # Legacy summarise_aoi_samples() returns one row per AOI
    # state within each subject/trial grouping.
    assert len(legacy) == 2

    assert set(legacy["aoi_current"]) == {
        "A",
        "B",
    }

    assert legacy["n_samples"].tolist() == [
        1,
        1,
    ]

    assert legacy["proportion"].sum() == pytest.approx(1.0)

    with pytest.raises(ValueError):
        aoi.summarise_gazepoint_aoi_trial_features(
            _aoi_samples(["outside", "outside"]),
            aoi_col="aoi_current",
            group_cols=["subject"],
            include_non_aoi=False,
            non_aoi_values=["outside"],
        )


# ============================================================================
# AOI — WINDOWS
# ============================================================================


def _window_data():
    return pd.DataFrame(
        {
            "subject": ["S1", "S1", "S2", "S2"],
            "time": [0.0, 50.0, 0.0, 50.0],
            "aoi_current": [
                "target",
                "other",
                "other",
                "other",
            ],
        }
    )


def test_h8_aoi_windows_dict_and_missing_condition():
    legacy = aoi.summarise_gazepoint_aoi_windows(
        _window_data(),
        aoi_col="aoi_current",
        time_col="time",
        windows={
            "early": (0, 25),
            "late": (25, 100),
        },
        group_cols=["subject"],
    )

    assert set(legacy["window"]) == {
        "early",
        "late",
    }

    result = aoi.summarise_gazepoint_aoi_windows(
        _window_data(),
        aoi_col="aoi_current",
        windows=[0, 100],
        group_cols=["subject"],
        condition_col="condition",
        target_aoi_values=["target"],
    )

    assert "condition" in result
    assert set(result["condition"]) == {"all_data"}


def test_h8_aoi_windows_validation_matrix():
    frame = _window_data()

    with pytest.raises(ValueError):
        aoi.summarise_gazepoint_aoi_windows(
            frame,
            aoi_col="aoi_current",
            windows=None,
            group_cols=["subject"],
            condition_col=None,
        )

    with pytest.raises(ValueError):
        aoi.summarise_gazepoint_aoi_windows(
            frame,
            aoi_col="aoi_current",
            windows=123,
            group_cols=["subject"],
            condition_col=None,
        )

    with pytest.raises(ValueError):
        aoi.summarise_gazepoint_aoi_windows(
            frame,
            aoi_col="aoi_current",
            windows=[0, 0],
            group_cols=["subject"],
            condition_col=None,
        )

    blank = pd.DataFrame(
        {
            "window_label": [""],
            "window_start_ms": [0],
            "window_end_ms": [100],
        }
    )

    with pytest.raises(ValueError):
        aoi.summarise_gazepoint_aoi_windows(
            frame,
            aoi_col="aoi_current",
            windows=blank,
            group_cols=["subject"],
            condition_col=None,
        )

    nonfinite = pd.DataFrame(
        {
            "window_label": ["x"],
            "window_start_ms": [0],
            "window_end_ms": [np.inf],
        }
    )

    with pytest.raises(ValueError):
        aoi.summarise_gazepoint_aoi_windows(
            frame,
            aoi_col="aoi_current",
            windows=nonfinite,
            group_cols=["subject"],
            condition_col=None,
        )

    backwards = pd.DataFrame(
        {
            "window_label": ["x"],
            "window_start_ms": [100],
            "window_end_ms": [0],
        }
    )

    with pytest.raises(ValueError):
        aoi.summarise_gazepoint_aoi_windows(
            frame,
            aoi_col="aoi_current",
            windows=backwards,
            group_cols=["subject"],
            condition_col=None,
        )

    bad_time = frame.copy()
    bad_time["time"] = np.nan

    with pytest.raises(ValueError):
        aoi.summarise_gazepoint_aoi_windows(
            bad_time,
            aoi_col="aoi_current",
            windows=[0, 100],
            group_cols=["subject"],
            condition_col=None,
        )

    with pytest.raises(ValueError):
        aoi.summarise_gazepoint_aoi_windows(
            frame,
            aoi_col="aoi_current",
            windows=[1000, 2000],
            group_cols=["subject"],
            condition_col=None,
        )

    with pytest.raises(ValueError):
        aoi.summarise_gazepoint_aoi_windows(
            frame,
            aoi_col="aoi_current",
            windows=[0, 100],
            group_cols=[],
            condition_col=None,
        )


# ============================================================================
# AOI — DENOMINATOR VARIABILITY / CONDITION IMBALANCE
# ============================================================================


def test_h8_denominator_variability_and_condition_statuses():
    frame = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S2",
                "S3",
                "S4",
                "S5",
            ],
            "condition": [
                "A",
                "B",
                "A",
                "A",
                "B",
            ],
            "window_label": [
                "ratio",
                "ratio",
                "single",
                "cv",
                "cv",
            ],
            "window_start_ms": [
                0,
                0,
                100,
                200,
                200,
            ],
            "window_end_ms": [
                100,
                100,
                200,
                300,
                300,
            ],
            "n_window_samples": [
                10,
                30,
                12,
                10,
                11,
            ],
            "n_valid_denominator_samples": [
                10,
                30,
                12,
                10,
                11,
            ],
            "n_target_samples": [
                1,
                1,
                1,
                1,
                1,
            ],
        }
    )

    result = aoi.audit_gazepoint_aoi_window_denominators(
        frame,
        max_denominator_cv=0.05,
        max_condition_ratio=2,
    )

    assert (
        result["overview"].loc[
            0,
            "denominator_audit_status",
        ]
        == "ok"
    )

    assert "high_denominator_variability" in set(
        result["window_summary"]["window_denominator_status"]
    )

    imbalance = set(result["denominator_imbalance"]["denominator_imbalance_status"])

    assert "condition_denominator_ratio_high" in imbalance
    assert "condition_denominator_cv_high" in imbalance
    assert "single_condition" in imbalance


def test_h8_denominator_missing_columns():
    with pytest.raises(KeyError):
        aoi.audit_gazepoint_aoi_window_denominators(
            pd.DataFrame(
                {
                    "window_label": ["x"],
                }
            ),
            group_cols=["subject"],
        )


# ============================================================================
# PUPIL — HAMPEL
# ============================================================================


def _hampel_data():
    return pd.DataFrame(
        {
            "subject": ["S1"] * 6,
            "time": [
                0,
                1,
                2,
                3,
                4,
                5,
            ],
            "pupil": [
                1.0,
                1.0,
                100.0,
                1.0,
                1.0,
                np.nan,
            ],
        }
    )


def test_h8_hampel_statuses_and_corrected_values():
    result = pupil.flag_gazepoint_pupil_hampel(
        _hampel_data(),
        pupil_col="pupil",
        time_col="time",
        grouping_cols=["subject"],
        window_size_samples=3,
        min_valid_samples=2,
        corrected_col="pupil_corrected",
    )

    assert result["pupil_hampel_outlier"].any()

    assert "complete_zero_mad" in set(result["pupil_hampel_status"])

    assert "missing_or_nonfinite_pupil" in set(result["pupil_hampel_status"])

    flagged = result["pupil_hampel_outlier"]

    assert result.loc[
        flagged,
        "pupil_corrected",
    ].iloc[0] == pytest.approx(1.0)

    sparse = pd.DataFrame(
        {
            "pupil": [
                1.0,
                np.nan,
                np.nan,
            ]
        }
    )

    sparse_result = pupil.flag_gazepoint_pupil_hampel(
        sparse,
        pupil_col="pupil",
        window_size_samples=3,
        min_valid_samples=3,
    )

    assert "insufficient_valid_window" in set(sparse_result["pupil_hampel_status"])

    legacy = pupil.flag_gazepoint_pupil_hampel(
        _hampel_data(),
        pupil_col="pupil",
        window=3,
        n_sigma=3,
        output_col="legacy_flag",
    )

    assert "legacy_flag" in legacy


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "window_size_samples": 2,
        },
        {
            "window_size_samples": 3,
            "min_valid_samples": 4,
        },
        {
            "k": 0,
        },
        {
            "scale_mad": 0,
        },
    ],
)
def test_h8_hampel_validation(kwargs):
    with pytest.raises(ValueError):
        pupil.flag_gazepoint_pupil_hampel(
            _hampel_data(),
            pupil_col="pupil",
            **kwargs,
        )


def test_h8_hampel_column_validation():
    with pytest.raises(KeyError):
        pupil.flag_gazepoint_pupil_hampel(
            _hampel_data(),
            pupil_col="pupil",
            grouping_cols=["missing"],
        )

    with pytest.raises(KeyError):
        pupil.flag_gazepoint_pupil_hampel(
            _hampel_data(),
            pupil_col="pupil",
            time_col="missing",
        )

    with pytest.raises(ValueError):
        pupil.flag_gazepoint_pupil_hampel(
            _hampel_data(),
            pupil_col="pupil",
            flag_col="same",
            median_col="same",
        )

    existing = _hampel_data()
    existing["pupil_hampel_outlier"] = False

    with pytest.raises(ValueError):
        pupil.flag_gazepoint_pupil_hampel(
            existing,
            pupil_col="pupil",
        )

    nonfinite = _hampel_data()
    nonfinite.loc[0, "time"] = np.nan

    with pytest.raises(ValueError):
        pupil.flag_gazepoint_pupil_hampel(
            nonfinite,
            pupil_col="pupil",
            time_col="time",
        )


# ============================================================================
# PUPIL — SMOOTHING VALIDATION
# ============================================================================


def test_h8_smoothing_missing_role_and_custom_group():
    no_media = pd.DataFrame(
        {
            "subject": ["S1", "S1"],
            "time": [0, 1],
            "pupil": [1.0, 2.0],
        }
    )

    with pytest.raises(ValueError):
        pupil.smooth_gazepoint_pupil(
            no_media,
            pupil_col="pupil",
            time_col="time",
        )

    frame = no_media.assign(media_id="M1")

    with pytest.raises(ValueError):
        pupil.smooth_gazepoint_pupil(
            frame,
            pupil_col="pupil",
            time_col="time",
            group_cols=["missing_group"],
        )

    with pytest.raises(ValueError):
        pupil.smooth_gazepoint_pupil(
            frame.drop(columns=["pupil"]),
            time_col="time",
            group_cols=["subject"],
        )

    with pytest.raises(ValueError):
        pupil.smooth_gazepoint_pupil(
            frame.drop(columns=["time"]),
            pupil_col="pupil",
            group_cols=["subject"],
        )


# ============================================================================
# PUPIL — GP IMPUTATION
# ============================================================================


def test_h8_gp_legacy_small_and_imputation():
    too_small = pd.DataFrame(
        {
            "time": [0, 1, 2],
            "pupil": [
                1.0,
                np.nan,
                np.nan,
            ],
        }
    )

    out = pupil.impute_gazepoint_pupil_gp(
        too_small,
        pupil_col="pupil",
        time_col="time",
    )

    assert pd.isna(
        out.loc[
            1,
            "pupil_gp",
        ]
    )

    frame = pd.DataFrame(
        {
            "time": [
                0,
                1,
                2,
                3,
                4,
                5,
            ],
            "pupil": [
                1.0,
                1.1,
                1.2,
                np.nan,
                1.4,
                1.5,
            ],
        }
    )

    fitted = pupil.impute_gazepoint_pupil_gp(
        frame,
        pupil_col="pupil",
        time_col="time",
        max_points=3,
    )

    assert np.isfinite(
        fitted.loc[
            3,
            "pupil_gp",
        ]
    )


def test_h8_gp_grouped_r_mode():
    frame = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
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
                1,
                1,
            ],
            "time": [
                0,
                1,
                2,
                3,
                0,
                1,
            ],
            "pupil": [
                1.0,
                1.1,
                1.2,
                np.nan,
                2.0,
                np.nan,
            ],
        }
    )

    result = pupil.impute_gazepoint_pupil_gp(
        frame,
        pupil_col="pupil",
        time_col="time",
        subject="subject",
        trial="trial",
        max_points=3,
        flag="gp_flag",
    )

    assert bool(
        result.loc[
            3,
            "gp_flag",
        ]
    )

    assert not bool(
        result.loc[
            5,
            "gp_flag",
        ]
    )


def test_h8_gp_singular_and_validation():
    singular = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S1",
                "S1",
            ],
            "time": [
                0,
                0,
                0,
                1,
            ],
            "pupil": [
                1.0,
                1.0,
                1.0,
                np.nan,
            ],
        }
    )

    result = pupil.impute_gazepoint_pupil_gp(
        singular,
        pupil_col="pupil",
        time_col="time",
        subject="subject",
        noise=0,
    )

    assert not bool(
        result.loc[
            3,
            "pupil_was_gp_imputed",
        ]
    )

    with pytest.raises(KeyError):
        pupil.impute_gazepoint_pupil_gp(
            singular,
            pupil_col="pupil",
            time_col="time",
            subject="missing",
        )

    with pytest.raises(ValueError):
        pupil.impute_gazepoint_pupil_gp(
            singular,
            pupil_col="pupil",
            time_col="time",
            subject="subject",
            max_points=0,
        )

    with pytest.raises(ValueError):
        pupil.impute_gazepoint_pupil_gp(
            singular,
            pupil_col="pupil",
            time_col="time",
            subject="subject",
            noise=-1,
        )

    with pytest.raises(ValueError):
        pupil.impute_gazepoint_pupil_gp(
            singular,
            pupil_col="pupil",
            time_col="time",
            subject="subject",
            length_scale=0,
        )


# ============================================================================
# PUPIL — TRIAL FEATURES
# ============================================================================


def test_h8_pupil_trial_features_auto_detection_and_missing_status():
    frame = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S2",
                "S2",
            ],
            "trial_global": [
                "T1",
                "T1",
                "T1",
                "T1",
            ],
            "time": [
                0,
                1000,
                0,
                1000,
            ],
            "pupil_smoothed": [
                1.0,
                2.0,
                np.nan,
                np.nan,
            ],
        }
    )

    result = pupil.summarise_gazepoint_pupil_trial_features(
        frame,
        time_col="time",
        group_cols=[
            "subject",
            "trial_global",
        ],
        min_valid_samples=0,
    )

    assert len(result) == 2

    assert "missing_pupil" in set(result["pupil_feature_status"])

    s1 = result.loc[result["subject"].eq("S1")].iloc[0]

    assert np.isfinite(s1["pupil_auc"])


def test_h8_pupil_trial_features_artifact_flag_and_validation():
    frame = pd.DataFrame(
        {
            "subject": ["S1", "S1"],
            "trial_global": ["T1", "T1"],
            "time": [0, 100],
            "pupil": [1.0, 2.0],
            "interp": [0, 1],
            "artifact": [False, True],
        }
    )

    result = pupil.summarise_gazepoint_pupil_trial_features(
        frame,
        pupil_col="pupil",
        time_col="time",
        group_cols=[
            "subject",
            "trial_global",
        ],
        interpolated_col="interp",
        artifact_col="artifact",
    )

    assert (
        result.loc[
            0,
            "n_interpolated_samples",
        ]
        == 1
    )

    assert (
        result.loc[
            0,
            "n_artifact_samples",
        ]
        == 1
    )

    with pytest.raises(ValueError):
        pupil.summarise_gazepoint_pupil_trial_features(
            frame,
            pupil_col="pupil",
            time_col="time",
            group_cols=[
                "subject",
                "subject",
            ],
        )

    with pytest.raises(KeyError):
        pupil.summarise_gazepoint_pupil_trial_features(
            frame.drop(columns=["pupil"]),
            pupil_col=None,
            time_col="time",
            group_cols=[
                "subject",
                "trial_global",
            ],
        )

    with pytest.raises(KeyError):
        pupil.summarise_gazepoint_pupil_trial_features(
            frame,
            pupil_col="pupil",
            time_col="missing",
            group_cols=[
                "subject",
                "trial_global",
            ],
        )


# ============================================================================
# PUPIL — OVERLAP RISK
# ============================================================================


def _overlap_rows():
    return pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S2",
                "S2",
            ],
            "trial_global": [
                "T1",
                "T1",
                "T1",
                "T1",
            ],
            "time": [
                0,
                1000,
                0,
                1000,
            ],
            "event1": [
                0,
                0,
                np.nan,
                np.nan,
            ],
            "event2": [
                500,
                500,
                np.nan,
                np.nan,
            ],
            "excluded_trial": [
                False,
                False,
                False,
                False,
            ],
        }
    )


def test_h8_overlap_risk_event_statuses():
    result = pupil.audit_gazepoint_pupil_overlap_risk(
        _overlap_rows(),
        trial_col="trial_global",
        time_col="time",
        group_cols=["subject"],
        event_time_cols=[
            "event1",
            "event2",
        ],
        window_start_ms=0,
        window_end_ms=1000,
        min_event_gap_ms=600,
    )

    assert (
        result["summary"].loc[
            0,
            "overlap_assessment_status",
        ]
        == "possible_overlap_risk"
    )

    assert "overlap_and_short_gap" in set(result["event_gaps"]["event_gap_status"])

    no_events = _overlap_rows()
    no_events[
        [
            "event1",
            "event2",
        ]
    ] = np.nan

    empty = pupil.audit_gazepoint_pupil_overlap_risk(
        no_events,
        trial_col="trial_global",
        time_col="time",
        group_cols=["subject"],
        event_time_cols=[
            "event1",
            "event2",
        ],
    )

    assert (
        empty["summary"].loc[
            0,
            "overlap_assessment_status",
        ]
        == "no_usable_event_times"
    )


@pytest.mark.parametrize(
    "values",
    [
        [0, 1, 0, 0],
        [
            "no",
            "yes",
            "no",
            "no",
        ],
    ],
)
def test_h8_overlap_exclusion_types(values):
    frame = _overlap_rows()
    frame["excluded_trial"] = values

    result = pupil.audit_gazepoint_pupil_overlap_risk(
        frame,
        trial_col="trial_global",
        time_col="time",
        group_cols=["subject"],
        event_time_cols=[
            "event1",
            "event2",
        ],
        exclude_col="excluded_trial",
    )

    assert not result["by_trial"].empty


def test_h8_overlap_validation():
    frame = _overlap_rows()

    with pytest.raises(ValueError):
        pupil.audit_gazepoint_pupil_overlap_risk(
            frame,
            group_cols=[],
            trial_col="trial_global",
            time_col="time",
            event_time_cols=[
                "event1",
                "event2",
            ],
        )

    with pytest.raises(ValueError):
        pupil.audit_gazepoint_pupil_overlap_risk(
            frame,
            group_cols=["subject"],
            trial_col="trial_global",
            time_col="time",
            event_time_cols=[
                "event1",
                "event1",
            ],
        )

    with pytest.raises(ValueError):
        pupil.audit_gazepoint_pupil_overlap_risk(
            frame,
            group_cols=["subject"],
            trial_col="trial_global",
            time_col="time",
            event_time_cols=[
                "event1",
                "event2",
            ],
            window_start_ms=1000,
            window_end_ms=100,
        )

    with pytest.raises(ValueError):
        pupil.audit_gazepoint_pupil_overlap_risk(
            frame,
            group_cols=["subject"],
            trial_col="trial_global",
            time_col="time",
            event_time_cols=[
                "event1",
                "event2",
            ],
            min_event_gap_ms=np.inf,
        )

    with pytest.raises(KeyError):
        pupil.audit_gazepoint_pupil_overlap_risk(
            frame,
            group_cols=["subject"],
            trial_col="trial_global",
            time_col="time",
            event_time_cols=["missing"],
        )


# ============================================================================
# QC — MASTER COERCION
# ============================================================================


def _raw_master():
    return pd.DataFrame(
        {
            "USER_FILE": [
                "S1_all_gaze.csv",
                "S1_all_gaze.csv",
            ],
            "MEDIA_ID": [
                "M1",
                "M1",
            ],
            "MEDIA_NAME": [
                "stimulus",
                "stimulus",
            ],
            "TIME": [
                0.0,
                0.1,
            ],
            "BPOGX": [
                0.5,
                0.6,
            ],
            "BPOGY": [
                0.4,
                0.5,
            ],
            "BPOGV": [
                1,
                1,
            ],
            "LPMM": [
                3.0,
                3.1,
            ],
            "RPMM": [
                3.2,
                3.3,
            ],
            "LPMMV": [
                1,
                1,
            ],
            "RPMMV": [
                1,
                1,
            ],
        }
    )


def test_h8_as_master_coordinate_modes():
    norm = qc.as_gazepoint_master(
        _raw_master(),
        screen_width_px=1000,
        screen_height_px=800,
        coordinate_unit="auto",
        event_latency_offset_ms=5,
    )

    assert (
        norm.loc[
            0,
            "coordinate_unit_detected",
        ]
        == "normalised"
    )

    assert norm.loc[
        0,
        "x",
    ] == pytest.approx(500)

    assert norm.loc[
        0,
        "time_ms",
    ] == pytest.approx(5)

    pixels = _raw_master()
    pixels["BPOGX"] = [
        500,
        600,
    ]
    pixels["BPOGY"] = [
        400,
        500,
    ]

    pixel_master = qc.as_gazepoint_master(
        pixels,
        screen_width_px=1000,
        screen_height_px=800,
        coordinate_unit="pixels",
    )

    assert pixel_master.loc[
        0,
        "x",
    ] == pytest.approx(500)

    minimal = qc.as_gazepoint_master(
        pd.DataFrame(
            {
                "TIME": [0.0],
            }
        ),
        coordinate_unit="auto",
    )

    assert (
        minimal.loc[
            0,
            "coordinate_unit_detected",
        ]
        == "pixels"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "coordinate_unit": "bad",
        },
        {
            "screen_width_px": np.inf,
        },
        {
            "screen_height_px": np.inf,
        },
        {
            "event_latency_offset_ms": np.inf,
        },
    ],
)
def test_h8_as_master_validation(kwargs):
    with pytest.raises(ValueError):
        qc.as_gazepoint_master(
            _raw_master(),
            **kwargs,
        )

    with pytest.raises(KeyError):
        qc.as_gazepoint_master(
            _raw_master().drop(columns=["TIME"]),
            coordinate_unit="auto",
        )


# ============================================================================
# QC — HARMONIZATION
# ============================================================================


def test_h8_coordinate_harmonization_legacy_defaults():
    normalised = qc.harmonize_gazepoint_screen_coordinates(
        pd.DataFrame(
            {
                "x": [
                    0.25,
                    0.5,
                ],
                "y": [
                    0.25,
                    0.5,
                ],
            }
        )
    )

    assert normalised.loc[
        1,
        "x_norm",
    ] == pytest.approx(0.5)

    pixels = qc.harmonize_gazepoint_screen_coordinates(
        pd.DataFrame(
            {
                "x": [
                    50.0,
                    100.0,
                ],
                "y": [
                    25.0,
                    50.0,
                ],
            }
        )
    )

    assert pixels.loc[
        1,
        "x_norm",
    ] == pytest.approx(1.0)

    assert pixels.loc[
        1,
        "y_norm",
    ] == pytest.approx(1.0)

    same_columns = qc.harmonize_gazepoint_screen_coordinates(
        pd.DataFrame(
            {
                "x": [10.0],
                "y": [20.0],
            }
        ),
        from_width=100,
        from_height=100,
        to_width=200,
        to_height=200,
        output_x_col="x",
        output_y_col="y",
        keep_original=False,
    )

    assert same_columns.loc[
        0,
        "x",
    ] == pytest.approx(20)


# ============================================================================
# QC — FILE PAIRS
# ============================================================================


def test_h8_file_pair_status_matrix(tmp_path):
    root = Path(tmp_path)

    (root / "S2_fixations.csv").write_text(
        "x\n1\n",
        encoding="utf-8",
    )

    (root / "S3_all_gaze.csv").write_text(
        "x\n1\n",
        encoding="utf-8",
    )

    (root / "S4_all_gaze.csv").write_text(
        "x\n1\n",
        encoding="utf-8",
    )

    (root / "S4_fixations.csv").write_text(
        "x\n1\n",
        encoding="utf-8",
    )

    (root / "a").mkdir()
    (root / "b").mkdir()

    (root / "a" / "S1_all_gaze.csv").write_text(
        "x\n1\n",
        encoding="utf-8",
    )

    (root / "b" / "S1_all_gaze.csv").write_text(
        "x\n1\n",
        encoding="utf-8",
    )

    (root / "S1_fixations.csv").write_text(
        "x\n1\n",
        encoding="utf-8",
    )

    result = qc.check_gazepoint_file_pairs(
        root,
        recursive=True,
    )

    status = dict(
        zip(
            result["participant"],
            result["status"],
            strict=True,
        )
    )

    assert status["S1"] == "duplicate_files"
    assert status["S2"] == "missing_all_gaze"
    assert status["S3"] == "missing_fixation"
    assert status["S4"] == "complete"

    with pytest.raises(ValueError):
        qc.check_gazepoint_file_pairs(root / "missing")

    empty = root / "empty"
    empty.mkdir()

    with pytest.raises(ValueError):
        qc.check_gazepoint_file_pairs(empty)


# ============================================================================
# QC — EXCLUSION VALIDATION / EXPLICIT MODE
# ============================================================================


def _exclusion_frame():
    return pd.DataFrame(
        {
            "participant": [
                "S1",
                "S1",
                "S2",
                "S2",
            ],
            "trial": [
                "T1",
                "T1",
                "T1",
                "T1",
            ],
            "condition": [
                "A",
                "A",
                "B",
                "B",
            ],
            "validity": [
                "yes",
                "no",
                "yes",
                "yes",
            ],
            "x": [
                0.1,
                np.nan,
                0.2,
                0.3,
            ],
            "y": [
                0.1,
                0.2,
                0.2,
                0.3,
            ],
            "pupil": [
                3.0,
                np.nan,
                3.1,
                3.2,
            ],
            "artifact": [
                0,
                1,
                0,
                0,
            ],
        }
    )


def test_h8_explicit_exclusion_contract():
    result = qc.recommend_gazepoint_exclusions(
        _exclusion_frame(),
        participant_col="participant",
        trial_col="trial",
        condition_col="condition",
        validity_col="validity",
        x_col="x",
        y_col="y",
        pupil_col="pupil",
        artifact_col="artifact",
        min_trial_samples=2,
        max_trial_missing_prop=0.2,
        max_trial_artifact_prop=0.2,
        min_participant_trials=1,
        min_participant_valid_trials=1,
        max_participant_missing_prop=0.2,
        max_participant_artifact_prop=0.2,
    )

    assert not result["trial_recommendations"].empty

    assert not result["participant_recommendations"].empty

    assert result["exclusion_table"]["recommend_exclude"].any()


def test_h8_explicit_exclusion_validation():
    frame = _exclusion_frame()

    with pytest.raises(ValueError):
        qc.recommend_gazepoint_exclusions(
            [],
            participant_col="participant",
            validity_col="validity",
        )

    with pytest.raises(ValueError):
        qc.recommend_gazepoint_exclusions(
            frame.iloc[0:0],
            participant_col="participant",
            validity_col="validity",
        )

    with pytest.raises(ValueError):
        qc.recommend_gazepoint_exclusions(
            frame,
            participant_col="missing",
            validity_col="validity",
        )

    with pytest.raises(ValueError):
        qc.recommend_gazepoint_exclusions(
            frame,
            participant_col="participant",
        )

    with pytest.raises(ValueError):
        qc.recommend_gazepoint_exclusions(
            frame,
            participant_col="participant",
            x_col="x",
            require_both_gaze_coordinates=True,
        )

    with pytest.raises(ValueError):
        qc.recommend_gazepoint_exclusions(
            frame,
            participant_col="participant",
            validity_col="validity",
            min_trial_samples=True,
        )

    with pytest.raises(ValueError):
        qc.recommend_gazepoint_exclusions(
            frame,
            participant_col="participant",
            validity_col="validity",
            min_trial_samples=0,
        )

    with pytest.raises(ValueError):
        qc.recommend_gazepoint_exclusions(
            frame,
            participant_col="participant",
            validity_col="validity",
            max_trial_missing_prop=2,
        )

    with pytest.raises(ValueError):
        qc.recommend_gazepoint_exclusions(
            frame,
            participant_col="participant",
            validity_col="validity",
            name="",
        )


# ============================================================================
# QC — NAMING AUDIT WRITER
# ============================================================================


def test_h8_naming_writer_validation(tmp_path):
    pairs = pd.DataFrame(
        {
            "stem": ["x"],
            "status": ["paired"],
        }
    )

    obj = {
        "pairs": pairs,
    }

    out = qc.write_gazepoint_naming_audit(
        x=obj,
        output_file=(tmp_path / "nested" / "audit.csv"),
    )

    assert out.exists()

    with pytest.raises(ValueError):
        qc.write_gazepoint_naming_audit(
            x=obj,
        )

    with pytest.raises(ValueError):
        qc.write_gazepoint_naming_audit(
            output_file=(tmp_path / "audit.csv"),
        )

    with pytest.raises(TypeError):
        qc.write_gazepoint_naming_audit(
            path=(tmp_path / "legacy.csv"),
            x=obj,
            output_file=(tmp_path / "r.csv"),
        )

    with pytest.raises(TypeError):
        qc.write_gazepoint_naming_audit(
            x={},
            output_file=(tmp_path / "bad.csv"),
        )

    with pytest.raises(TypeError):
        qc.write_gazepoint_naming_audit()

    legacy = qc.write_gazepoint_naming_audit(
        tmp_path / "legacy" / "audit.csv",
        names=[
            "summarise_x",
            "summarize_x",
        ],
    )

    assert legacy.exists()
