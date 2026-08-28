import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3


def _velocity_samples():
    return pd.DataFrame(
        {
            "USER_ID": ["P01"] * 8,
            "trial": [1] * 8,
            "TIME": np.arange(8) * 0.01,
            "FPOGX": [0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8],
            "FPOGY": [0.5] * 8,
        }
    )


def test_velocity_r_events_samples_and_both():
    data = _velocity_samples()
    events = gp3.detect_gazepoint_fixations_velocity(
        data,
        id_col="USER_ID",
        group_cols=["trial"],
        time_unit="seconds",
        x_scale=1,
        y_scale=1,
        velocity_threshold=5,
        min_duration_ms=20,
        **{"return": "events"},
    )
    assert isinstance(events, pd.DataFrame)
    assert {"fixation_id", "duration_ms", "mean_x", "algorithm"} <= set(events)
    assert events.attrs["_gp3_class"] == "gp3_velocity_fixations"
    assert len(events) >= 1

    samples = gp3.detect_gazepoint_fixations_velocity(
        data,
        id_col="USER_ID",
        time_unit="seconds",
        velocity_threshold=5,
        min_duration_ms=20,
        **{"return": "samples"},
    )
    assert {"gaze_velocity", "velocity_fixation", "velocity_fixation_id"} <= set(samples)

    both = gp3.detect_gazepoint_fixations_velocity(
        data,
        id_col="USER_ID",
        time_unit="seconds",
        velocity_threshold=5,
        min_duration_ms=20,
        **{"return": "both"},
    )
    assert set(both) >= {"events", "samples", "_gp3_class"}


def test_velocity_r_time_units_validation_and_single_sample():
    ms = _velocity_samples()
    ms["TIME"] *= 1000
    out = gp3.detect_gazepoint_fixations_velocity(
        ms,
        id_col="USER_ID",
        time_unit="milliseconds",
        velocity_threshold=5,
        min_duration_ms=0,
        keep_single_sample=True,
        **{"return": "events"},
    )
    assert isinstance(out, pd.DataFrame)
    with pytest.raises(ValueError):
        gp3.detect_gazepoint_fixations_velocity(ms, id_col="USER_ID", time_unit="bad")
    with pytest.raises(TypeError):
        gp3.detect_gazepoint_fixations_velocity(ms, id_col="USER_ID", nonsense=True)


def test_velocity_legacy_path_is_retained():
    legacy = pd.DataFrame(
        {"subject": ["S1"] * 5, "TIME": np.arange(5) / 60, "FPOGX": [0.2] * 5, "FPOGY": [0.5] * 5}
    )
    out = gp3.detect_gazepoint_fixations_velocity(
        legacy, velocity_threshold=2.0, min_duration_ms=10
    )
    assert {"event_velocity", "fixation", "fixation_id"} <= set(out)


def _blink_samples():
    return pd.DataFrame(
        {
            "USER_ID": ["P01"] * 10,
            "trial": [1] * 10,
            "TIME": np.arange(10) * 0.01,
            "pupil": [3.0, 3.0, 3.0, np.nan, np.nan, np.nan, 3.0, 3.0, 3.0, 3.0],
        }
    )


def test_blinks_r_events_samples_both_and_reasons():
    data = _blink_samples()
    events = gp3.detect_gazepoint_blinks(
        data,
        pupil_col="pupil",
        id_col="USER_ID",
        group_cols=["trial"],
        time_unit="seconds",
        z_thresh=4,
        zero_threshold=0,
        merge_gap_ms=20,
        include_rapid_changes=False,
        min_duration_ms=20,
        **{"return": "events"},
    )
    assert len(events) == 1
    assert events.loc[0, "reason"] == "missing"
    assert events.attrs["_gp3_class"] == "gp3_blink_events"

    samples = gp3.detect_gazepoint_blinks(
        data,
        pupil_col="pupil",
        id_col="USER_ID",
        time_unit="seconds",
        include_rapid_changes=False,
        min_duration_ms=20,
        **{"return": "samples"},
    )
    assert {"blink_detected", "blink_id", "blink_reason"} <= set(samples)
    assert samples["blink_detected"].sum() == 3

    both = gp3.detect_gazepoint_blinks(
        data,
        pupil_col=["pupil"],
        id_col="USER_ID",
        time_unit="seconds",
        include_rapid_changes=False,
        min_duration_ms=20,
        **{"return": "both"},
    )
    assert both["_gp3_class"] == "gp3_blink_detection_result"


def test_blinks_r_zero_merge_and_validation():
    data = _blink_samples()
    data.loc[3, "pupil"] = 0
    data.loc[4, "pupil"] = 3
    data.loc[5, "pupil"] = 0
    merged = gp3.detect_gazepoint_blinks(
        data,
        pupil_col="pupil",
        id_col="USER_ID",
        time_unit="seconds",
        include_rapid_changes=False,
        zero_threshold=0,
        merge_gap_ms=30,
        min_duration_ms=0,
        **{"return": "events"},
    )
    assert len(merged) == 1
    assert "zero" in merged.loc[0, "reason"]
    with pytest.raises(ValueError):
        gp3.detect_gazepoint_blinks(data, pupil_col="pupil", id_col="USER_ID", time_unit="bad")


def test_blinks_legacy_path_is_retained():
    data = pd.DataFrame({"TIME": np.arange(6) / 60, "pupil": [3, np.nan, np.nan, 3, 3, 3]})
    out = gp3.detect_gazepoint_blinks(data, pupil_col="pupil", time_col="TIME", min_duration_ms=10)
    assert "blink" in out


def _fixations():
    return pd.DataFrame(
        {
            "subject": ["S1"] * 4,
            "MEDIA_ID": ["M1"] * 4,
            "trial_global": [1] * 4,
            "FPOGID": [1, 2, 3, 4],
            "FPOGS": [0.10, 0.30, 0.50, 0.70],
            "FPOGD": [0.10, 0.10, 0.10, 0.10],
            "FPOGX": [0.2, 0.3, 0.7, 0.8],
            "FPOGY": [0.5, 0.5, 0.5, 0.5],
            "FPOGV": [1, 1, 1, 1],
            "AOI": ["target", "background", "distractor", "target"],
        }
    )


def test_fixation_trial_features_r_contract():
    out = gp3.summarise_gazepoint_fixation_trials(
        _fixations(),
        group_cols=["subject", "MEDIA_ID", "trial_global"],
        target_aoi_values=["target"],
        distractor_aoi_values=["distractor"],
    )
    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_fixations"] == 4
    assert row["target_fixation_count"] == 2
    assert row["target_revisits"] == 1
    assert row["distractor_fixation_count"] == 1
    assert row["n_non_aoi_fixations"] == 1
    assert row["fixation_trial_feature_status"] == "ok"
    assert np.isclose(row["target_fixation_duration_ms"], 200)


def test_fixation_trial_features_filter_units_and_statuses():
    data = _fixations()
    data.loc[1, "FPOGV"] = 0
    out = gp3.summarise_gazepoint_fixation_trials(
        data,
        group_cols=["subject", "MEDIA_ID", "trial_global"],
        valid_only=True,
        include_non_aoi=False,
        target_aoi_values=["never"],
        distractor_aoi_values=["distractor"],
        start_time_unit="s",
        duration_unit="s",
    )
    assert out.loc[0, "n_fixations"] == 3
    assert out.loc[0, "fixation_trial_feature_status"] == "target_not_observed"
    assert np.isclose(out.loc[0, "mean_fixation_duration_ms"], 100)


def test_fixation_trial_features_legacy_and_validation():
    legacy = pd.DataFrame(
        {"subject": ["S1", "S1"], "trial_global": [1, 1], "duration_ms": [100.0, 200.0]}
    )
    out = gp3.summarise_gazepoint_fixation_trials(
        legacy, trial_col="trial_global", subject_col="subject"
    )
    assert out.loc[0, "n_fixations"] == 2
    with pytest.raises(ValueError):
        gp3.summarise_gazepoint_fixation_trials(_fixations(), group_cols=["missing"])


def test_tranche_h_validation_and_auto_unit_branches():
    gaze = _velocity_samples()
    with pytest.raises(TypeError):
        gp3.detect_gazepoint_fixations_velocity(
            gaze,
            id_col="USER_ID",
            return_mode="events",
            **{"return": "events"},
        )
    for kwargs in (
        {"return_mode": "bad"},
        {"velocity_threshold": -1},
        {"min_duration_ms": -1},
        {"x_scale": 0},
        {"keep_single_sample": "yes"},
    ):
        with pytest.raises(ValueError):
            gp3.detect_gazepoint_fixations_velocity(
                gaze,
                id_col="USER_ID",
                time_unit="seconds",
                **kwargs,
            )
    with pytest.raises(ValueError):
        gp3.detect_gazepoint_fixations_velocity(
            gaze.drop(columns="FPOGX"), id_col="USER_ID", time_unit="seconds"
        )
    auto = gaze.copy()
    auto["TIME"] = np.arange(len(auto)) * 10.0
    auto_result = gp3.detect_gazepoint_fixations_velocity(
        auto,
        id_col="USER_ID",
        time_unit="auto",
        velocity_threshold=5,
        min_duration_ms=0,
        **{"return": "samples"},
    )
    assert "gaze_velocity" in auto_result

    blink = _blink_samples()
    with pytest.raises(TypeError):
        gp3.detect_gazepoint_blinks(
            blink,
            pupil_col="pupil",
            id_col="USER_ID",
            return_mode="events",
            **{"return": "events"},
        )
    for kwargs in (
        {"return_mode": "bad"},
        {"z_thresh": -1},
        {"zero_threshold": np.inf},
        {"include_rapid_changes": "yes"},
    ):
        with pytest.raises(ValueError):
            gp3.detect_gazepoint_blinks(
                blink,
                pupil_col="pupil",
                id_col="USER_ID",
                time_unit="seconds",
                **kwargs,
            )
    with pytest.raises(ValueError):
        gp3.detect_gazepoint_blinks(
            blink.drop(columns="pupil"),
            id_col="USER_ID",
            time_unit="seconds",
        )
    auto_blink = blink.copy()
    auto_blink["TIME"] = np.arange(len(auto_blink)) * 10.0
    auto_events = gp3.detect_gazepoint_blinks(
        auto_blink,
        pupil_col="pupil",
        id_col="USER_ID",
        time_unit="auto",
        include_rapid_changes=True,
        min_duration_ms=0,
        **{"return": "events"},
    )
    assert isinstance(auto_events, pd.DataFrame)


def test_fixation_trial_feature_additional_status_and_validation_branches():
    data = _fixations().drop(columns="AOI")
    out = gp3.summarise_gazepoint_fixation_trials(
        data,
        group_cols=["subject", "MEDIA_ID", "trial_global"],
        start_time_unit="ms",
        duration_unit="ms",
    )
    assert out.loc[0, "fixation_trial_feature_status"] == "no_aoi_column"

    with pytest.raises(ValueError):
        gp3.summarise_gazepoint_fixation_trials(_fixations(), group_cols=["subject", "subject"])
    with pytest.raises(ValueError):
        gp3.summarise_gazepoint_fixation_trials(
            _fixations(), group_cols=["subject"], start_time_unit="bad"
        )
    with pytest.raises(ValueError):
        gp3.summarise_gazepoint_fixation_trials(
            _fixations(), group_cols=["subject"], duration_unit="bad"
        )
    with pytest.raises(ValueError):
        gp3.summarise_gazepoint_fixation_trials(
            _fixations(), group_cols=["subject"], valid_only="yes"
        )
