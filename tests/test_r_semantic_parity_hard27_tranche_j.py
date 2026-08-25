import numpy as np
import pandas as pd

import gp3tools as gp3


def test_r_pupil_imbalance_grouped_semantics():
    df = pd.DataFrame(
        {
            "condition": ["A"] * 4 + ["B"] * 4,
            "pupil_interpolated": [1, 2, np.nan, 4, 1, 2, 3, 4],
            "pupil_was_interpolated": [False, True, False, False, False, False, True, False],
            "pupil_interpolation_status": [
                "observed",
                "interpolated",
                "missing_long_gap",
                "observed",
                "observed",
                "observed",
                "interpolated",
                "observed",
            ],
            "pupil_artifact_flag": [False, False, True, False, False, False, False, False],
        }
    )
    out = gp3.audit_gazepoint_pupil_imbalance(df, group_cols=["condition"])
    assert list(out["condition"]) == ["A", "B"]
    assert out.loc[0, "n_valid_samples"] == 3
    assert out.loc[0, "n_artifact_samples"] == 1
    assert out.loc[0, "n_missing_long_gap_samples"] == 1
    assert bool(out["preprocessing_imbalance_warning"].iloc[0])
    assert "valid_pct_diff" in out["preprocessing_imbalance_reason"].iloc[0]


def test_r_pupil_trial_features_auc_windows_and_status():
    df = pd.DataFrame(
        {
            "subject": ["S1"] * 5,
            "trial_global": [1] * 5,
            "time": [0, 250, 500, 1000, 1600],
            "pupil_smoothed": [1.0, 2.0, 3.0, 2.0, 1.0],
            "pupil_was_interpolated": [False, False, True, False, False],
            "pupil_artifact_reason": ["valid", "valid", "valid", "blink", "valid"],
        }
    )
    out = gp3.summarise_gazepoint_pupil_trial_features(
        df,
        time_col="time",
        artifact_reason_col="pupil_artifact_reason",
        early_window=(0, 500),
        middle_window=(500, 1500),
        late_window=(1500, 2000),
        min_valid_samples=3,
    )
    row = out.iloc[0]
    assert row["n_valid_pupil"] == 5
    assert row["peak_pupil"] == 3.0
    assert row["peak_time_ms"] == 500.0
    assert row["time_to_peak_ms"] == 500.0
    assert row["early_mean_pupil"] == 1.5
    assert row["n_artifact_samples"] == 1
    assert row["pupil_feature_status"] == "ok"
    assert row["pupil_feature_pupil_column"] == "pupil_smoothed"


def test_r_pupil_windows_numeric_breaks_and_role_mapping():
    df = pd.DataFrame(
        {
            "participant": ["S1"] * 6,
            "MEDIA_ID": ["M1"] * 6,
            "time_ms": [0, 250, 500, 750, 1000, 1250],
            "pupil": [1.0, 2.0, 3.0, np.nan, 5.0, 6.0],
        }
    )
    out = gp3.summarise_gazepoint_pupil_windows(
        df,
        windows=[0, 500, 1000, 1500],
        group_cols=["subject", "media_id"],
        min_valid_samples=2,
    )
    assert list(out["window_label"]) == ["0_500ms", "500_1000ms", "1000_1500ms"]
    assert set(out["subject"]) == {"S1"}
    assert set(out["media_id"]) == {"M1"}
    assert out.loc[0, "n_samples"] == 2
    assert out.loc[1, "n_valid_pupil"] == 1
    assert out.loc[1, "pupil_window_status"] == "insufficient_valid_pupil"
    assert out.loc[2, "pupil_window_status"] == "valid"


def test_r_pupil_windows_dataframe_and_right_endpoint():
    df = pd.DataFrame({"time": [0, 1, 2], "pupil": [1.0, 2.0, 3.0]})
    windows = pd.DataFrame({"label": ["w"], "start": [0], "end": [2]})
    out = gp3.summarise_gazepoint_pupil_windows(
        df,
        pupil_col="pupil",
        time_col="time",
        windows=windows,
        group_cols=[],
        include_window_end=True,
    )
    assert out.loc[0, "n_samples"] == 3
    assert out.loc[0, "mean_pupil"] == 2.0
    assert bool(out.loc[0, "pupil_window_include_end"])


def test_r_pupil_baseline_audit_metadata_and_reason():
    df = pd.DataFrame(
        {
            "subject": ["S1"] * 4,
            "media_id": ["M1"] * 4,
            "time": [-100, 0, 100, 200],
            "pupil_interpolated": [1.0, np.nan, 1.2, 1.3],
            "pupil_baseline_n": [1, 1, 1, 1],
            "pupil_baseline_status": ["ok"] * 4,
            "pupil_baseline_available": [True] * 4,
            "pupil_baseline_used": [True] * 4,
            "pupil_baseline_window_start": [-100] * 4,
            "pupil_baseline_window_end": [0] * 4,
            "pupil_was_interpolated": [False, True, False, False],
            "pupil_artifact_flag": [False, True, False, False],
        }
    )
    out = gp3.audit_gazepoint_pupil_baseline(
        df,
        baseline_n_col="pupil_baseline_n",
        max_missing_pct=40,
        max_interpolated_pct=40,
        max_artifact_pct=40,
    )
    row = out.iloc[0]
    assert row["n_baseline_rows"] == 2
    assert row["n_baseline_missing_samples"] == 1
    assert row["baseline_missing_pct"] == 50.0
    assert bool(row["low_quality_baseline_flag"])
    assert row["baseline_quality_reason"] == "high_baseline_missing_pct"


def test_legacy_pupil_summary_paths_remain_available():
    df = pd.DataFrame(
        {
            "condition": ["A", "A", "B", "B"],
            "trial_global": [1, 1, 2, 2],
            "time": [0, 1, 0, 1],
            "pupil": [1.0, 2.0, 3.0, 4.0],
        }
    )
    imbalance = gp3.audit_gazepoint_pupil_imbalance(
        df, pupil_col="pupil", condition_col="condition"
    )
    assert {"n", "n_valid", "mean_pupil"} <= set(imbalance)
    trials = gp3.summarise_gazepoint_pupil_trial_features(
        df, pupil_col="pupil", trial_col="trial_global"
    )
    assert "n_valid" in trials
    windows = gp3.summarise_gazepoint_pupil_windows(
        df, pupil_col="pupil", time_col="time", windows={"all": (0, 1)}
    )
    assert "window" in windows
    baseline = gp3.audit_gazepoint_pupil_baseline(
        df, pupil_col="pupil", time_col="time", baseline=(0, 1)
    )
    assert "n_baseline" in baseline
