import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3


def _base():
    return pd.DataFrame(
        {
            "subject": ["S1"] * 5,
            "media_id": ["M1"] * 5,
            "time_ms": [0.0, 10.0, 20.0, 30.0, 40.0],
            "pupil": [1.0, np.nan, 3.0, np.nan, 5.0],
        }
    )


def test_interpolate_r_internal_gaps_and_metadata():
    out = gp3.interpolate_gazepoint_pupil(_base(), max_gap_ms=25, max_gap_samples=1)
    assert out["pupil_interpolated"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert out["pupil_interpolation_status"].tolist() == [
        "observed",
        "interpolated",
        "observed",
        "interpolated",
        "observed",
    ]
    assert out["pupil_gap_id"].astype("Int64").tolist() == [pd.NA, 1, pd.NA, 2, pd.NA]
    assert out.loc[1, "pupil_gap_duration_ms"] == 20
    assert out.loc[3, "pupil_gap_duration_ms"] == 20
    assert out["pupil_interp_pupil_column"].eq("pupil").all()
    assert out["pupil_interp_time_column"].eq("time_ms").all()


def test_interpolate_r_long_edge_and_insufficient_statuses():
    long = pd.DataFrame(
        {
            "subject": ["S1"] * 4,
            "media_id": ["M1"] * 4,
            "time_ms": [0.0, 100.0, 200.0, 300.0],
            "pupil": [1.0, np.nan, np.nan, 4.0],
        }
    )
    out = gp3.interpolate_gazepoint_pupil(long, max_gap_ms=150)
    assert out.loc[1:2, "pupil_interpolation_status"].eq("missing_long_gap").all()

    edge = long.copy()
    edge["pupil"] = [np.nan, 2.0, 3.0, 4.0]
    out = gp3.interpolate_gazepoint_pupil(edge)
    assert out.loc[0, "pupil_interpolation_status"] == "missing_edge_gap"

    insufficient = long.iloc[:3].copy()
    insufficient["pupil"] = [1.0, np.nan, np.nan]
    out = gp3.interpolate_gazepoint_pupil(insufficient, min_valid_points=2)
    assert out.loc[1:2, "pupil_interpolation_status"].eq("missing_insufficient_valid").all()


def test_interpolate_r_missing_time_and_sample_limit():
    df = _base()
    df.loc[1, "time_ms"] = np.nan
    out = gp3.interpolate_gazepoint_pupil(df)
    assert out.loc[1, "pupil_interpolation_status"] == "missing_no_time"

    df = pd.DataFrame(
        {
            "subject": ["S1"] * 4,
            "media_id": ["M1"] * 4,
            "time_ms": [0.0, 10.0, 20.0, 30.0],
            "pupil": [1.0, np.nan, np.nan, 4.0],
        }
    )
    out = gp3.interpolate_gazepoint_pupil(df, max_gap_ms=100, max_gap_samples=1)
    assert out.loc[1:2, "pupil_interpolation_status"].eq("missing_long_gap").all()


def test_interpolate_r_group_validation_and_legacy_method():
    with pytest.raises(ValueError, match="can only contain"):
        gp3.interpolate_gazepoint_pupil(_base(), group_cols=["condition"])
    with pytest.raises(ValueError, match="greater than or equal to 2"):
        gp3.interpolate_gazepoint_pupil(_base(), min_valid_points=1)

    legacy = gp3.interpolate_gazepoint_pupil(
        _base(), pupil_col="pupil", time_col="time_ms", group_cols=[], method="linear"
    )
    assert "pupil_interpolated" in legacy


def test_smooth_r_center_mean_and_preserve_missing():
    df = _base()
    out = gp3.smooth_gazepoint_pupil(
        df, pupil_col="pupil", time_col="time_ms", group_cols=[], window_samples=3
    )
    assert out.loc[0, "pupil_smoothed"] == pytest.approx(1.0)
    assert np.isnan(out.loc[1, "pupil_smoothed"])
    assert out.loc[2, "pupil_smoothed"] == pytest.approx(3.0)
    assert out.loc[1, "pupil_smoothing_status"] == "missing_input"
    assert out.loc[1, "pupil_smoothing_window_n"] == 2


def test_smooth_r_alignment_median_and_missing_fill():
    df = pd.DataFrame(
        {
            "subject": ["S"] * 4,
            "media_id": ["M"] * 4,
            "time_ms": [0, 1, 2, 3],
            "pupil": [1.0, np.nan, 5.0, 9.0],
        }
    )
    right = gp3.smooth_gazepoint_pupil(
        df, group_cols=[], window_samples=2, align="right", preserve_missing=False
    )
    assert right.loc[1, "pupil_smoothed"] == pytest.approx(1.0)
    left = gp3.smooth_gazepoint_pupil(
        df, group_cols=[], window_samples=3, align="left", method="median", preserve_missing=False
    )
    assert left.loc[0, "pupil_smoothed"] == pytest.approx(3.0)
    assert left.loc[1, "pupil_smoothed"] == pytest.approx(7.0)


def test_smooth_r_min_points_and_validation():
    df = _base()
    out = gp3.smooth_gazepoint_pupil(
        df, group_cols=[], window_samples=3, min_points=3, preserve_missing=False
    )
    assert (
        out["pupil_smoothing_status"]
        .isin(["smoothed", "insufficient_window", "missing_input"])
        .all()
    )
    with pytest.raises(ValueError, match="less than or equal"):
        gp3.smooth_gazepoint_pupil(df, group_cols=[], window_samples=2, min_points=3)
    with pytest.raises(ValueError, match="Unknown smoothing method"):
        gp3.smooth_gazepoint_pupil(df, group_cols=[], method="bad")


def test_baseline_r_mean_outputs_and_statuses():
    df = pd.DataFrame(
        {
            "subject": ["S1"] * 5,
            "media_id": ["M1"] * 5,
            "time_ms": [-200.0, -100.0, 0.0, 100.0, 200.0],
            "pupil": [2.0, 4.0, 6.0, 8.0, np.nan],
        }
    )
    out = gp3.baseline_correct_gazepoint_pupil(df, baseline_window=(-200, 0), group_cols=[])
    assert out["pupil_baseline_value"].iloc[0] == pytest.approx(4.0)
    assert out["pupil_baseline_n"].iloc[0] == 3
    assert out.loc[3, "pupil_baseline_corrected"] == pytest.approx(4.0)
    assert out.loc[3, "pupil_baseline_percent_change"] == pytest.approx(100.0)
    assert out.loc[3, "pupil_baseline_ratio"] == pytest.approx(2.0)
    assert out.loc[4, "pupil_baseline_status"] == "missing_pupil"
    assert out.loc[0, "pupil_baseline_used"]


def test_baseline_r_median_flag_and_no_baseline():
    df = pd.DataFrame(
        {
            "subject": ["S1"] * 4,
            "media_id": ["M1"] * 4,
            "time_ms": [0.0, 1.0, 2.0, 3.0],
            "pupil": [1.0, 100.0, 3.0, 4.0],
            "is_base": [True, False, True, False],
        }
    )
    out = gp3.baseline_correct_gazepoint_pupil(
        df,
        baseline_flag_col="is_base",
        baseline_window=None,
        baseline_method="median",
        group_cols=[],
    )
    assert out["pupil_baseline_value"].iloc[0] == pytest.approx(2.0)
    assert out["pupil_baseline_flag_column"].iloc[0] == "is_base"

    no = gp3.baseline_correct_gazepoint_pupil(
        df, baseline_window=(10, 20), min_baseline_samples=2, group_cols=[]
    )
    assert no["pupil_baseline_status"].eq("no_baseline").all()


def test_baseline_legacy_mode_is_preserved():
    df = pd.DataFrame({"time_ms": [-1.0, 0.0, 1.0], "pupil": [2.0, 4.0, 6.0]})
    out = gp3.baseline_correct_gazepoint_pupil(
        df, pupil_col="pupil", time_col="time_ms", baseline=(-1, 0), group_cols=[], mode="percent"
    )
    assert "pupil_baseline" in out
    assert "pupil_baseline_corrected" in out


def test_gap_audit_r_counts_and_percentages():
    interpolated = gp3.interpolate_gazepoint_pupil(_base(), max_gap_ms=25, max_gap_samples=1)
    out = gp3.audit_gazepoint_pupil_gaps(interpolated, group_cols=[])
    row = out.iloc[0]
    assert row["n_rows"] == 5
    assert row["n_observed_samples"] == 3
    assert row["n_interpolated_samples"] == 2
    assert row["n_gaps_total"] == 2
    assert row["n_gaps_interpolated"] == 2
    assert row["mean_gap_duration_ms"] == pytest.approx(20.0)
    assert row["max_gap_n_samples"] == pytest.approx(1.0)
    assert row["pct_interpolated_samples"] == pytest.approx(40.0)


def test_gap_audit_r_grouped_and_status_aliases():
    df = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S2", "S2"],
            "media_id": ["M"] * 4,
            "pupil_interpolation_status": [
                "observed",
                "missing_no_time_gap",
                "observed",
                "missing_unfilled",
            ],
            "pupil_gap_id": [pd.NA, 1, pd.NA, 1],
            "pupil_gap_n_samples": [pd.NA, 1, pd.NA, 1],
            "pupil_gap_duration_ms": [np.nan, np.nan, np.nan, 20.0],
            "pupil_was_interpolated": [False, False, False, False],
            "pupil_interpolated": [1.0, np.nan, 2.0, np.nan],
        }
    )
    out = gp3.audit_gazepoint_pupil_gaps(df, group_cols=["subject", "media_id"])
    s1 = out.loc[out.subject.eq("S1")].iloc[0]
    s2 = out.loc[out.subject.eq("S2")].iloc[0]
    assert s1["n_missing_no_time_samples"] == 1
    assert s2["n_missing_unfilled_samples"] == 1


def test_gap_audit_legacy_raw_gap_mode():
    df = pd.DataFrame({"time": [0, 1, 2, 3], "pupil": [1.0, np.nan, np.nan, 4.0]})
    out = gp3.audit_gazepoint_pupil_gaps(df, pupil_col="pupil", time_col="time", group_cols=[])
    assert out.loc[0, "n_gaps"] == 1
    assert out.loc[0, "max_gap_samples"] == 2


def test_tranche_d_legacy_omitted_group_cols_with_uppercase_media_id():
    df = pd.DataFrame(
        {
            "subject": ["S1"] * 6,
            "MEDIA_ID": [1] * 6,
            "TIME": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            "pupil": [3.0, np.nan, 3.2, 3.1, np.nan, 3.0],
        }
    )

    interpolated = gp3.interpolate_gazepoint_pupil(
        df,
        pupil_col="pupil",
        time_col="TIME",
        method="linear",
    )
    assert len(interpolated) == len(df)
    assert "pupil_interpolated" in interpolated.columns

    centered = df.copy()
    centered["time_ms"] = centered["TIME"] * 1000.0 - 100.0

    corrected = gp3.baseline_correct_gazepoint_pupil(
        centered,
        pupil_col="pupil",
        time_col="time_ms",
        baseline=(-100.0, 0.0),
    )
    assert len(corrected) == len(df)
    assert "pupil_baseline_corrected" in corrected.columns

    gaps = gp3.audit_gazepoint_pupil_gaps(
        df,
        pupil_col="pupil",
        time_col="TIME",
    )
    assert len(gaps) == 1
    assert int(gaps.loc[0, "n_gaps"]) == 2
