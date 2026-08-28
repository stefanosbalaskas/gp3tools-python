import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3


def test_r_flag_pupil_quality_columns_and_iqr():
    df = pd.DataFrame(
        {
            "subject": ["S1"] * 7,
            "MEDIA_ID": [1] * 7,
            "TIME": np.arange(7),
            "pupil": [np.nan, 2.0, 2.0, 2.1, 2.2, 4.0, np.inf],
        }
    )
    out = gp3.flag_gazepoint_pupil(
        df,
        pupil_col="pupil",
        time_col="TIME",
        group_cols=["subject", "media_id"],
        physiological_min=1.0,
        physiological_max=9.0,
        outlier_k=1.5,
    )
    assert out.loc[0, "pupil_flag_reason"] == "missing"
    assert out.loc[6, "pupil_flag_reason"] == "nonfinite"
    assert bool(out.loc[5, "pupil_flag_iqr_outlier"])
    assert np.isnan(out.loc[5, "pupil_for_preprocessing"])
    assert out.loc[1, "pupil_flag_reason"] == "valid"


def test_r_flag_pupil_missing_source_and_validation():
    df = pd.DataFrame(
        {
            "subject": ["S1"] * 4,
            "MEDIA_ID": [1] * 4,
            "TIME": range(4),
            "pupil": [2.0] * 4,
            "miss": [0, 1, 0, 0],
        }
    )
    out = gp3.flag_gazepoint_pupil(
        df, pupil_col="pupil", time_col="TIME", missing_pupil_col="miss", flag_iqr_outliers=False
    )
    assert out.loc[1, "pupil_flag_reason"] == "missing"
    with pytest.raises(ValueError):
        gp3.flag_gazepoint_pupil(df, pupil_col="pupil", time_col="TIME", group_cols=["trial"])


def test_legacy_flag_pupil_preserved():
    out = gp3.flag_gazepoint_pupil(
        pd.DataFrame({"pupil": [0.5, 2.0, 10.0]}),
        pupil_col="pupil",
        physiological_min=1,
        physiological_max=9,
    )
    assert out["pupil_flag"].tolist() == ["below_min", "ok", "above_max"]


def test_r_hampel_structured_outputs_and_attrs():
    df = pd.DataFrame(
        {"subject": ["S1"] * 7, "time": range(7), "pupil": [2, 2, 2, 8, 2, np.nan, 2]}
    )
    out = gp3.flag_gazepoint_pupil_hampel(
        df,
        pupil_col="pupil",
        time_col="time",
        grouping_cols=["subject"],
        window_size_samples=5,
        k=3,
        corrected_col="corrected",
    )
    assert bool(out.loc[3, "pupil_hampel_outlier"])
    assert out.loc[3, "corrected"] == pytest.approx(2.0)
    assert out.loc[5, "pupil_hampel_status"] == "missing_or_nonfinite_pupil"
    assert out.attrs["gp3_hampel_overview"].loc[0, "n_flagged"] >= 1


def test_r_hampel_zero_mad_and_errors():
    df = pd.DataFrame({"pupil": [1, 1, 2, 1, 1]})
    out = gp3.flag_gazepoint_pupil_hampel(df, pupil_col="pupil", window_size_samples=5, k=2)
    assert out.loc[2, "pupil_hampel_status"] == "complete_zero_mad"
    assert bool(out.loc[2, "pupil_hampel_outlier"])
    with pytest.raises(ValueError):
        gp3.flag_gazepoint_pupil_hampel(df, pupil_col="pupil", window_size_samples=4)


def test_legacy_hampel_preserved():
    out = gp3.flag_gazepoint_pupil_hampel(
        pd.DataFrame({"pupil": [1, 1, 8, 1, 1]}), pupil_col="pupil", window=3
    )
    assert "pupil_hampel_flag" in out


def test_r_gp_grouped_imputation():
    df = pd.DataFrame(
        {
            "subject": ["S1"] * 6 + ["S2"] * 6,
            "trial": [1] * 12,
            "time": list(range(6)) * 2,
            "pupil": [1, 1.2, np.nan, 1.6, 1.8, 2, 3, 3.2, np.nan, 3.6, 3.8, 4],
        }
    )
    out = gp3.impute_gazepoint_pupil_gp(
        df,
        pupil="pupil",
        time="time",
        subject="subject",
        trial="trial",
        length_scale=2,
        noise=1e-4,
        max_train=4,
        output="gp",
        flag="was",
    )
    assert np.isfinite(out.loc[[2, 8], "gp"]).all()
    assert out.loc[[2, 8], "was"].all()


def test_r_gp_insufficient_and_invalid_length_scale():
    df = pd.DataFrame({"time": [0.0, 1.0, 2.0], "pupil": [1.0, np.nan, 2.0]})
    out = gp3.impute_gazepoint_pupil_gp(df, pupil="pupil", time="time", length_scale=1.0)
    assert out["pupil_gp_imputed"].isna().sum() == 1
    assert not out["pupil_was_gp_imputed"].any()
    with pytest.raises(ValueError):
        gp3.impute_gazepoint_pupil_gp(df, pupil="pupil", time="time", length_scale=0)
