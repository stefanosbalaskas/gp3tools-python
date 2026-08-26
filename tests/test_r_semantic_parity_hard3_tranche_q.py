from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3


def _workflow(n=24):
    return pd.DataFrame(
        {
            "USER_ID": ["U1"] * n,
            "TIME": np.arange(n, dtype=float) * 10.0,
            "FPOGX": np.linspace(0.1, 0.3, n),
            "FPOGY": np.linspace(0.2, 0.4, n),
            "LPupil": np.linspace(3.0, 4.0, n),
            "RPupil": np.linspace(3.1, 4.1, n),
        }
    )


def _artifact(n=14):
    pupil = np.linspace(3.0, 3.8, n)
    pupil[3] = np.nan
    pupil[8] = 15.0
    left = np.linspace(3.0, 3.8, n)
    right = left.copy()
    right[9] = 8.0
    return pd.DataFrame(
        {
            "subject": ["S1"] * n,
            "media_id": ["M1"] * n,
            "time_ms": np.arange(n, dtype=float) * 20.0,
            "pupil": pupil,
            "left_pupil": left,
            "right_pupil": right,
            "pupil_unit": ["diameter_mm"] * n,
            "blink": [False, False, True] + [False] * (n - 3),
            "trackloss": [False] * (n - 2) + [True, False],
        }
    )


def test_luminance_legacy_modes():
    df = pd.DataFrame({"luminance": [0.1, 0.2, 0.3, 0.4], "pupil": [4.0, 3.5, 3.0, 2.5]})
    out = gp3.audit_gazepoint_stimulus_luminance(df)
    assert out.loc[0, "n"] == 4
    assert out.loc[0, "correlation"] < 0
    missing = gp3.audit_gazepoint_stimulus_luminance(df.drop(columns="luminance"))
    assert missing.loc[0, "status"] == "luminance_column_missing"


def test_r_luminance_reads_images_and_missing_files(tmp_path):
    from matplotlib import pyplot as plt

    nested = tmp_path / "nested"
    nested.mkdir()
    image = nested / "bright.png"
    plt.imsave(image, np.array([[0.0, 1.0], [0.25, 0.5]]), cmap="gray")
    df = pd.DataFrame(
        {
            "stimulus_file": ["bright.png", "missing.png", None],
            "stimulus_id": ["a", "b", "c"],
            "condition": ["A", "B", None],
        }
    )
    out = gp3.audit_gazepoint_stimulus_luminance(
        df,
        stimulus_file_col="stimulus_file",
        stimulus_id_col="stimulus_id",
        condition_col="condition",
        image_dir=str(tmp_path),
        recursive=True,
        name="lum",
    )
    assert set(out) >= {
        "overview",
        "stimulus_index",
        "stimulus_luminance",
        "condition_summary",
        "balance_summary",
        "settings",
    }
    lum = out["stimulus_luminance"].set_index("stimulus_id")
    assert lum.loc["a", "luminance_status"] == "available"
    assert lum.loc["a", "n_pixels"] == 4
    assert lum.loc["b", "luminance_status"] == "file_missing"
    assert lum.loc["c", "luminance_status"] == "missing_file_name"
    assert out["overview"].loc[0, "audit_status"] == "partial_luminance_available"
    assert "missing_condition" in set(out["condition_summary"]["condition"])
    assert out["balance_summary"].loc[0, "n_conditions"] == 3


def test_r_luminance_read_error_and_validation(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_text("not an image", encoding="utf-8")
    df = pd.DataFrame({"file_name": ["bad.png"]})
    out = gp3.audit_gazepoint_stimulus_luminance(
        df, stimulus_file_col="file_name", image_dir=str(tmp_path), recursive=False
    )
    assert out["stimulus_luminance"].loc[0, "luminance_status"] == "read_error"
    assert out["balance_summary"].loc[0, "luminance_balance_status"] == "no_luminance_available"
    with pytest.raises(ValueError):
        gp3.audit_gazepoint_stimulus_luminance(df.iloc[0:0], stimulus_file_col="file_name")
    with pytest.raises(ValueError):
        gp3.audit_gazepoint_stimulus_luminance(df, stimulus_file_col="file_name", recursive="yes")
    with pytest.raises(ValueError):
        gp3.audit_gazepoint_stimulus_luminance(pd.DataFrame({"x": [1]}), name="x")
    with pytest.raises(ValueError):
        gp3.audit_gazepoint_stimulus_luminance(df, stimulus_file_col="missing")


def test_artifact_legacy_unchanged():
    df = pd.DataFrame(
        {"TIME": np.arange(8) / 60.0, "PUPIL": [3.0, 3.1, np.nan, 3.2, 3.3, 20.0, 3.4, 3.5]}
    )
    out = gp3.flag_gazepoint_pupil_artifacts(df, pupil_col="PUPIL", time_col="TIME")
    assert "pupil_artifact" in out
    assert "pupil_speed" in out


def test_r_artifact_flags_sources_and_reasons():
    df = _artifact()
    out = gp3.flag_gazepoint_pupil_artifacts(
        df,
        pupil_col="pupil",
        left_pupil_col="left_pupil",
        right_pupil_col="right_pupil",
        pupil_unit_col="pupil_unit",
        blink_col="blink",
        trackloss_col="trackloss",
        group_cols=["subject", "media_id"],
        blink_padding_pre_ms=0,
        blink_padding_post_ms=0,
        pupil_speed_mad_k=6,
        binocular_mad_k=2,
        max_physio_outlier_prop=0.8,
    )
    assert out.loc[3, "pupil_flag_missing_source"]
    assert out.loc[2, "pupil_flag_blink_source"]
    assert out.loc[len(out) - 2, "pupil_flag_trackloss_source"]
    assert out.loc[8, "pupil_physio_outlier"]
    assert out.loc[9, "pupil_binocular_disagreement"]
    assert out["pupil_artifact_flag"].any()
    assert out.loc[3, "pupil_artifact_reason"].startswith("missing_pupil")
    assert pd.isna(out.loc[3, "pupil_clean"])
    assert out.loc[0, "pupil_artifact_pupil_column"] == "pupil"


def test_r_artifact_physio_suppression_registry_and_switches():
    df = _artifact(10)
    df["pupil"] = 20.0
    registry = pd.DataFrame(
        {
            "parameter": [
                "pupil_physiological_min",
                "pupil_physiological_max",
                "binocular_mad_k",
                "blink_padding_pre_ms",
                "blink_padding_post_ms",
                "pupil_speed_mad_k",
            ],
            "value": [1, 9, 6, 0, 0, 6],
        }
    )
    out = gp3.flag_gazepoint_pupil_artifacts(
        df,
        pupil_col="pupil",
        group_cols=["subject", "media_id"],
        pupil_unit_col="pupil_unit",
        registry=registry,
        max_physio_outlier_prop=0.5,
        flag_speed_outliers=False,
        flag_binocular_disagreement=False,
        flag_physiological_outliers=True,
    )
    assert out["pupil_physio_rule_suppressed"].all()
    assert not out["pupil_physio_outlier"].any()
    assert np.isinf(out["pupil_binocular_disagreement_threshold"]).all()
    assert not out["pupil_speed_outlier"].any()


def test_r_artifact_non_role_group_and_validation():
    df = _artifact()
    df["session"] = "A"
    out = gp3.flag_gazepoint_pupil_artifacts(
        df,
        pupil_col="pupil",
        group_cols=["session"],
        pupil_unit_col="pupil_unit",
        flag_physiological_outliers=False,
    )
    assert len(out) == len(df)
    with pytest.raises(ValueError):
        gp3.flag_gazepoint_pupil_artifacts(df, pupil_col="pupil", group_cols=["missing"])
    with pytest.raises(ValueError):
        gp3.flag_gazepoint_pupil_artifacts(
            df, pupil_col="pupil", group_cols=["subject"], pupil_min_mm=5, pupil_max_mm=2
        )
    with pytest.raises(ValueError):
        gp3.flag_gazepoint_pupil_artifacts(
            df, pupil_col="pupil", group_cols=["subject"], max_physio_outlier_prop=2
        )
    with pytest.raises(ValueError):
        gp3.flag_gazepoint_pupil_artifacts(
            df, pupil_col="pupil", group_cols=["subject"], flag_speed_outliers="yes"
        )
    with pytest.raises(ValueError):
        gp3.flag_gazepoint_pupil_artifacts(
            df, pupil_col="pupil", group_cols=["subject"], registry=pd.DataFrame({"x": [1]})
        )


def test_preprocess_legacy_unchanged():
    df = pd.DataFrame({"TIME": np.arange(10) / 60.0, "PUPIL": np.linspace(3, 4, 10)})
    out = gp3.preprocess_gazepoint_signals(
        df, pupil_col="PUPIL", time_col="TIME", interpolate=False, smooth=False
    )
    assert isinstance(out, pd.DataFrame)
    assert "pupil_artifact" in out


def test_r_preprocess_none_disabled_and_downsample():
    df = _workflow()
    df["pupil"] = np.linspace(3, 4, len(df))
    out = gp3.preprocess_gazepoint_signals(
        df,
        pupil_col="pupil",
        pupil_mode="none",
        detect_blinks=False,
        interpolate_blinks=False,
        smooth_pupil=False,
        smooth_coordinates=False,
        detect_fixations=False,
        downsample_factor=2,
    )
    assert out["_gp3_class"] == "gp3_signal_preprocessing_result"
    assert len(out["data"]) == len(df) // 2
    assert out["diagnostics"]["overview"].loc[0, "pupil_mode"] == "none"
    assert out["decision_log"].loc[0, "status"] == "skipped"
    assert out["decision_log"].iloc[-1]["operation"] == "downsampling"


def test_r_preprocess_mean_full_pipeline():
    df = _workflow(30)
    df.loc[10:12, ["LPupil", "RPupil"]] = np.nan
    out = gp3.preprocess_gazepoint_signals(
        df,
        pupil_mode="mean",
        detect_blinks=True,
        interpolate_blinks=True,
        smooth_pupil=True,
        smooth_coordinates=True,
        detect_fixations=True,
        blink_args={"min_duration_ms": 0},
        interpolation_args={"suffix": "_interp", "max_gap_ms": 500},
        pupil_smoothing_args={"window_samples": 3},
        coordinate_smoothing_args={"window": 3, "suffix": "_s"},
        fixation_args={"min_duration_ms": 0},
    )
    assert out["settings"]["final_pupil_col"] == "pupil_smoothed"
    assert "FPOGX_s" in out["data"]
    assert isinstance(out["blinks"], pd.DataFrame)
    assert isinstance(out["fixations"], pd.DataFrame)
    assert len(out["decision_log"]) == 7
    assert set(out["diagnostics"]) == {
        "overview",
        "signal_summary",
        "blink_summary",
        "fixation_summary",
    }


def test_r_preprocess_regression_branch():
    df = _workflow(30)
    out = gp3.preprocess_gazepoint_signals(
        df,
        pupil_mode="regression",
        detect_blinks=False,
        interpolate_blinks=False,
        smooth_pupil=False,
        smooth_coordinates=False,
        detect_fixations=False,
        pupil_args={"min_complete": 5},
    )
    assert "gp3_pupil_fused" in out["data"]
    assert out["decision_log"].loc[0, "operation"] == "binocular_pupil_regression"


def test_r_preprocess_validation_and_protected_overrides():
    df = _workflow()
    with pytest.raises(ValueError):
        gp3.preprocess_gazepoint_signals(df, pupil_mode="bad")
    with pytest.raises(ValueError):
        gp3.preprocess_gazepoint_signals(df, pupil_mode="mean", detect_blinks="yes")
    with pytest.raises(ValueError):
        gp3.preprocess_gazepoint_signals(df, pupil_mode="mean", downsample_factor=0)
    with pytest.raises(ValueError):
        gp3.preprocess_gazepoint_signals(df, pupil_mode="mean", blink_args=[])
    with pytest.raises(ValueError):
        gp3.preprocess_gazepoint_signals(
            df,
            pupil_mode="mean",
            detect_blinks=False,
            interpolate_blinks=True,
            smooth_coordinates=False,
            detect_fixations=False,
        )
    with pytest.raises(ValueError):
        gp3.preprocess_gazepoint_signals(df.drop(columns="FPOGX"), pupil_mode="mean")
    with pytest.raises(ValueError):
        gp3.preprocess_gazepoint_signals(df, pupil_mode="mean", pupil_args={"left_col": "x"})
    with pytest.raises(ValueError):
        gp3.preprocess_gazepoint_signals(df, pupil_mode="mean", blink_args={"pupil_col": "x"})
