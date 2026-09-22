from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest
import scipy.interpolate

pupil = importlib.import_module("gp3tools.pupil")
qc = importlib.import_module("gp3tools.qc")


def test_final_pupil_hampel_accepts_scalar_grouping_column():
    result = pupil.flag_gazepoint_pupil_hampel(
        pd.DataFrame(
            {
                "subject": ["S1"] * 5,
                "pupil": [3.0, 3.1, 3.2, 3.1, 3.0],
            }
        ),
        pupil_col="pupil",
        grouping_cols="subject",
    )
    assert "pupil_hampel_outlier" in result


def test_final_pupil_blink_short_delta_series_uses_no_rapid_change_threshold():
    result = pupil.detect_gazepoint_blinks(
        pd.DataFrame(
            {
                "USER_ID": ["S1", "S1", "S1"],
                "TIME": [0.0, 0.01, 0.02],
                "pupil": [3.0, 3.1, 3.2],
            }
        ),
        pupil_col="pupil",
        id_col="USER_ID",
        time_unit="seconds",
        include_rapid_changes=True,
        return_mode="samples",
    )
    assert "blink_detected" in result


def test_final_pupil_artifact_negative_parameter_and_missing_group_role():
    frame = pd.DataFrame(
        {
            "subject": ["S1", "S1"],
            "time": [0.0, 1.0],
            "pupil": [3.0, 3.1],
        }
    )
    with pytest.raises(ValueError, match="finite and non-negative"):
        pupil.flag_gazepoint_pupil_artifacts(
            frame,
            pupil_col="pupil",
            time_col="time",
            blink_padding_pre_ms=-1,
            group_cols=[],
        )

    with pytest.raises(ValueError, match="grouping column role not found"):
        pupil.flag_gazepoint_pupil_artifacts(
            frame,
            pupil_col="pupil",
            time_col="time",
            group_cols=["trial"],
        )


def test_final_pupil_legacy_interpolation_defaults_to_60_hz_without_time():
    result = pupil.interpolate_gazepoint_pupil(
        pd.DataFrame({"pupil": [1.0, np.nan, 3.0]}),
        pupil_col="pupil",
        output_col="filled",
        method="linear",
        max_gap_ms=100,
    )
    assert result.loc[1, "filled"] == pytest.approx(2.0)


def test_final_pupil_interpolation_unfilled_branch_and_no_groups(monkeypatch):
    monkeypatch.setattr(
        pupil.np,
        "interp",
        lambda x, xp, fp, left=None, right=None: np.full(len(np.atleast_1d(x)), np.nan),
    )
    result = pupil.interpolate_gazepoint_pupil(
        pd.DataFrame(
            {
                "time": [0.0, 1.0, 2.0],
                "pupil": [1.0, np.nan, 3.0],
            }
        ),
        pupil_col="pupil",
        time_col="time",
        group_cols=[],
        max_gap_ms=10_000,
    )
    assert result.loc[1, "pupil_interpolation_status"] == "missing_unfilled"


def test_final_pupil_pchip_failure_is_recorded(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("forced PCHIP failure")

    monkeypatch.setattr(scipy.interpolate, "PchipInterpolator", fail)
    result = pupil.interpolate_gazepoint_pupil_pchip(
        pd.DataFrame(
            {
                "time": [0.0, 1.0, 2.0, 3.0],
                "pupil": [1.0, np.nan, 2.0, 3.0],
            }
        ),
        pupil_col="pupil",
        time_col="time",
        grouping_cols=[],
        min_valid_points=3,
        max_gap_ms=None,
    )
    assert result.loc[1, "pchip_interpolation_status"] == "missing_pchip_failed"


def test_final_pupil_blink_interpolation_legacy_detects_mask_and_rejects_extra_kwargs():
    result = pupil.interpolate_gazepoint_blinks(
        data=pd.DataFrame(
            {
                "pupil": [1.0, np.nan, 3.0],
                "time": [0.0, 1.0, 2.0],
                "subject": ["S1"] * 3,
                "media_id": ["M1"] * 3,
            }
        ),
        pupil_col="pupil",
        time_col="time",
        group_cols=[],
    )
    assert "pupil_interpolated" in result

    with pytest.raises(TypeError, match="Unexpected keyword"):
        pupil.interpolate_gazepoint_blinks(
            data=pd.DataFrame({"USER_ID": ["S1"], "TIME": [0.0], "pupil": [1.0]}),
            blink_df=pd.DataFrame({"USER_ID": ["S1"], "start_time": [0.0], "end_time": [0.1]}),
            unexpected=True,
        )


def test_final_pupil_smoothing_scalar_group_and_custom_group_copy():
    frame = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S1"],
            "custom": ["A", "A", "A"],
            "time": [0.0, 1.0, 2.0],
            "pupil": [1.0, 2.0, 3.0],
        }
    )
    result = pupil.smooth_gazepoint_pupil(
        frame,
        pupil_col="pupil",
        time_col="time",
        group_cols="custom",
        window_samples=3,
        min_points=1,
    )
    assert len(result) == 3


def test_final_pupil_coordinate_legacy_median_and_r_invalid_method():
    legacy = pupil.smooth_gazepoint_coordinate(
        pd.DataFrame({"x": [1.0, 100.0, 3.0]}),
        column="x",
        window=3,
        method="not-moving-average",
    )
    assert legacy.loc[1, "x_smoothed"] == pytest.approx(3.0)

    with pytest.raises(ValueError, match="method must be"):
        pupil.smooth_gazepoint_coordinate(
            pd.DataFrame({"FPOGX": [0.1], "FPOGY": [0.2]}),
            x_col="FPOGX",
            y_col="FPOGY",
            method="moving_average",
        )


def test_final_pupil_baseline_window_is_converted_to_floats():
    result = pupil.baseline_correct_gazepoint_pupil(
        pd.DataFrame(
            {
                "subject": ["S1", "S1", "S1"],
                "media_id": ["M1", "M1", "M1"],
                "time": [-100.0, 0.0, 100.0],
                "pupil": [2.0, 2.0, 3.0],
            }
        ),
        pupil_col="pupil",
        time_col="time",
        baseline_window=(-100, 0),
        group_cols=["subject", "media_id"],
    )
    assert len(result) == 3


def test_final_pupil_gap_audit_uses_default_groups():
    result = pupil.audit_gazepoint_pupil_gaps(
        pd.DataFrame(
            {
                "subject": ["S1"],
                "media_id": ["M1"],
                "pupil_interpolation_status": ["observed"],
                "pupil_gap_id": pd.array([pd.NA], dtype="Int64"),
                "pupil_gap_n_samples": pd.array([pd.NA], dtype="Int64"),
                "pupil_gap_duration_ms": [np.nan],
                "pupil_was_interpolated": [False],
                "pupil_interpolated": [3.0],
            }
        )
    )
    assert not result.empty


def _baseline_audit_frame(baseline_n=1):
    return pd.DataFrame(
        {
            "subject": ["S1"],
            "media_id": ["M1"],
            "time": [0.0],
            "pupil_interpolated": [3.0],
            "pupil_baseline_n": [baseline_n],
            "pupil_baseline_status": ["complete"],
            "pupil_baseline_available": [True],
            "pupil_baseline_used": [True],
            "pupil_baseline_window_start": [-100.0],
            "pupil_baseline_window_end": [0.0],
            "pupil_was_interpolated": [False],
        }
    )


def test_final_pupil_baseline_audit_distinguishes_too_few_samples():
    result = pupil.audit_gazepoint_pupil_baseline(
        _baseline_audit_frame(1),
        min_baseline_samples=2,
    )
    assert result.loc[0, "baseline_quality_reason"] == "too_few_baseline_samples"


def test_final_pupil_drift_handles_all_missing_condition():
    result = pupil.audit_gazepoint_pupil_drift(
        pd.DataFrame(
            {
                "subject": ["S1", "S1", "S1"],
                "trial": [1, 2, 3],
                "condition": [pd.NA, pd.NA, pd.NA],
                "time": [0.0, 100.0, 200.0],
                "pupil": [3.0, 3.1, 3.2],
            }
        ),
        pupil_col="pupil",
        time_col="time",
        condition_col="condition",
    )
    assert result["condition_balance"].loc[0, "condition_balance_reason"] == "no_non_missing_conditions"


def test_final_pupil_summary_scalar_group_numeric_validation_and_empty_values():
    master = pd.DataFrame(
        {
            "subject": ["S1"],
            "media_id": ["M1"],
            "time": [np.nan],
            "pupil": [np.nan],
        }
    )
    result = pupil.summarise_gazepoint_pupil(
        master=master,
        group_cols="subject",
    )
    assert pd.isna(result.loc[0, "mean_pupil"])
    assert pd.isna(result.loc[0, "median_pupil"])
    assert pd.isna(result.loc[0, "iqr_outlier_pct"])

    with pytest.raises(ValueError, match="single numeric value"):
        pupil.summarise_gazepoint_pupil(
            master=master,
            min_pupil=object(),
        )


def test_final_pupil_windows_legacy_default_and_custom_group_copy():
    legacy = pupil.summarise_gazepoint_pupil_windows(
        pd.DataFrame({"time": [0.0, 100.0], "pupil": [3.0, 3.1]}),
        pupil_col="pupil",
        time_col="time",
        windows=None,
    )
    assert legacy.loc[0, "window"] == "window"

    r_result = pupil.summarise_gazepoint_pupil_windows(
        pd.DataFrame(
            {
                "time": [0.0, 50.0],
                "pupil": [3.0, 3.1],
                "custom": ["A", "A"],
            }
        ),
        pupil_col="pupil",
        time_col="time",
        windows=[0.0, 100.0],
        group_cols=["custom"],
    )
    assert r_result.loc[0, "custom"] == "A"


def test_final_pupil_group_cols_scalar_helper():
    assert pupil._gp3_pupil_r_group_cols("USER_ID", "trial") == ["USER_ID", "trial"]


def test_final_pupil_downsample_missing_required_and_empty_output():
    with pytest.raises(ValueError, match="missing required column"):
        pupil.downsample_gazepoint_pupil(
            master_df=pd.DataFrame({"USER_ID": ["S1"], "pupil": [3.0]}),
            pupil_cols=["pupil"],
            ts_col="TIME",
        )

    result = pupil.downsample_gazepoint_pupil(
        master_df=pd.DataFrame(
            {
                "USER_ID": pd.Series(dtype="object"),
                "TIME": pd.Series(dtype=float),
                "pupil": pd.Series(dtype=float),
            }
        ),
        pupil_cols=["pupil"],
        ts_col="TIME",
    )
    assert result.empty


def test_final_pupil_fit_line_and_binocular_fit_defensive_fallbacks(monkeypatch):
    monkeypatch.setattr(
        pupil.np.linalg,
        "lstsq",
        lambda *args, **kwargs: (np.array([np.nan, np.nan]), None, None, None),
    )
    coef = pupil._gp3_pupil_r_fit_line(
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 4.0, 6.0]),
    )
    assert np.isfinite(coef).all()

    fit = pupil._gp3_binoc_r_fit_one(
        np.array([1.0, 2.0, 3.0]),
        np.array([2.0, 4.0, 6.0]),
        min_pairs=2,
        min_unique=2,
        min_r2=None,
        allow_negative_slope=True,
        max_abs_slope=None,
    )
    assert fit["reason"] == "unstable_linear_fit"


def test_final_pupil_binocular_gap_with_nonfinite_time():
    result = pupil._gp3_binoc_r_gaps(
        pd.DataFrame({"time": [0.0, np.nan, 2.0]}),
        np.array([False, True, False]),
        [],
        time_col="time",
        time_unit="milliseconds",
    )
    assert pd.isna(result["gaps"].loc[0, "gap_ms"])


def test_final_pupil_binocular_level_specs_scalar_vector_fallback():
    specs = pupil._gp3_binoc_r_level_specs(["subject"], ["media_id"])
    assert specs == [["subject"], ["media_id"]]


def test_final_pupil_assign_models_skips_empty_level():
    calibration = {
        "models": pd.DataFrame(
            {
                "model_id": ["m1"],
                "direction": ["left_from_right"],
                "eligible": [True],
            }
        ),
        "levels": [
            {
                "group_cols": ["subject"],
                "models": pd.DataFrame(),
            }
        ],
    }
    selected = pupil._gp3_binoc_r_assign_models(
        pd.DataFrame({"subject": ["S1"]}),
        calibration,
        "left_from_right",
    )
    assert selected.tolist() == [-1]


def test_final_pupil_binocular_diagnosis_without_time():
    result = pupil.diagnose_gazepoint_binocular_pupil(
        pd.DataFrame(
            {
                "left": [1.0, 2.0, 3.0],
                "right": [1.1, 2.1, 3.1],
            }
        ),
        left_col="left",
        right_col="right",
        min_pairs=2,
        min_unique=2,
    )
    assert result["_gp3_class"] == "gp3_binocular_diagnostics"


def _pooled_calibration():
    models = pd.DataFrame(
        {
            "model_id": ["m1"],
            "direction": ["left_from_right"],
            "calibration_level": ["pooled"],
            "group_key": ["__pooled__"],
            "eligible": [True],
            "intercept": [100.0],
            "slope": [1.0],
            "r_squared": [1.0],
            "predictor_min": [0.0],
            "predictor_max": [10.0],
        }
    )
    return {
        "_gp3_class": "gp3_binocular_calibration",
        "models": models,
        "levels": [{"group_cols": [], "models": models.copy()}],
        "settings": {
            "left_col": "left",
            "right_col": "right",
            "group_cols": [],
        },
    }


def test_final_pupil_reconstruction_negative_gap_and_blocked_bounds():
    with pytest.raises(ValueError, match="max_gap_ms"):
        pupil.reconstruct_gazepoint_binocular_pupil(
            pd.DataFrame({"left": [1.0], "right": [1.0]}),
            left_col="left",
            right_col="right",
            max_gap_ms=-1,
        )

    result = pupil.reconstruct_gazepoint_binocular_pupil(
        pd.DataFrame({"left": [np.nan], "right": [3.0]}),
        left_col="left",
        right_col="right",
        calibration=_pooled_calibration(),
        min_pairs=2,
        min_unique=2,
        valid_max=10.0,
        max_gap_ms=np.inf,
    )
    assert result.loc[0, "gp3_binocular_status"] == "reconstruction_blocked_bounds"


def test_final_pupil_reconstruction_validation_skips_group_without_bilateral_rows():
    result = pupil.validate_gazepoint_binocular_reconstruction(
        pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "left": [1.0, 2.0, np.nan, np.nan],
                "right": [1.1, 2.1, 3.1, 4.1],
            }
        ),
        left_col="left",
        right_col="right",
        group_cols=["group"],
        gap_group_cols=["group"],
        fallback_group_cols=[[]],
        repeats=1,
        min_pairs=2,
        min_unique=2,
        mask_prop=0.5,
    )
    assert result is not None


def test_final_pupil_reconstruction_validation_contiguous_break():
    result = pupil.validate_gazepoint_binocular_reconstruction(
        pd.DataFrame(
            {
                "left": [1.0, 2.0, 3.0, 4.0],
                "right": [1.1, 2.1, 3.1, 4.1],
            }
        ),
        left_col="left",
        right_col="right",
        repeats=1,
        min_pairs=2,
        min_unique=2,
        mask_prop=0.25,
        mask_mode="contiguous",
        block_size=4,
        seed=1,
    )
    assert result is not None


def test_final_pupil_reconstruction_validation_forces_both_eye_directions():
    result = pupil.validate_gazepoint_binocular_reconstruction(
        pd.DataFrame(
            {
                "left": [1.0, 2.0, 3.0, 4.0],
                "right": [1.1, 2.1, 3.1, 4.1],
            }
        ),
        left_col="left",
        right_col="right",
        repeats=1,
        min_pairs=2,
        min_unique=2,
        mask_prop=0.5,
        mask_mode="random",
        direction="both",
        seed=0,
    )
    assert result is not None


def test_final_qc_overview_none_columns_are_nullable():
    result = qc.recommend_gazepoint_exclusions(
        pd.DataFrame(
            {
                "subject": ["S1", "S1"],
                "valid": [True, True],
            }
        ),
        participant_col="subject",
        validity_col="valid",
        min_trial_samples=1,
        min_participant_trials=1,
        min_participant_valid_trials=1,
    )
    overview = result["overview"].iloc[0]
    assert pd.isna(overview["trial_col"])
    assert pd.isna(overview["condition_col"])

def test_final_pupil_baseline_scalar_group_and_overlap_residuals():
    baseline = pupil.baseline_correct_gazepoint_pupil(
        pd.DataFrame(
            {
                "subject": ["S1", "S1"],
                "time": [-100.0, 50.0],
                "pupil": [3.0, 3.2],
            }
        ),
        pupil_col="pupil",
        time_col="time",
        baseline_window=(-100.0, 0.0),
        group_cols="subject",
    )
    assert len(baseline) == 2

    legacy = pupil.audit_gazepoint_pupil_overlap_risk(
        pd.DataFrame({"pupil": [3.0]})
    )
    assert legacy.loc[0, "status"] == "insufficient_columns"


def test_final_pupil_drift_condition_column_not_available():
    result = pupil.audit_gazepoint_pupil_drift(
        pd.DataFrame(
            {
                "subject": ["S1", "S1", "S1"],
                "trial": [1, 2, 3],
                "time": [0.0, 100.0, 200.0],
                "pupil": [3.0, 3.1, 3.2],
            }
        ),
        pupil_col="pupil",
        time_col="time",
        condition_col="",
    )
    assert (
        result["condition_balance"].loc[0, "condition_balance_reason"]
        == "condition_col_not_available"
    )


def test_final_pupil_overlap_risk_all_rows_excluded():
    result = pupil.audit_gazepoint_pupil_overlap_risk(
        pd.DataFrame(
            {
                "subject": ["S1"],
                "trial_global": [1],
                "time": [0.0],
                "stimulus_onset_time": [0.0],
                "target_onset_time": [500.0],
                "response_time": [1000.0],
                "excluded_trial": [True],
            }
        ),
        exclude_col="excluded_trial",
    )
    assert result["by_trial"].empty
    assert result["summary"].loc[0, "n_trials"] == 0


def test_final_pupil_contiguous_mask_break_when_target_covers_group(monkeypatch):
    monkeypatch.setattr(
        pupil,
        "_gp3_binoc_r_calibration",
        lambda *args, **kwargs: {},
    )

    def fake_reconstruct(data, left_col, right_col, **kwargs):
        out = data.copy()
        left = pd.to_numeric(out[left_col], errors="coerce")
        right = pd.to_numeric(out[right_col], errors="coerce")
        out["gp3_binocular_left_final"] = left.fillna(right)
        out["gp3_binocular_right_final"] = right.fillna(left)
        out["gp3_binocular_status"] = "test_reconstruction"
        out["gp3_binocular_model_id"] = pd.NA
        out["gp3_binocular_calibration_level"] = pd.NA
        out["gp3_binocular_r_squared"] = np.nan
        out["gp3_binocular_extrapolated"] = False
        out["gp3_binocular_gap_ms"] = np.nan
        return out

    monkeypatch.setattr(
        pupil,
        "reconstruct_gazepoint_binocular_pupil",
        fake_reconstruct,
    )

    result = pupil.validate_gazepoint_binocular_reconstruction(
        pd.DataFrame(
            {
                "left": [1.0, 2.0],
                "right": [1.1, 2.1],
            }
        ),
        left_col="left",
        right_col="right",
        repeats=1,
        mask_prop=0.75,
        mask_mode="contiguous",
        block_size=2,
        seed=1,
        min_pairs=1,
        min_unique=1,
    )
    assert result["metrics"]["n_requested"].sum() == 2

