import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3


def _linear(n=20, groups=False):
    time = np.arange(n, dtype=float) * 10.0
    left = np.linspace(2.0, 5.0, n)
    right = 0.4 + 1.1 * left
    data = {"time": time, "left": left, "right": right}
    if groups:
        data["subject"] = np.where(np.arange(n) < n // 2, "S1", "S2")
    return pd.DataFrame(data)


def test_legacy_binocular_paths_are_preserved():
    df = _linear(12)
    diag = gp3.diagnose_gazepoint_binocular_pupil(df, "left", "right")
    cal = gp3.fit_gazepoint_binocular_calibration(df, "left", "right")
    rec = gp3.reconstruct_gazepoint_binocular_pupil(df, "left", "right")
    val = gp3.validate_gazepoint_binocular_reconstruction(df, "left", "right", fraction=0.2)
    assert isinstance(diag, pd.DataFrame)
    assert set(cal) == {"right_from_left", "left_from_right"}
    assert "pupil_combined" in rec
    assert isinstance(val, pd.DataFrame)


def test_r_diagnostics_summary_gaps_and_settings():
    df = _linear(20, groups=True)
    df.loc[3:4, "left"] = np.nan
    df.loc[15, "right"] = np.nan
    out = gp3.diagnose_gazepoint_binocular_pupil(
        df,
        "left",
        "right",
        time_col="time",
        group_cols=["subject"],
        min_pairs=5,
        min_unique=4,
    )
    assert out["_gp3_class"] == "gp3_binocular_diagnostics"
    assert len(out["summary"]) == 2
    assert {"correlation", "rank_correlation", "calibration_eligible", "status"} <= set(
        out["summary"]
    )
    assert set(out["gaps"]["eye"]) == {"left", "right"}
    assert out["settings"]["time_unit"] == "auto"
    assert out["summary"]["longest_left_gap_ms"].max() == pytest.approx(20.0)


def test_r_diagnostics_bounds_seconds_and_time_diagnostics():
    df = pd.DataFrame(
        {
            "g": ["A"] * 6,
            "time": [0.0, 0.1, 0.1, 0.05, 0.4, 0.5],
            "left": [2.0, 2.1, 99.0, 2.3, 2.4, 2.5],
            "right": [2.1, 2.2, 2.3, 2.4, 2.5, 2.6],
        }
    )
    out = gp3.diagnose_gazepoint_binocular_pupil(
        df,
        "left",
        "right",
        time_col="time",
        group_cols=["g"],
        time_unit="seconds",
        valid_max=10,
        min_pairs=2,
        min_unique=2,
        disagreement_mad_k=0,
    )
    row = out["summary"].iloc[0]
    assert row["n_left"] == 5
    assert row["duplicate_time_count"] == 1
    assert bool(row["time_unsorted"])
    assert row["disagreement_fraction"] >= 0


def test_r_calibration_levels_models_and_fallback():
    df = _linear(20, groups=True)
    out = gp3.fit_gazepoint_binocular_calibration(
        df,
        "left",
        "right",
        group_cols=["subject"],
        min_pairs=5,
        min_unique=4,
        min_r2=0.8,
    )
    assert out["_gp3_class"] == "gp3_binocular_calibration"
    assert len(out["levels"]) == 2
    assert set(out["models"]["calibration_level"]) == {"subject", "pooled"}
    assert out["models"]["eligible"].all()
    assert out["models"]["model_index"].tolist() == list(range(1, len(out["models"]) + 1))


def test_r_calibration_ineligibility_reasons_and_limits():
    flat = pd.DataFrame({"left": [1.0] * 6, "right": np.arange(6.0)})
    out = gp3.fit_gazepoint_binocular_calibration(
        flat,
        "left",
        "right",
        group_cols=[],
        fallback_group_cols=[],
        min_pairs=2,
        min_unique=2,
    )
    assert "insufficient_unique_values" in set(out["models"]["reason"])

    neg = pd.DataFrame({"left": np.arange(1.0, 8.0), "right": -np.arange(1.0, 8.0)})
    out2 = gp3.fit_gazepoint_binocular_calibration(
        neg,
        "left",
        "right",
        group_cols=[],
        min_pairs=3,
        min_unique=3,
    )
    assert set(out2["models"]["reason"]) == {"non_positive_slope"}


def test_r_linear_reconstruction_and_metadata():
    df = _linear(20)
    original = df.copy()
    df.loc[5:6, "left"] = np.nan
    df.loc[12, "right"] = np.nan
    out = gp3.reconstruct_gazepoint_binocular_pupil(
        df,
        "left",
        "right",
        time_col="time",
        group_cols=[],
        min_pairs=5,
        min_unique=4,
    )
    assert out.loc[5:6, "gp3_binocular_left_reconstructed"].all()
    assert bool(out.loc[12, "gp3_binocular_right_reconstructed"])
    assert out.loc[5:6, "gp3_binocular_left_final"].notna().all()
    assert pd.isna(df.loc[5, "left"])
    assert original.loc[5, "left"] == pytest.approx(original.loc[5, "left"])
    assert out.attrs["gp3_binocular_reconstruction"]["method"] == "linear_regression"
    assert (
        out.attrs["gp3_binocular_reconstruction"]["calibration"]["_gp3_class"]
        == "gp3_binocular_calibration"
    )


def test_r_available_eye_and_none_do_not_synthesize_missing_eye():
    df = _linear(10)
    df.loc[3, "left"] = np.nan
    for method in ("available_eye", "none"):
        out = gp3.reconstruct_gazepoint_binocular_pupil(
            df,
            "left",
            "right",
            method=method,
            group_cols=[],
        )
        assert pd.isna(out.loc[3, "gp3_binocular_left_final"])
        assert not bool(out.loc[3, "gp3_binocular_reconstructed"])
        assert out.loc[3, "gp3_binocular_status"] == "right_only_observed"


def test_r_reconstruction_gap_edge_and_exclusion_blocks():
    df = _linear(20)
    df.loc[5:6, "left"] = np.nan
    blocked = gp3.reconstruct_gazepoint_binocular_pupil(
        df,
        "left",
        "right",
        time_col="time",
        group_cols=[],
        min_pairs=5,
        max_gap_ms=10,
    )
    assert set(blocked.loc[5:6, "gp3_binocular_status"]) == {"reconstruction_blocked_gap"}

    edge = _linear(20)
    edge.loc[0, "left"] = np.nan
    blocked_edge = gp3.reconstruct_gazepoint_binocular_pupil(
        edge,
        "left",
        "right",
        time_col="time",
        group_cols=[],
        min_pairs=5,
        allow_edge_gaps=False,
    )
    assert blocked_edge.loc[0, "gp3_binocular_status"] == "reconstruction_blocked_edge"

    ex = _linear(20)
    ex["bad"] = False
    ex.loc[7, "left"] = np.nan
    ex.loc[7, "bad"] = True
    blocked_ex = gp3.reconstruct_gazepoint_binocular_pupil(
        ex,
        "left",
        "right",
        group_cols=[],
        min_pairs=5,
        exclude_flag_cols=["bad"],
    )
    assert blocked_ex.loc[7, "gp3_binocular_status"] == "reconstruction_blocked_exclusion"


def test_r_reconstruction_extrapolation_and_overwrite_controls():
    train = _linear(12)
    cal = gp3.fit_gazepoint_binocular_calibration(
        train,
        "left",
        "right",
        group_cols=[],
        min_pairs=5,
        min_unique=4,
    )
    apply = train.copy()
    apply.loc[3, "left"] = np.nan
    apply.loc[3, "right"] = 100.0
    out = gp3.reconstruct_gazepoint_binocular_pupil(
        apply,
        "left",
        "right",
        group_cols=[],
        calibration=cal,
    )
    assert out.loc[3, "gp3_binocular_status"] == "reconstruction_blocked_extrapolation"

    with pytest.raises(ValueError):
        gp3.reconstruct_gazepoint_binocular_pupil(
            out,
            "left",
            "right",
            group_cols=[],
            calibration=cal,
        )
    overwritten = gp3.reconstruct_gazepoint_binocular_pupil(
        out,
        "left",
        "right",
        group_cols=[],
        calibration=cal,
        overwrite=True,
    )
    assert "gp3_binocular_status" in overwritten


def test_r_reconstruction_ineligible_and_bad_calibration_paths():
    df = _linear(8)
    df.loc[2, "left"] = np.nan
    out = gp3.reconstruct_gazepoint_binocular_pupil(
        df,
        "left",
        "right",
        group_cols=[],
        min_pairs=20,
    )
    assert out.loc[2, "gp3_binocular_status"] == "reconstruction_ineligible"
    with pytest.raises(TypeError):
        gp3.reconstruct_gazepoint_binocular_pupil(
            df, "left", "right", group_cols=[], calibration={}
        )
    other = gp3.fit_gazepoint_binocular_calibration(
        df.rename(columns={"left": "L2"}),
        "L2",
        "right",
        group_cols=[],
        min_pairs=3,
    )
    with pytest.raises(ValueError):
        gp3.reconstruct_gazepoint_binocular_pupil(
            df, "left", "right", group_cols=[], calibration=other
        )


def test_r_reconstruction_argument_validation():
    df = _linear(8)
    with pytest.raises(ValueError):
        gp3.reconstruct_gazepoint_binocular_pupil(df, "left", "right", group_cols=[], max_gap_ms=1)
    with pytest.raises(ValueError):
        gp3.reconstruct_gazepoint_binocular_pupil(df, "left", "right", group_cols=[], method="bad")
    bad = df.assign(flag="bad")
    with pytest.raises(TypeError):
        gp3.reconstruct_gazepoint_binocular_pupil(
            bad, "left", "right", group_cols=[], exclude_flag_cols=["flag"]
        )


def test_r_reconstruction_audit_descriptive_and_thresholds():
    df = _linear(20, groups=True)
    df.loc[[3, 12], "left"] = np.nan
    rec = gp3.reconstruct_gazepoint_binocular_pupil(
        df,
        "left",
        "right",
        group_cols=[],
        min_pairs=5,
    )
    descriptive = gp3.audit_gazepoint_binocular_reconstruction(rec)
    assert descriptive["_gp3_class"] == "gp3_binocular_audit"
    assert descriptive["audit"].iloc[0]["status"] == "descriptive"
    reviewed = gp3.audit_gazepoint_binocular_reconstruction(
        rec,
        by=["subject"],
        max_reconstruction_prop=0.01,
        max_group_rate_difference=0.01,
    )
    assert reviewed["audit"].iloc[0]["status"] == "review"
    assert len(reviewed["by_group"]) == 2
    assert reviewed["status_counts"]["n"].sum() == len(rec)


def test_r_audit_requires_metadata_and_valid_thresholds():
    with pytest.raises(ValueError):
        gp3.audit_gazepoint_binocular_reconstruction(
            _linear(8),
            by=[],
        )
    df = _linear(10)
    df.loc[3, "left"] = np.nan
    rec = gp3.reconstruct_gazepoint_binocular_pupil(df, "left", "right", group_cols=[], min_pairs=3)
    with pytest.raises(ValueError):
        gp3.audit_gazepoint_binocular_reconstruction(rec, max_reconstruction_prop=2)


def test_r_validation_random_left_direction():
    df = _linear(30)
    out = gp3.validate_gazepoint_binocular_reconstruction(
        df,
        "left",
        "right",
        group_cols=[],
        direction="left_from_right",
        mask_prop=0.2,
        repeats=2,
        seed=7,
        min_pairs=8,
        min_unique=5,
    )
    assert out["_gp3_class"] == "gp3_binocular_validation"
    assert set(out["summary"]["direction"]) == {"left_from_right"}
    assert len(out["metrics"]) == 2
    assert out["predictions"]["predicted"].notna().any()
    assert out["settings"]["seed"] == 7


def test_r_validation_both_and_contiguous_modes():
    df = _linear(30, groups=True)
    both = gp3.validate_gazepoint_binocular_reconstruction(
        df,
        "left",
        "right",
        group_cols=["subject"],
        direction="both",
        mask_prop=0.2,
        repeats=1,
        seed=2,
        min_pairs=5,
    )
    assert set(both["metrics"]["direction"]) == {"left_from_right", "right_from_left"}

    contiguous = gp3.validate_gazepoint_binocular_reconstruction(
        df,
        "left",
        "right",
        time_col="time",
        group_cols=["subject"],
        gap_group_cols=["subject"],
        mask_mode="contiguous",
        block_size=2,
        mask_prop=0.2,
        repeats=1,
        seed=3,
        min_pairs=5,
        max_gap_ms=30,
    )
    assert not contiguous["metrics"].empty
    assert contiguous["settings"]["mask_mode"] == "contiguous"


def test_r_validation_errors():
    df = _linear(10)
    with pytest.raises(ValueError):
        gp3.validate_gazepoint_binocular_reconstruction(
            df, "left", "right", group_cols=[], mask_prop=1.0
        )
    with pytest.raises(ValueError):
        gp3.validate_gazepoint_binocular_reconstruction(
            df, "left", "right", group_cols=[], direction="wrong"
        )
    sparse = df.copy()
    sparse.loc[1:, "left"] = np.nan
    with pytest.raises(ValueError):
        gp3.validate_gazepoint_binocular_reconstruction(sparse, "left", "right", group_cols=[])


def test_r_calibration_argument_validation():
    df = _linear(8)
    with pytest.raises(ValueError):
        gp3.fit_gazepoint_binocular_calibration(df, "left", "right", group_cols=[], min_pairs=1)
    with pytest.raises(ValueError):
        gp3.fit_gazepoint_binocular_calibration(df, "left", "right", group_cols=[], min_r2=2)
    with pytest.raises(ValueError):
        gp3.fit_gazepoint_binocular_calibration(
            df, "left", "right", group_cols=[], max_abs_slope=-1
        )
