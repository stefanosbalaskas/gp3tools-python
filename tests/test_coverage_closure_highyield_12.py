from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

aoi = importlib.import_module("gp3tools.aoi")
pupil = importlib.import_module("gp3tools.pupil")
qc = importlib.import_module("gp3tools.qc")


# ============================================================================
# AOI — DYNAMIC AOI VALIDATION
# ============================================================================


def _dynamic_frame():
    return pd.DataFrame(
        {
            "TIME": [0.0, 1.0],
            "FPOGX": [0.5, 0.6],
            "FPOGY": [0.5, 0.6],
        }
    )


def _dynamic_rect():
    return pd.DataFrame(
        {
            "aoi_time": [0.0],
            "aoi_name": ["A"],
            "left": [0.0],
            "right": [1.0],
            "top": [0.0],
            "bottom": [1.0],
        }
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shape": "bad"},
        {"match": "bad"},
        {"output": "bad"},
        {"overlap": "bad"},
        {"boundary": "bad"},
        {"max_time_gap": -1},
    ],
)
def test_h12_dynamic_aoi_validation(kwargs):
    with pytest.raises(ValueError):
        aoi.add_gazepoint_dynamic_aoi(
            _dynamic_frame(),
            _dynamic_rect(),
            **kwargs,
        )


def test_h12_dynamic_aoi_auto_shape_failure():
    defs = pd.DataFrame(
        {
            "aoi_time": [0.0],
            "aoi_name": ["A"],
        }
    )

    with pytest.raises(ValueError):
        aoi.add_gazepoint_dynamic_aoi(
            _dynamic_frame(),
            defs,
            shape="auto",
        )


def test_h12_dynamic_aoi_missing_geometry_columns():
    defs = _dynamic_rect().drop(columns=["right"])

    with pytest.raises(KeyError):
        aoi.add_gazepoint_dynamic_aoi(
            _dynamic_frame(),
            defs,
            shape="rectangle",
        )


def test_h12_dynamic_aoi_blank_name():
    defs = _dynamic_rect()
    defs["aoi_name"] = ""

    with pytest.raises(ValueError):
        aoi.add_gazepoint_dynamic_aoi(
            _dynamic_frame(),
            defs,
            shape="rectangle",
        )


def test_h12_dynamic_aoi_missing_frame_column():
    with pytest.raises(KeyError):
        aoi.add_gazepoint_dynamic_aoi(
            _dynamic_frame().drop(columns=["FPOGX"]),
            _dynamic_rect(),
            shape="rectangle",
        )


# ============================================================================
# AOI — OVERLAP / SCREEN COVERAGE VALIDATION
# ============================================================================


def _geometry_frame():
    return pd.DataFrame(
        {
            "aoi": ["A", "B"],
            "x_min": [0.0, 0.5],
            "x_max": [0.6, 1.1],
            "y_min": [0.0, 0.0],
            "y_max": [1.0, 1.0],
        }
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_overlap_area": True},
        {"min_overlap_area": -1},
        {"min_overlap_prop": -0.1},
        {"min_overlap_prop": 1.1},
        {"ignore_invalid_geometry": "yes"},
    ],
)
def test_h12_overlap_r_validation(kwargs):
    with pytest.raises(ValueError):
        aoi.audit_gazepoint_aoi_overlap(
            _geometry_frame(),
            aoi_col="aoi",
            **kwargs,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 0},
        {"height": 0},
        {"width": np.inf},
        {"height": np.inf},
        {"margin": -1},
    ],
)
def test_h12_screen_coverage_validation(kwargs):
    with pytest.raises(ValueError):
        aoi.audit_gazepoint_aoi_screen_coverage(
            _geometry_frame(),
            aoi_col="aoi",
            **kwargs,
        )


def test_h12_screen_coverage_missing_geometry():
    with pytest.raises(ValueError):
        aoi.audit_gazepoint_aoi_screen_coverage(
            _geometry_frame().drop(columns=["x_max"]),
            aoi_col="aoi",
        )


def test_h12_screen_coverage_missing_invalid_offscreen():
    frame = pd.DataFrame(
        {
            "aoi": [
                "missing",
                "invalid",
                "offscreen",
            ],
            "x_min": [
                np.nan,
                0.8,
                -0.2,
            ],
            "x_max": [
                np.nan,
                0.2,
                0.4,
            ],
            "y_min": [
                0.0,
                0.0,
                0.0,
            ],
            "y_max": [
                1.0,
                1.0,
                1.2,
            ],
        }
    )

    result = aoi.audit_gazepoint_aoi_screen_coverage(
        frame,
        aoi_col="aoi",
    )

    summary = result["overall_summary"].iloc[0]

    assert summary["n_missing_geometry"] == 1
    assert summary["n_invalid_rectangles"] == 1
    assert summary["n_outside_screen"] >= 1


# ============================================================================
# AOI — EMPIRICAL LOGIT RESIDUALS
# ============================================================================


def test_h12_empirical_logit_empty():
    with pytest.raises(ValueError):
        aoi.transform_gazepoint_aoi_empirical_logit(pd.DataFrame())


def test_h12_empirical_logit_missing_mode():
    with pytest.raises(ValueError):
        aoi.transform_gazepoint_aoi_empirical_logit(pd.DataFrame({"x": [1]}))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"correction": True},
        {"correction": 0},
        {"correction": np.inf},
        {"pseudo_denominator": 0},
        {"overwrite": "yes"},
        {"name": ""},
    ],
)
def test_h12_empirical_logit_parameter_validation(kwargs):
    with pytest.raises(ValueError):
        aoi.transform_gazepoint_aoi_empirical_logit(
            pd.DataFrame({"prop": [0.5]}),
            proportion_col="prop",
            **kwargs,
        )


def test_h12_empirical_logit_bad_column():
    with pytest.raises(ValueError):
        aoi.transform_gazepoint_aoi_empirical_logit(
            pd.DataFrame({"prop": [0.5]}),
            proportion_col="missing",
        )


def test_h12_empirical_logit_out_of_bounds():
    result = aoi.transform_gazepoint_aoi_empirical_logit(
        pd.DataFrame(
            {
                "prop": [
                    -0.1,
                    1.1,
                ]
            }
        ),
        proportion_col="prop",
        pseudo_denominator=10,
    )

    assert set(result["aoi_empirical_logit_status"]) == {"proportion_out_of_bounds"}


def test_h12_empirical_logit_all_nonfinite_overview():
    result = aoi.transform_gazepoint_aoi_empirical_logit(
        pd.DataFrame(
            {
                "prop": [
                    np.nan,
                    np.inf,
                ]
            }
        ),
        proportion_col="prop",
    )

    overview = result.attrs["gp3_empirical_logit_overview"].iloc[0]

    assert np.isnan(overview["min_raw_proportion"])

    assert np.isnan(overview["max_empirical_logit"])


# ============================================================================
# AOI — DENOMINATOR AUDIT
# ============================================================================


def _denominator_frame():
    return pd.DataFrame(
        {
            "window_label": ["w1"],
            "n_window_samples": [10],
            "n_valid_denominator_samples": [8],
            "n_target_samples": [3],
        }
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_denominator_samples": 0},
        {"min_valid_denominator_prop": -0.1},
        {"min_valid_denominator_prop": 1.1},
        {"max_denominator_cv": 0},
        {"max_condition_ratio": 0},
    ],
)
def test_h12_denominator_threshold_validation(kwargs):
    with pytest.raises(ValueError):
        aoi.audit_gazepoint_aoi_window_denominators(
            _denominator_frame(),
            group_cols=[],
            **kwargs,
        )


def test_h12_denominator_missing_condition_fallback():
    result = aoi.audit_gazepoint_aoi_window_denominators(
        _denominator_frame(),
        group_cols=[],
    )

    assert result["condition_window_summary"]["condition"].iloc[0] == "all_data"


def test_h12_denominator_unknown_window():
    frame = _denominator_frame()
    frame["window_label"] = ""

    result = aoi.audit_gazepoint_aoi_window_denominators(
        frame,
        group_cols=[],
    )

    row = result["row_audit"].iloc[0]

    # Raw source label is intentionally preserved.
    assert row["window_label"] == ""

    # Normalised audit label is stored separately.
    assert row[".gp3_window"] == "unknown_window"

    assert (
        result["window_summary"].loc[
            0,
            "window_label",
        ]
        == "unknown_window"
    )


def test_h12_luminance_float_above_one(monkeypatch, tmp_path):
    import matplotlib.image as mpimg

    path = tmp_path / "float255.png"
    path.write_bytes(b"x")

    monkeypatch.setattr(
        mpimg,
        "imread",
        lambda _: np.array(
            [
                [
                    [255.0, 255.0, 255.0],
                    [0.0, 0.0, 0.0],
                ]
            ],
            dtype=float,
        ),
    )

    result = pupil._gp3_final_read_luminance(
        "stim",
        str(path),
        None,
        True,
    )

    assert result["luminance_available"]
    assert result["max_luminance"] <= 1


def test_h12_luminance_no_finite_values(monkeypatch, tmp_path):
    import matplotlib.image as mpimg

    path = tmp_path / "nan.png"
    path.write_bytes(b"x")

    monkeypatch.setattr(
        mpimg,
        "imread",
        lambda _: np.full(
            (
                2,
                2,
                3,
            ),
            np.nan,
        ),
    )

    result = pupil._gp3_final_read_luminance(
        "stim",
        str(path),
        None,
        True,
    )

    assert result["luminance_status"] == "read_error"


# ============================================================================
# PUPIL — DRIFT
# ============================================================================


def _drift_frame(excluded):
    return pd.DataFrame(
        {
            "subject": ["S1"] * 6,
            "condition": [
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
            ],
            "trial": [
                1,
                2,
                3,
                4,
                5,
                6,
            ],
            "time": [
                0,
                1000,
                2000,
                3000,
                4000,
                5000,
            ],
            "pupil": [
                3.0,
                3.1,
                3.2,
                3.3,
                3.4,
                3.5,
            ],
            "excluded": excluded,
        }
    )


@pytest.mark.parametrize(
    "excluded",
    [
        [
            False,
            True,
            False,
            False,
            False,
            False,
        ],
        [
            0,
            1,
            0,
            0,
            0,
            0,
        ],
        [
            "no",
            "yes",
            "no",
            "no",
            "no",
            "no",
        ],
    ],
)
def test_h12_drift_exclusion_encodings(excluded):
    result = pupil.audit_gazepoint_pupil_drift(
        _drift_frame(excluded),
        pupil_col="pupil",
        time_col="time",
        group_cols="subject",
        order_col="trial",
        condition_col="condition",
        exclude_col="excluded",
        min_valid_samples=2,
    )

    assert not result["by_group"].empty


@pytest.mark.parametrize(
    "kwargs",
    [
        {"group_cols": []},
        {"group_cols": ["subject", "subject"]},
        {"min_valid_samples": 0},
        {"max_abs_slope_per_min": np.inf},
        {"max_condition_time_mean_diff_ms": np.inf},
        {"max_condition_order_mean_diff": np.inf},
    ],
)
def test_h12_drift_validation(kwargs):
    call = {
        "pupil_col": "pupil",
        "time_col": "time",
        "group_cols": "subject",
        "order_col": "trial",
        "condition_col": "condition",
        "min_valid_samples": 2,
    }

    call.update(kwargs)

    with pytest.raises(ValueError):
        pupil.audit_gazepoint_pupil_drift(
            _drift_frame([False] * 6),
            **call,
        )


# ============================================================================
# PUPIL — OVERLAP RISK STATUS MATRIX
# ============================================================================


def _overlap_status_frame(
    event2,
):
    return pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
            ],
            "trial_global": [
                "T1",
                "T1",
            ],
            "time": [
                0.0,
                1000.0,
            ],
            "event1": [
                0.0,
                0.0,
            ],
            "event2": [
                event2,
                event2,
            ],
        }
    )


def test_h12_overlap_short_only():
    result = pupil.audit_gazepoint_pupil_overlap_risk(
        _overlap_status_frame(500.0),
        trial_col="trial_global",
        time_col="time",
        group_cols="subject",
        event_time_cols=[
            "event1",
            "event2",
        ],
        window_start_ms=0,
        window_end_ms=100,
        min_event_gap_ms=1000,
    )

    assert "short_event_gap" in set(result["event_gaps"]["event_gap_status"])


def test_h12_overlap_window_only():
    result = pupil.audit_gazepoint_pupil_overlap_risk(
        _overlap_status_frame(500.0),
        trial_col="trial_global",
        time_col="time",
        group_cols="subject",
        event_time_cols=[
            "event1",
            "event2",
        ],
        window_start_ms=0,
        window_end_ms=1000,
        min_event_gap_ms=100,
    )

    assert "overlapping_response_window" in set(result["event_gaps"]["event_gap_status"])


def test_h12_overlap_ok():
    result = pupil.audit_gazepoint_pupil_overlap_risk(
        _overlap_status_frame(500.0),
        trial_col="trial_global",
        time_col="time",
        group_cols="subject",
        event_time_cols=[
            "event1",
            "event2",
        ],
        window_start_ms=0,
        window_end_ms=100,
        min_event_gap_ms=100,
    )

    assert "ok" in set(result["event_gaps"]["event_gap_status"])


# ============================================================================
# PUPIL — LUMINANCE AUDIT STATUS MATRIX
# ============================================================================


def _fake_luminance_row(
    stimulus_id,
    stimulus_file,
    *,
    available,
    mean=np.nan,
):
    return {
        "stimulus_id": stimulus_id,
        "stimulus_file": stimulus_file,
        "resolved_path": stimulus_file,
        "file_exists": available,
        "luminance_available": available,
        "image_width_px": 10 if available else np.nan,
        "image_height_px": 10 if available else np.nan,
        "n_pixels": 100 if available else np.nan,
        "mean_luminance": mean,
        "median_luminance": mean,
        "sd_luminance": 0.1 if available else np.nan,
        "min_luminance": mean,
        "max_luminance": mean,
        "mean_brightness": mean,
        "rms_contrast": 0.1 if available else np.nan,
        "michelson_contrast": 0.1 if available else np.nan,
        "luminance_status": ("available" if available else "file_missing"),
        "error_message": None,
    }


def test_h12_luminance_audit_no_available(monkeypatch):
    monkeypatch.setattr(
        pupil,
        "_gp3_final_read_luminance",
        lambda sid, file, image_dir, recursive: _fake_luminance_row(
            sid,
            file,
            available=False,
        ),
    )

    result = pupil.audit_gazepoint_stimulus_luminance(
        pd.DataFrame(
            {
                "stimulus_file": ["a.png"],
                "stimulus_id": ["A"],
                "condition": ["C1"],
            }
        ),
        stimulus_file_col="stimulus_file",
        stimulus_id_col="stimulus_id",
        condition_col="condition",
    )

    assert (
        result["overview"].loc[
            0,
            "audit_status",
        ]
        == "no_luminance_available"
    )

    assert (
        result["balance_summary"].loc[
            0,
            "luminance_balance_status",
        ]
        == "no_luminance_available"
    )


def test_h12_luminance_audit_single_condition(monkeypatch):
    monkeypatch.setattr(
        pupil,
        "_gp3_final_read_luminance",
        lambda sid, file, image_dir, recursive: _fake_luminance_row(
            sid,
            file,
            available=True,
            mean=0.5,
        ),
    )

    result = pupil.audit_gazepoint_stimulus_luminance(
        pd.DataFrame(
            {
                "stimulus_file": ["a.png"],
                "stimulus_id": ["A"],
                "condition": ["C1"],
            }
        ),
        stimulus_file_col="stimulus_file",
        stimulus_id_col="stimulus_id",
        condition_col="condition",
    )

    assert (
        result["balance_summary"].loc[
            0,
            "luminance_balance_status",
        ]
        == "single_condition_available"
    )


def test_h12_luminance_audit_two_conditions(monkeypatch):
    values = {
        "A": 0.3,
        "B": 0.7,
    }

    monkeypatch.setattr(
        pupil,
        "_gp3_final_read_luminance",
        lambda sid, file, image_dir, recursive: _fake_luminance_row(
            sid,
            file,
            available=True,
            mean=values[sid],
        ),
    )

    result = pupil.audit_gazepoint_stimulus_luminance(
        pd.DataFrame(
            {
                "stimulus_file": [
                    "a.png",
                    "b.png",
                ],
                "stimulus_id": [
                    "A",
                    "B",
                ],
                "condition": [
                    "C1",
                    "C2",
                ],
            }
        ),
        stimulus_file_col="stimulus_file",
        stimulus_id_col="stimulus_id",
        condition_col="condition",
    )

    assert (
        result["balance_summary"].loc[
            0,
            "luminance_balance_status",
        ]
        == "condition_luminance_summarised"
    )


# ============================================================================
# PUPIL — WINDOW DEFINITIONS / STATUS
# ============================================================================


def _window_frame():
    return pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S1",
            ],
            "time": [
                0.0,
                250.0,
                500.0,
            ],
            "pupil": [
                1.0,
                np.nan,
                2.0,
            ],
        }
    )


def test_h12_pupil_windows_generated_labels():
    windows = pd.DataFrame(
        {
            "start": [
                0,
                250,
            ],
            "end": [
                250,
                500,
            ],
        }
    )

    result = pupil.summarise_gazepoint_pupil_windows(
        _window_frame(),
        windows=windows,
        group_cols="subject",
    )

    assert set(result["window_label"]) == {
        "0_250ms",
        "250_500ms",
    }


def test_h12_pupil_windows_blank_label():
    windows = pd.DataFrame(
        {
            "label": [""],
            "start": [0],
            "end": [500],
        }
    )

    with pytest.raises(ValueError):
        pupil.summarise_gazepoint_pupil_windows(
            _window_frame(),
            windows=windows,
            group_cols="subject",
        )


def test_h12_pupil_windows_backwards():
    windows = pd.DataFrame(
        {
            "label": ["x"],
            "start": [500],
            "end": [0],
        }
    )

    with pytest.raises(ValueError):
        pupil.summarise_gazepoint_pupil_windows(
            _window_frame(),
            windows=windows,
            group_cols="subject",
        )


def test_h12_pupil_windows_min_valid_status():
    result = pupil.summarise_gazepoint_pupil_windows(
        _window_frame(),
        windows=[
            0,
            500,
        ],
        group_cols="subject",
        min_valid_samples=3,
        include_window_end=True,
    )

    assert (
        result.loc[
            0,
            "pupil_window_status",
        ]
        != "ok"
    )


# ============================================================================
# PUPIL — BINOCULAR DIAGNOSTICS
# ============================================================================


def _binocular_diag():
    return pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S1",
                "S1",
            ],
            "time": [
                0.0,
                1.0,
                1.0,
                0.5,
            ],
            "left_pupil": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            "right_pupil": [
                1.1,
                2.1,
                3.1,
                4.1,
            ],
        }
    )


def test_h12_binocular_diagnostic_eligible():
    result = pupil.diagnose_gazepoint_binocular_pupil(
        _binocular_diag(),
        group_cols="subject",
        time_col="time",
        min_pairs=2,
        min_unique=2,
    )

    row = result["summary"].iloc[0]

    assert row["calibration_eligible"]

    assert row["duplicate_time_count"] == 1

    assert row["time_unsorted"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"time_unit": "bad"},
        {"min_pairs": 1},
        {"min_unique": 1},
        {"disagreement_mad_k": -1},
    ],
)
def test_h12_binocular_diagnostic_validation(kwargs):
    call = {
        "time_col": "time",
        "group_cols": "subject",
    }

    call.update(kwargs)

    with pytest.raises(ValueError):
        pupil.diagnose_gazepoint_binocular_pupil(
            _binocular_diag(),
            **call,
        )


# ============================================================================
# PUPIL — LEGACY AVAILABLE-EYE RECONSTRUCTION
# ============================================================================


def test_h12_available_eye_reconstruction():
    frame = pd.DataFrame(
        {
            "left_pupil": [
                1.0,
                np.nan,
            ],
            "right_pupil": [
                np.nan,
                2.0,
            ],
        }
    )

    result = pupil.reconstruct_gazepoint_binocular_pupil(
        frame,
        method="available_eye",
    )

    assert result.loc[
        0,
        "right_pupil_reconstructed",
    ] == pytest.approx(1.0)

    assert result.loc[
        1,
        "left_pupil_reconstructed",
    ] == pytest.approx(2.0)


# ============================================================================
# PUPIL — RECONSTRUCTION AUDIT
# ============================================================================


def _auditable_reconstruction():
    frame = pd.DataFrame(
        {
            "group": [
                "A",
                "A",
                "B",
                "B",
            ],
            "gp3_binocular_status": [
                "bilateral_observed",
                "left_reconstructed",
                "bilateral_observed",
                "right_reconstructed",
            ],
            "gp3_binocular_reconstructed": [
                False,
                True,
                False,
                True,
            ],
            "gp3_binocular_left_observed": [
                1.0,
                np.nan,
                2.0,
                3.0,
            ],
            "gp3_binocular_right_observed": [
                1.0,
                1.5,
                2.0,
                np.nan,
            ],
            "gp3_binocular_left_final": [
                1.0,
                1.4,
                2.0,
                3.0,
            ],
            "gp3_binocular_right_final": [
                1.0,
                1.5,
                2.0,
                3.1,
            ],
        }
    )

    frame.attrs["gp3_binocular_reconstruction"] = {
        "calibration": None,
    }

    return frame


def test_h12_reconstruction_audit_review():
    result = pupil.audit_gazepoint_binocular_reconstruction(
        _auditable_reconstruction(),
        by="group",
        max_reconstruction_prop=0.1,
        max_group_rate_difference=0.1,
    )

    assert (
        result["audit"].loc[
            0,
            "status",
        ]
        == "review"
    )

    assert result["audit"].loc[
        0,
        "burden_flag",
    ]


def test_h12_reconstruction_audit_ok():
    result = pupil.audit_gazepoint_binocular_reconstruction(
        _auditable_reconstruction(),
        by="group",
        max_reconstruction_prop=1.0,
        max_group_rate_difference=1.0,
    )

    assert (
        result["audit"].loc[
            0,
            "status",
        ]
        == "ok"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_reconstruction_prop": -0.1},
        {"max_reconstruction_prop": 1.1},
        {"max_group_rate_difference": -0.1},
        {"max_group_rate_difference": 1.1},
    ],
)
def test_h12_reconstruction_audit_validation(kwargs):
    with pytest.raises(ValueError):
        pupil.audit_gazepoint_binocular_reconstruction(
            _auditable_reconstruction(),
            by="group",
            **kwargs,
        )


# ============================================================================
# PUPIL — VALIDATION CONTRACT
# ============================================================================


def _validation_binocular():
    return pd.DataFrame(
        {
            "time": [
                0,
                1,
                2,
                3,
            ],
            "left_pupil": [
                1.0,
                2.0,
                3.0,
                4.0,
            ],
            "right_pupil": [
                1.1,
                2.1,
                3.1,
                4.1,
            ],
        }
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"direction": "bad"},
        {"mask_mode": "bad"},
        {"mask_prop": 0},
        {"mask_prop": 1},
        {"block_size": 0},
        {"repeats": 0},
    ],
)
def test_h12_validation_settings(kwargs):
    call = {
        "time_col": "time",
        "min_pairs": 2,
        "min_unique": 2,
    }

    call.update(kwargs)

    with pytest.raises(ValueError):
        pupil.validate_gazepoint_binocular_reconstruction(
            _validation_binocular(),
            **call,
        )


def test_h12_validation_requires_bilateral():
    frame = pd.DataFrame(
        {
            "time": [
                0,
                1,
            ],
            "left_pupil": [
                1.0,
                np.nan,
            ],
            "right_pupil": [
                np.nan,
                2.0,
            ],
        }
    )

    with pytest.raises(ValueError):
        pupil.validate_gazepoint_binocular_reconstruction(
            frame,
            time_col="time",
            min_pairs=2,
            min_unique=2,
        )


# ============================================================================
# QC — COORDINATE COVERAGE GROUPED LEGACY
# ============================================================================


def test_h12_coordinate_coverage_grouped():
    result = qc.summarise_gazepoint_coordinate_coverage(
        pd.DataFrame(
            {
                "subject": [
                    "S1",
                    "S1",
                    "S2",
                ],
                "x": [
                    0.1,
                    np.nan,
                    0.5,
                ],
                "y": [
                    0.2,
                    0.3,
                    0.6,
                ],
            }
        ),
        group_cols="subject",
    )

    assert len(result) == 2

    s1 = result.loc[result["subject"].eq("S1")].iloc[0]

    assert s1["n_xy"] == 1


# ============================================================================
# QC — POST-EXCLUSION EMPTY R MODE
# ============================================================================


def test_h12_post_exclusion_empty_r_mode():
    frame = pd.DataFrame(
        {
            "subject": pd.Series(dtype="string"),
            "condition": pd.Series(dtype="string"),
        }
    )

    with pytest.raises(ValueError):
        qc.audit_gazepoint_post_exclusion_balance(
            frame,
            subject_col="subject",
            condition_col="condition",
        )


# ============================================================================
# QC — EXCLUSION RATIO NEGATIVE VALUES
# ============================================================================


def test_h12_exclusion_ratio_no_positive_values():
    assert np.isnan(
        qc._gp3_exclusion_r_ratio(
            [
                -1,
                -2,
            ]
        )
    )

    assert qc._gp3_exclusion_r_ratio(
        [
            -1,
            -2,
        ],
        zero_returns_one=True,
    ) == pytest.approx(1.0)


# ============================================================================
# QC — READINESS WITH NO TRIAL / CONDITION
# ============================================================================


def test_h12_readiness_no_trial_or_condition():
    frame = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S2",
            ],
            "time": [
                0.0,
                1.0,
            ],
            "x": [
                0.1,
                0.2,
            ],
            "y": [
                0.1,
                0.2,
            ],
            "pupil": [
                3.0,
                3.1,
            ],
        }
    )

    result = qc.check_gazepoint_real_data_readiness(
        frame,
        analysis_type="general",
        participant_col="subject",
        trial_col=None,
        time_col="time",
        condition_col=None,
        min_rows=1,
        min_participants=1,
        min_trials=1,
    )

    checks = result["checks"]

    assert "condition_detected" in set(checks["check_id"])

    trial_check = checks.loc[checks["check_id"].eq("minimum_trials")].iloc[0]

    assert trial_check["status"] == "fail"


# ============================================================================
# QC — LEGACY RECOMMENDATION VALIDITY BRANCH
# ============================================================================


def test_h12_recommend_exclusions_non_trackloss_validity():
    result = qc.recommend_gazepoint_exclusions(
        pd.DataFrame(
            {
                "subject": [
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
                ],
                "valid": [
                    True,
                    False,
                    True,
                    True,
                ],
                "x": [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                ],
                "y": [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                ],
                "pupil": [
                    3.0,
                    3.1,
                    3.2,
                    3.3,
                ],
            }
        ),
        validity_col="valid",
        min_trial_samples=1,
        min_participant_trials=1,
        min_participant_valid_trials=1,
    )

    assert {
        "name",
        "overview",
        "trial_recommendations",
        "participant_recommendations",
        "exclusions",
        "settings",
    }.issubset(result)

    trial = result["trial_recommendations"]

    participant = result["participant_recommendations"]

    assert len(trial) == 2

    assert len(participant) == 2

    assert trial.loc[
        trial["subject"].eq("S1"),
        "usable_prop",
    ].iloc[0] == pytest.approx(0.5)

    assert trial.loc[
        trial["subject"].eq("S2"),
        "usable_prop",
    ].iloc[0] == pytest.approx(1.0)
