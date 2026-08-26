from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3

ORACLE_PATH = Path(__file__).parent / "oracles" / "r_v2_3_0_behavioral_r2.csv"

ORACLE = pd.read_csv(
    ORACLE_PATH,
    dtype=str,
    keep_default_na=False,
)


def _missing(value):
    try:
        result = pd.isna(value)
    except Exception:
        return False

    return isinstance(result, (bool, np.bool_)) and bool(result)


def _type(value):
    if isinstance(value, (bool, np.bool_)):
        return "logical"

    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return "integer"

    if isinstance(value, (float, np.floating)):
        return "numeric"

    if isinstance(value, str):
        return "character"

    if _missing(value):
        return "missing"

    return type(value).__name__


def _text(value):
    if _missing(value):
        return "<NA>"

    if isinstance(value, (bool, np.bool_)):
        return "TRUE" if bool(value) else "FALSE"

    if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
        return str(int(value))

    if isinstance(value, (float, np.floating)):
        value = float(value)

        if math.isnan(value):
            return "<NA>"

        if math.isinf(value):
            return "<Inf>" if value > 0 else "<-Inf>"

        return format(value, ".15g")

    return str(value)


def _flatten(x, path="result", out=None):
    if out is None:
        out = {}

    if isinstance(x, pd.DataFrame):
        out[f"{path}.__type__"] = (
            "metadata",
            "data.frame",
        )
        out[f"{path}.__nrow__"] = (
            "integer",
            str(len(x)),
        )
        out[f"{path}.__ncol__"] = (
            "integer",
            str(len(x.columns)),
        )
        out[f"{path}.__columns__"] = (
            "metadata",
            "|".join(map(str, x.columns)),
        )

        for row_number, (_, row) in enumerate(
            x.iterrows(),
            start=1,
        ):
            for column in x.columns:
                _flatten(
                    row[column],
                    f"{path}[{row_number},{column}]",
                    out,
                )

        return out

    if isinstance(x, dict):
        out[f"{path}.__type__"] = (
            "metadata",
            "list",
        )
        out[f"{path}.__length__"] = (
            "integer",
            str(len(x)),
        )

        for key, value in x.items():
            _flatten(
                value,
                f"{path}.{key}",
                out,
            )

        return out

    if isinstance(
        x,
        (
            list,
            tuple,
            np.ndarray,
            pd.Series,
        ),
    ):
        values = list(x)

        out[f"{path}.__type__"] = (
            "metadata",
            "vector",
        )
        out[f"{path}.__length__"] = (
            "integer",
            str(len(values)),
        )

        for i, value in enumerate(
            values,
            start=1,
        ):
            _flatten(
                value,
                f"{path}[{i}]",
                out,
            )

        return out

    out[path] = (
        _type(x),
        _text(x),
    )

    return out


def _result_class(value, fallback=None):
    if hasattr(value, "r_class"):
        return value.r_class

    if isinstance(value, pd.DataFrame):
        return value.attrs.get(
            "r_class",
            fallback or "data.frame",
        )

    if isinstance(value, dict):
        return fallback or "list"

    if isinstance(value, list):
        return fallback or "character"

    return fallback or type(value).__name__


def _oracle_for(function_name):
    return ORACLE.loc[ORACLE["function_name"].eq(function_name)].copy()


def _assert_r_oracle(
    function_name,
    value,
    *,
    result_class=None,
    numeric_tol=1e-9,
):
    expected = _oracle_for(function_name)

    assert not expected.empty

    actual = _flatten(value)

    actual["result.__class__"] = (
        "metadata",
        result_class or _result_class(value),
    )

    expected_paths = expected["path"].tolist()

    missing_paths = [path for path in expected_paths if path not in actual]

    extra_paths = [path for path in actual if path not in set(expected_paths)]

    errors = []

    if missing_paths:
        errors.append("MISSING PATHS:\n  " + "\n  ".join(missing_paths[:40]))

    if extra_paths:
        errors.append("EXTRA PATHS:\n  " + "\n  ".join(extra_paths[:40]))

    for row in expected.itertuples(index=False):
        if row.path not in actual:
            continue

        actual_type, actual_value = actual[row.path]
        expected_type = row.value_type
        expected_value = row.value

        if expected_value in {
            "<NA>",
            "<NaN>",
        }:
            if actual_value not in {
                "<NA>",
                "<NaN>",
            }:
                errors.append(f"{row.path}: expected missing {expected_value}, got {actual_value}")
            continue

        if expected_value in {
            "<Inf>",
            "<-Inf>",
        }:
            if actual_value != expected_value:
                errors.append(f"{row.path}: expected {expected_value}, got {actual_value}")
            continue

        if expected_type in {
            "numeric",
            "integer",
        }:
            try:
                expected_number = float(expected_value)
                actual_number = float(actual_value)
            except Exception:
                if actual_value != expected_value:
                    errors.append(f"{row.path}: expected {expected_value}, got {actual_value}")
                continue

            if not np.isclose(
                actual_number,
                expected_number,
                atol=numeric_tol,
                rtol=numeric_tol,
                equal_nan=True,
            ):
                errors.append(f"{row.path}: expected {expected_number!r}, got {actual_number!r}")
        else:
            if actual_value != expected_value:
                errors.append(f"{row.path}: expected {expected_value!r}, got {actual_value!r}")

        if len(errors) >= 80:
            break

    if errors:
        pytest.fail(
            "\n\n".join(errors[:80]),
            pytrace=False,
        )


def test_r2_condition_quality():
    data = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S2",
                "S3",
                "S1",
                "S2",
                "S3",
            ],
            "condition": [
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
            ],
            "missing_gaze_prop": [
                0.05,
                0.10,
                0.08,
                0.35,
                0.40,
                0.30,
            ],
            "offscreen_prop": [
                0.02,
                0.03,
                0.01,
                0.20,
                0.25,
                0.18,
            ],
            "pupil_valid_prop": [
                0.95,
                0.90,
                0.92,
                0.70,
                0.72,
                0.68,
            ],
        }
    )

    out = gp3.audit_gazepoint_condition_quality_imbalance(
        data,
        condition_col="condition",
        quality_cols=[
            "missing_gaze_prop",
            "offscreen_prop",
            "pupil_valid_prop",
        ],
        subject_col="subject",
        min_units_per_condition=2,
        max_mean_difference=0.10,
        max_condition_ratio=2,
        lower_is_better=[
            "missing_gaze_prop",
            "offscreen_prop",
        ],
    )

    _assert_r_oracle(
        "audit_gazepoint_condition_quality_imbalance",
        out,
    )


def test_r2_event_sync():
    data = pd.DataFrame(
        {
            "subject": ["S1"] * 6 + ["S2"] * 6,
            "media_id": ["M1"] * 12,
            "trial_global": ["T1"] * 6 + ["T2"] * 6,
            "condition": ["A"] * 6 + ["B"] * 6,
            "time": [
                0,
                10,
                20,
                30,
                40,
                50,
                0,
                10,
                20,
                100,
                110,
                120,
            ],
            "event": [
                "onset",
                None,
                None,
                "response",
                None,
                None,
                "onset",
                None,
                None,
                None,
                None,
                None,
            ],
        }
    )

    out = gp3.audit_gazepoint_event_sync(
        data=data,
        time_col="time",
        event_col="event",
        group_cols=[
            "subject",
            "media_id",
            "trial_global",
        ],
        condition_col="condition",
        expected_event_labels=[
            "onset",
            "response",
        ],
        onset_event_label="onset",
        response_event_label="response",
        min_samples_per_unit=4,
        max_time_gap_ms=50,
    )

    _assert_r_oracle(
        "audit_gazepoint_event_sync",
        out,
    )


def test_r2_face_sync():
    data = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S1",
                "S2",
                "S2",
            ],
            "face_sync_method": ["nearest_time"] * 5,
            "face_sync_status": [
                "matched",
                "matched",
                "outside_tolerance",
                "matched",
                "unmatched",
            ],
            "face_sync_within_tolerance": [
                True,
                True,
                False,
                True,
                False,
            ],
            "face_sync_diff_sec": [
                0.001,
                -0.004,
                0.090,
                0.010,
                np.nan,
            ],
            "face_sync_abs_diff_sec": [
                0.001,
                0.004,
                0.090,
                0.010,
                np.nan,
            ],
        }
    )

    out = gp3.audit_gazepoint_face_sync(
        data=data,
        group_cols="subject",
        min_matched_percent=60,
        warning_matched_percent=80,
        max_abs_diff_sec=0.05,
    )

    _assert_r_oracle(
        "audit_gazepoint_face_sync",
        out,
    )


def test_r2_gaze_signal_quality():
    data = pd.DataFrame(
        {
            "subject": ["S1"] * 4 + ["S2"] * 4,
            "media_id": ["M1"] * 8,
            "trial_global": ["T1"] * 4 + ["T2"] * 4,
            "condition": ["A"] * 4 + ["B"] * 4,
            "x": [
                0.2,
                0.3,
                np.nan,
                1.2,
                0.4,
                0.5,
                0.6,
                0.7,
            ],
            "y": [
                0.2,
                0.3,
                np.nan,
                0.5,
                0.4,
                0.5,
                0.6,
                0.7,
            ],
            "FPOGV": [
                1,
                1,
                0,
                1,
                1,
                1,
                1,
                1,
            ],
            "pupil": [
                3.1,
                3.2,
                np.nan,
                3.0,
                3.3,
                3.4,
                3.5,
                3.6,
            ],
        }
    )

    out = gp3.audit_gazepoint_gaze_signal_quality(
        data,
        subject_col="subject",
        condition_col="condition",
        group_cols=[
            "subject",
            "media_id",
            "trial_global",
        ],
        x_col="x",
        y_col="y",
        validity_cols="FPOGV",
        pupil_col="pupil",
        screen_x_range=(0, 1),
        screen_y_range=(0, 1),
        min_gaze_valid_prop=0.75,
        max_missing_gaze_prop=0.25,
        max_offscreen_prop=0.20,
        min_pupil_valid_prop=0.75,
    )

    _assert_r_oracle(
        "audit_gazepoint_gaze_signal_quality",
        out,
    )


def test_r2_timecourse_grid():
    data = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S1",
                "S1",
                "S2",
                "S2",
                "S2",
            ],
            "condition": [
                "A",
                "A",
                "B",
                "B",
                "A",
                "B",
                "B",
            ],
            "time_bin": [
                0,
                1,
                0,
                1,
                0,
                0,
                0,
            ],
            "value": [
                1.0,
                1.2,
                1.1,
                1.3,
                0.8,
                0.9,
                1.0,
            ],
        }
    )

    out = gp3.audit_gazepoint_timecourse_grid(
        data,
        subject_col="subject",
        condition_col="condition",
        time_col="time_bin",
        outcome_col="value",
    )

    _assert_r_oracle(
        "audit_gazepoint_timecourse_grid",
        out,
    )


def test_r2_event_detector_comparison():
    n = 80

    data = pd.DataFrame(
        {
            "USER_ID": ["P01"] * n,
            "trial": ["T01"] * n,
            "TIME": np.arange(n) * 0.01,
            "FPOGX": np.r_[
                np.repeat(0.20, 30),
                np.linspace(0.20, 0.80, 10),
                np.repeat(0.80, 40),
            ],
            "FPOGY": np.repeat(0.50, n),
        }
    )

    result = gp3.compare_gazepoint_event_detectors(
        data,
        id_col="USER_ID",
        trial_col="trial",
        x_col="FPOGX",
        y_col="FPOGY",
        time_col="TIME",
        methods=["velocity"],
        velocity_thresholds=[5, 10],
        min_duration=20,
        hmm_states=3,
        run_optional_eyetools=False,
        min_overlap=0.5,
    )

    subset = type(result)(
        {
            "events": result["events"],
            "runs": result["runs"],
            "detector_summary": result["detector_summary"],
            "pairwise_agreement": result["pairwise_agreement"],
            "unmatched_events": result["unmatched_events"],
            "settings": result["settings"],
        },
        r_class="gp3_event_detector_comparison|list",
    )

    _assert_r_oracle(
        "compare_gazepoint_event_detectors",
        subset,
    )


def test_r2_cross_package_report():
    workflow = {
        "audit": pd.DataFrame(
            {
                "engine": ["nearest"],
                "gaze_rows": [100],
                "biometric_rows": [95],
                "matched_rows": [90],
                "unmatched_rows": [10],
                "matched_rate": [0.90],
                "median_absolute_difference_ms": [4.125],
                "maximum_absolute_difference_ms": [18.750],
            }
        ),
        "report_text": ("Synthetic cross-package workflow. Used for frozen behavioral parity."),
    }

    out = gp3.create_gazepoint_cross_package_report(
        x=workflow,
        output_file=None,
    )

    _assert_r_oracle(
        "create_gazepoint_cross_package_report",
        out,
        result_class="character",
    )


def test_r2_preprocessing_multiverse():
    out = gp3.create_gazepoint_preprocessing_multiverse(
        pupil_max_gap_ms=[75, 150],
        pupil_smoothing_window_samples=[3, 5],
        pupil_baseline_windows=[(-200, 0)],
        pupil_artifact_padding_ms=0,
        aoi_denominators=["valid", "all"],
        aoi_min_denominator_samples=[1, 5],
        include_pupil=True,
        include_aoi=True,
        label_prefix="r2",
    )

    _assert_r_oracle(
        "create_gazepoint_preprocessing_multiverse",
        out,
    )


def test_r2_blink_interpolation():
    master = pd.DataFrame(
        {
            "USER_ID": ["P01"] * 10,
            "trial": ["T01"] * 10,
            "TIME": np.arange(10) * 0.01,
            "mean_pupil": [
                3.00,
                3.05,
                3.10,
                3.15,
                3.20,
                3.25,
                3.30,
                3.35,
                3.40,
                3.45,
            ],
        }
    )

    blink = pd.DataFrame(
        {
            "USER_ID": ["P01"],
            "trial": ["T01"],
            "start_time": [0.03],
            "end_time": [0.05],
        }
    )

    out = gp3.interpolate_gazepoint_blinks(
        master_df=master,
        blink_df=blink,
        pupil_cols="mean_pupil",
        id_col="USER_ID",
        group_cols="trial",
        ts_col="TIME",
        start_col="start_time",
        end_col="end_time",
        method="linear",
        max_gap_ms=500,
        suffix="_blink_interp",
        keep_mask=True,
        time_unit="seconds",
    )

    _assert_r_oracle(
        "interpolate_gazepoint_blinks",
        out,
        result_class="data.frame",
    )


def test_r2_fixation_alignment():
    data = pd.DataFrame(
        {
            "participant": ["P1"] * 5 + ["P2"] * 5,
            "trial": ["T1"] * 5 + ["T2"] * 5,
            "time": [0, 100, 200, 300, 400] * 2,
            "aoi": [
                "other",
                "target",
                "target",
                "other",
                "other",
                "other",
                "other",
                "other",
                "other",
                "other",
            ],
            "fixation": [
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
                False,
            ],
            "saccade": [
                False,
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                False,
                False,
            ],
        }
    )

    out = gp3.prepare_gazepoint_fixation_aligned_data(
        data,
        time_col="time",
        participant_col="participant",
        trial_col="trial",
        aoi_col="aoi",
        target_aoi="target",
        fixation_col="fixation",
        saccade_col="saccade",
        alignment_event="first_fixation_to_target",
        baseline_window=(-100, 0),
        analysis_window=(0, 300),
        keep_unaligned=True,
        name="r2_fixalign",
    )

    _assert_r_oracle(
        "prepare_gazepoint_fixation_aligned_data",
        out,
    )


def test_r2_multimodal():
    face = pd.DataFrame(
        {
            "participant_id": ["P001", "P002", "P003"],
            "trial_id": [1, 1, 1],
            "AU12_r_mean": [0.20, 0.30, 0.40],
            "face_confidence_mean": [0.95, 0.94, 0.92],
        }
    )

    gaze = pd.DataFrame(
        {
            "participant_id": ["P001", "P002", "P003"],
            "trial_id": [1, 1, 1],
            "dwell": [0.40, 0.55, 0.60],
        }
    )

    response = pd.DataFrame(
        {
            "participant_id": ["P001", "P002", "P003"],
            "trial_id": [1, 1, 1],
            "rating": [4.0, np.nan, 5.0],
        }
    )

    out = gp3.prepare_gazepoint_multimodal_data(
        face_windows=face,
        gaze_data=gaze,
        response_data=response,
        by=["participant_id", "trial_id"],
        predictor_cols=[
            "AU12_r_mean",
            "face_confidence_mean",
            "dwell",
        ],
        outcome_cols="rating",
        covariate_cols=None,
        scale_predictors=True,
        scaled_suffix="_z",
        drop_missing_outcomes=True,
        keep_all=True,
    )

    wrapper = {
        "data": out.copy(),
        "settings": out.attrs["gp3_multimodal_settings"],
        "scaling": out.attrs["gp3_multimodal_scaling"],
        "original_class": out.attrs["r_class"].split("|"),
    }

    wrapper["data"].attrs.clear()

    _assert_r_oracle(
        "prepare_gazepoint_multimodal_data",
        wrapper,
        result_class="list",
    )


def test_r2_recalibration():
    data = pd.DataFrame(
        {
            "subject": ["S1"] * 4 + ["S2"] * 4,
            "time": [1, 2, 3, 4] * 2,
            "gaze_x": [
                0.40,
                0.42,
                0.44,
                0.46,
                0.10,
                0.12,
                0.14,
                0.16,
            ],
            "gaze_y": [
                0.45,
                0.47,
                0.49,
                0.51,
                0.10,
                0.12,
                0.14,
                0.16,
            ],
            "target_x": [0.50] * 8,
            "target_y": [0.50] * 8,
            "calibration": [
                True,
                True,
                True,
                False,
                True,
                True,
                True,
                False,
            ],
        }
    )

    out = gp3.recalibrate_gazepoint_gaze(
        data,
        x_col="gaze_x",
        y_col="gaze_y",
        target_x_col="target_x",
        target_y_col="target_y",
        time_col="time",
        grouping_cols="subject",
        calibration_col="calibration",
        calibration_value=True,
        method="median_shift",
        min_valid_points=3,
        max_shift=0.30,
        overwrite=False,
        name="r2_recalibration",
    )

    wrapper = {
        "data": out.copy(),
        "overview": out.attrs["gp3_gaze_recalibration_overview"],
        "group_summary": out.attrs["gp3_gaze_recalibration_group_summary"],
        "status_summary": out.attrs["gp3_gaze_recalibration_status_summary"],
        "settings": out.attrs["gp3_gaze_recalibration_settings"],
        "original_class": out.attrs["r_class"].split("|"),
    }

    wrapper["data"].attrs.clear()

    _assert_r_oracle(
        "recalibrate_gazepoint_gaze",
        wrapper,
        result_class="list",
    )
