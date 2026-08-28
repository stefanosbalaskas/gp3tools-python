import inspect

import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3


def test_r_aliases_are_visible_in_public_signatures():
    expected = {
        "add_gazepoint_aoi": {"master_df", "aoi_defs"},
        "add_gazepoint_dynamic_aoi": {"master_df", "aoi_defs"},
        "audit_gazepoint_aoi_geometry": {"data"},
        "audit_gazepoint_aoi_margin_sensitivity": {"gaze_data"},
        "audit_gazepoint_aoi_overlap": {"data"},
        "audit_gazepoint_aoi_screen_coverage": {
            "data",
            "screen_width",
            "screen_height",
        },
        "extract_gazepoint_representative_scanpaths": {"x"},
        "detect_gazepoint_blinks": {"all_gaze", "ts_col"},
        "flag_gazepoint_pupil": {"master"},
        "impute_gazepoint_pupil_gp": {"pupil", "time"},
        "mean_gazepoint_pupil": {"master_df", "lp_col", "rp_col"},
        "audit_gazepoint_master": {"master"},
        "audit_gazepoint_screen_bounds": {
            "screen_width",
            "screen_height",
        },
        "summarise_gazepoint_qc_status": {"qc_bundle"},
        "validate_gazepoint_master": {"master"},
        "detect_gazepoint_fixations_velocity": {
            "all_gaze",
            "ts_col",
            "vmax",
        },
        "inspect_gazepoint_columns": {"x"},
        "export_gazepoint_cluster_results": {"outdir"},
        "report_gazepoint_multiverse": {"multiverse_results"},
        "report_gazepoint_qc_overview": {"qc_bundle"},
        "summarise_gazepoint_workflow": {"results"},
        "sync_gazepoint_face_data": {
            "gazepoint_data",
            "face_data",
        },
    }

    for name, aliases in expected.items():
        parameters = set(inspect.signature(getattr(gp3, name)).parameters)
        assert aliases <= parameters, name


def test_r_alias_conflict_is_explicit():
    df = pd.DataFrame({"x": [1]})

    with pytest.raises(
        TypeError,
        match="both 'data'.*R-compatible alias 'x'",
    ):
        gp3.inspect_gazepoint_columns(
            df,
            x=df,
        )


def test_inspect_columns_r_alias():
    df = pd.DataFrame(
        {
            "x": [1.0, np.nan],
            "y": [2.0, 3.0],
        }
    )

    canonical = gp3.inspect_gazepoint_columns(df)
    via_r = gp3.inspect_gazepoint_columns(x=df)

    pd.testing.assert_frame_equal(canonical, via_r)


def test_audit_master_r_alias():
    df = pd.DataFrame(
        {
            "subject": ["S1", "S1"],
            "time": [0.0, 1.0],
        }
    )

    canonical = gp3.audit_gazepoint_master(df)
    via_r = gp3.audit_gazepoint_master(master=df)

    assert canonical.keys() == via_r.keys()

    for key in canonical:
        pd.testing.assert_frame_equal(
            canonical[key],
            via_r[key],
        )


def test_validate_master_r_alias():
    df = pd.DataFrame(
        {
            "subject": ["S1"],
            "time": [0.0],
        }
    )

    canonical = gp3.validate_gazepoint_master(df)
    via_r = gp3.validate_gazepoint_master(master=df)

    assert canonical["valid"] == via_r["valid"]

    pd.testing.assert_frame_equal(
        canonical["checks"],
        via_r["checks"],
    )


def test_rectangular_aoi_r_input_aliases():
    samples = pd.DataFrame(
        {
            "x": [0.25, 0.75],
            "y": [0.25, 0.75],
        }
    )

    geometry = pd.DataFrame(
        {
            "aoi": ["A", "B"],
            "xmin": [0.0, 0.5],
            "xmax": [0.5, 1.0],
            "ymin": [0.0, 0.5],
            "ymax": [0.5, 1.0],
        }
    )

    canonical = gp3.add_gazepoint_aoi(
        samples,
        x_col="x",
        y_col="y",
        aoi_geometry=geometry,
    )

    via_r = gp3.add_gazepoint_aoi(
        master_df=samples,
        aoi_defs=geometry,
        x_col="x",
        y_col="y",
    )

    pd.testing.assert_series_equal(
        canonical["aoi_current"],
        via_r["aoi_current"],
    )


def test_dynamic_aoi_r_input_aliases():
    samples = pd.DataFrame(
        {
            "TIME": [0.0, 1.0],
            "x": [0.25, 0.75],
            "y": [0.25, 0.75],
        }
    )

    geometry = pd.DataFrame(
        {
            "TIME": [0.0, 1.0],
            "aoi": ["A", "B"],
            "xmin": [0.0, 0.5],
            "xmax": [0.5, 1.0],
            "ymin": [0.0, 0.5],
            "ymax": [0.5, 1.0],
        }
    )

    canonical = gp3.add_gazepoint_dynamic_aoi(
        samples,
        geometry,
        time_col="TIME",
        aoi_time_col="TIME",
        x_col="x",
        y_col="y",
    )

    via_r = gp3.add_gazepoint_dynamic_aoi(
        master_df=samples,
        aoi_defs=geometry,
        time_col="TIME",
        aoi_time_col="TIME",
        x_col="x",
        y_col="y",
    )

    pd.testing.assert_series_equal(
        canonical["aoi_current"],
        via_r["aoi_current"],
    )


def test_aoi_geometry_r_data_alias():
    geometry = pd.DataFrame(
        {
            "aoi": ["A"],
            "xmin": [0.0],
            "xmax": [1.0],
            "ymin": [0.0],
            "ymax": [1.0],
        }
    )

    canonical = gp3.audit_gazepoint_aoi_geometry(geometry)

    via_r = gp3.audit_gazepoint_aoi_geometry(data=geometry)

    assert canonical["valid"] == via_r["valid"]

    pd.testing.assert_frame_equal(
        canonical["summary"],
        via_r["summary"],
    )


def test_aoi_overlap_r_data_alias():
    geometry = pd.DataFrame(
        {
            "aoi": ["A", "B"],
            "xmin": [0.0, 0.5],
            "xmax": [0.75, 1.0],
            "ymin": [0.0, 0.5],
            "ymax": [0.75, 1.0],
        }
    )

    canonical = gp3.audit_gazepoint_aoi_overlap(geometry)

    via_r = gp3.audit_gazepoint_aoi_overlap(data=geometry)

    pd.testing.assert_frame_equal(canonical, via_r)


def test_aoi_screen_coverage_r_aliases():
    geometry = pd.DataFrame(
        {
            "xmin": [0.0],
            "xmax": [0.5],
            "ymin": [0.0],
            "ymax": [0.5],
        }
    )

    canonical = gp3.audit_gazepoint_aoi_screen_coverage(
        geometry,
        width=2.0,
        height=1.0,
    )

    via_r = gp3.audit_gazepoint_aoi_screen_coverage(
        data=geometry,
        screen_width=2.0,
        screen_height=1.0,
    )

    pd.testing.assert_frame_equal(canonical, via_r)


def test_screen_bounds_r_dimension_aliases():
    data = pd.DataFrame(
        {
            "x": [0.0, 50.0, 101.0],
            "y": [0.0, 50.0, 50.0],
        }
    )

    canonical = gp3.audit_gazepoint_screen_bounds(
        data,
        x_col="x",
        y_col="y",
        width=100,
        height=100,
        normalized=False,
    )

    via_r = gp3.audit_gazepoint_screen_bounds(
        data,
        x_col="x",
        y_col="y",
        screen_width=100,
        screen_height=100,
        normalized=False,
    )

    pd.testing.assert_frame_equal(canonical, via_r)


def test_blink_r_input_aliases():
    df = pd.DataFrame(
        {
            "TIME": [0.0, 0.1, 0.2, 0.3],
            "pupil": [1.0, np.nan, np.nan, 1.0],
        }
    )

    canonical = gp3.detect_gazepoint_blinks(
        df,
        pupil_col="pupil",
        time_col="TIME",
        min_duration_ms=50,
    )

    via_r = gp3.detect_gazepoint_blinks(
        all_gaze=df,
        pupil_col="pupil",
        ts_col="TIME",
        min_duration=50,
    )

    pd.testing.assert_series_equal(
        canonical["blink"],
        via_r["blink"],
    )


def test_velocity_fixation_r_aliases():
    df = pd.DataFrame(
        {
            "TIME": [0.0, 0.1, 0.2],
            "x": [0.1, 0.1, 0.1],
            "y": [0.2, 0.2, 0.2],
        }
    )

    canonical = gp3.detect_gazepoint_fixations_velocity(
        df,
        x_col="x",
        y_col="y",
        time_col="TIME",
        velocity_threshold=10,
        min_duration_ms=50,
    )

    via_r = gp3.detect_gazepoint_fixations_velocity(
        all_gaze=df,
        x_col="x",
        y_col="y",
        ts_col="TIME",
        vmax=10,
        min_duration=50,
    )

    pd.testing.assert_series_equal(
        canonical["fixation"],
        via_r["fixation"],
    )


def test_flag_pupil_r_aliases():
    df = pd.DataFrame(
        {
            "pupil": [0.5, 2.0, 10.0],
        }
    )

    canonical = gp3.flag_gazepoint_pupil(
        df,
        pupil_col="pupil",
        physiological_min=1.0,
        physiological_max=9.0,
    )

    via_r = gp3.flag_gazepoint_pupil(
        master=df,
        pupil_col="pupil",
        min_pupil=1.0,
        max_pupil=9.0,
    )

    pd.testing.assert_series_equal(
        canonical["pupil_flag"],
        via_r["pupil_flag"],
    )


def test_gp_imputation_required_r_aliases():
    df = pd.DataFrame(
        {
            "TIME": [0.0, 1.0, 2.0],
            "pupil": [1.0, np.nan, 2.0],
        }
    )

    canonical = gp3.impute_gazepoint_pupil_gp(
        df,
        pupil_col="pupil",
        time_col="TIME",
    )

    via_r = gp3.impute_gazepoint_pupil_gp(
        df,
        pupil="pupil",
        time="TIME",
    )

    pd.testing.assert_frame_equal(canonical, via_r)


def test_mean_pupil_r_aliases():
    df = pd.DataFrame(
        {
            "LPupil": [1.0, 2.0],
            "RPupil": [3.0, 4.0],
        }
    )

    canonical = gp3.mean_gazepoint_pupil(
        df,
        left_col="LPupil",
        right_col="RPupil",
    )

    via_r = gp3.mean_gazepoint_pupil(
        master_df=df,
        lp_col="LPupil",
        rp_col="RPupil",
    )

    pd.testing.assert_series_equal(
        canonical["pupil_mean"],
        via_r["pupil_mean"],
    )


def test_qc_status_r_alias():
    qc = {
        "sampling": pd.DataFrame({"x": [1]}),
        "other": None,
    }

    canonical = gp3.summarise_gazepoint_qc_status(qc)

    via_r = gp3.summarise_gazepoint_qc_status(qc_bundle=qc)

    pd.testing.assert_frame_equal(canonical, via_r)


def test_workflow_summary_r_alias():
    result = {
        "master": pd.DataFrame({"x": [1]}),
        "sampling_error": "failure",
    }

    canonical = gp3.summarise_gazepoint_workflow(result)

    via_r = gp3.summarise_gazepoint_workflow(results=result)

    pd.testing.assert_frame_equal(canonical, via_r)


def test_export_cluster_outdir_alias(tmp_path):
    data = pd.DataFrame(
        {
            "start": [1],
            "end": [2],
        }
    )

    out = gp3.export_gazepoint_cluster_results(
        data,
        outdir=tmp_path / "r-out",
    )

    assert out


def test_unknown_keyword_is_not_silently_swallowed():
    with pytest.raises(TypeError):
        gp3.inspect_gazepoint_columns(
            x=pd.DataFrame({"a": [1]}),
            definitely_not_an_argument=True,
        )


def test_us_spelling_qc_status_inherits_r_alias():
    qc = {
        "sampling": pd.DataFrame({"x": [1]}),
        "other": None,
    }

    british = gp3.summarise_gazepoint_qc_status(qc_bundle=qc)

    american = gp3.summarize_gazepoint_qc_status(qc_bundle=qc)

    pd.testing.assert_frame_equal(
        british,
        american,
    )
