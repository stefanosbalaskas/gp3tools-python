from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3
from gp3tools import _behavioral_r4 as r4


def _canonical_master() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S1",
                "S1",
                "S2",
                "S2",
                "S2",
                "S2",
            ],
            "media_id": [
                "M1",
                "M1",
                "M2",
                "M2",
                "M1",
                "M1",
                "M2",
                "M2",
            ],
            "time_ms": [
                0.0,
                16.0,
                32.0,
                48.0,
                0.0,
                16.0,
                32.0,
                48.0,
            ],
            "x": [
                0.2,
                0.3,
                np.nan,
                0.8,
                0.4,
                0.5,
                1.2,
                0.6,
            ],
            "y": [
                0.2,
                0.3,
                np.nan,
                0.8,
                0.4,
                0.5,
                0.2,
                0.6,
            ],
            "valid_sample": [
                True,
                True,
                False,
                True,
                True,
                True,
                True,
                True,
            ],
            "missing_gaze": [
                False,
                False,
                True,
                False,
                False,
                False,
                False,
                False,
            ],
            "missing_pupil": [
                False,
                True,
                True,
                False,
                False,
                False,
                True,
                False,
            ],
            "gaze_offscreen": [
                False,
                False,
                False,
                False,
                False,
                False,
                True,
                False,
            ],
            "mean_pupil": [
                3.0,
                np.nan,
                np.nan,
                3.4,
                3.1,
                3.2,
                np.nan,
                3.5,
            ],
            "aoi_current": [
                "target",
                "target",
                "missing",
                "outside",
                "distractor",
                "",
                "offscreen",
                "target",
            ],
            "aoi_count": [
                1,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
            ],
            "raw_x": [
                0.2,
                0.3,
                np.nan,
                0.8,
                0.4,
                0.5,
                1.2,
                0.6,
            ],
            "raw_y": [
                0.2,
                0.3,
                np.nan,
                0.8,
                0.4,
                0.5,
                0.2,
                0.6,
            ],
        }
    )


def _rectangles() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "AOI": [
                "target",
                "distractor",
            ],
            "x_min": [
                0.10,
                0.40,
            ],
            "x_max": [
                0.35,
                0.70,
            ],
            "y_min": [
                0.10,
                0.30,
            ],
            "y_max": [
                0.35,
                0.60,
            ],
        }
    )


def test_r4_virtual_class_key_does_not_pollute_r_list() -> None:
    value = r4.R4List(
        {
            "a": 1,
            "b": 2,
        },
        r_class="gp3_demo|list",
    )

    assert value["_gp3_class"] == "gp3_demo"

    assert list(value) == [
        "a",
        "b",
    ]

    assert len(value) == 2


def test_r4_master_audit_full_grouped_contract() -> None:
    out = gp3.audit_gazepoint_master(master=_canonical_master())

    assert list(out) == [
        "overview",
        "by_subject",
        "by_media",
        "by_subject_media",
        "aoi_states",
        "pupil_summary",
        "coordinate_summary",
    ]

    overview = out["overview"]

    assert overview.attrs["r_class"] == "tbl_df|tbl|data.frame"

    assert (
        overview.loc[
            0,
            "n_rows",
        ]
        == 8
    )

    assert (
        overview.loc[
            0,
            "n_subjects",
        ]
        == 2
    )

    assert (
        overview.loc[
            0,
            "n_media",
        ]
        == 2
    )

    assert len(out["by_subject"]) == 2

    assert len(out["by_media"]) == 2

    assert len(out["by_subject_media"]) == 4

    assert {
        "target",
        "missing",
        "offscreen",
        "unclassified",
    } <= set(out["aoi_states"]["aoi_state"])

    assert len(out["pupil_summary"]) == 4


def test_r4_static_aoi_all_output_modes_and_overlap() -> None:
    master = pd.DataFrame(
        {
            "FPOGX": [
                0.20,
                0.50,
                0.30,
                np.nan,
            ],
            "FPOGY": [
                0.20,
                0.40,
                0.30,
                0.20,
            ],
        }
    )

    geometry = pd.DataFrame(
        {
            "AOI": [
                "target",
                "other",
            ],
            "x_min": [
                0.10,
                0.25,
            ],
            "x_max": [
                0.35,
                0.60,
            ],
            "y_min": [
                0.10,
                0.20,
            ],
            "y_max": [
                0.35,
                0.50,
            ],
        }
    )

    both = gp3.add_gazepoint_aoi(
        master_df=master,
        aoi_defs=geometry,
        output="both",
        overlap="last",
    )

    assert {
        "aoi_target",
        "aoi_other",
        "aoi_current",
        "aoi_overlap_count",
    } <= set(both.columns)

    assert (
        both.loc[
            2,
            "aoi_overlap_count",
        ]
        == 2.0
    )

    assert (
        both.loc[
            2,
            "aoi_current",
        ]
        == "other"
    )

    assert pd.isna(
        both.loc[
            3,
            "aoi_current",
        ]
    )

    selected = gp3.add_gazepoint_aoi(
        master_df=master,
        aoi_defs=geometry,
        aoi_name="target",
    )

    assert "aoi_target" in selected
    assert "aoi_other" not in selected

    with pytest.raises(
        ValueError,
        match="overlapping",
    ):
        gp3.add_gazepoint_aoi(
            master_df=master,
            aoi_defs=geometry,
            output="both",
            overlap="error",
        )

    with pytest.raises(
        ValueError,
        match="did not match",
    ):
        gp3.add_gazepoint_aoi(
            master_df=master,
            aoi_defs=geometry,
            aoi_name="not-present",
        )

    duplicate = pd.concat(
        [
            geometry.iloc[[0]],
            geometry.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        gp3.add_gazepoint_aoi(
            master_df=master,
            aoi_defs=duplicate,
        )


def test_r4_geometry_bounds_origin_size_and_validation() -> None:
    bounds = _rectangles()

    out = gp3.audit_gazepoint_aoi_geometry(data=bounds)

    assert out["_gp3_class"] == "gp3_aoi_geometry_audit"

    assert (
        out["overview"].loc[
            0,
            "coordinate_format",
        ]
        == "bounds"
    )

    origin = pd.DataFrame(
        {
            "aoi": [
                "a",
                "b",
                "duplicate",
            ],
            "stimulus": [
                "S1",
                "S1",
                "S1",
            ],
            "x": [
                0.10,
                0.75,
                0.10,
            ],
            "y": [
                0.10,
                0.75,
                0.10,
            ],
            "width": [
                0.20,
                0.40,
                0.20,
            ],
            "height": [
                0.20,
                0.40,
                0.20,
            ],
        }
    )

    out2 = gp3.audit_gazepoint_aoi_geometry(
        data=origin,
        stimulus_col="stimulus",
        require_within_screen=True,
    )

    assert (
        out2["overview"].loc[
            0,
            "coordinate_format",
        ]
        == "origin_size"
    )

    assert (
        out2["overview"].loc[
            0,
            "aoi_geometry_status",
        ]
        == "review"
    )

    assert len(out2["duplicate_geometry"]) >= 1

    assert len(out2["flagged_aois"]) >= 1

    for bad in (
        -0.01,
        1.01,
        np.inf,
    ):
        with pytest.raises(
            ValueError,
            match="max_area_prop",
        ):
            gp3.audit_gazepoint_aoi_geometry(
                data=bounds,
                max_area_prop=bad,
            )

    with pytest.raises(
        ValueError,
        match="min_width",
    ):
        gp3.audit_gazepoint_aoi_geometry(
            data=bounds,
            min_width=-1,
        )

    with pytest.raises(
        ValueError,
        match="screen_x_range",
    ):
        gp3.audit_gazepoint_aoi_geometry(
            data=bounds,
            screen_x_range=(
                1,
                0,
            ),
        )

    with pytest.raises(
        ValueError,
        match="require_within_screen",
    ):
        gp3.audit_gazepoint_aoi_geometry(
            data=bounds,
            require_within_screen="yes",
        )


def test_r4_workflow_optional_file_outputs_and_pairs() -> None:
    results = {
        "all_gaze": pd.DataFrame({"x": range(6)}),
        "all_fix": pd.DataFrame({"x": [1, 2]}),
        "sampling": pd.DataFrame({"hz": [60, 60]}),
        "quality": pd.DataFrame({"quality": [1, 2]}),
        "flagged_quality": pd.DataFrame(
            {
                "review_required": [
                    True,
                    False,
                    True,
                ]
            }
        ),
        "aoi_table": pd.DataFrame({"aoi": ["a", "b"]}),
        "file_pairs": pd.DataFrame(
            {
                "status": [
                    "complete",
                    "missing_fixations",
                    "complete",
                ]
            }
        ),
        "written_files": [
            "a.csv",
            "b.csv",
        ],
        "written_plots": [
            "a.png",
        ],
        "written_report": "report.html",
    }

    out = gp3.summarise_gazepoint_workflow(results=results)

    assert (
        out.loc[
            0,
            "all_gaze_rows",
        ]
        == 6
    )

    assert (
        out.loc[
            0,
            "fixation_rows",
        ]
        == 2
    )

    assert (
        out.loc[
            0,
            "review_required_rows",
        ]
        == 2
    )

    assert (
        out.loc[
            0,
            "file_pair_rows",
        ]
        == 3
    )

    assert (
        out.loc[
            0,
            "complete_file_pairs",
        ]
        == 2
    )

    assert (
        out.loc[
            0,
            "problem_file_pairs",
        ]
        == 1
    )

    assert (
        out.loc[
            0,
            "output_table_files",
        ]
        == 2
    )

    assert (
        out.loc[
            0,
            "output_plot_files",
        ]
        == 1
    )

    assert bool(
        out.loc[
            0,
            "report_created",
        ]
    )


def test_r4_pupil_response_multiple_trials_and_missingness() -> None:
    data = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S1",
                "S1",
                "S1",
                "S1",
                "S1",
                "S1",
            ],
            "trial": [
                1,
                1,
                1,
                1,
                2,
                2,
                2,
                2,
            ],
            "time": [
                -0.1,
                0.0,
                0.1,
                0.2,
                -0.1,
                0.0,
                0.1,
                0.2,
            ],
            "pupil": [
                3.0,
                3.2,
                3.5,
                3.4,
                np.nan,
                np.nan,
                3.3,
                np.nan,
            ],
            "condition": [
                "A",
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
                "B",
            ],
            "interpolated": [
                False,
                False,
                True,
                False,
                False,
                False,
                True,
                False,
            ],
        }
    )

    out = gp3.summarize_gazepoint_pupil_response_features(
        data=data,
        pupil="pupil",
        time="time",
        subject="subject",
        trial="trial",
        baseline_window=[
            -0.1,
            0.0,
        ],
        response_window=[
            0.0,
            0.2,
        ],
        condition="condition",
        interpolated="interpolated",
    )

    assert len(out) == 2

    assert (
        out.loc[
            0,
            "condition",
        ]
        == "A"
    )

    assert np.isclose(
        out.loc[
            0,
            "baseline_mean",
        ],
        3.1,
    )

    assert np.isnan(
        out.loc[
            1,
            "baseline_mean",
        ]
    )

    assert (
        out.loc[
            1,
            "missing_percent",
        ]
        > 0
    )

    with pytest.raises(
        ValueError,
        match="Missing columns",
    ):
        gp3.summarize_gazepoint_pupil_response_features(
            data=data.drop(
                columns=[
                    "pupil",
                ]
            ),
            pupil="pupil",
            time="time",
            subject="subject",
            trial="trial",
            baseline_window=[
                -0.1,
                0.0,
            ],
            response_window=[
                0.0,
                0.2,
            ],
        )


def test_r4_qc_direct_object_summary_all_statuses() -> None:
    object_summary = pd.DataFrame(
        {
            "object_name": [
                "one",
                "two",
                "three",
                "four",
                "five",
            ],
            "qc_status": [
                "pass",
                "warn",
                "fail",
                "info",
                "unknown",
            ],
        }
    )

    out = gp3.summarise_gazepoint_qc_status(qc_bundle=object_summary)

    assert out["_gp3_class"] == "gp3_qc_status_summary"

    overview = out["overview"]

    assert (
        overview.loc[
            0,
            "n_objects",
        ]
        == 5
    )

    assert (
        overview.loc[
            0,
            "n_pass",
        ]
        == 1
    )

    assert (
        overview.loc[
            0,
            "n_warn",
        ]
        == 1
    )

    assert (
        overview.loc[
            0,
            "n_fail",
        ]
        == 1
    )

    assert (
        overview.loc[
            0,
            "n_info",
        ]
        == 1
    )

    assert (
        overview.loc[
            0,
            "n_unknown",
        ]
        == 1
    )

    assert (
        overview.loc[
            0,
            "qc_overview_status",
        ]
        == "fail"
    )
