from __future__ import annotations

import numpy as np
import pandas as pd

import gp3tools as gp3


def test_r4_pupil_response_contract() -> None:
    data = pd.DataFrame(
        {
            "subject": [
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
            ],
            "time": [
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
            ],
            "condition": [
                "A",
                "A",
                "A",
                "A",
            ],
            "interpolated": [
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

    assert list(out.columns) == [
        "subject",
        "trial",
        "baseline_mean",
        "peak_dilation",
        "latency_to_peak",
        "auc",
        "missing_percent",
        "interpolated_percent",
        "condition",
    ]

    assert np.isclose(
        out.loc[
            0,
            "baseline_mean",
        ],
        3.1,
    )

    assert np.isclose(
        out.loc[
            0,
            "peak_dilation",
        ],
        0.4,
    )

    assert np.isclose(
        out.loc[
            0,
            "latency_to_peak",
        ],
        0.1,
    )


def test_r4_static_aoi_contract() -> None:
    master = pd.DataFrame(
        {
            "FPOGX": [
                0.2,
                0.5,
                np.nan,
            ],
            "FPOGY": [
                0.2,
                0.4,
                0.2,
            ],
        }
    )

    geometry = pd.DataFrame(
        {
            "AOI": [
                "target",
                "distractor",
            ],
            "x_min": [
                0.1,
                0.4,
            ],
            "x_max": [
                0.35,
                0.7,
            ],
            "y_min": [
                0.1,
                0.3,
            ],
            "y_max": [
                0.35,
                0.6,
            ],
        }
    )

    out = gp3.add_gazepoint_aoi(
        master_df=master,
        aoi_defs=geometry,
    )

    assert list(out.columns) == [
        "FPOGX",
        "FPOGY",
        "aoi_target",
        "aoi_distractor",
        "aoi_overlap_count",
    ]

    assert out["aoi_target"].tolist() == [
        True,
        False,
        False,
    ]

    assert out["aoi_distractor"].tolist() == [
        False,
        True,
        False,
    ]

    assert out["aoi_overlap_count"].tolist() == [
        1.0,
        1.0,
        0.0,
    ]


def test_r4_geometry_structure() -> None:
    geometry = pd.DataFrame(
        {
            "AOI": [
                "target",
                "distractor",
            ],
            "x_min": [
                0.1,
                0.4,
            ],
            "x_max": [
                0.35,
                0.7,
            ],
            "y_min": [
                0.1,
                0.3,
            ],
            "y_max": [
                0.35,
                0.6,
            ],
        }
    )

    out = gp3.audit_gazepoint_aoi_geometry(data=geometry)

    assert out.r_class == ("gp3_aoi_geometry_audit|list")

    assert list(out) == [
        "overview",
        "geometry_summary",
        "size_summary",
        "duplicate_geometry",
        "flagged_aois",
        "settings",
    ]

    assert out["overview"].attrs["r_class"] == "tbl_df|tbl|data.frame"

    assert (
        out["overview"].loc[
            0,
            "coordinate_format",
        ]
        == "bounds"
    )


def test_r4_workflow_structure() -> None:
    results = {
        "all_gaze": pd.DataFrame({"x": range(24)}),
        "all_fix": pd.DataFrame(),
        "sampling": pd.DataFrame({"x": [1, 2]}),
        "quality": pd.DataFrame({"x": [1, 2]}),
        "flagged_quality": pd.DataFrame(
            {
                "review_required": [
                    False,
                    True,
                ]
            }
        ),
        "aoi_table": pd.DataFrame({"x": [1, 2]}),
        "written_files": None,
        "written_plots": None,
        "written_report": None,
    }

    out = gp3.summarise_gazepoint_workflow(results=results)

    assert len(out) == 1

    assert (
        out.loc[
            0,
            "all_gaze_rows",
        ]
        == 24
    )

    assert (
        out.loc[
            0,
            "review_required_rows",
        ]
        == 1
    )

    assert out.attrs["r_class"] == "tbl_df|tbl|data.frame"
