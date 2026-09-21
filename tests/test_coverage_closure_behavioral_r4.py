from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gp3tools import _behavioral_r4 as r4


def test_r4_metadata_character_and_group_key_helpers() -> None:
    bundle = r4.R4List(
        {"x": 1},
        r_class="demo|list",
    )

    assert bundle.gp3_r_class == "demo|list"

    assert r4._r_character(None) is pd.NA
    assert r4._r_character(pd.NA) is pd.NA
    assert r4._r_character("target") == "target"

    assert r4._make_name("") == "X"

    assert r4._group_key_tuple(("S1",)) == ("S1",)
    assert r4._group_key_tuple("S1") == ("S1",)


def test_r4_master_audit_supports_missing_optional_raw_coordinates() -> None:
    master = pd.DataFrame(
        {
            "subject": ["S1", "S1"],
            "media_id": ["M1", "M1"],
            "time_ms": [0.0, 16.0],
            "x": [0.25, 0.50],
            "y": [0.25, 0.50],
            "valid_sample": [True, True],
            "missing_gaze": [False, False],
            "missing_pupil": [False, False],
            "gaze_offscreen": [False, False],
            "mean_pupil": [3.0, 3.1],
            "aoi_current": ["target", "target"],
            "aoi_count": [1, 1],
        }
    )

    assert r4._canonical_master(master)

    out = r4._audit_master(master)

    coordinate = out["coordinate_summary"].iloc[0]

    assert np.isnan(coordinate["raw_x_min"])
    assert np.isnan(coordinate["raw_x_max"])
    assert np.isnan(coordinate["raw_y_min"])
    assert np.isnan(coordinate["raw_y_max"])

    assert out["overview"].attrs["r_class"] == "tbl_df|tbl|data.frame"


def test_r4_qc_reserved_keys_and_workflow_missing_contract() -> None:
    reserved = pd.DataFrame(
        {
            "object_name": ["reserved"],
            "qc_status": ["fail"],
        }
    )

    bundle = {
        "overview": reserved,
        "status_counts": reserved,
        "object_summary": reserved,
        "sampling": pd.DataFrame(
            {
                "qc_status": ["pass"],
            }
        ),
    }

    derived = r4._derive_object_summary(bundle)

    assert derived is not None
    assert derived["object_name"].tolist() == ["sampling"]
    assert derived["qc_status"].tolist() == ["pass"]

    with pytest.raises(
        ValueError,
        match="missing required elements",
    ):
        r4._workflow_summary(
            {
                "all_gaze": pd.DataFrame(),
            }
        )


def test_r4_static_aoi_unresolved_fields_and_duplicate_names() -> None:
    master = pd.DataFrame(
        {
            "FPOGX": [0.25, 0.75],
            "FPOGY": [0.25, 0.75],
        }
    )

    with pytest.raises(
        ValueError,
        match="Could not resolve AOI definition fields",
    ):
        r4._static_aoi(
            master,
            pd.DataFrame({"bad": [1]}),
            x_col="FPOGX",
            y_col="FPOGY",
            aoi_name=None,
            output="logical",
            prefix="aoi_",
            label_col="aoi_current",
            outside_label="outside",
            overlap="first",
            include_overlap_count=False,
        )

    duplicate_definitions = pd.DataFrame(
        {
            "AOI": ["target", "target"],
            "x_min": [0.0, 0.5],
            "x_max": [0.5, 1.0],
            "y_min": [0.0, 0.5],
            "y_max": [0.5, 1.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="AOI names must be unique",
    ):
        r4._static_aoi(
            master,
            duplicate_definitions,
            x_col="FPOGX",
            y_col="FPOGY",
            aoi_name=None,
            output="logical",
            prefix="aoi_",
            label_col="aoi_current",
            outside_label="outside",
            overlap="first",
            include_overlap_count=False,
        )

    # Raw AOI names are distinct, satisfying the scientific/API contract,
    # but their Python-safe column names collide after sanitisation:
    # "target 1" -> "target.1"
    # "target.1" -> "target.1"
    #
    # The second logical column must therefore receive a deterministic
    # suffix rather than silently overwriting the first one.
    collision_definitions = pd.DataFrame(
        {
            "AOI": ["target 1", "target.1"],
            "x_min": [0.0, 0.5],
            "x_max": [0.5, 1.0],
            "y_min": [0.0, 0.5],
            "y_max": [0.5, 1.0],
        }
    )

    out = r4._static_aoi(
        master,
        collision_definitions,
        x_col="FPOGX",
        y_col="FPOGY",
        aoi_name=None,
        output="logical",
        prefix="aoi_",
        label_col="aoi_current",
        outside_label="outside",
        overlap="first",
        include_overlap_count=False,
    )

    assert "aoi_target.1" in out.columns
    assert "aoi_target.1.1" in out.columns

    assert out["aoi_target.1"].tolist() == [True, False]
    assert out["aoi_target.1.1"].tolist() == [False, True]


def test_r4_workflow_result_alias_uses_canonical_summary() -> None:
    def legacy_workflow(
        result=None,
    ):
        return "legacy"

    wrapped = r4.wrap_r4(
        legacy_workflow,
        name="summarise_gazepoint_workflow",
    )

    bundle = {
        "all_gaze": pd.DataFrame(),
        "all_fix": pd.DataFrame(),
        "sampling": pd.DataFrame(),
        "quality": pd.DataFrame(),
        "flagged_quality": pd.DataFrame(
            {
                "review_required": [True, False],
            }
        ),
        "aoi_table": pd.DataFrame(),
    }

    out = wrapped(
        result=bundle,
    )

    assert isinstance(out, pd.DataFrame)
    assert out.loc[0, "all_gaze_rows"] == 0
    assert out.loc[0, "review_required_rows"] == 1
    assert out.attrs["r_class"] == "tbl_df|tbl|data.frame"


def test_r4_qc_data_alias_falls_back_to_original_result() -> None:
    sentinel = object()

    def legacy_qc(
        data=None,
    ):
        return sentinel

    wrapped = r4.wrap_r4(
        legacy_qc,
        name="summarize_gazepoint_qc_status",
    )

    out = wrapped(
        data=object(),
    )

    assert out is sentinel


def test_r4_legacy_detector_preserves_existing_python_class_key() -> None:
    result = {
        "events": pd.DataFrame(),
        "_gp3_class": "already-classified",
    }

    def detector(
        *args,
        **kwargs,
    ):
        return result

    wrapped = r4.legacy_detector_result_bridge(
        detector,
        result_class="replacement-class",
    )

    out = wrapped(
        data=pd.DataFrame(),
        return_mode="both",
    )

    assert out is result
    assert out["_gp3_class"] == "already-classified"


def test_r4_geometry_validation_rejects_bool_nonscalar_and_bad_range_cast() -> None:
    def implementation(
        *args,
        **kwargs,
    ):
        return "ok"

    wrapped = r4.geometry_validation_bridge(
        implementation
    )

    with pytest.raises(
        ValueError,
        match="min_width",
    ):
        wrapped(
            min_width=True,
        )

    with pytest.raises(
        ValueError,
        match="min_height",
    ):
        wrapped(
            min_height=[0.1],
        )

    with pytest.raises(
        ValueError,
        match="max_area_prop",
    ):
        wrapped(
            max_area_prop=True,
        )

    with pytest.raises(
        ValueError,
        match="screen_x_range",
    ):
        wrapped(
            screen_x_range=["not-numeric", 1],
        )
