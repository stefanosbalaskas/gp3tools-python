import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3


def test_dynamic_rectangle_nearest_groups_outputs_and_gap():
    gaze = pd.DataFrame(
        {
            "subject": ["S1"] * 4,
            "TIME": [0.0, 4.0, 6.0, 20.0],
            "FPOGX": [0.2, 0.8, 0.8, 0.2],
            "FPOGY": [0.2, 0.2, 0.2, 0.2],
        }
    )
    defs = pd.DataFrame(
        {
            "subject": ["S1"] * 4,
            "aoi_time": [0.0, 0.0, 10.0, 10.0],
            "aoi_name": ["A", "B", "A", "B"],
            "left": [0, 0.5, 0.5, 0],
            "right": [0.5, 1, 1, 0.5],
            "top": [0, 0, 0, 0],
            "bottom": [1, 1, 1, 1],
        }
    )
    out = gp3.add_gazepoint_dynamic_aoi(
        gaze,
        defs,
        shape="rectangle",
        group_cols=["subject"],
        match="nearest",
        max_time_gap=6,
        output="both",
        include_overlap_count=True,
    )
    assert out["aoi_current"].tolist()[:3] == ["A", "B", "A"]
    assert pd.isna(out.loc[3, "aoi_current"])
    assert out["aoi_A"].dtype == bool
    assert out["aoi_overlap_count"].tolist() == [1, 1, 1, 0]
    assert out["aoi_definition_time"].tolist()[:3] == [0, 0, 10]
    assert out.attrs["gazepoint_dynamic_aoi_settings"]["shape"] == "rectangle"


def test_dynamic_previous_next_overlap_and_polygon_boundary():
    gaze = pd.DataFrame(
        {"TIME": [5.0, 5.0, 5.0], "FPOGX": [0.25, 0.75, 1.0], "FPOGY": [0.25, 0.25, 0.5]}
    )
    defs = pd.DataFrame(
        {
            "aoi_time": [0.0, 0.0, 10.0, 10.0],
            "aoi_name": ["A", "B", "A", "B"],
            "left": [0, 0.2, 0.5, 0.5],
            "right": [0.8, 1, 1, 1],
            "top": [0, 0, 0, 0],
            "bottom": [1, 1, 1, 1],
        }
    )
    prev = gp3.add_gazepoint_dynamic_aoi(
        gaze, defs, shape="rectangle", match="previous", overlap="first", output="label"
    )
    nxt = gp3.add_gazepoint_dynamic_aoi(
        gaze, defs, shape="rectangle", match="next", overlap="last", output="label"
    )
    assert prev["aoi_definition_time"].eq(0).all()
    assert nxt["aoi_definition_time"].eq(10).all()
    assert prev.loc[0, "aoi_current"] == "A"
    with pytest.raises(ValueError):
        gp3.add_gazepoint_dynamic_aoi(
            gaze, defs, shape="rectangle", match="previous", overlap="error", output="label"
        )
    vertices = pd.DataFrame(
        {
            "aoi_time": [0, 0, 0],
            "aoi_name": ["P"] * 3,
            "vertex_x": [0, 1, 0],
            "vertex_y": [0, 0, 1],
            "ord": [1, 2, 3],
        }
    )
    polygon_gaze = pd.DataFrame({"TIME": [5.0, 5.0], "FPOGX": [0.25, 0.5], "FPOGY": [0.25, 0.5]})
    pin = gp3.add_gazepoint_dynamic_aoi(
        polygon_gaze,
        vertices,
        shape="polygon",
        vertex_order_col="ord",
        boundary="inside",
        output="label",
    )
    pout = gp3.add_gazepoint_dynamic_aoi(
        polygon_gaze,
        vertices,
        shape="polygon",
        vertex_order_col="ord",
        boundary="outside",
        output="label",
    )
    assert pin.iloc[0]["aoi_current"] == "P"
    assert pin.iloc[1]["aoi_current"] == "P"
    assert pout.iloc[1]["aoi_current"] == "outside"


def test_dynamic_validation_and_legacy():
    legacy_gaze = pd.DataFrame({"TIME": [0, 1], "FPOGX": [0.2, 0.8], "FPOGY": [0.2, 0.2]})
    legacy_defs = pd.DataFrame(
        {
            "TIME": [0, 1],
            "aoi": ["A", "B"],
            "xmin": [0, 0.5],
            "xmax": [0.5, 1],
            "ymin": [0, 0],
            "ymax": [1, 1],
        }
    )
    legacy = gp3.add_gazepoint_dynamic_aoi(legacy_gaze, legacy_defs)
    assert legacy["aoi_current"].tolist() == ["A", "B"]
    with pytest.raises(ValueError):
        gp3.add_gazepoint_dynamic_aoi(legacy_gaze, legacy_defs, shape="bad")
    with pytest.raises(ValueError):
        gp3.add_gazepoint_dynamic_aoi(legacy_gaze, legacy_defs, shape="rectangle", max_time_gap=-1)


def _margin_geometry():
    return pd.DataFrame(
        {
            "stimulus": ["s"] * 2,
            "aoi": ["A", "B"],
            "xmin": [0, 0.5],
            "xmax": [0.5, 1.0],
            "ymin": [0, 0],
            "ymax": [1, 1],
        }
    )


def test_margin_structured_base_changes_ambiguity_and_settings():
    gaze = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "stimulus": ["s"] * 4,
            "x": [0.49, 0.51, 0.1, np.nan],
            "y": [0.5, 0.5, 0.5, 0.5],
        }
    )
    out = gp3.audit_gazepoint_aoi_margin_sensitivity(
        gaze,
        _margin_geometry(),
        gaze_x_col="x",
        gaze_y_col="y",
        gaze_stimulus_col="stimulus",
        sample_id_cols=["id"],
        geometry_aoi_col="aoi",
        geometry_stimulus_col="stimulus",
        x_min_col="xmin",
        y_min_col="ymin",
        x_max_col="xmax",
        y_max_col="ymax",
        margins=[-0.02, 0, 0.02],
        max_margin_change_prop=0.1,
        max_ambiguous_prop=0,
    )
    assert out["_gp3_class"] == "gp3_aoi_margin_sensitivity_audit"
    assert len(out["sample_sensitivity"]) == 12
    assert set(out["margin_summary"]["margin"]) == {-0.02, 0, 0.02}
    assert out["sample_sensitivity"]["margin_assignment_status"].eq("missing_coordinate").any()
    assert (
        out["margin_summary"]["margin_sensitivity_status"]
        .isin(["margin_sensitive", "ambiguous_margin", "base_ambiguous"])
        .any()
    )
    assert out["overview"].loc[0, "aoi_margin_sensitivity_status"] == "review"
    assert {"setting", "value"}.issubset(out["settings"])


def test_margin_tie_first_invalid_geometry_and_multistimulus_guard():
    gaze = pd.DataFrame({"stim": ["s"], "x": [0.5], "y": [0.5]})
    geom = pd.DataFrame(
        {
            "stim": ["s", "s"],
            "aoi": ["A", "B"],
            "xmin": [0, 0.4],
            "xmax": [0.6, 1],
            "ymin": [0, 0],
            "ymax": [1, 1],
        }
    )
    out = gp3.audit_gazepoint_aoi_margin_sensitivity(
        gaze,
        geom,
        gaze_x_col="x",
        gaze_y_col="y",
        gaze_stimulus_col="stim",
        geometry_aoi_col="aoi",
        geometry_stimulus_col="stim",
        x_min_col="xmin",
        y_min_col="ymin",
        x_max_col="xmax",
        y_max_col="ymax",
        margins=[0],
        tie_method="first",
    )
    assert out["sample_sensitivity"].loc[0, "assigned_aoi"] == "A"
    assert out["sample_sensitivity"].loc[0, "margin_assignment_status"] == "multiple_aoi_resolved"
    multi = pd.concat([geom.assign(stim="s1"), geom.assign(stim="s2")], ignore_index=True)
    with pytest.raises(ValueError):
        gp3.audit_gazepoint_aoi_margin_sensitivity(
            gaze.drop(columns="stim"),
            multi,
            gaze_x_col="x",
            gaze_y_col="y",
            geometry_aoi_col="aoi",
            geometry_stimulus_col="stim",
            x_min_col="xmin",
            y_min_col="ymin",
            x_max_col="xmax",
            y_max_col="ymax",
            margins=[0],
            tie_method="first",
        )


def test_margin_legacy():
    gaze = pd.DataFrame({"x": [0.1, 0.9], "y": [0.1, 0.1]})
    geom = pd.DataFrame(
        {"aoi": ["A", "B"], "xmin": [0, 0.5], "xmax": [0.5, 1], "ymin": [0, 0], "ymax": [1, 1]}
    )
    out = gp3.audit_gazepoint_aoi_margin_sensitivity(gaze, geom, margins=[0])
    assert list(out.columns) == ["margin", "assigned_prop", "outside_prop"]


def _denom_data():
    return pd.DataFrame(
        {
            "subject": ["S1", "S2", "S1", "S2", "S1"],
            "condition": ["A", "A", "B", "B", "B"],
            "window_label": ["early", "early", "early", "early", "late"],
            "window_start_ms": [0, 0, 0, 0, 500],
            "window_end_ms": [500, 500, 500, 500, 1000],
            "n_valid_denominator_samples": [10, 8, 2, 0, 5],
            "n_window_samples": [10, 10, 10, 10, 10],
            "n_target_samples": [3, 4, 1, 0, 6],
        }
    )


def test_denominator_structured_counts_summaries_imbalance_and_settings():
    out = gp3.audit_gazepoint_aoi_window_denominators(
        _denom_data(),
        window_col="window_label",
        window_start_col="window_start_ms",
        window_end_col="window_end_ms",
        denominator_col="n_valid_denominator_samples",
        total_col="n_window_samples",
        target_col="n_target_samples",
        condition_col="condition",
        min_denominator_samples=5,
        min_valid_denominator_prop=0.7,
        max_denominator_cv=0.25,
        max_condition_ratio=2,
    )
    assert out["_gp3_class"] == "gp3_aoi_window_denominator_audit"
    assert out["overview"].loc[0, "n_zero_denominator"] == 1
    assert out["overview"].loc[0, "n_target_exceeds_denominator"] == 1
    assert out["overview"].loc[0, "denominator_audit_status"] == "invalid_counts"
    assert set(out["row_audit"]["denominator_audit_status"]) >= {
        "ok",
        "low_denominator",
        "zero_denominator",
        "target_exceeds_denominator",
    }
    assert len(out["window_summary"]) == 2
    assert len(out["condition_window_summary"]) == 3
    assert (
        out["denominator_imbalance"]["denominator_imbalance_status"]
        .isin(
            [
                "condition_denominator_ratio_high",
                "single_condition",
                "condition_denominator_cv_high",
                "ok",
            ]
        )
        .all()
    )
    assert len(out["flagged_rows"]) >= 3
    assert out["settings"]["denominator_col"] == "n_valid_denominator_samples"


def test_denominator_missing_lowprop_single_condition_and_validation():
    d = _denom_data().iloc[:2].copy()
    d.loc[0, "n_valid_denominator_samples"] = np.nan
    out = gp3.audit_gazepoint_aoi_window_denominators(
        d,
        window_col="window_label",
        denominator_col="n_valid_denominator_samples",
        target_col="n_target_samples",
        condition_col="condition",
        min_valid_denominator_prop=0.95,
    )
    assert out["overview"].loc[0, "denominator_audit_status"] == "missing_denominators"
    assert out["denominator_imbalance"].loc[0, "denominator_imbalance_status"] == "single_condition"
    with pytest.raises(ValueError):
        gp3.audit_gazepoint_aoi_window_denominators(
            d,
            window_col="window_label",
            denominator_col="n_valid_denominator_samples",
            target_col="n_target_samples",
            min_denominator_samples=0,
        )


def test_denominator_legacy():
    d = pd.DataFrame({"success": [1, 3, -1], "total": [2, 2, 2]})
    out = gp3.audit_gazepoint_aoi_window_denominators(d)
    assert out.loc[0, "n_invalid"] == 2
