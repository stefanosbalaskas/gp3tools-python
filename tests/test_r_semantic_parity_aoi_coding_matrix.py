import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3


def geometry_bounds():
    return pd.DataFrame(
        {
            "aoi_name": ["A", "B"],
            "x_min": [0.0, 0.5],
            "y_min": [0.0, 0.0],
            "x_max": [0.5, 1.0],
            "y_max": [1.0, 1.0],
        }
    )


def gaze_basic():
    return pd.DataFrame(
        {
            "AOI": ["A", "B", "outside", "A", None],
            "FPOGX": [0.25, 0.75, 1.5, 0.75, np.nan],
            "FPOGY": [0.25, 0.25, 0.5, 0.75, 0.5],
            "USER_ID": ["u1"] * 5,
        }
    )


def test_legacy_frequency_table_preserved():
    data = pd.DataFrame({"AOI": ["A", "A", "B"], "trial": [1, 1, 1]})
    out = gp3.audit_gazepoint_aoi_coding_matrix(data, aoi_col="AOI", group_cols=["trial"])
    assert list(out.columns) == ["trial", "AOI", "n"]
    assert out["n"].sum() == 3


def test_r_basic_structure_and_counts():
    out = gp3.audit_gazepoint_aoi_coding_matrix(
        gaze_data=gaze_basic(), aoi_geometry=geometry_bounds()
    )
    assert out["_gp3_class"] == "gp3_aoi_coding_matrix_audit"
    assert {
        "overview",
        "geometry_summary",
        "sample_coding",
        "coding_matrix",
        "observed_summary",
        "derived_summary",
        "flagged_samples",
        "settings",
    }.issubset(out)
    overview = out["overview"].iloc[0]
    assert overview["n_gaze_rows"] == 5
    assert overview["n_geometry_rows"] == 2
    assert overview["n_aois"] == 2
    assert overview["n_coded_samples"] == 5
    assert overview["n_comparable_samples"] == 4
    assert overview["n_mismatched_samples"] == 1
    assert overview["n_missing_coordinate_samples"] == 1
    assert overview["n_flagged_samples"] == 2


def test_r_positional_two_dataframe_interface():
    out = gp3.audit_gazepoint_aoi_coding_matrix(gaze_basic(), geometry_bounds())
    assert isinstance(out, dict)
    assert len(out["sample_coding"]) == 5


def test_r_sample_index_is_one_based():
    gaze = gaze_basic().copy()
    gaze.index = [10, 20, 30, 40, 50]
    out = gp3.audit_gazepoint_aoi_coding_matrix(gaze_data=gaze, aoi_geometry=geometry_bounds())
    assert out["sample_coding"][".gp3_sample_index"].tolist() == [1, 2, 3, 4, 5]


def test_r_aliases_and_sample_id_columns():
    gaze = pd.DataFrame(
        {
            "AOI": ["A", "B"],
            "MEDIA_ID": ["s1", "s1"],
            "FPOGX": [0.25, 0.75],
            "FPOGY": [0.5, 0.5],
        }
    )
    out = gp3.audit_gazepoint_aoi_coding_matrix(
        gaze_data=gaze,
        aoi_geometry=geometry_bounds(),
        sample_id_cols=["MEDIA_ID", "missing"],
    )
    sample = out["sample_coding"]
    assert "media_id" in sample.columns
    assert "missing" not in sample.columns
    settings = out["settings"].set_index("setting")["value"]
    assert settings["observed_aoi_col"] == "aoi"
    assert settings["gaze_stimulus_col"] == "media_id"
    assert settings["sample_id_cols"] == "media_id"


def test_r_outside_values_and_blank_observed():
    gaze = pd.DataFrame(
        {
            "observed_aoi": ["NONE", "  background  ", "   "],
            "x": [2.0, 2.0, 2.0],
            "y": [0.5, 0.5, 0.5],
        }
    )
    out = gp3.audit_gazepoint_aoi_coding_matrix(gaze_data=gaze, aoi_geometry=geometry_bounds())
    sample = out["sample_coding"]
    assert sample["observed_aoi"].iloc[0] == "outside"
    assert sample["observed_aoi"].iloc[1] == "outside"
    assert pd.isna(sample["observed_aoi"].iloc[2])
    assert sample["aoi_coding_status"].iloc[2] == "observed_missing"


def test_r_ambiguous_tie():
    gaze = pd.DataFrame({"aoi": ["A"], "x": [0.5], "y": [0.5]})
    out = gp3.audit_gazepoint_aoi_coding_matrix(
        gaze_data=gaze, aoi_geometry=geometry_bounds(), tie_method="ambiguous"
    )
    row = out["sample_coding"].iloc[0]
    assert row["derived_aoi"] == "ambiguous"
    assert row["n_matching_aois"] == 2
    assert row["derived_assignment_status"] == "ambiguous_aoi"
    assert row["aoi_coding_status"] == "ambiguous_derived"
    assert pd.isna(row["coding_match"])


def test_r_first_tie_resolution():
    gaze = pd.DataFrame({"aoi": ["A"], "x": [0.5], "y": [0.5]})
    out = gp3.audit_gazepoint_aoi_coding_matrix(
        gaze_data=gaze, aoi_geometry=geometry_bounds(), tie_method="first"
    )
    row = out["sample_coding"].iloc[0]
    assert row["derived_aoi"] == "A"
    assert row["derived_assignment_status"] == "multiple_aoi_resolved"
    assert bool(row["coding_match"])


def test_r_missing_coordinate_status():
    gaze = pd.DataFrame({"aoi": ["A"], "x": [np.nan], "y": [0.5]})
    out = gp3.audit_gazepoint_aoi_coding_matrix(gaze_data=gaze, aoi_geometry=geometry_bounds())
    row = out["sample_coding"].iloc[0]
    assert row["derived_aoi"] == "missing_coordinate"
    assert row["derived_assignment_status"] == "missing_coordinate"
    assert row["aoi_coding_status"] == "missing_coordinate"


def test_r_no_aoi_assignment_is_outside():
    gaze = pd.DataFrame({"aoi": ["outside"], "x": [2.0], "y": [2.0]})
    out = gp3.audit_gazepoint_aoi_coding_matrix(gaze_data=gaze, aoi_geometry=geometry_bounds())
    row = out["sample_coding"].iloc[0]
    assert row["derived_aoi"] == "outside"
    assert row["derived_assignment_status"] == "no_aoi"
    assert row["aoi_coding_status"] == "ok"


def test_r_threshold_status_ok_and_review():
    gaze = pd.DataFrame({"aoi": ["A", "A"], "x": [0.25, 0.75], "y": [0.5, 0.5]})
    review = gp3.audit_gazepoint_aoi_coding_matrix(gaze_data=gaze, aoi_geometry=geometry_bounds())
    ok = gp3.audit_gazepoint_aoi_coding_matrix(
        gaze_data=gaze,
        aoi_geometry=geometry_bounds(),
        max_mismatch_prop=1,
        max_ambiguous_prop=1,
        max_missing_coordinate_prop=1,
        tie_method="first",
    )
    assert review["overview"].iloc[0]["aoi_coding_matrix_status"] == "review"
    assert ok["overview"].iloc[0]["aoi_coding_matrix_status"] == "ok"


def test_r_invalid_geometry_ignored():
    geometry = geometry_bounds()
    geometry.loc[1, "x_max"] = geometry.loc[1, "x_min"]
    gaze = pd.DataFrame({"aoi": ["A"], "x": [0.25], "y": [0.5]})
    ignored = gp3.audit_gazepoint_aoi_coding_matrix(
        gaze_data=gaze, aoi_geometry=geometry, ignore_invalid_geometry=True
    )
    retained = gp3.audit_gazepoint_aoi_coding_matrix(
        gaze_data=gaze, aoi_geometry=geometry, ignore_invalid_geometry=False
    )
    assert ignored["overview"].iloc[0]["n_aois_used"] == 1
    assert retained["overview"].iloc[0]["n_aois_used"] == 2
    assert retained["geometry_summary"].loc[1, "aoi_geometry_status"] == "invalid_dimension"


def test_r_invalid_coordinate_geometry():
    geometry = geometry_bounds()
    geometry.loc[1, "x_min"] = np.nan
    out = gp3.audit_gazepoint_aoi_coding_matrix(gaze_data=gaze_basic(), aoi_geometry=geometry)
    assert out["overview"].iloc[0]["n_aois_used"] == 1
    assert out["geometry_summary"].loc[1, "aoi_geometry_status"] == "invalid_coordinate"


def test_r_origin_size_geometry():
    geometry = pd.DataFrame(
        {
            "aoi": ["A", "B"],
            "x": [0.0, 0.5],
            "y": [0.0, 0.0],
            "width": [0.5, 0.5],
            "height": [1.0, 1.0],
        }
    )
    gaze = pd.DataFrame({"aoi": ["A", "B"], "FPOGX": [0.25, 0.75], "FPOGY": [0.5, 0.5]})
    out = gp3.audit_gazepoint_aoi_coding_matrix(gaze_data=gaze, aoi_geometry=geometry)
    summary = out["geometry_summary"]
    assert np.allclose(summary["x_max"], [0.5, 1.0])
    assert np.allclose(summary["area"], [0.5, 0.5])


def test_r_geometry_outside_screen_is_recorded_not_invalidated():
    geometry = geometry_bounds()
    geometry.loc[0, "x_min"] = -0.1
    out = gp3.audit_gazepoint_aoi_coding_matrix(gaze_data=gaze_basic(), aoi_geometry=geometry)
    row = out["geometry_summary"].iloc[0]
    assert bool(row["outside_screen"])
    assert row["aoi_geometry_status"] == "ok"


def test_r_geometry_too_large_status():
    geometry = pd.DataFrame(
        {
            "aoi": ["A"],
            "x_min": [0.0],
            "y_min": [0.0],
            "x_max": [2.0],
            "y_max": [2.0],
        }
    )
    out = gp3.audit_gazepoint_aoi_coding_matrix(
        gaze_data=pd.DataFrame({"aoi": ["A"], "x": [0.2], "y": [0.2]}),
        aoi_geometry=geometry,
    )
    assert out["geometry_summary"].iloc[0]["aoi_geometry_status"] == "too_large"


def test_r_multiple_stimuli_requires_gaze_stimulus():
    geometry = pd.DataFrame(
        {
            "aoi": ["A", "A"],
            "stimulus": ["s1", "s2"],
            "x_min": [0.0, 0.0],
            "y_min": [0.0, 0.0],
            "x_max": [1.0, 1.0],
            "y_max": [1.0, 1.0],
        }
    )
    gaze = pd.DataFrame({"aoi": ["A"], "x": [0.5], "y": [0.5]})
    with pytest.raises(ValueError, match="gaze_stimulus_col"):
        gp3.audit_gazepoint_aoi_coding_matrix(
            gaze_data=gaze,
            aoi_geometry=geometry,
            geometry_stimulus_col="stimulus",
        )


def test_r_stimulus_specific_assignment():
    geometry = pd.DataFrame(
        {
            "aoi": ["A", "B"],
            "stimulus": ["s1", "s2"],
            "x_min": [0.0, 0.0],
            "y_min": [0.0, 0.0],
            "x_max": [1.0, 1.0],
            "y_max": [1.0, 1.0],
        }
    )
    gaze = pd.DataFrame(
        {
            "aoi": ["A", "B"],
            "stimulus": ["s1", "s2"],
            "x": [0.5, 0.5],
            "y": [0.5, 0.5],
        }
    )
    out = gp3.audit_gazepoint_aoi_coding_matrix(
        gaze_data=gaze,
        aoi_geometry=geometry,
        gaze_stimulus_col="stimulus",
        geometry_stimulus_col="stimulus",
    )
    assert out["sample_coding"]["derived_aoi"].tolist() == ["A", "B"]


def test_r_geometry_alias_columns():
    geometry = pd.DataFrame(
        {
            "AOI": ["A"],
            "MEDIA_ID": ["s1"],
            "xmin": [0.0],
            "ymin": [0.0],
            "xmax": [1.0],
            "ymax": [1.0],
        }
    )
    gaze = pd.DataFrame({"AOI": ["A"], "MEDIA_ID": ["s1"], "FPOGX": [0.5], "FPOGY": [0.5]})
    out = gp3.audit_gazepoint_aoi_coding_matrix(
        gaze_data=gaze,
        aoi_geometry=geometry,
        geometry_stimulus_col="MEDIA_ID",
    )
    settings = out["settings"].set_index("setting")["value"]
    assert settings["geometry_aoi_col"] == "aoi"
    assert settings["geometry_stimulus_col"] == "media_id"


def test_r_coding_matrix_proportions_sum_to_one():
    out = gp3.audit_gazepoint_aoi_coding_matrix(
        gaze_data=gaze_basic(), aoi_geometry=geometry_bounds()
    )
    assert np.isclose(out["coding_matrix"]["sample_prop"].sum(), 1.0)


def test_r_observed_summary_excludes_missing_observed():
    out = gp3.audit_gazepoint_aoi_coding_matrix(
        gaze_data=gaze_basic(), aoi_geometry=geometry_bounds()
    )
    assert out["observed_summary"]["n_samples"].sum() == 4
    assert out["derived_summary"]["n_samples"].sum() == 5


def test_r_settings_empty_sample_ids_are_missing():
    out = gp3.audit_gazepoint_aoi_coding_matrix(
        gaze_data=pd.DataFrame({"aoi": ["A"], "x": [0.2], "y": [0.2]}),
        aoi_geometry=geometry_bounds(),
    )
    settings = out["settings"].set_index("setting")["value"]
    assert pd.isna(settings["sample_id_cols"])
    assert settings["ignore_invalid_geometry"] == "TRUE"
    assert settings["screen_x_range"] == "0, 1"


@pytest.mark.parametrize("tie_method", ["bad", "", None])
def test_r_tie_method_validation(tie_method):
    with pytest.raises(ValueError, match="tie_method"):
        gp3.audit_gazepoint_aoi_coding_matrix(
            gaze_data=gaze_basic(),
            aoi_geometry=geometry_bounds(),
            tie_method=tie_method,
        )


@pytest.mark.parametrize(
    "name,value",
    [
        ("outside_label", ""),
        ("ambiguous_label", None),
        ("missing_label", 1),
    ],
)
def test_r_label_validation(name, value):
    kwargs = {"gaze_data": gaze_basic(), "aoi_geometry": geometry_bounds(), name: value}
    with pytest.raises(ValueError):
        gp3.audit_gazepoint_aoi_coding_matrix(**kwargs)


@pytest.mark.parametrize(
    "name,value",
    [
        ("max_mismatch_prop", -0.1),
        ("max_mismatch_prop", 1.1),
        ("max_ambiguous_prop", np.inf),
        ("max_missing_coordinate_prop", True),
    ],
)
def test_r_proportion_validation(name, value):
    kwargs = {"gaze_data": gaze_basic(), "aoi_geometry": geometry_bounds(), name: value}
    with pytest.raises(ValueError):
        gp3.audit_gazepoint_aoi_coding_matrix(**kwargs)


@pytest.mark.parametrize("value", [[], [""], [None], 3])
def test_r_observed_outside_values_validation(value):
    with pytest.raises(ValueError, match="observed_outside_values"):
        gp3.audit_gazepoint_aoi_coding_matrix(
            gaze_data=gaze_basic(),
            aoi_geometry=geometry_bounds(),
            observed_outside_values=value,
        )


@pytest.mark.parametrize(
    "name,value",
    [
        ("screen_x_range", (1, 0)),
        ("screen_y_range", (0, np.inf)),
        ("screen_x_range", (0,)),
    ],
)
def test_r_screen_range_validation(name, value):
    kwargs = {"gaze_data": gaze_basic(), "aoi_geometry": geometry_bounds(), name: value}
    with pytest.raises(ValueError):
        gp3.audit_gazepoint_aoi_coding_matrix(**kwargs)


def test_r_ignore_invalid_geometry_validation():
    with pytest.raises(ValueError, match="ignore_invalid_geometry"):
        gp3.audit_gazepoint_aoi_coding_matrix(
            gaze_data=gaze_basic(),
            aoi_geometry=geometry_bounds(),
            ignore_invalid_geometry="yes",
        )


def test_r_empty_input_validation():
    with pytest.raises(ValueError, match="gaze_data"):
        gp3.audit_gazepoint_aoi_coding_matrix(
            gaze_data=pd.DataFrame(), aoi_geometry=geometry_bounds()
        )
    with pytest.raises(ValueError, match="aoi_geometry"):
        gp3.audit_gazepoint_aoi_coding_matrix(gaze_data=gaze_basic(), aoi_geometry=pd.DataFrame())


def test_r_missing_geometry_columns_validation():
    with pytest.raises(ValueError, match="AOI geometry requires"):
        gp3.audit_gazepoint_aoi_coding_matrix(
            gaze_data=gaze_basic(),
            aoi_geometry=pd.DataFrame({"aoi": ["A"], "x": [0.0]}),
        )


def test_r_required_gaze_columns_validation():
    with pytest.raises(ValueError, match="observed_aoi_col"):
        gp3.audit_gazepoint_aoi_coding_matrix(
            gaze_data=pd.DataFrame({"x": [0.1], "y": [0.1]}),
            aoi_geometry=geometry_bounds(),
        )
    with pytest.raises(ValueError, match="gaze_x_col"):
        gp3.audit_gazepoint_aoi_coding_matrix(
            gaze_data=pd.DataFrame({"aoi": ["A"], "y": [0.1]}),
            aoi_geometry=geometry_bounds(),
        )
    with pytest.raises(ValueError, match="gaze_y_col"):
        gp3.audit_gazepoint_aoi_coding_matrix(
            gaze_data=pd.DataFrame({"aoi": ["A"], "x": [0.1]}),
            aoi_geometry=geometry_bounds(),
        )


def test_r_explicit_missing_column_validation():
    with pytest.raises(ValueError, match="observed_aoi_col"):
        gp3.audit_gazepoint_aoi_coding_matrix(
            gaze_data=gaze_basic(),
            aoi_geometry=geometry_bounds(),
            observed_aoi_col="not_here",
        )


def test_r_conflicting_data_and_gaze_data():
    with pytest.raises(TypeError, match="either data or gaze_data"):
        gp3.audit_gazepoint_aoi_coding_matrix(
            gaze_basic(),
            gaze_data=gaze_basic(),
            aoi_geometry=geometry_bounds(),
        )
