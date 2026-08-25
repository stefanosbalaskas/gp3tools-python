import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3


def test_geometry_legacy_contract_is_preserved():
    geometry = pd.DataFrame(
        {
            "aoi": ["A", "B"],
            "xmin": [0.0, 0.5],
            "xmax": [0.4, 0.8],
            "ymin": [0.0, 0.2],
            "ymax": [0.4, 0.6],
        }
    )
    out = gp3.audit_gazepoint_aoi_geometry(geometry)
    assert out["valid"] is True
    assert out["summary"].loc[0, "n_aois"] == 2

    overlap = gp3.audit_gazepoint_aoi_overlap(geometry)
    assert list(overlap.columns) == ["aoi1", "aoi2", "overlap_area"]


def test_geometry_r_bounds_status_precedence_settings_and_duplicates():
    geometry = pd.DataFrame(
        {
            "AOI": ["A", "B", "C", "D", "E"],
            "MEDIA_ID": [1, 1, 1, 1, 1],
            "xmin": [0.0, 0.0, 0.9, 0.4, np.nan],
            "ymin": [0.0, 0.0, 0.9, 0.4, 0.1],
            "xmax": [0.2, 0.2, 1.1, 0.3, 0.4],
            "ymax": [0.2, 0.2, 1.1, 0.7, 0.4],
        }
    )
    out = gp3.audit_gazepoint_aoi_geometry(
        data=geometry,
        stimulus_col="MEDIA_ID",
        min_width=0.25,
        min_height=0.25,
        min_area=0.05,
        max_area_prop=0.5,
        require_within_screen=True,
    )
    assert out["_gp3_class"] == "gp3_aoi_geometry_audit"
    assert set(out) >= {
        "overview",
        "geometry_summary",
        "size_summary",
        "duplicate_geometry",
        "flagged_aois",
        "settings",
    }

    summary = out["geometry_summary"]
    assert list(summary["aoi"]) == ["A", "B", "C", "D", "E"]
    assert list(summary["aoi_geometry_status"]) == [
        "too_small",
        "too_small",
        "outside_screen",
        "invalid_dimension",
        "invalid_coordinate",
    ]
    assert bool(summary.loc[2, "outside_screen"]) is True
    assert out["overview"].loc[0, "n_duplicate_geometry_groups"] == 1
    assert out["overview"].loc[0, "aoi_geometry_status"] == "review"
    assert out["duplicate_geometry"].loc[0, "aoi_values"] == "A, B"

    settings = dict(zip(out["settings"]["setting"], out["settings"]["value"], strict=True))
    assert settings["aoi_col"] == "aoi"
    assert settings["stimulus_col"] == "media_id"
    assert settings["require_within_screen"] == "TRUE"


def test_geometry_r_origin_size_and_screen_requirement_toggle():
    geometry = pd.DataFrame(
        {
            "aoi_name": ["A", "B"],
            "x": [0.1, 0.9],
            "y": [0.2, 0.9],
            "width": [0.2, 0.3],
            "height": [0.2, 0.3],
        }
    )
    out = gp3.audit_gazepoint_aoi_geometry(
        data=geometry,
        require_within_screen=False,
    )
    summary = out["geometry_summary"]
    assert out["overview"].loc[0, "coordinate_format"] == "origin_size"
    assert summary.loc[1, "outside_screen"]
    assert summary.loc[1, "aoi_geometry_status"] == "ok"
    assert np.isclose(summary.loc[0, "area"], 0.04)


def test_geometry_r_validation():
    with pytest.raises(ValueError, match="at least one row"):
        gp3.audit_gazepoint_aoi_geometry(data=pd.DataFrame(), aoi_col="aoi")
    with pytest.raises(ValueError, match="requires either"):
        gp3.audit_gazepoint_aoi_geometry(
            data=pd.DataFrame({"aoi": ["A"], "x": [0.1]}),
            aoi_col="aoi",
        )
    with pytest.raises(ValueError, match="max_area_prop"):
        gp3.audit_gazepoint_aoi_geometry(
            data=pd.DataFrame(
                {
                    "aoi": ["A"],
                    "xmin": [0],
                    "ymin": [0],
                    "xmax": [1],
                    "ymax": [1],
                }
            ),
            max_area_prop=1.1,
        )


def test_overlap_r_pairwise_thresholds_and_summary():
    geometry = pd.DataFrame(
        {
            "aoi": ["A", "B", "C"],
            "stimulus": ["S1", "S1", "S1"],
            "xmin": [0.0, 0.25, 0.8],
            "ymin": [0.0, 0.0, 0.0],
            "xmax": [0.5, 0.75, 1.0],
            "ymax": [0.5, 0.5, 0.2],
        }
    )
    out = gp3.audit_gazepoint_aoi_overlap(
        data=geometry,
        stimulus_col="stimulus",
        min_overlap_area=0.10,
        min_overlap_prop=0.30,
    )
    assert out["_gp3_class"] == "gp3_aoi_overlap_audit"
    pairs = out["pairwise_overlap"]
    assert len(pairs) == 3
    ab = pairs.loc[(pairs["aoi_1"] == "A") & (pairs["aoi_2"] == "B")].iloc[0]
    assert np.isclose(ab["overlap_area"], 0.125)
    assert np.isclose(ab["overlap_prop_smaller"], 0.5)
    assert ab["aoi_overlap_status"] == "overlap"
    assert out["overview"].loc[0, "n_overlapping_pairs"] == 1
    assert out["overview"].loc[0, "n_flagged_overlaps"] == 1
    assert out["overview"].loc[0, "aoi_overlap_status"] == "review"
    assert out["overlap_summary"].loc[0, "aoi_overlap_summary_status"] == "review"


def test_overlap_r_ignore_invalid_geometry_and_empty_pairs():
    geometry = pd.DataFrame(
        {
            "aoi": ["A", "B"],
            "xmin": [0.0, 0.5],
            "ymin": [0.0, 0.5],
            "xmax": [0.4, 0.4],
            "ymax": [0.4, 0.8],
        }
    )
    out = gp3.audit_gazepoint_aoi_overlap(
        data=geometry,
        aoi_col="aoi",
        ignore_invalid_geometry=True,
    )
    assert out["overview"].loc[0, "n_aois_used"] == 1
    assert out["pairwise_overlap"].empty
    assert out["overlap_summary"].empty

    out_all = gp3.audit_gazepoint_aoi_overlap(
        data=geometry,
        ignore_invalid_geometry=False,
    )
    assert out_all["overview"].loc[0, "n_aois_used"] == 2
    assert len(out_all["pairwise_overlap"]) == 1


def test_overlap_r_validation():
    geometry = pd.DataFrame(
        {
            "aoi": ["A"],
            "xmin": [0.0],
            "ymin": [0.0],
            "xmax": [0.5],
            "ymax": [0.5],
        }
    )
    with pytest.raises(ValueError, match="min_overlap_area"):
        gp3.audit_gazepoint_aoi_overlap(data=geometry, min_overlap_area=-1)
    with pytest.raises(ValueError, match="min_overlap_prop"):
        gp3.audit_gazepoint_aoi_overlap(data=geometry, min_overlap_prop=1.1)
