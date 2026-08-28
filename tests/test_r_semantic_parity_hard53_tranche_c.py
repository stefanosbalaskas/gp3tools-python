import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3


def test_add_aoi_r_mode_outputs_and_overlap():
    samples = pd.DataFrame({"x": [0.25, 0.5, np.nan], "y": [0.5, 0.5, 0.5]})
    defs = pd.DataFrame(
        {
            "name": ["A one", "B"],
            "left": [0.0, 0.4],
            "right": [0.6, 1.0],
            "top": [0.0, 0.0],
            "bottom": [1.0, 1.0],
        }
    )
    out = gp3.add_gazepoint_aoi(
        master_df=samples,
        aoi_defs=defs,
        x_col="x",
        y_col="y",
        output="both",
        overlap="first",
        include_overlap_count=True,
    )
    assert out["aoi_current"].iloc[0] == "A one"
    assert out["aoi_current"].iloc[1] == "A one"
    assert pd.isna(out["aoi_current"].iloc[2])
    assert out["aoi_overlap_count"].tolist() == [1, 2, 0]
    assert "aoi_A.one" in out.columns
    last = gp3.add_gazepoint_aoi(
        master_df=samples,
        aoi_defs=defs,
        x_col="x",
        y_col="y",
        output="label",
        overlap="last",
    )
    assert last["aoi_current"].iloc[1] == "B"
    with pytest.raises(ValueError, match="overlapping AOIs"):
        gp3.add_gazepoint_aoi(
            master_df=samples,
            aoi_defs=defs,
            x_col="x",
            y_col="y",
            output="label",
            overlap="error",
        )
    filtered = gp3.add_gazepoint_aoi(
        master_df=samples,
        aoi_defs=defs,
        x_col="x",
        y_col="y",
        aoi_name=["B"],
        output="label",
    )
    assert filtered["aoi_current"].tolist()[:2] == ["outside", "B"]


def test_add_aoi_legacy_mode_unchanged():
    samples = pd.DataFrame({"x": [0.5], "y": [0.5]})
    defs = pd.DataFrame(
        {
            "aoi": ["A", "B"],
            "xmin": [0.0, 0.4],
            "xmax": [0.6, 1.0],
            "ymin": [0.0, 0.0],
            "ymax": [1.0, 1.0],
        }
    )
    out = gp3.add_gazepoint_aoi(samples, x_col="x", y_col="y", aoi_geometry=defs)
    assert out["aoi_current"].iloc[0] == "B"
    assert "aoi_overlap_count" not in out.columns


def test_dynamic_aoi_coverage_r_mode():
    data = pd.DataFrame(
        {
            "subject": ["s1", "s1", "s2", "s2"],
            "aoi_current": ["A", "outside", "B", "A"],
            "aoi_definition_time": [0.0, 0.0, np.nan, 1.0],
            "aoi_time_gap": [0.0, 2.0, np.nan, 0.1],
            "x": [0.2, 0.8, 0.4, np.nan],
            "y": [0.2, 0.8, 0.4, 0.2],
        }
    )
    out = gp3.audit_gazepoint_dynamic_aoi_coverage(
        data,
        label_col="aoi_current",
        definition_time_col="aoi_definition_time",
        time_gap_col="aoi_time_gap",
        group_cols=["subject"],
        max_time_gap=1,
        x_col="x",
        y_col="y",
    )
    assert out["overview"].iloc[0]["audit_status"] == "review"
    assert out["overview"].iloc[0]["n_excessive_gap"] == 1
    assert set(out["flagged_rows"]["dynamic_aoi_issue"]) == {
        "definition_gap_exceeds_threshold",
        "no_dynamic_definition",
        "missing_gaze",
    }
    assert len(out["group_summary"]) == 2


def test_face_standardization_openface_and_validity():
    face = pd.DataFrame(
        {
            "frame": [1, 2, 3],
            "timestamp": [0.0, 0.1, 0.2],
            "confidence": [0.95, 0.5, 0.9],
            "success": [1, 1, 0],
            "AU12_r": [0.1, 0.2, 0.3],
            "pose_Tx": [1, 2, 3],
        }
    )
    out = gp3.standardize_gazepoint_face_columns(face, confidence_threshold=0.8)
    assert out["face_source"].unique().tolist() == ["openface"]
    assert out["face_frame"].tolist() == [1, 2, 3]
    assert out["face_time_ms"].tolist() == [0.0, 100.0, 200.0]
    assert out["face_valid"].tolist() == [True, False, False]
    assert out["face_pose_tx"].tolist() == [1, 2, 3]
    assert "timestamp" in out.columns


def test_face_standardization_string_success_and_drop_originals(tmp_path):
    face = pd.DataFrame(
        {
            "time": [0.0, 0.1],
            "tracking_success": ["yes", "no"],
        }
    )
    out = gp3.standardize_gazepoint_face_columns(face, keep_original_columns=False)
    assert out["face_success"].tolist() == [True, False]
    assert out["face_valid"].tolist() == [True, False]
    assert "tracking_success" not in out.columns
    path = tmp_path / "face.csv"
    face.to_csv(path, index=False)
    from_path = gp3.standardize_gazepoint_face_columns(path)
    assert len(from_path) == 2


def test_face_sync_nearest_time_keeps_outside_tolerance():
    gaze = pd.DataFrame({"time": [0.0, 1.0, 4.0]})
    face = pd.DataFrame(
        {
            "frame": [1, 2],
            "timestamp": [0.02, 1.2],
            "confidence": [0.9, 0.95],
            "success": [1, 1],
            "AU12_r": [0.1, 0.2],
        }
    )
    out = gp3.sync_gazepoint_face_data(
        gaze,
        face,
        gaze_time_col="time",
        face_time_col="face_time_sec",
        tolerance_sec=0.05,
    )
    assert out["face_sync_status"].tolist() == ["matched", "outside_tolerance", "outside_tolerance"]
    assert np.isclose(out["face_sync_diff_sec"].iloc[0], 0.02)
    assert out["face_frame"].notna().all()
    kept = gp3.sync_gazepoint_face_data(
        gaze,
        face,
        gaze_time_col="time",
        face_time_col="face_time_sec",
        tolerance_sec=0.05,
        keep_unmatched=False,
    )
    assert len(kept) == 1


def test_face_sync_frame_exact_and_group_mapping():
    gaze = pd.DataFrame({"trial": ["a", "a", "b"], "frame": [1, 2, 1]})
    face = pd.DataFrame(
        {
            "trial_face": ["a", "a", "b"],
            "frame": [1, 2, 1],
            "confidence": [0.9, 0.9, 0.9],
            "success": [1, 1, 1],
        }
    )
    out = gp3.sync_gazepoint_face_data(
        gaze,
        face,
        method="frame_exact",
        by={"trial": "trial_face"},
        gaze_frame_col="frame",
        face_frame_col="face_frame",
    )
    assert out["face_sync_status"].tolist() == ["matched", "matched", "matched"]
    assert out["face_sync_method"].eq("frame_exact").all()


def _cluster_result():
    timecourse = pd.DataFrame(
        {
            ".gp3_cluster_time_bin": [0.0, 10.0, 20.0],
            "n_subjects": [10, 10, 10],
            "mean_difference": [0.1, 0.4, 0.2],
            "statistic": [1.0, 3.0, 1.5],
            "cluster_id": [np.nan, 1, 1],
            "point_candidate": [False, True, True],
        }
    )
    clusters = pd.DataFrame(
        {
            "cluster_id": [1],
            "cluster_direction": ["positive"],
            "start_time_bin": [10.0],
            "end_time_bin": [20.0],
            "n_time_bins": [2],
            "cluster_statistic": [4.5],
            "max_abs_statistic": [3.0],
            "mean_difference": [0.3],
            "p_value": [0.04],
        }
    )
    permutation = pd.DataFrame({"permutation": [1, 2, 3], "max_cluster_statistic": [1.0, 2.0, 5.0]})
    return {
        "timecourse": timecourse,
        "clusters": clusters,
        "permutation_distribution": permutation,
        "settings": {
            "n_permutations": 3,
            "condition_1": "A",
            "condition_2": "B",
            "difference": "A - B",
            "cluster_threshold": 2.0,
            "tail": "two-sided",
            "cluster_stat": "sum",
            "min_time_bins": 2,
        },
        "model_status": "ok",
        "n_subjects": 10,
        "n_time_bins": 3,
    }


def test_cluster_summary_r_mode_and_legacy_mode():
    result = _cluster_result()
    legacy = gp3.summarise_gazepoint_clusters(result)
    assert isinstance(legacy, pd.DataFrame)
    out = gp3.summarise_gazepoint_clusters(
        result,
        alpha=0.05,
        round_digits=2,
        include_timecourse=False,
    )
    assert out["overview"].iloc[0]["report_status"] == "significant_cluster_evidence"
    assert out["clusters"].iloc[0]["cluster_duration_ms"] == 20.0
    assert bool(out["clusters"].iloc[0]["significant_alpha"])
    assert "timecourse" not in out
    assert out["permutation_summary"].iloc[0]["n_permutations"] == 3


def test_cluster_summary_validation():
    with pytest.raises(ValueError, match="required element"):
        gp3.summarise_gazepoint_clusters({}, alpha=0.05)
    result = _cluster_result()
    with pytest.raises(ValueError, match="alpha"):
        gp3.summarise_gazepoint_clusters(result, alpha=1)
    with pytest.raises(ValueError, match="round_digits"):
        gp3.summarise_gazepoint_clusters(result, alpha=0.05, round_digits=-1)
