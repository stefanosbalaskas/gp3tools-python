import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3


def test_aoi_sample_r_timing_mode():
    data = pd.DataFrame({"MEDIA_ID": ["m"] * 3, "AOI": ["A", "A", "B"], "TIME": [0.0, 1.0, 2.0]})
    out = gp3.summarise_aoi_samples(data, time_col="TIME")
    assert list(out["AOI"]) == ["A", "B"]
    assert out.loc[out["AOI"].eq("A"), "aoi_sample_count"].iloc[0] == 2
    assert np.isclose(out.loc[out["AOI"].eq("A"), "approx_time_viewed_sec"].iloc[0], 2)


def test_sequence_distance_r_weighted_mode():
    out = gp3.compute_gazepoint_sequence_distance(
        ["A", None, "B"], ["A", "C"], ignore_missing=True, substitution_cost=2
    )
    assert out.iloc[0]["edit_distance"] == 2
    assert out.iloc[0]["sequence_a_length"] == 2


def test_sequence_recurrence_r_mode():
    out = gp3.compute_gazepoint_sequence_recurrence(sequence=["A", "B", "A"], min_line=1)
    assert out.iloc[0]["recurrence_points"] == 1
    assert out.iloc[0]["recurrence_status"] == "ok"


def test_aoi_screen_coverage_r_mode():
    geometry = pd.DataFrame(
        {"name": ["A"], "x_min": [0.0], "x_max": [2.0], "y_min": [0.0], "y_max": [1.0]}
    )
    out = gp3.audit_gazepoint_aoi_screen_coverage(
        data=geometry, screen_width=1, screen_height=1, aoi_col="name"
    )
    assert out["overall_summary"].iloc[0]["n_outside_screen"] == 1
    assert np.isclose(out["overall_summary"].iloc[0]["total_clipped_area"], 1)


def test_fixation_summary_r_mode():
    data = pd.DataFrame(
        {
            "MEDIA_ID": ["m", "m"],
            "AOI": ["A", "A"],
            "FPOGD": [0.1, 0.2],
            "FPOGS": [0.2, 0.3],
        }
    )
    out = gp3.summarise_fixations(data)
    assert out.iloc[0]["fixation_count"] == 2
    assert np.isclose(out.iloc[0]["fixation_duration_sum_sec"], 0.3)


def test_saccade_metrics_r_mode():
    data = pd.DataFrame({"x": [0.0, 1.0, 1.0], "y": [0.0, 0.0, 1.0], "t": [0, 1, 2]})
    out = gp3.compute_gazepoint_saccade_metrics(
        data, x_col="x", y_col="y", time_col="t", distance_scale=2
    )
    assert len(out) == 2
    assert np.allclose(out["saccade_amplitude"], [2, 2])


def test_mean_pupil_min_eyes_and_r_aliases():
    data = pd.DataFrame({"LPupil": [1.0, np.nan], "RPupil": [3.0, 4.0]})
    out = gp3.mean_gazepoint_pupil(master_df=data, lp_col="LPupil", rp_col="RPupil", min_eyes=2)
    assert out["pupil_mean"].iloc[0] == 2
    assert pd.isna(out["pupil_mean"].iloc[1])


def test_combine_eyes_r_methods_and_bounds():
    data = pd.DataFrame({"l": [1.0, 10.0, np.nan], "r": [2.0, 3.0, 4.0]})
    out = gp3.combine_gazepoint_eyes(
        data, left_col="l", right_col="r", method="prefer_left", valid_max=5
    )
    assert out["pupil_combined"].tolist() == [1.0, 3.0, 4.0]


def test_construct_combined_pupil_provenance():
    data = pd.DataFrame({"l": [1.0, np.nan], "r": [3.0, 4.0]})
    out = gp3.construct_gazepoint_combined_pupil(
        data, left_col="l", right_col="r", policy="available_eye"
    )
    assert out["pupil_combined"].tolist() == [2.0, 4.0]
    assert out["pupil_binocular_status"].tolist() == [
        "bilateral_observed",
        "right_only_observed",
    ]


def test_screen_bounds_r_detailed_mode():
    data = pd.DataFrame({"g": ["a", "a"], "x": [0.0, 11.0], "y": [0.0, 5.0]})
    out = gp3.audit_gazepoint_screen_bounds(
        data,
        x_col="x",
        y_col="y",
        screen_width=10,
        screen_height=10,
        group_cols=["g"],
    )
    overall = out["overall_summary"].iloc[0]
    assert overall["n_zero_zero"] == 1
    assert overall["n_outside_bounds"] == 1
    assert overall["n_invalid_coordinate"] == 2


def test_naming_audit_r_and_legacy_fields():
    out = gp3.audit_gazepoint_naming_consistency(["summarise_x", "summarize_x"])
    assert out["summary"].iloc[0]["status"] == "pass"
    assert out["summary"].iloc[0]["n_names"] == 2
    assert out["pairs"].iloc[0]["status"] == "paired"


def test_file_pair_r_and_legacy_fields(tmp_path):
    (tmp_path / "User 1_all_gaze.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "User 1_fixations.csv").write_text("x\n1\n", encoding="utf-8")
    (tmp_path / "User 2_all_gaze.csv").write_text("x\n1\n", encoding="utf-8")
    out = gp3.check_gazepoint_file_pairs(tmp_path)
    assert out.loc[out["participant"].eq("User 1"), "status"].iloc[0] == "complete"
    assert out.loc[out["user"].eq("1"), "paired"].iloc[0]
    assert out.loc[out["participant"].eq("User 2"), "status"].iloc[0] == "missing_fixation"


def test_missingness_r_fields_and_alias():
    data = pd.DataFrame({"g": ["a", "a"], "x": [1.0, np.nan]})
    out = gp3.summarize_gazepoint_missingness(
        data, cols=["x"], group_cols=["g"], include_group_cols=False
    )
    assert out.iloc[0]["variable"] == "x"
    assert out.iloc[0]["n_missing"] == 1
    assert np.isclose(out.iloc[0]["missing_rate"], 0.5)


def test_phase_coverage_r_fields_and_alias():
    data = pd.DataFrame(
        {"task_phase": ["a", "a", "b"], "t": [0.0, 1.0, 2.0], "x": [1.0, np.nan, 2.0]}
    )
    out = gp3.summarize_gazepoint_phase_coverage(data, time_col="t", value_cols=["x"])
    a = out.loc[out["phase"].eq("a")].iloc[0]
    assert a["n_rows"] == 2
    assert a["time_span"] == 1
    assert np.isclose(a["complete_value_rate"], 0.5)


def test_cluster_timecourse_r_mode_shape_and_columns():
    out = gp3.simulate_gazepoint_cluster_timecourse_data(
        n_subjects=2, n_time_bins=3, conditions=["c", "t"], seed=1
    )
    assert len(out) == 12
    assert list(out.columns) == ["subject", "condition", "time_bin", "outcome"]
    assert out["subject"].iloc[0] == "S001"
    assert out["time_bin"].min() == 1


def test_time_cluster_summary_and_report_r_mode():
    result = {
        "clusters": pd.DataFrame(
            {
                "cluster_id": [1],
                "cluster_direction": ["positive"],
                "start_time_bin": [2],
                "end_time_bin": [4],
                "cluster_statistic": [5.0],
                "p_value": [0.01],
            }
        ),
        "settings": {"tail": "two-sided"},
    }
    summary = gp3.summarize_gazepoint_time_clusters(result, alpha=0.05)
    assert bool(summary.iloc[0]["cluster_significant"])
    assert summary.iloc[0]["n_time_bins"] == 3
    report = gp3.report_gazepoint_cluster_permutation(result, alpha=0.05)
    assert report["report_status"] == "ok"
    assert "2-4" in report["report_text"]


def test_missingness_and_phase_reports_r_mode():
    miss = pd.DataFrame({"x": [1.0, np.nan]})
    m = gp3.report_gazepoint_missingness(miss, cols=["x"], digits=1)
    assert m["overall"].iloc[0]["overall_missing_rate"] == 0.5

    phase = pd.DataFrame({"task_phase": ["a", "b"], "t": [0.0, 1.0], "x": [1.0, np.nan]})
    p = gp3.report_gazepoint_phase_coverage(phase, time_col="t", value_cols=["x"], digits=1)
    assert p["overall"].iloc[0]["n_phases"] == 2


def test_export_master_output_dir_is_optional(tmp_path, monkeypatch):
    # Only validate the fixed optionality at the native signature/runtime layer.
    import inspect

    signature = inspect.signature(gp3.export_gazepoint_master_audit)
    assert signature.parameters["output_dir"].default == "."


def test_model_prediction_model_is_optional_in_signature():
    import inspect

    signature = inspect.signature(gp3.plot_gazepoint_model_predictions)
    assert signature.parameters["model"].default is None


def test_residual_plot_data_mode():
    import matplotlib.pyplot as plt

    data = pd.DataFrame({"fitted": [1.0, 2.0, 3.0], "residual": [0.1, -0.2, 0.05]})
    fig = gp3.plot_gazepoint_model_residuals(data=data)
    assert fig.axes[0].get_xlabel() == "Fitted values"
    plt.close(fig)


def test_representatives_r_mode_multiple_per_cluster():
    distance = pd.DataFrame(
        [[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )
    fit = {
        "distance": distance,
        "assignments": pd.DataFrame({"sequence_id": ["a", "b", "c"], "cluster": [1, 1, 1]}),
        "medoids": ["b"],
    }
    out = gp3.extract_gazepoint_representative_scanpaths(x=fit, n_per_cluster=2)
    assert out["sequence_id"].tolist() == ["b", "a"]
    assert out["is_model_medoid"].tolist() == [True, False]


def test_recurrence_grouped_data_mode_and_too_short():
    data = pd.DataFrame(
        {
            "g": ["a", "a", "a", "b"],
            "t": [2, 1, 3, 1],
            "AOI": ["A", "A", "B", "Z"],
        }
    )
    out = gp3.compute_gazepoint_sequence_recurrence(
        data=data,
        aoi_col="AOI",
        group_cols=["g"],
        time_col="t",
        min_line=1,
    )
    assert set(out["g"]) == {"a", "b"}
    assert out.loc[out["g"].eq("b"), "recurrence_status"].iloc[0] == "too_short"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_line": 0},
        {"include_missing": "yes"},
        {"missing_label": ""},
    ],
)
def test_recurrence_validation_paths(kwargs):
    with pytest.raises(ValueError):
        gp3.compute_gazepoint_sequence_recurrence(sequence=["A", "B"], **kwargs)


def test_recurrence_missing_inclusion():
    out = gp3.compute_gazepoint_sequence_recurrence(
        sequence=["A", None, None, "A"],
        include_missing=True,
        min_line=1,
    )
    assert out.iloc[0]["sequence_length"] == 4


def test_saccade_start_end_grouped_branch_and_short_group():
    data = pd.DataFrame(
        {
            "g": ["a", "a", "a", "b"],
            "x": [0.0, 1.0, 2.0, 0.0],
            "y": [0.0, 0.0, 0.0, 1.0],
            "start": [0.0, 2.0, 4.0, 0.0],
            "end": [1.0, 3.0, 5.0, 0.5],
        }
    )
    out = gp3.compute_gazepoint_saccade_metrics(
        data,
        x_col="x",
        y_col="y",
        group_cols=["g"],
        start_time_col="start",
        end_time_col="end",
        distance_scale=1,
    )
    assert len(out) == 2
    assert set(out["g"]) == {"a"}
    assert set(out["time_delta_kind"]) == {"next_start_minus_current_end"}


def test_saccade_r_validation_paths():
    data = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    with pytest.raises(ValueError):
        gp3.compute_gazepoint_saccade_metrics(data, distance_scale=2)
    with pytest.raises(ValueError):
        gp3.compute_gazepoint_saccade_metrics(data, x_col="x", y_col="y", distance_scale=-1)
    with pytest.raises(ValueError):
        gp3.compute_gazepoint_saccade_metrics(data, x_col="x", y_col="y", start_time_col="missing")


def test_construct_combined_pupil_all_policies_and_reconstruction():
    base = pd.DataFrame({"l": [1.0, np.nan], "r": [3.0, 4.0]})
    left = gp3.construct_gazepoint_combined_pupil(base, "l", "r", policy="left_only")
    assert left["pupil_binocular_status"].iloc[0] == "left_observed"

    right = gp3.construct_gazepoint_combined_pupil(base, "l", "r", policy="right_only")
    assert right["pupil_binocular_status"].iloc[0] == "right_observed"

    complete = gp3.construct_gazepoint_combined_pupil(
        base, "l", "r", policy="complete_case", valid_min=0.5, valid_max=5
    )
    assert pd.isna(complete["pupil_combined"].iloc[1])

    reconstructed = pd.DataFrame(
        {
            "l": [1.0, np.nan],
            "r": [3.0, 4.0],
            "gp3_binocular_left_final": [1.0, 2.0],
            "gp3_binocular_right_final": [3.0, 4.0],
            "gp3_binocular_left_reconstructed": [False, True],
            "gp3_binocular_right_reconstructed": [False, False],
        }
    )
    out = gp3.construct_gazepoint_combined_pupil(
        reconstructed,
        "l",
        "r",
        policy="reconstructed_mean",
    )
    assert out["pupil_binocular_status"].iloc[1] == "bilateral_with_reconstruction"


def test_construct_combined_pupil_validation_paths():
    base = pd.DataFrame({"l": [1.0], "r": [2.0]})
    with pytest.raises(ValueError):
        gp3.construct_gazepoint_combined_pupil(base, "l", "r", policy="bad")
    with pytest.raises(ValueError):
        gp3.construct_gazepoint_combined_pupil(
            base.assign(pupil_combined=1.5),
            "l",
            "r",
        )
    with pytest.raises(ValueError):
        gp3.construct_gazepoint_combined_pupil(
            base,
            "l",
            "r",
            policy="reconstructed_mean",
        )


def test_combine_eyes_r_method_branches():
    data = pd.DataFrame({"l": [1.0, np.nan], "r": [2.0, 3.0]})
    assert (
        gp3.combine_gazepoint_eyes(data, left_col="l", right_col="r", method="left")[
            "pupil_combined"
        ].iloc[0]
        == 1
    )
    assert (
        gp3.combine_gazepoint_eyes(data, left_col="l", right_col="r", method="right")[
            "pupil_combined"
        ].iloc[0]
        == 2
    )
    assert (
        gp3.combine_gazepoint_eyes(data, left_col="l", right_col="r", method="prefer_right")[
            "pupil_combined"
        ].iloc[0]
        == 2
    )
    best = gp3.combine_gazepoint_eyes(data, left_col="l", right_col="r", method="best")
    assert best["pupil_combined"].notna().all()
    with pytest.raises(ValueError):
        gp3.combine_gazepoint_eyes(data, left_col="l", right_col="r", method="bad")


def test_residual_plot_qq_and_error_paths():
    import matplotlib.pyplot as plt

    data = pd.DataFrame({"fitted_value": [1.0, 2.0], "residuals": [0.1, -0.1]})
    fig = gp3.plot_gazepoint_model_residuals(data=data, type="qq", title="QQ")
    assert fig.axes[0].get_title() == "QQ"
    plt.close(fig)

    with pytest.raises(ValueError):
        gp3.plot_gazepoint_model_residuals(type="bad")
    with pytest.raises(ValueError):
        gp3.plot_gazepoint_model_residuals(data=pd.DataFrame({"x": [1]}))
    with pytest.raises(ValueError):
        gp3.plot_gazepoint_model_residuals()
    with pytest.raises(ValueError):
        gp3.plot_gazepoint_model_residuals(
            data=pd.DataFrame({"fitted": [np.nan], "residual": [np.nan]})
        )
