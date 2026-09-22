from __future__ import annotations

import importlib

import numpy as np
import pandas as pd
import pytest

aoi = importlib.import_module("gp3tools.aoi")


def test_final_aoi_polygon_dataframe_legacy_route():
    result = aoi.add_gazepoint_polygon_aoi(
        pd.DataFrame({"x": [0.25], "y": [0.25]}),
        pd.DataFrame(
            {
                "aoi": ["A"],
                "polygon": [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]],
            }
        ),
        x_col="x",
        y_col="y",
    )
    assert result.loc[0, "aoi_current"] == "A"


def test_final_aoi_dynamic_missing_definition_column_and_rectangle_auto():
    gaze = pd.DataFrame({"TIME": [0.0], "FPOGX": [0.25], "FPOGY": [0.25]})

    with pytest.raises(KeyError, match="Missing required AOI column"):
        aoi.add_gazepoint_dynamic_aoi(
            gaze,
            pd.DataFrame(
                {
                    "aoi_time": [0.0],
                    "left": [0.0],
                    "right": [1.0],
                    "top": [0.0],
                    "bottom": [1.0],
                }
            ),
            shape="rectangle",
        )

    result = aoi.add_gazepoint_dynamic_aoi(
        gaze,
        pd.DataFrame(
            {
                "aoi_time": [0.0],
                "aoi_name": ["A"],
                "left": [0.0],
                "right": [1.0],
                "top": [0.0],
                "bottom": [1.0],
            }
        ),
        shape="auto",
    )
    assert result.loc[0, "aoi_current"] == "A"


def test_final_aoi_geometry_and_overlap_public_contracts():
    legacy = pd.DataFrame(
        {
            "aoi": ["A"],
            "xmin": [0.0],
            "xmax": [1.0],
            "ymin": [0.0],
            "ymax": [1.0],
        }
    )
    geometry = pd.DataFrame(
        {
            "aoi": ["A"],
            "x_min": [0.0],
            "x_max": [1.0],
            "y_min": [0.0],
            "y_max": [1.0],
        }
    )

    legacy_result = aoi.audit_gazepoint_aoi_geometry(legacy)
    assert legacy_result["valid"]

    r_result = aoi.audit_gazepoint_aoi_geometry(
        data=geometry,
        aoi_col="aoi",
        x_min_col="x_min",
        x_max_col="x_max",
        y_min_col="y_min",
        y_max_col="y_max",
    )
    assert r_result["_gp3_class"] == "gp3_aoi_geometry_audit"

    legacy_callable = aoi.audit_gazepoint_aoi_geometry.__wrapped__._gp3_r4_legacy
    with pytest.raises(TypeError, match="either aoi_geometry or data"):
        legacy_callable(legacy, data=legacy)

    with pytest.raises(TypeError, match="either aoi_geometry or data"):
        aoi.audit_gazepoint_aoi_overlap(legacy, data=legacy)


def test_final_aoi_coding_resolve_rejects_empty_explicit_column():
    with pytest.raises(ValueError, match="non-missing character scalar"):
        aoi._gp3_aoi_coding_r_resolve("", pd.Index(["x"]), "column")


def test_final_aoi_coding_matrix_data_alias_and_scalar_sample_id():
    result = aoi.audit_gazepoint_aoi_coding_matrix(
        data=pd.DataFrame(
            {
                "AOI": ["A"],
                "FPOGX": [0.5],
                "FPOGY": [0.5],
                "sample_id": [1],
            }
        ),
        aoi_geometry=pd.DataFrame(
            {
                "aoi": ["A"],
                "x_min": [0.0],
                "x_max": [1.0],
                "y_min": [0.0],
                "y_max": [1.0],
            }
        ),
        sample_id_cols="sample_id",
    )
    assert result is not None


def test_final_aoi_summary_data_alias_with_fixations():
    result = aoi.summarise_gazepoint_aoi(
        data=pd.DataFrame(
            {
                "USER_FILE": ["user_1.csv", "user_1.csv"],
                "MEDIA_ID": ["M1", "M1"],
                "MEDIA_NAME": ["stimulus", "stimulus"],
                "AOI": ["A", "A"],
                "TIME": [0.0, 0.1],
            }
        ),
        fixation_data=pd.DataFrame(
            {
                "USER_FILE": ["user_1.csv"],
                "MEDIA_ID": ["M1"],
                "MEDIA_NAME": ["stimulus"],
                "AOI": ["A"],
                "FPOGD": [0.1],
                "FPOGS": [0.0],
            }
        ),
    )
    assert not result.empty


def test_final_aoi_entropy_sequence_frame_and_validation_paths():
    data = pd.DataFrame(
        {
            "subject": ["S1", "S1"],
            "time": [0.0, 1.0],
            "aoi": ["A", pd.NA],
        }
    )
    result = aoi.compute_gazepoint_aoi_entropy(
        data=data,
        aoi_col="aoi",
        group_cols="subject",
        time_col="time",
        include_missing=True,
    )
    assert result.loc[0, "n_observations"] == 2

    with pytest.raises(ValueError, match="time_col"):
        aoi.compute_gazepoint_aoi_entropy(
            data=data,
            aoi_col="aoi",
            group_cols="subject",
            time_col="",
        )

    metrics = aoi.compute_gazepoint_aoi_sequence_metrics(
        data=data,
        aoi_col="aoi",
        group_cols="subject",
        time_col="time",
        include_missing=True,
    )
    assert not metrics.empty

    complexity = aoi.compute_gazepoint_sequence_complexity(
        data=data,
        aoi_col="aoi",
        group_cols="subject",
        time_col="time",
        include_missing=True,
    )
    assert not complexity.empty


def test_final_aoi_transition_summary_missing_required_columns(monkeypatch):
    monkeypatch.setattr(
        aoi,
        "_gp3_aoi_r_sequences",
        lambda *args, **kwargs: pd.DataFrame({"aoi_state": ["A"]}),
    )
    with pytest.raises(ValueError, match="Missing required columns"):
        aoi.summarise_gazepoint_aoi_transitions(
            pd.DataFrame({"aoi": ["A"]}),
            aoi_col="aoi",
            include_non_aoi=True,
        )


def test_final_aoi_transition_summary_classifies_missing_origin():
    prepared = pd.DataFrame(
        {
            "aoi_state": ["A"],
            "transition_from": [pd.NA],
            "transition_to": ["A"],
            "entry_duration_ms": [10.0],
            "dwell_before_transition_ms": [10.0],
            "is_non_aoi": [False],
            "is_terminal_state": [False],
        }
    )
    result = aoi.summarise_gazepoint_aoi_transitions(
        prepared,
        group_cols=[],
        include_non_aoi=True,
        target_aoi_values=["A"],
    )
    assert not result.empty


def _prepared_sequence_rows():
    return pd.DataFrame(
        {
            "aoi_state": ["A", "B"],
            "transition_from": ["A", "B"],
            "transition_to": ["B", pd.NA],
            "entry_start_time": [0.0, 1.0],
            "is_terminal_state": [False, True],
        }
    )


def test_final_aoi_transition_matrix_prepared_rows_and_missing_by_group():
    result = aoi.compute_gazepoint_aoi_transition_matrix(
        data=_prepared_sequence_rows(),
        group_cols=[],
        include_self_transitions=True,
    )
    assert result is not None

    with pytest.raises(ValueError, match="Missing required columns"):
        aoi.compute_gazepoint_aoi_transition_matrix(
            data=_prepared_sequence_rows(),
            group_cols=[],
            by_cols=["subject"],
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"complete_states": "yes"}, "complete_states"),
        ({"drop_self_transitions": "no"}, "drop_self_transitions"),
        ({"normalise": "bad"}, "normalise"),
    ],
)
def test_final_aoi_time_varying_transition_validation(kwargs, message):
    frame = pd.DataFrame(
        {
            "from": ["A", "B"],
            "to": ["B", "A"],
            "time": [0.0, 100.0],
        }
    )
    with pytest.raises(ValueError, match=message):
        aoi.compute_gazepoint_time_varying_transition_matrix(
            frame,
            from_col="from",
            to_col="to",
            time_col="time",
            window_size_ms=100,
            **kwargs,
        )


def test_final_aoi_time_varying_detection_failure_and_complete_drop_self():
    with pytest.raises(ValueError, match="could not be detected"):
        aoi.compute_gazepoint_time_varying_transition_matrix(
            pd.DataFrame({"x": [1]}),
            window_size_ms=100,
        )

    result = aoi.compute_gazepoint_time_varying_transition_matrix(
        pd.DataFrame(
            {
                "from": ["A", "A", "B"],
                "to": ["A", "B", "A"],
                "window": ["w1", "w1", "w1"],
            }
        ),
        from_col="from",
        to_col="to",
        window_col="window",
        complete_states=True,
        drop_self_transitions=True,
        normalise="row",
    )
    table = result["matrix_long"]
    assert not table[".gp3_from"].eq(table[".gp3_to"]).any()


def test_final_aoi_scanpath_geometry_legacy_sort_and_r_validation():
    legacy = aoi.compute_gazepoint_scanpath_geometry(
        pd.DataFrame(
            {
                "g": ["S1", "S1"],
                "time": [1.0, 0.0],
                "x": [1.0, 0.0],
                "y": [0.0, 0.0],
            }
        ),
        x_col="x",
        y_col="y",
        time_col="time",
        group_cols="g",
    )
    assert legacy.loc[0, "path_length"] == pytest.approx(1.0)

    with pytest.raises(ValueError, match="time must be None or a non-empty string"):
        aoi.compute_gazepoint_scanpath_geometry(
            pd.DataFrame({"x": [0.0], "y": [0.0], "subject": ["S1"], "trial": [1]}),
            x="x",
            y="y",
            subject="subject",
            trial="trial",
            time="",
        )


def test_final_aoi_scanpath_geometry_no_finite_points_and_nonfinite_dispersion():
    empty_points = aoi.compute_gazepoint_scanpath_geometry(
        pd.DataFrame(
            {
                "x": [np.nan],
                "y": [np.nan],
                "subject": ["S1"],
                "trial": [1],
            }
        ),
        x="x",
        y="y",
        subject="subject",
        trial="trial",
    )
    assert pd.isna(empty_points.loc[0, "spatial_dispersion"])

    huge = aoi.compute_gazepoint_scanpath_geometry(
        pd.DataFrame(
            {
                "x": [1e308, -1e308],
                "y": [1e308, -1e308],
                "subject": ["S1", "S1"],
                "trial": [1, 1],
            }
        ),
        x="x",
        y="y",
        subject="subject",
        trial="trial",
    )
    assert pd.isna(huge.loc[0, "spatial_dispersion"])


def test_final_aoi_scanpath_distance_and_cluster_validation():
    with pytest.raises(ValueError, match="square"):
        aoi._gp3_scanpath_r_validate_distance_matrix(
            pd.DataFrame(np.zeros((2, 3)))
        )

    distance = pd.DataFrame(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]],
        index=["s1", "s2", "s3"],
        columns=["s1", "s2", "s3"],
    )
    with pytest.raises(ValueError, match="finite integer"):
        aoi._gp3_scanpath_r_cluster_matrix(
            distance,
            k="bad",
            method="hierarchical",
            linkage="average",
        )


def test_final_aoi_scanpath_similarity_dataframe_alias_and_missing_legacy_path_b():
    frame = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S2", "S2"],
            "time": [0, 1, 0, 1],
            "aoi": ["A", "B", "A", "C"],
        }
    )
    result = aoi.compute_gazepoint_scanpath_similarity(
        frame,
        aoi_col="aoi",
        group_cols=["subject"],
        time_col="time",
    )
    assert not result.empty

    with pytest.raises(TypeError, match="path_b is required"):
        aoi.compute_gazepoint_scanpath_similarity(["A", "B"])


def test_final_aoi_transition_network_skips_missing_pairs():
    result = aoi.compute_gazepoint_transition_network_metrics(
        data=pd.DataFrame(
            {
                "from": ["A", pd.NA],
                "to": ["B", "A"],
            }
        ),
        from_col="from",
        to_col="to",
    )
    assert result is not None


def test_final_aoi_anomaly_missing_column_and_zero_length_spread():
    with pytest.raises(ValueError, match="missing required column"):
        aoi.flag_gazepoint_sequence_anomalies(
            pd.DataFrame({"subject": ["S1"], "aoi": ["A"]}),
            aoi_col="aoi",
            group_cols=["subject"],
            time_col="time",
        )

    result = aoi.flag_gazepoint_sequence_anomalies(
        pd.DataFrame(
            {
                "subject": ["S1", "S2"],
                "aoi": ["A", "B"],
            }
        ),
        aoi_col="aoi",
        group_cols=["subject"],
        min_length=0,
    )
    assert result["length_z"].eq(0).all()


def _entry_frame(rows):
    columns = [
        "subject",
        "MEDIA_ID",
        "trial_global",
        "aoi_state",
        "entry_start_time",
        "entry_end_time",
        "entry_duration_ms",
        "n_samples",
        "time",
    ]
    return pd.DataFrame(rows, columns=columns)


def test_final_aoi_trial_features_empty_and_fallback_non_aoi():
    with pytest.raises(ValueError, match="No AOI entries"):
        aoi.summarise_gazepoint_aoi_trial_features(_entry_frame([]))

    result = aoi.summarise_gazepoint_aoi_trial_features(
        _entry_frame(
            [
                ["S1", "M1", 1, "outside", 0.0, 10.0, 10.0, 2, 0.0],
            ]
        )
    )
    assert result.loc[0, "n_non_aoi_entries"] == 1


def test_final_aoi_windows_missing_window_column():
    data = pd.DataFrame(
        {
            "subject": ["S1"],
            "time": [0.0],
            "aoi_current": ["A"],
        }
    )
    with pytest.raises(ValueError, match="Missing required window columns"):
        aoi.summarise_gazepoint_aoi_windows(
            data,
            windows=pd.DataFrame(
                {
                    "window_label": ["w1"],
                    "window_start_ms": [0.0],
                }
            ),
            condition_col=None,
        )


def _denominator_frame(denominators):
    n = len(denominators)
    return pd.DataFrame(
        {
            "window_label": ["w1"] * n,
            "window_start_ms": [0.0] * n,
            "window_end_ms": [100.0] * n,
            "n_valid_denominator_samples": denominators,
            "n_window_samples": [10.0] * n,
            "n_target_samples": [0.0] * n,
            "condition": ["A", "B"][:n],
        }
    )


def test_final_aoi_denominator_zero_review_and_ok_condition_statuses():
    zero = aoi.audit_gazepoint_aoi_window_denominators(
        _denominator_frame([0.0, 0.0]),
        window_col="window_label",
    )
    assert zero["overview"].loc[0, "denominator_audit_status"] == "zero_denominators"
    assert zero["denominator_imbalance"].loc[0, "denominator_imbalance_status"] == "ok"

    low = aoi.audit_gazepoint_aoi_window_denominators(
        _denominator_frame([2.0, 2.0]),
        window_col="window_label",
    )
    assert low["overview"].loc[0, "denominator_audit_status"] == "review_denominators"


def test_final_aoi_cluster_empty_legacy_and_non_dataframe_r_source(monkeypatch):
    monkeypatch.setattr(
        aoi,
        "prepare_gazepoint_aoi_sequences",
        lambda *args, **kwargs: pd.DataFrame({"sequence": pd.Series(dtype=object)}),
    )
    empty = aoi.cluster_gazepoint_scanpaths(
        pd.DataFrame({"aoi": []}),
        aoi_col="aoi",
        group_cols=[],
    )
    assert empty.empty
    assert "cluster" in empty

    result = aoi.cluster_gazepoint_scanpaths(
        x=[[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]],
        k=2,
        method="hierarchical",
    )
    assert result["_gp3_class"] == "gp3_scanpath_clusters"


def test_final_aoi_select_clusters_swallows_failed_candidate(monkeypatch):
    monkeypatch.setattr(
        aoi,
        "cluster_gazepoint_scanpaths",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("bad candidate")),
    )
    result = aoi.select_gazepoint_scanpath_clusters(pd.DataFrame({"aoi": ["A"]}), max_clusters=2)
    assert result.empty


def test_final_aoi_representative_singleton_cluster():
    distance = pd.DataFrame(
        [[0.0, 2.0], [2.0, 0.0]],
        index=["s1", "s2"],
        columns=["s1", "s2"],
    )
    result = aoi.extract_gazepoint_representative_scanpaths(
        {
            "distance": distance,
            "assignments": pd.DataFrame(
                {
                    "sequence_id": ["s1", "s2"],
                    "cluster": [1, 2],
                }
            ),
            "medoids": ["s1"],
        }
    )
    assert result["mean_within_cluster_distance"].eq(0).all()


def test_final_aoi_bootstrap_legacy_records_failed_resample(monkeypatch):
    calls = {"n": 0}

    def fake_cluster(data, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return pd.DataFrame({"cluster": [0, 0]})
        raise ValueError("resample failed")

    monkeypatch.setattr(aoi, "cluster_gazepoint_scanpaths", fake_cluster)
    result = aoi.bootstrap_gazepoint_scanpath_clusters(
        pd.DataFrame({"x": [1, 2]}),
        n_boot=1,
        random_state=1,
    )
    assert pd.isna(result.loc[0, "n_clusters"])
