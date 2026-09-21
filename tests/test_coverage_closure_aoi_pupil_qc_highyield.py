from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gp3tools.aoi as aoi
import gp3tools.pupil as pupil
import gp3tools.qc as qc


# ============================================================================
# QC — EXCLUSION RECOMMENDATIONS
# ============================================================================


def test_qc_exclusion_recommendations_convenience_mode():
    data = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S2", "S2"],
            "TRACKLOSS": [0, 1, 0, 0],
            "FPOGX": [0.1, np.nan, 0.2, 0.3],
            "FPOGY": [0.1, 0.2, 0.2, 0.3],
            "pupil": [3.0, 3.1, np.nan, 3.3],
            "artifact": [False, True, False, False],
        }
    )

    out = qc.recommend_gazepoint_exclusions(
        data,
        artifact_col="artifact",
        min_trial_samples=1,
        min_participant_trials=1,
        min_participant_valid_trials=1,
    )

    assert "trial_recommendations" in out
    assert "participant_recommendations" in out
    assert len(out["trial_recommendations"]) == 2


def _explicit_exclusion_data():
    return pd.DataFrame(
        {
            "participant": ["S1", "S1", "S1", "S2", "S2", "S2"],
            "trial": ["T1", "T1", "T2", "T1", "T1", "T2"],
            "condition": ["A", "A", "B", "A", "A", pd.NA],
            "valid_bool": [True, False, True, True, True, True],
            "valid_num": [1, 0, 1, 1, 1, 1],
            "valid_text": ["yes", "no", "valid", "good", "bad", "true"],
            "x": [0.1, np.nan, 0.2, 0.2, 0.3, 0.4],
            "y": [0.1, 0.2, 0.2, np.nan, 0.3, 0.4],
            "pupil": [3.0, np.nan, 3.2, 3.0, 3.1, 3.2],
            "artifact": [False, True, False, False, False, False],
        }
    )


def test_qc_exclusion_recommendations_explicit_contracts():
    data = _explicit_exclusion_data()

    out = qc.recommend_gazepoint_exclusions(
        data,
        participant_col="participant",
        trial_col="trial",
        condition_col="condition",
        validity_col="valid_text",
        x_col="x",
        y_col="y",
        pupil_col="pupil",
        artifact_col="artifact",
        min_trial_samples=2,
        max_trial_missing_prop=0.25,
        max_trial_artifact_prop=0.25,
        min_participant_trials=2,
        min_participant_valid_trials=1,
        max_participant_missing_prop=0.25,
        max_participant_artifact_prop=0.25,
        require_both_gaze_coordinates=True,
    )

    assert out["overview"].loc[0, "recommendation_status"] == "complete"
    assert set(out["exclusion_table"]["exclusion_level"]) == {"participant", "trial"}

    numeric = qc.recommend_gazepoint_exclusions(
        data,
        participant_col="participant",
        validity_col="valid_num",
        x_col="x",
        require_both_gaze_coordinates=False,
        min_trial_samples=1,
        min_participant_trials=1,
        min_participant_valid_trials=1,
    )
    assert len(numeric["participant_recommendations"]) == 2

    boolean = qc.recommend_gazepoint_exclusions(
        data,
        participant_col="participant",
        validity_col="valid_bool",
        y_col="y",
        require_both_gaze_coordinates=False,
        min_trial_samples=1,
        min_participant_trials=1,
        min_participant_valid_trials=1,
    )
    assert len(boolean["participant_recommendations"]) == 2


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"participant_col": ""}, "participant_col"),
        ({"participant_col": "missing"}, "participant_col"),
        ({"participant_col": "participant"}, "quality indicator"),
        (
            {
                "participant_col": "participant",
                "x_col": "x",
                "require_both_gaze_coordinates": True,
            },
            "supply both",
        ),
        (
            {
                "participant_col": "participant",
                "validity_col": "valid_bool",
                "min_trial_samples": 0,
            },
            "positive integer",
        ),
        (
            {
                "participant_col": "participant",
                "validity_col": "valid_bool",
                "max_trial_missing_prop": 2,
            },
            "between 0 and 1",
        ),
        (
            {
                "participant_col": "participant",
                "validity_col": "valid_bool",
                "require_both_gaze_coordinates": "yes",
            },
            "TRUE or FALSE",
        ),
        (
            {
                "participant_col": "participant",
                "validity_col": "valid_bool",
                "name": "",
            },
            "name",
        ),
    ],
)
def test_qc_exclusion_validation_matrix(kwargs, match):
    with pytest.raises(ValueError, match=match):
        qc.recommend_gazepoint_exclusions(
            _explicit_exclusion_data(),
            **kwargs,
        )


def test_qc_exclusion_top_level_validation():
    with pytest.raises(TypeError, match="Unexpected keyword"):
        qc.recommend_gazepoint_exclusions(
            _explicit_exclusion_data(),
            impossible=True,
        )
    with pytest.raises(ValueError, match="data frame"):
        qc.recommend_gazepoint_exclusions([])
    with pytest.raises(ValueError, match="at least one row"):
        qc.recommend_gazepoint_exclusions(pd.DataFrame())


# ============================================================================
# PUPIL SUMMARY
# ============================================================================


def test_pupil_summary_legacy_grouped_and_ungrouped():
    frame = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S2"],
            "pupil": [1.0, np.nan, 3.0],
        }
    )

    overall = pupil.summarise_gazepoint_pupil(
        frame,
        pupil_col="pupil",
    )
    assert overall.loc[0, "n_samples"] == 3

    grouped = pupil.summarise_gazepoint_pupil(
        frame,
        pupil_col="pupil",
        group_cols=["subject"],
    )
    assert len(grouped) == 2

    with pytest.raises(TypeError, match="data or master"):
        pupil.summarise_gazepoint_pupil()


def _pupil_master():
    return pd.DataFrame(
        {
            "subject": ["S1"] * 5 + ["S2"] * 5,
            "media_id": ["M1"] * 10,
            "time_ms": list(range(5)) * 2,
            "pupil": [1.0, 2.0, 3.0, 100.0, np.nan, 2.0, 2.0, 2.0, 2.0, 2.0],
            "missing_bool": [False, False, False, False, True] * 2,
            "missing_num": [0, 0, 0, 0, 1] * 2,
            "missing_text": ["false", "0", "f", "false", "true"] * 2,
        }
    )


def test_pupil_summary_r_contract_success_and_missing_types():
    master = _pupil_master()

    for missing_col in ["missing_bool", "missing_num", "missing_text"]:
        out = pupil.summarise_gazepoint_pupil(
            master=master,
            pupil_col="pupil",
            time_col="time_ms",
            missing_pupil_col=missing_col,
            group_cols=["subject", "media_id"],
            min_pupil=0,
            max_pupil=10,
            outlier_k=1.5,
        )
        assert len(out) == 2
        assert "n_iqr_outliers" in out

    no_groups = pupil.summarise_gazepoint_pupil(
        master=master,
        pupil_col="pupil",
        time_col="time_ms",
        group_cols=[],
    )
    assert len(no_groups) == 1


def test_pupil_summary_r_validation_matrix():
    master = _pupil_master()

    with pytest.raises(TypeError, match="either data or master"):
        pupil.summarise_gazepoint_pupil(master=master, data=master)

    with pytest.raises(TypeError, match="DataFrame"):
        pupil.summarise_gazepoint_pupil(master=[])

    for bad in [1, [1, "subject"]]:
        with pytest.raises(ValueError, match="group_cols"):
            pupil.summarise_gazepoint_pupil(
                master=master,
                group_cols=bad,
            )

    with pytest.raises(ValueError, match="group_cols can only"):
        pupil.summarise_gazepoint_pupil(
            master=master,
            group_cols=["bad"],
        )

    with pytest.raises(ValueError, match="pupil_col"):
        pupil.summarise_gazepoint_pupil(
            master=master,
            pupil_col="",
        )

    with pytest.raises(ValueError, match="single numeric"):
        pupil.summarise_gazepoint_pupil(
            master=master,
            min_pupil=True,
        )

    with pytest.raises(ValueError, match="greater than"):
        pupil.summarise_gazepoint_pupil(
            master=master,
            min_pupil=10,
            max_pupil=1,
        )

    with pytest.raises(ValueError, match="No subject"):
        pupil.summarise_gazepoint_pupil(
            master=master.drop(columns=["subject"]),
        )

    with pytest.raises(ValueError, match="No media"):
        pupil.summarise_gazepoint_pupil(
            master=master.drop(columns=["media_id"]),
        )

    with pytest.raises(ValueError, match="No pupil"):
        pupil.summarise_gazepoint_pupil(
            master=master.drop(columns=["pupil"]),
        )

    with pytest.raises(ValueError, match="No time"):
        pupil.summarise_gazepoint_pupil(
            master=master.drop(columns=["time_ms"]),
        )

    with pytest.raises(ValueError, match="missing_pupil_col"):
        pupil.summarise_gazepoint_pupil(
            master=master,
            missing_pupil_col="missing",
        )


# ============================================================================
# PUPIL WINDOWS
# ============================================================================


def _window_data():
    return pd.DataFrame(
        {
            "subject": ["S1"] * 4,
            "media_id": ["M1"] * 4,
            "time_ms": [0.0, 100.0, 200.0, 300.0],
            "pupil": [1.0, np.nan, 3.0, 4.0],
        }
    )


def test_pupil_windows_legacy_and_r_paths():
    data = _window_data()

    legacy = pupil.summarise_gazepoint_pupil_windows(
        data,
        pupil_col="pupil",
        time_col="time_ms",
        windows={"early": (0, 200), "late": (200, 400)},
        group_cols=["subject"],
    )
    assert set(legacy["window"]) == {"early", "late"}

    numeric = pupil.summarise_gazepoint_pupil_windows(
        data,
        pupil_col="pupil",
        time_col="time_ms",
        windows=[0, 200, 400],
        group_cols=["subject", "media_id"],
        include_window_end=True,
        min_valid_samples=2,
    )
    assert len(numeric) == 2
    assert set(numeric["pupil_window_status"]).issubset(
        {"valid", "insufficient_valid_pupil", "no_valid_pupil"}
    )

    no_piece = pupil.summarise_gazepoint_pupil_windows(
        data,
        pupil_col="pupil",
        time_col="time_ms",
        windows=[1000, 2000],
        group_cols=["subject", "media_id"],
    )
    assert no_piece.empty


def test_pupil_windows_validation_matrix():
    data = _window_data()

    with pytest.raises(ValueError, match="min_valid_samples"):
        pupil.summarise_gazepoint_pupil_windows(
            data,
            windows=[0, 1],
            min_valid_samples=0,
        )

    with pytest.raises(KeyError, match="No pupil"):
        pupil.summarise_gazepoint_pupil_windows(
            data.drop(columns=["pupil"]),
            windows=[0, 1],
        )

    with pytest.raises(KeyError, match="No time"):
        pupil.summarise_gazepoint_pupil_windows(
            data.drop(columns=["time_ms"]),
            windows=[0, 1],
        )

    with pytest.raises(KeyError, match="Missing grouping"):
        pupil.summarise_gazepoint_pupil_windows(
            data,
            windows=[0, 1],
            group_cols=["trial"],
        )

    with pytest.raises(ValueError, match="strictly increasing"):
        pupil.summarise_gazepoint_pupil_windows(
            data,
            windows=[0, 0, 1],
            group_cols=["subject"],
        )

    with pytest.raises(KeyError, match="start and end"):
        pupil.summarise_gazepoint_pupil_windows(
            data,
            windows=pd.DataFrame({"label": ["x"]}),
            group_cols=["subject"],
        )

    with pytest.raises(ValueError, match="numeric and non-missing"):
        pupil.summarise_gazepoint_pupil_windows(
            data,
            windows=pd.DataFrame(
                {
                    "start": ["bad"],
                    "end": [1],
                }
            ),
            group_cols=["subject"],
        )

    with pytest.raises(ValueError, match="Invalid window definitions"):
        pupil.summarise_gazepoint_pupil_windows(
            data,
            windows=pd.DataFrame(
                {
                    "start": [2],
                    "end": [1],
                    "label": ["bad"],
                }
            ),
            group_cols=["subject"],
        )


# ============================================================================
# AOI POLYGONS / HELPERS
# ============================================================================


def _square_vertices(name, x0, y0, x1, y1):
    return pd.DataFrame(
        {
            "aoi_name": [name] * 4,
            "vertex_x": [x0, x1, x1, x0],
            "vertex_y": [y0, y0, y1, y1],
            "order": [1, 2, 3, 4],
        }
    )


def test_aoi_polygon_legacy_and_r_contracts():
    data = pd.DataFrame(
        {
            "x": [0.5, 2.0, np.nan],
            "y": [0.5, 2.0, 0.5],
        }
    )

    legacy = aoi.add_gazepoint_polygon_aoi(
        data,
        polygons={
            "inside": [(0, 0), (1, 0), (1, 1), (0, 1)],
        },
        x_col="x",
        y_col="y",
    )
    assert legacy["aoi_current"].iloc[0] == "inside"
    assert legacy["aoi_current"].iloc[1] == "outside"
    assert pd.isna(legacy["aoi_current"].iloc[2])

    with pytest.raises(ValueError, match="Polygon table"):
        aoi.add_gazepoint_polygon_aoi(
            data,
            polygons=pd.DataFrame({"bad": [1]}),
            x_col="x",
            y_col="y",
        )

    frame = pd.DataFrame(
        {
            "FPOGX": [0.5, 1.5, np.nan],
            "FPOGY": [0.5, 1.5, 0.5],
        }
    )
    vertices = pd.concat(
        [
            _square_vertices("A", 0, 0, 1, 1),
            _square_vertices("B", 0.25, 0.25, 2, 2),
        ],
        ignore_index=True,
    )

    both = aoi.add_gazepoint_polygon_aoi(
        master_df=frame,
        vertices=vertices,
        vertex_order_col="order",
        output="both",
        overlap="last",
        boundary="inside",
    )
    assert both.loc[0, "aoi_current"] == "B"
    assert "aoi_overlap_count" in both

    with pytest.raises(ValueError, match="overlapping"):
        aoi.add_gazepoint_polygon_aoi(
            master_df=frame,
            vertices=vertices,
            vertex_order_col="order",
            output="label",
            overlap="error",
        )


def test_aoi_polygon_validation_and_helper_contracts():
    frame = pd.DataFrame({"FPOGX": [0.5], "FPOGY": [0.5]})
    vertices = _square_vertices("A", 0, 0, 1, 1)

    with pytest.raises(TypeError, match="either data or master_df"):
        aoi.add_gazepoint_polygon_aoi(
            data=frame,
            master_df=frame,
            vertices=vertices,
        )

    with pytest.raises(TypeError, match="either polygons or vertices"):
        aoi.add_gazepoint_polygon_aoi(
            master_df=frame,
            polygons=vertices,
            vertices=vertices,
        )

    for kwargs, match in [
        ({"output": "bad"}, "output"),
        ({"overlap": "bad"}, "overlap"),
        ({"boundary": "bad"}, "boundary"),
    ]:
        with pytest.raises(ValueError, match=match):
            aoi.add_gazepoint_polygon_aoi(
                master_df=frame,
                vertices=vertices,
                **kwargs,
            )

    with pytest.raises(ValueError, match="missing required"):
        aoi.add_gazepoint_polygon_aoi(
            master_df=frame.drop(columns=["FPOGY"]),
            vertices=vertices,
        )

    assert pd.isna(aoi._gp3_dynamic_match_time(np.nan, [1, 2], "nearest"))
    assert pd.isna(aoi._gp3_dynamic_match_time(1, [], "nearest"))
    assert aoi._gp3_dynamic_match_time(1.5, [1, 2], "nearest") == 1.0
    assert aoi._gp3_dynamic_match_time(1.5, [1, 2], "previous") == 1.0
    assert aoi._gp3_dynamic_match_time(1.5, [1, 2], "next") == 2.0

    grouped = pd.DataFrame({"g": ["a", pd.NA]})
    assert aoi._gp3_dynamic_group_keys(grouped, []).eq("__all__").all()
    assert len(aoi._gp3_dynamic_group_keys(grouped, ["g"])) == 2

    with pytest.raises(ValueError, match="overlapping"):
        aoi._gp3_dynamic_apply_membership(
            frame,
            np.array([[True, True]]),
            ["A", "B"],
            output="label",
            prefix="aoi_",
            label_col="label",
            outside_label="outside",
            overlap="error",
            include_overlap_count=True,
            valid_xy=np.array([True]),
        )

    applied = aoi._gp3_dynamic_apply_membership(
        pd.DataFrame({"x": [1, 2]}),
        np.array([[True, False], [False, True]]),
        ["A", "B"],
        output="both",
        prefix="aoi_",
        label_col="label",
        outside_label="outside",
        overlap="first",
        include_overlap_count=True,
        valid_xy=np.array([True, True]),
    )
    assert applied["label"].tolist() == ["A", "B"]

    with pytest.raises(ValueError, match="non-empty column"):
        aoi._gp3_margin_resolve_column(
            frame,
            "",
            [],
            required=True,
            arg="x",
        )

    with pytest.raises(KeyError, match="Missing required"):
        aoi._gp3_margin_resolve_column(
            frame,
            "missing",
            [],
            required=True,
            arg="x",
        )

    assert (
        aoi._gp3_margin_resolve_column(
            frame,
            "missing",
            [],
            required=False,
            arg="x",
        )
        is None
    )

    with pytest.raises(KeyError, match="Could not resolve"):
        aoi._gp3_margin_resolve_column(
            frame,
            None,
            ["missing"],
            required=True,
            arg="x",
        )

    assert pd.isna(aoi._gp3_margin_safe_stat([np.nan], np.mean))


# ============================================================================
# AOI TRANSITIONS
# ============================================================================


def test_aoi_transitions_legacy_empty_and_self_paths():
    data = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S1"],
            "time": [0, 1, 2],
            "aoi": ["A", "A", "B"],
        }
    )

    collapsed = aoi.summarise_gazepoint_aoi_transitions(
        data,
        aoi_col="aoi",
        group_cols=["subject"],
        time_col="time",
        include_self=False,
    )
    assert collapsed["n_transitions"].sum() == 1

    with_self = aoi.summarise_gazepoint_aoi_transitions(
        data,
        aoi_col="aoi",
        group_cols=["subject"],
        time_col="time",
        include_self=True,
    )
    assert with_self["n_transitions"].sum() == 2

    empty = aoi.summarise_gazepoint_aoi_transitions(
        data.iloc[:1],
        aoi_col="aoi",
        group_cols=["subject"],
        time_col="time",
    )
    assert empty.empty


def _sequence_frame():
    return pd.DataFrame(
        {
            "aoi_state": ["target", "background", "distractor", "other"],
            "transition_from": ["target", "background", "distractor", "other"],
            "transition_to": ["background", "distractor", "target", pd.NA],
            "entry_duration_ms": [10.0, 20.0, 30.0, 40.0],
            "dwell_before_transition_ms": [10.0, 20.0, np.nan, np.nan],
            "is_non_aoi": [False, True, False, False],
            "is_terminal_state": [False, False, False, True],
        }
    )


def test_aoi_transitions_r_status_matrix():
    seq = _sequence_frame()

    complete = aoi.summarise_gazepoint_aoi_transitions(
        seq,
        group_cols=[],
        include_non_aoi=True,
        target_aoi_values=["target"],
        distractor_aoi_values=["distractor"],
        non_aoi_values=["background"],
    )
    assert complete.loc[0, "transition_feature_status"] == "ok"
    assert complete.loc[0, "target_to_background"] == 1
    assert complete.loc[0, "background_to_distractor"] == 1
    assert complete.loc[0, "distractor_to_target"] == 1

    no_defs = aoi.summarise_gazepoint_aoi_transitions(
        seq,
        group_cols=[],
        include_non_aoi=True,
        target_aoi_values=None,
        distractor_aoi_values=None,
        non_aoi_values=["background"],
    )
    assert no_defs.loc[0, "transition_feature_status"] == "no_target_or_distractor_defined"

    terminal = seq.iloc[[-1]].copy()
    none = aoi.summarise_gazepoint_aoi_transitions(
        terminal,
        group_cols=[],
        include_non_aoi=True,
        target_aoi_values=["target"],
        distractor_aoi_values=["distractor"],
    )
    assert none.loc[0, "transition_feature_status"] == "no_transitions"

    with pytest.raises(ValueError, match="Missing required columns"):
        aoi.summarise_gazepoint_aoi_transitions(
            pd.DataFrame({"aoi_state": ["A"]}),
            group_cols=[],
            include_non_aoi=True,
        )

    with pytest.raises(ValueError, match="No AOI sequence"):
        aoi.summarise_gazepoint_aoi_transitions(
            _sequence_frame().iloc[0:0],
            group_cols=[],
            include_non_aoi=True,
        )
