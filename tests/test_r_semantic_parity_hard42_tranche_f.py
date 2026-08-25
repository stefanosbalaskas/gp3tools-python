from __future__ import annotations

import pandas as pd
import pytest

import gp3tools as gp3


def _sample_aoi_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": ["S1"] * 5,
            "MEDIA_ID": ["M1"] * 5,
            "trial_global": [1] * 5,
            "condition": ["A"] * 5,
            "time": [0.0, 100.0, 200.0, 300.0, 400.0],
            "aoi_current": ["A", "A", "background", "B", "B"],
        }
    )


def test_r_entries_episode_durations_and_neighbors():
    out = gp3.summarise_gazepoint_aoi_entries(
        _sample_aoi_data(),
        aoi_col="aoi_current",
        time_col="time",
        group_cols=["subject", "MEDIA_ID", "trial_global"],
        include_non_aoi=True,
    )

    assert out["aoi_state"].tolist() == ["A", "background", "B"]
    assert out["entry_id"].tolist() == [1, 2, 3]
    assert out["entry_order"].tolist() == [1, 2, 3]
    assert out["entry_start_time"].tolist() == [0.0, 200.0, 300.0]
    assert out["entry_end_time"].tolist() == [200.0, 300.0, 500.0]
    assert out["entry_duration_ms"].tolist() == [200.0, 100.0, 200.0]
    assert out["n_samples"].tolist() == [2, 1, 2]
    assert pd.isna(out.loc[0, "previous_aoi_state"])
    assert out.loc[0, "next_aoi_state"] == "background"
    assert out.loc[2, "previous_aoi_state"] == "background"
    assert bool(out.loc[1, "is_non_aoi"])


def test_r_entries_filter_happens_after_neighbor_fields():
    out = gp3.summarise_gazepoint_aoi_entries(
        _sample_aoi_data(),
        aoi_col="aoi_current",
        time_col="time",
        group_cols=["subject", "MEDIA_ID", "trial_global"],
        include_non_aoi=False,
    )

    assert out["aoi_state"].tolist() == ["A", "B"]
    assert out["entry_order"].tolist() == [1, 3]
    assert out.loc[0, "next_aoi_state"] == "background"
    assert out.loc[1, "previous_aoi_state"] == "background"


def test_r_sequences_remove_background_before_transition_construction():
    out = gp3.prepare_gazepoint_aoi_sequences(
        _sample_aoi_data(),
        aoi_col="aoi_current",
        time_col="time",
        group_cols=["subject", "MEDIA_ID", "trial_global"],
        include_non_aoi=False,
        include_terminal=True,
    )

    assert out["aoi_state"].tolist() == ["A", "B"]
    assert out["transition_to"].iloc[0] == "B"
    assert bool(out["is_terminal_state"].iloc[-1])
    assert pd.isna(out["transition_order"].iloc[-1])


def test_r_transition_feature_summary():
    out = gp3.summarise_gazepoint_aoi_transitions(
        _sample_aoi_data(),
        aoi_col="aoi_current",
        time_col="time",
        group_cols=["subject", "MEDIA_ID", "trial_global"],
        include_non_aoi=True,
        target_aoi_values=["B"],
        distractor_aoi_values=["A"],
    )

    row = out.iloc[0]
    assert row["n_states"] == 3
    assert row["total_transitions"] == 2
    assert row["distractor_to_background"] == 1
    assert row["background_to_target"] == 1
    assert row["target_to_distractor"] == 0
    assert row["total_pre_transition_dwell_ms"] == pytest.approx(300.0)
    assert row["transition_feature_status"] == "ok"


def test_r_trial_features_and_transition_join():
    out = gp3.summarise_gazepoint_aoi_trial_features(
        _sample_aoi_data(),
        aoi_col="aoi_current",
        time_col="time",
        group_cols=["subject", "MEDIA_ID", "trial_global"],
        include_non_aoi=True,
        target_aoi_values=["B"],
        distractor_aoi_values=["A"],
    )

    row = out.iloc[0]
    assert row["trial_start_time"] == 0.0
    assert row["trial_end_time"] == 500.0
    assert row["trial_duration_ms"] == 500.0
    assert row["n_entries"] == 3
    assert row["n_aoi_entries"] == 2
    assert row["target_entries"] == 1
    assert row["distractor_entries"] == 1
    assert row["target_dwell_ms"] == 200.0
    assert row["distractor_dwell_ms"] == 200.0
    assert row["target_ttff_ms"] == 300.0
    assert row["distractor_ttff_ms"] == 0.0
    assert row["background_to_target"] == 1
    assert row["aoi_trial_feature_status"] == "ok"


def test_r_aoi_windows_numeric_breakpoints_and_endpoint_policy():
    out = gp3.summarise_gazepoint_aoi_windows(
        _sample_aoi_data(),
        windows=[0.0, 200.0, 500.0],
        time_col="time",
        aoi_col="aoi_current",
        subject_col="subject",
        condition_col="condition",
        group_cols=["subject", "condition"],
        target_aoi_values=["B"],
        include_right_endpoint=False,
    )

    assert out["window_label"].tolist() == ["0_200ms", "200_500ms"]
    assert out["n_window_samples"].tolist() == [2, 3]
    assert out["n_target_samples"].tolist() == [0, 2]
    assert out["aoi_window_status"].tolist() == ["target_not_observed", "ok"]
    assert out.attrs["gp3_class"] == "gp3_aoi_window_summary"


def test_r_transition_matrix_global_and_by_group_forms():
    data = _sample_aoi_data()
    global_result = gp3.compute_gazepoint_aoi_transition_matrix(
        data=data,
        aoi_col="aoi_current",
        time_col="time",
        group_cols=["subject", "MEDIA_ID", "trial_global"],
        include_non_aoi=True,
        states=["A", "background", "B"],
    )

    assert global_result["count_matrix"].loc["A", "background"] == 1
    assert global_result["count_matrix"].loc["background", "B"] == 1
    assert global_result["probability_matrix"].loc["A", "background"] == 1.0
    assert global_result["count_matrices"] is None
    assert list(global_result["states"]) == ["A", "background", "B"]

    by_result = gp3.compute_gazepoint_aoi_transition_matrix(
        data=data,
        aoi_col="aoi_current",
        time_col="time",
        group_cols=["subject", "MEDIA_ID", "trial_global"],
        by_cols=["condition"],
        include_non_aoi=True,
    )
    assert by_result["count_matrix"] is None
    assert "condition=A" in by_result["count_matrices"]
    assert by_result["long_table"]["n"].sum() == 2


def test_r_time_varying_transition_matrix_complete_grid_and_row_probabilities():
    transitions = pd.DataFrame(
        {
            "from_aoi": ["A", "A", "B", "B"],
            "to_aoi": ["B", "C", "A", "C"],
            "time_ms": [0.0, 100.0, 600.0, 700.0],
            "condition": ["x", "x", "x", "x"],
        }
    )

    out = gp3.compute_gazepoint_time_varying_transition_matrix(
        transitions,
        from_col="from_aoi",
        to_col="to_aoi",
        time_col="time_ms",
        window_size_ms=500.0,
        by_cols=["condition"],
        states=["A", "B", "C"],
        complete_states=True,
        drop_self_transitions=False,
        normalise="row",
        name="tv",
    )

    assert out["overview"].loc[0, "n_time_windows"] == 2
    assert out["overview"].loc[0, "n_matrix_rows"] == 18
    assert out["overview"].loc[0, "total_transition_count"] == 4
    assert out["matrix_long"]["transition_count"].sum() == 4
    first = out["matrix_long"]
    first = first.loc[first[".gp3_time_window"].eq("0-500") & first[".gp3_from"].eq("A")]
    assert first["transition_probability"].sum(skipna=True) == pytest.approx(1.0)
    assert out["settings"].set_index("setting").loc["normalise", "value"] == "row"


def test_legacy_aoi_interfaces_remain_available():
    data = _sample_aoi_data().rename(columns={"time": "TIME"})
    entries = gp3.summarise_gazepoint_aoi_entries(
        data,
        aoi_col="aoi_current",
        group_cols=["trial_global"],
        time_col="TIME",
    )
    assert {"aoi", "n_entries"} <= set(entries.columns)

    transitions = gp3.summarise_gazepoint_aoi_transitions(
        data,
        aoi_col="aoi_current",
        group_cols=["trial_global"],
        time_col="TIME",
    )
    assert {"from_aoi", "to_aoi", "n_transitions"} <= set(transitions.columns)

    windows = gp3.summarise_gazepoint_aoi_windows(
        data,
        aoi_col="aoi_current",
        time_col="TIME",
        windows={"early": (0.0, 200.0)},
        group_cols=["condition"],
    )
    assert "window" in windows.columns

    matrix = gp3.compute_gazepoint_aoi_transition_matrix(
        data,
        aoi_col="aoi_current",
        group_cols=["subject"],
        time_col="TIME",
    )
    assert isinstance(matrix, pd.DataFrame)
    assert "value" in matrix.columns

    tv = gp3.compute_gazepoint_time_varying_transition_matrix(
        data,
        aoi_col="aoi_current",
        time_col="TIME",
        bin_width=500,
        normalize=True,
    )
    assert isinstance(tv, pd.DataFrame)
    assert "time_bin" in tv.columns


def test_r_sequences_terminal_filter_and_matrix_time_window():
    data = _sample_aoi_data()
    seq = gp3.prepare_gazepoint_aoi_sequences(
        data,
        aoi_col="aoi_current",
        time_col="time",
        group_cols=["subject", "MEDIA_ID", "trial_global"],
        include_non_aoi=True,
        include_terminal=False,
    )
    assert len(seq) == 2
    assert not seq["is_terminal_state"].any()

    matrix = gp3.compute_gazepoint_aoi_transition_matrix(
        data=data,
        aoi_col="aoi_current",
        time_col="time",
        group_cols=["subject", "MEDIA_ID", "trial_global"],
        include_non_aoi=True,
        include_self_transitions=False,
        time_window=[0.0, 250.0],
    )
    assert matrix["long_table"]["n"].sum() == 2


def test_r_windows_dataframe_and_right_endpoint():
    windows = pd.DataFrame(
        {
            "label": ["early", "late"],
            "start": [0.0, 200.0],
            "end": [200.0, 400.0],
        }
    )
    out = gp3.summarise_gazepoint_aoi_windows(
        _sample_aoi_data(),
        windows=windows,
        time_col="time",
        aoi_col="aoi_current",
        subject_col="subject",
        condition_col="condition",
        group_cols=["subject", "condition"],
        target_aoi_values=["B"],
        window_label_col="label",
        window_start_col="start",
        window_end_col="end",
        include_right_endpoint=True,
    )
    assert out["window_label"].tolist() == ["early", "late"]
    assert out["n_window_samples"].tolist() == [3, 2]


def test_r_time_varying_existing_windows_counts_and_global_normalisation():
    transitions = pd.DataFrame(
        {
            "from": ["A", "A", "B"],
            "to": ["B", "C", "A"],
            "window": ["w1", "w1", None],
            "weight": [2.0, 1.0, 3.0],
        }
    )
    out = gp3.compute_gazepoint_time_varying_transition_matrix(
        transitions,
        from_col="from",
        to_col="to",
        window_col="window",
        count_col="weight",
        states=["A", "B", "C"],
        complete_states=False,
        normalise="global",
    )
    assert set(out["time_windows"][".gp3_time_window"]) == {"w1", "missing_window"}
    w1 = out["matrix_long"].loc[out["matrix_long"][".gp3_time_window"].eq("w1")]
    assert w1["transition_count"].sum() == pytest.approx(3.0)
    assert w1["transition_probability"].sum() == pytest.approx(1.0)
