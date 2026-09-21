from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gp3tools import aoi, pupil, qc


# ============================================================================
# PUPIL PURE HELPERS
# ============================================================================


def test_pupil_interpolation_helper_contracts():
    x = pd.Series([1.0, np.nan, 3.0, np.nan, np.nan, 6.0])

    linear = pupil._interpolate_series(x, "linear")
    assert linear.iloc[1] == pytest.approx(2.0)

    pchip = pupil._interpolate_series(x, "pchip")
    assert np.isfinite(pchip.iloc[1])

    cubic = pupil._interpolate_series(x, "cubic")
    assert np.isfinite(cubic.iloc[1])

    limited = pupil._interpolate_series(x, "pchip", limit=1)
    assert pd.isna(limited.iloc[3])
    assert pd.isna(limited.iloc[4])

    too_sparse = pupil._interpolate_series(
        pd.Series([np.nan, 1.0, np.nan]),
        "pchip",
    )
    assert too_sparse.isna().sum() == 2

    with pytest.raises(ValueError, match="Unsupported interpolation method"):
        pupil._interpolate_series(x, "bad")


def test_pupil_shared_r_helpers():
    frame = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S2"],
            "pupil": [1.0, 2.0, 3.0],
        }
    )

    assert pupil._gp3_r_first_present(frame, ["missing", "pupil"]) == "pupil"
    assert pupil._gp3_r_first_present(frame, ["missing"]) is None

    assert pupil._gp3_r_bool(pd.Series([True, False])).tolist() == [True, False]
    assert pupil._gp3_r_bool(pd.Series([1, 0, np.nan])).tolist() == [True, False, False]
    assert pupil._gp3_r_bool(pd.Series(["yes", "no", "t", "x"])).tolist() == [
        True,
        False,
        True,
        False,
    ]

    pooled = pupil._gp3_r_group_parts(frame, [])
    assert len(pooled) == 1

    grouped = list(pupil._gp3_r_group_parts(frame, ["subject"]))
    assert len(grouped) == 2

    assert pupil._gp3_r_group_row([], ()) == {}
    assert pupil._gp3_r_group_row(["subject"], "S1") == {"subject": "S1"}
    assert pupil._gp3_r_group_row(["subject", "trial"], ("S1", "T1")) == {
        "subject": "S1",
        "trial": "T1",
    }


def test_pupil_downsampling_and_binocular_helpers():
    frame = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S2"],
            "pupil": [1.0, 2.0, 3.0],
        }
    )

    assert pupil._gp3_pupil_r_group_cols("USER_ID", ["trial", "USER_ID"]) == [
        "USER_ID",
        "trial",
    ]
    assert pupil._gp3_pupil_r_group_positions(frame, []).pop().tolist() == [0, 1, 2]
    assert len(pupil._gp3_pupil_r_group_positions(frame, ["subject"])) == 2

    assert pupil._gp3_pupil_r_detect_columns(frame, "pupil") == ["pupil"]
    assert pupil._gp3_pupil_r_detect_columns(frame, ["pupil", "pupil"]) == ["pupil"]

    with pytest.raises(ValueError, match="missing required"):
        pupil._gp3_pupil_r_detect_columns(frame, ["missing"])

    with pytest.raises(ValueError, match="No pupil column"):
        pupil._gp3_pupil_r_detect_columns(pd.DataFrame({"x": [1]}), None)

    numeric = pupil._gp3_pupil_r_numeric(pd.Series(["1", "bad", 3]))
    assert numeric[0] == 1
    assert np.isnan(numeric[1])

    means = pupil._gp3_pupil_r_row_mean_two(
        [1.0, np.nan, np.nan],
        [3.0, 4.0, np.nan],
    )
    assert means[0] == 2.0
    assert means[1] == 4.0
    assert np.isnan(means[2])

    coeff = pupil._gp3_pupil_r_fit_line([0, 1, 2], [1, 3, 5])
    assert coeff[0] == pytest.approx(1.0)
    assert coeff[1] == pytest.approx(2.0)


def test_pupil_binocular_helper_matrix():
    assert pupil._gp3_binoc_r_cols(None) == []
    assert pupil._gp3_binoc_r_cols("x") == ["x"]
    assert pupil._gp3_binoc_r_cols(["x", "x", None, ""]) == ["x"]

    frame = pd.DataFrame(
        {
            "subject": ["S1", "S1", pd.NA],
            "time": [0.0, 0.1, 0.2],
            "x": [1.0, np.nan, 3.0],
        }
    )

    pupil._gp3_binoc_r_check_cols(frame, ["x"])
    with pytest.raises(KeyError, match="Missing columns"):
        pupil._gp3_binoc_r_check_cols(frame, ["missing"])

    pupil._gp3_binoc_r_bounds(None, None)
    pupil._gp3_binoc_r_bounds(0, 10)
    with pytest.raises(ValueError, match="finite"):
        pupil._gp3_binoc_r_bounds(np.inf, None)
    with pytest.raises(ValueError, match="smaller"):
        pupil._gp3_binoc_r_bounds(2, 1)

    observed = pupil._gp3_binoc_r_observed(
        pd.Series([1.0, np.inf, -1.0, 5.0]),
        valid_min=0,
        valid_max=4,
    )
    assert np.isfinite(observed).sum() == 1

    pooled = pupil._gp3_binoc_r_group_key(frame, [])
    assert set(pooled) == {"__pooled__"}

    keys = pupil._gp3_binoc_r_group_key(frame, ["subject"])
    assert "subject=<NA>" in keys

    groups = pupil._gp3_binoc_r_groups(frame, ["subject"])
    assert len(groups) == 2

    assert pupil._gp3_binoc_r_time_scale_ms(np.array([0.0]), "auto") == 1.0
    assert pupil._gp3_binoc_r_time_scale_ms(np.array([0.0, 0.1]), "auto") == 1000.0
    assert pupil._gp3_binoc_r_time_scale_ms(np.array([0.0, 10.0]), "auto") == 1.0
    assert pupil._gp3_binoc_r_time_scale_ms(np.array([0.0]), "seconds") == 1000.0
    assert pupil._gp3_binoc_r_time_scale_ms(np.array([0.0]), "milliseconds") == 1.0

    gaps = pupil._gp3_binoc_r_gaps(
        frame,
        np.array([False, True, True]),
        ["subject"],
        time_col="time",
        time_unit="seconds",
    )
    assert len(gaps["gaps"]) == 2
    assert gaps["edge_gap"].sum() >= 1

    assert pd.isna(pupil._gp3_binoc_r_mad([np.nan]))
    assert pupil._gp3_binoc_r_mad([1, 2, 100]) == pytest.approx(1.0)


# ============================================================================
# QC PURE HELPERS
# ============================================================================


def test_qc_boolean_and_exclusion_helpers():
    idx = pd.RangeIndex(3)

    assert qc._gp3_qc_r_list(None) == []
    assert qc._gp3_qc_r_list("x") == ["x"]
    assert qc._gp3_qc_r_list(("x", "y")) == ["x", "y"]

    b = qc._gp3_qc_r_as_bool_series([True, False, pd.NA], idx)
    assert bool(b.iloc[0])
    assert not bool(b.iloc[1])
    assert pd.isna(b.iloc[2])

    n = qc._gp3_qc_r_as_bool_series([1, 0, np.nan], idx)
    assert bool(n.iloc[0])
    assert not bool(n.iloc[1])
    assert pd.isna(n.iloc[2])

    t = qc._gp3_qc_r_as_bool_series(["yes", "bad", "other"], idx)
    assert bool(t.iloc[0])
    assert not bool(t.iloc[1])
    assert pd.isna(t.iloc[2])

    assert qc._gp3_exclusion_r_bool([True, False], "x").tolist() == [True, False]
    assert qc._gp3_exclusion_r_bool([1, 0, np.nan], "x").tolist()[:2] == [True, False]

    with pytest.raises(ValueError, match="0, 1"):
        qc._gp3_exclusion_r_bool([2], "x")

    assert qc._gp3_exclusion_r_bool(["keep", "drop"], "x").tolist() == [True, False]
    with pytest.raises(ValueError, match="interpretable"):
        qc._gp3_exclusion_r_bool(["mystery"], "x")

    aliases = qc._gp3_exclusion_r_aliases(
        pd.DataFrame({"MEDIA_ID": ["M1"], "USER_FILE": ["S1"]})
    )
    assert {"media_id", "subject"}.issubset(aliases.columns)

    assert qc._gp3_exclusion_r_col(aliases, None, "x", optional=True) is None
    assert qc._gp3_exclusion_r_col(aliases, "MEDIA_ID", "x") == "media_id"
    assert qc._gp3_exclusion_r_col(aliases, "USER_FILE", "x") == "subject"

    with pytest.raises(ValueError, match="non-empty string"):
        qc._gp3_exclusion_r_col(aliases, "", "x")
    with pytest.raises(KeyError, match="present in data"):
        qc._gp3_exclusion_r_col(aliases, "missing", "x")


def test_qc_exclusion_units_and_ratio_paths():
    frame = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S2", "S2", "S3"],
            "condition": ["A", "A", "B", "B", "C"],
        }
    )

    flags = pd.Series([True, False, False, False, pd.NA], dtype="boolean")
    reasons = pd.Series(["ok", "bad", "", None, None])

    out = qc._gp3_exclusion_r_units(
        frame,
        flags,
        "subject",
        "condition",
        [],
        reason=reasons,
    )

    statuses = set(out["exclusion_flow_status"])
    assert {"conflicting_flags", "excluded", "unclear_status"}.issubset(statuses)

    no_ids = qc._gp3_exclusion_r_units(
        pd.DataFrame({"x": [1, 2]}),
        pd.Series([True, True], dtype="boolean"),
        None,
        None,
        [],
    )
    assert len(no_ids) == 2

    assert pd.isna(qc._gp3_exclusion_r_ratio([1]))
    assert pd.isna(qc._gp3_exclusion_r_ratio([0, 0]))
    assert qc._gp3_exclusion_r_ratio([0, 0], zero_returns_one=True) == 1.0
    assert np.isinf(qc._gp3_exclusion_r_ratio([0, 1]))
    assert qc._gp3_exclusion_r_ratio([1, 2, 4]) == 4.0


def test_qc_overview_collection_helpers():
    overview = pd.DataFrame(
        {
            "status": ["pass", "warn"],
            "message": ["ok", "review"],
        }
    )
    obj = {"overview": overview}

    assert qc._gp3_qc_has_overview(obj)
    assert not qc._gp3_qc_has_overview({})

    with pytest.raises(ValueError, match="at least one"):
        qc._gp3_qc_normalise_objects(None)
    with pytest.raises(ValueError, match="at least one"):
        qc._gp3_qc_normalise_objects({})
    with pytest.raises(ValueError, match="at least one"):
        qc._gp3_qc_normalise_objects([])

    assert len(qc._gp3_qc_normalise_objects(overview)) == 1
    assert len(qc._gp3_qc_normalise_objects(obj)) == 1
    assert len(qc._gp3_qc_normalise_objects({"a": obj, "b": overview})) == 2
    assert len(qc._gp3_qc_normalise_objects([obj, overview])) == 2

    assert qc._gp3_qc_extract_overview(overview) is overview
    assert qc._gp3_qc_extract_overview(obj).equals(overview)
    assert qc._gp3_qc_extract_overview("x") is None

    assert qc._gp3_qc_status_columns(overview) == ["status"]
    assert qc._gp3_qc_message_columns(overview) == ["message"]

    assert qc._gp3_qc_worse_status("pass", "warn") == "warn"
    assert qc._gp3_qc_worse_status("fail", "pass") == "fail"


# ============================================================================
# AOI PURE HELPERS
# ============================================================================


def test_aoi_r_list_label_and_resolution_helpers():
    assert aoi._gp3_aoi_r_list(None) == []
    assert aoi._gp3_aoi_r_list("x") == ["x"]
    assert aoi._gp3_aoi_r_list(["x", "y"]) == ["x", "y"]

    with pytest.raises(ValueError, match="character vector"):
        aoi._gp3_aoi_r_list(3)
    with pytest.raises(ValueError, match="non-missing"):
        aoi._gp3_aoi_r_list([""])
    with pytest.raises(ValueError, match="unique"):
        aoi._gp3_aoi_r_list(["x", "x"], unique=True)

    assert aoi._gp3_aoi_r_scalar_label(None, "x", allow_none=True) is None
    assert aoi._gp3_aoi_r_scalar_label("x", "x") == "x"
    with pytest.raises(ValueError, match="non-missing"):
        aoi._gp3_aoi_r_scalar_label("", "x")

    frame = pd.DataFrame({"AOI": ["a"]})
    assert (
        aoi._gp3_aoi_r_resolve_column(
            frame,
            None,
            ["AOI"],
            name="aoi_col",
        )
        == "AOI"
    )
    with pytest.raises(ValueError, match="automatically detect"):
        aoi._gp3_aoi_r_resolve_column(
            frame,
            None,
            ["missing"],
            name="x",
        )
    with pytest.raises(ValueError, match="Missing required"):
        aoi._gp3_aoi_r_resolve_column(
            frame,
            "missing",
            [],
            name="x",
        )


def test_aoi_entries_and_sequences_core_contract():
    data = pd.DataFrame(
        {
            "subject": ["S1"] * 5,
            "time": [0.0, 100.0, 200.0, 300.0, np.nan],
            "aoi": ["A", "A", "B", pd.NA, "C"],
        }
    )

    entries = aoi._gp3_aoi_r_entries(
        data,
        aoi_col="aoi",
        time_col="time",
        group_cols=["subject"],
        include_non_aoi=True,
        non_aoi_values=["missing_aoi"],
        missing_aoi_label="missing_aoi",
    )
    assert len(entries) == 3
    assert entries["entry_order"].tolist() == [1, 2, 3]

    filtered = aoi._gp3_aoi_r_entries(
        data,
        aoi_col="aoi",
        time_col="time",
        group_cols=["subject"],
        include_non_aoi=False,
        non_aoi_values=["missing_aoi"],
        missing_aoi_label="missing_aoi",
    )
    assert "missing_aoi" not in filtered["aoi_state"].tolist()

    sequences = aoi._gp3_aoi_r_sequences(
        entries,
        group_cols=["subject"],
        include_non_aoi=True,
        non_aoi_values=["missing_aoi"],
        missing_aoi_label="missing_aoi",
        include_terminal=True,
    )
    assert bool(sequences.iloc[-1]["is_terminal_state"])

    no_terminal = aoi._gp3_aoi_r_sequences(
        entries,
        group_cols=["subject"],
        include_non_aoi=True,
        non_aoi_values=["missing_aoi"],
        missing_aoi_label="missing_aoi",
        include_terminal=False,
    )
    assert not no_terminal["is_terminal_state"].any()

    with pytest.raises(ValueError, match="include_non_aoi"):
        aoi._gp3_aoi_r_sequences(
            entries,
            group_cols=["subject"],
            include_non_aoi="yes",
        )

    with pytest.raises(ValueError, match="include_terminal"):
        aoi._gp3_aoi_r_sequences(
            entries,
            group_cols=["subject"],
            include_terminal="yes",
        )


def test_aoi_distance_and_scanpath_helpers():
    assert aoi._levenshtein([], []) == 0
    assert aoi._levenshtein(["A", "B"], ["A", "C"]) == 1

    assert aoi._gp3_scanpath_r_group_id("S1") == "S1"
    assert aoi._gp3_scanpath_r_group_id(("S1", pd.NA)) == "S1|<NA>"

    data = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S2", "S2"],
            "time": [2, 1, 1, 2],
            "aoi": ["A", "A", "B", pd.NA],
        }
    )

    ids, seqs = aoi._gp3_scanpath_r_sequences(
        data,
        aoi_col="aoi",
        group_cols=["subject"],
        time_col="time",
        include_missing=True,
        missing_label="missing",
        collapse_repeats=True,
        max_sequences=10,
    )
    assert len(ids) == 2
    assert seqs[0] == ["A"]

    with pytest.raises(ValueError, match="aoi_col"):
        aoi._gp3_scanpath_r_sequences(
            data,
            aoi_col="",
            group_cols=["subject"],
            time_col=None,
            include_missing=False,
            missing_label="missing",
            collapse_repeats=False,
            max_sequences=10,
        )

    with pytest.raises(ValueError, match="group_cols"):
        aoi._gp3_scanpath_r_sequences(
            data,
            aoi_col="aoi",
            group_cols=[],
            time_col=None,
            include_missing=False,
            missing_label="missing",
            collapse_repeats=False,
            max_sequences=10,
        )

    with pytest.raises(ValueError, match="Missing columns"):
        aoi._gp3_scanpath_r_sequences(
            data,
            aoi_col="aoi",
            group_cols=["missing"],
            time_col=None,
            include_missing=False,
            missing_label="missing",
            collapse_repeats=False,
            max_sequences=10,
        )

    with pytest.raises(ValueError, match="at least 2"):
        aoi._gp3_scanpath_r_sequences(
            data,
            aoi_col="aoi",
            group_cols=["subject"],
            time_col=None,
            include_missing=False,
            missing_label="missing",
            collapse_repeats=False,
            max_sequences=1,
        )


def test_aoi_distance_matrix_helpers_and_clustering():
    matrix = pd.DataFrame(
        [[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )

    validated = aoi._gp3_scanpath_r_validate_distance_matrix(matrix)
    assert validated.equals(matrix)

    for bad, match in [
        (np.ones((2, 3)), "square"),
        (np.array([[0.0, np.nan], [np.nan, 0.0]]), "finite"),
        (np.array([[0.0, -1.0], [-1.0, 0.0]]), "non-negative"),
        (np.array([[0.0, 1.0], [2.0, 0.0]]), "symmetric"),
        (np.array([[1.0, 0.0], [0.0, 1.0]]), "diagonal"),
    ]:
        with pytest.raises(ValueError, match=match):
            aoi._gp3_scanpath_r_validate_distance_matrix(bad)

    with pytest.raises(ValueError, match="labels"):
        aoi._gp3_scanpath_r_validate_distance_matrix(
            np.zeros((2, 2)),
            labels=["a", "a"],
        )

    pairs = pd.DataFrame(
        {
            "sequence_a": ["a", "a", "b"],
            "sequence_b": ["b", "c", "c"],
            "distance": [1.0, 2.0, 1.5],
        }
    )
    pair_matrix = aoi._gp3_scanpath_r_pairs_to_matrix(pairs, "distance")
    assert pair_matrix.loc["a", "c"] == 2.0

    with pytest.raises(ValueError, match="Missing columns"):
        aoi._gp3_scanpath_r_pairs_to_matrix(
            pairs.drop(columns=["distance"]),
            "distance",
        )

    incomplete = pairs.iloc[:2]
    with pytest.raises(ValueError, match="every sequence pair"):
        aoi._gp3_scanpath_r_pairs_to_matrix(incomplete, "distance")

    labels, model, medoids = aoi._gp3_scanpath_r_cluster_matrix(
        matrix,
        k=2,
        method="hierarchical",
        linkage="average",
    )
    assert len(labels) == 3
    assert model is not None
    assert medoids is None

    pam_labels, pam_model, medoids = aoi._gp3_scanpath_r_cluster_matrix(
        matrix,
        k=2,
        method="pam",
        linkage="average",
    )
    assert len(pam_labels) == 3
    assert pam_model["method"] == "pam"
    assert len(medoids) == 2

    with pytest.raises(ValueError, match="At least three"):
        aoi._gp3_scanpath_r_cluster_matrix(
            matrix.iloc[:2, :2],
            k=1,
            method="pam",
            linkage="average",
        )

    with pytest.raises(ValueError, match="k must"):
        aoi._gp3_scanpath_r_cluster_matrix(
            matrix,
            k=3,
            method="pam",
            linkage="average",
        )

    with pytest.raises(ValueError, match="linkage"):
        aoi._gp3_scanpath_r_cluster_matrix(
            matrix,
            k=2,
            method="hierarchical",
            linkage="bad",
        )

    with pytest.raises(ValueError, match="method"):
        aoi._gp3_scanpath_r_cluster_matrix(
            matrix,
            k=2,
            method="bad",
            linkage="average",
        )


def test_aoi_scanpath_representatives_mapping_and_transition_helpers():
    distance = pd.DataFrame(
        [[0.0, 1.0, 3.0], [1.0, 0.0, 2.0], [3.0, 2.0, 0.0]],
        index=["a", "b", "c"],
        columns=["a", "b", "c"],
    )
    fit = {
        "distance": distance,
        "assignments": pd.DataFrame(
            {
                "sequence_id": ["a", "b", "c"],
                "cluster": [1, 1, 2],
            }
        ),
    }
    reps = aoi._gp3_scanpath_r_representatives(fit)
    assert len(reps) == 2
    assert reps.loc[reps["cluster"].eq(2), "cluster_size"].iloc[0] == 1

    mapping = aoi._gp3_scanpath_r_map_clusters(
        np.array([1, 1, 2, 2]),
        np.array([2, 2, 1, 1]),
    )
    assert mapping == {1: 2, 2: 1}

    assert aoi._gp3_transition_r_scalar_string("aoi", "aoi_col") == "aoi"
    with pytest.raises(ValueError, match="non-empty"):
        aoi._gp3_transition_r_scalar_string("", "aoi_col")

    assert aoi._gp3_transition_r_group_cols(None) is None
    assert aoi._gp3_transition_r_group_cols("subject") == ["subject"]
    assert aoi._gp3_transition_r_group_cols(["subject", "trial"]) == [
        "subject",
        "trial",
    ]
    with pytest.raises(ValueError, match="group_cols"):
        aoi._gp3_transition_r_group_cols([])

    assert aoi._gp3_transition_r_prepare_aoi(
        ["A", pd.NA, "", " B "]
    ) == ["A", " B "]
