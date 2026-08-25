import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3


def fixture():
    sequences = {
        "S1": ["A", "B", "C", "A"],
        "S2": ["A", "B", "C", "B"],
        "S3": ["A", "B", "C", "C"],
        "S4": ["C", "B", "A", "C"],
        "S5": ["C", "B", "A", "A"],
    }

    return pd.DataFrame(
        [
            {
                "sequence_id": sequence_id,
                "AOI": aoi,
                "TIME": time,
            }
            for sequence_id, sequence in sequences.items()
            for time, aoi in enumerate(sequence, start=1)
        ]
    )


def test_legacy_similarity():
    assert np.isclose(
        gp3.compute_gazepoint_scanpath_similarity(
            ["A", "B", "C"],
            ["A", "B", "D"],
        ),
        2 / 3,
    )


def test_r_similarity_table():
    out = gp3.compute_gazepoint_scanpath_similarity(
        data=fixture(),
        aoi_col="AOI",
        group_cols=["sequence_id"],
        time_col="TIME",
    )

    assert len(out) == 15

    self_rows = out.loc[out["sequence_a"].eq(out["sequence_b"])]

    assert self_rows["similarity"].eq(1).all()
    assert self_rows["edit_distance"].eq(0).all()
    assert out["n_sequences"].eq(5).all()


def test_similarity_time_order_and_guard():
    data = pd.DataFrame(
        {
            "id": ["S1", "S1", "S2", "S2"],
            "AOI": ["B", "A", "B", "A"],
            "TIME": [2, 1, 1, 2],
        }
    )

    out = gp3.compute_gazepoint_scanpath_similarity(
        data=data,
        aoi_col="AOI",
        group_cols=["id"],
        time_col="TIME",
    )

    pair = out.loc[out["sequence_a"].eq("S1") & out["sequence_b"].eq("S2")].iloc[0]

    assert pair["edit_distance"] == 2

    with pytest.raises(
        ValueError,
        match="Too many grouped sequences",
    ):
        gp3.compute_gazepoint_scanpath_similarity(
            data=fixture(),
            aoi_col="AOI",
            group_cols=["sequence_id"],
            time_col="TIME",
            max_sequences=4,
        )


def test_similarity_missing_collapse():
    data = pd.DataFrame(
        {
            "id": ["S1"] * 4 + ["S2"] * 4,
            "AOI": [
                "A",
                "A",
                None,
                "B",
                "A",
                "",
                "missing",
                "B",
            ],
            "TIME": list(range(1, 9)),
        }
    )

    out = gp3.compute_gazepoint_scanpath_similarity(
        data=data,
        aoi_col="AOI",
        group_cols=["id"],
        time_col="TIME",
        include_missing=True,
        collapse_repeats=True,
    )

    pair = out.loc[out["sequence_a"].eq("S1") & out["sequence_b"].eq("S2")].iloc[0]

    assert pair["normalized_distance"] == 0


def test_legacy_cluster():
    out = gp3.cluster_gazepoint_scanpaths(
        fixture(),
        aoi_col="AOI",
        group_cols=["sequence_id"],
        time_col="TIME",
        n_clusters=2,
    )

    assert isinstance(out, pd.DataFrame)
    assert out["cluster"].nunique() == 2


def test_r_cluster_long_and_pairwise():
    long_fit = gp3.cluster_gazepoint_scanpaths(
        x=fixture(),
        k=2,
        method="hierarchical",
        linkage="average",
        aoi_col="AOI",
        group_cols=["sequence_id"],
        time_col="TIME",
    )

    assert long_fit["_gp3_class"] == "gp3_scanpath_clusters"
    assert long_fit["distance_source"] == "long_aoi_data"
    assert long_fit["distance"].shape == (5, 5)
    assert long_fit["assignments"]["cluster"].nunique() == 2

    pair_fit = gp3.cluster_gazepoint_scanpaths(
        x=long_fit["pairwise_distances"],
        k=2,
        method="hierarchical",
        linkage="complete",
    )

    assert pair_fit["distance_source"] == "pairwise_distance_table"


def test_r_cluster_matrix_and_pam():
    matrix = np.array(
        [
            [0, 0.1, 0.8, 0.9],
            [0.1, 0, 0.9, 0.8],
            [0.8, 0.9, 0, 0.1],
            [0.9, 0.8, 0.1, 0],
        ],
        dtype=float,
    )

    hierarchical = gp3.cluster_gazepoint_scanpaths(
        x=matrix,
        k=2,
        method="hierarchical",
        linkage="average",
    )

    assert len(hierarchical["assignments"]) == 4

    pam = gp3.cluster_gazepoint_scanpaths(
        x=matrix,
        k=2,
        method="pam",
    )

    assert len(pam["medoids"]) == 2


def test_cluster_validation():
    with pytest.raises(
        ValueError,
        match="at least 2",
    ):
        gp3.cluster_gazepoint_scanpaths(
            x=np.array(
                [
                    [0.0, 0.1, 0.8, 0.9],
                    [0.1, 0.0, 0.9, 0.8],
                    [0.8, 0.9, 0.0, 0.1],
                    [0.9, 0.8, 0.1, 0.0],
                ]
            ),
            k=1,
            method="hierarchical",
        )

    with pytest.raises(
        ValueError,
        match="symmetric",
    ):
        gp3.cluster_gazepoint_scanpaths(
            x=np.array(
                [
                    [0, 1, 2],
                    [0, 0, 1],
                    [2, 1, 0.0],
                ]
            ),
            k=2,
            method="hierarchical",
        )


def test_legacy_bootstrap_and_stability():
    bootstrap = gp3.bootstrap_gazepoint_scanpath_clusters(
        fixture(),
        n_boot=3,
        random_state=9,
        aoi_col="AOI",
        group_cols=["sequence_id"],
        time_col="TIME",
        n_clusters=2,
    )

    assert isinstance(bootstrap, pd.DataFrame)
    assert len(bootstrap) == 3

    summary = gp3.summarise_gazepoint_scanpath_cluster_stability(bootstrap)

    assert summary.iloc[0]["n_boot"] == 3


def test_r_bootstrap_structure_and_reproducibility():
    kwargs = {
        "x": fixture(),
        "k": 2,
        "n_boot": 5,
        "sample_fraction": 0.8,
        "method": "hierarchical",
        "linkages": ["average"],
        "seed": 17,
        "aoi_col": "AOI",
        "group_cols": ["sequence_id"],
        "time_col": "TIME",
    }

    first = gp3.bootstrap_gazepoint_scanpath_clusters(**kwargs)

    second = gp3.bootstrap_gazepoint_scanpath_clusters(**kwargs)

    assert first["_gp3_class"] == "gp3_scanpath_cluster_bootstrap"
    assert first["settings"]["sample_size"] == 4
    assert len(first["iteration_summary"]) == 5

    specification = first["specifications"].iloc[0]["specification"]

    assert first["co_clustering"][specification].shape == (5, 5)

    assert np.allclose(
        np.diag(first["co_clustering"][specification].to_numpy()),
        1,
    )

    pd.testing.assert_frame_equal(
        first["iteration_summary"],
        second["iteration_summary"],
    )

    pd.testing.assert_frame_equal(
        first["co_clustering"][specification],
        second["co_clustering"][specification],
    )


def test_r_bootstrap_multiple_linkages_and_validation():
    result = gp3.bootstrap_gazepoint_scanpath_clusters(
        x=fixture(),
        k=2,
        n_boot=2,
        sample_fraction=1,
        method="hierarchical",
        linkages=["average", "complete"],
        seed=5,
        aoi_col="AOI",
        group_cols=["sequence_id"],
        time_col="TIME",
    )

    assert len(result["specifications"]) == 2
    assert len(result["iteration_summary"]) == 4

    with pytest.raises(
        ValueError,
        match="sample_fraction",
    ):
        gp3.bootstrap_gazepoint_scanpath_clusters(
            x=fixture(),
            k=2,
            sample_fraction=1.5,
            aoi_col="AOI",
            group_cols=["sequence_id"],
            time_col="TIME",
        )


def test_r_stability_structure_and_thresholds():
    bootstrap = gp3.bootstrap_gazepoint_scanpath_clusters(
        x=fixture(),
        k=2,
        n_boot=6,
        sample_fraction=0.8,
        seed=101,
        aoi_col="AOI",
        group_cols=["sequence_id"],
        time_col="TIME",
    )

    out = gp3.summarise_gazepoint_scanpath_cluster_stability(
        x=bootstrap,
        min_pair_coverage=0.5,
        stable_threshold=0.5,
    )

    assert out["_gp3_class"] == "gp3_scanpath_cluster_stability_summary"

    assert len(out["overview"]) == 1
    assert len(out["sequence_summary"]) == 5
    assert len(out["pairwise_summary"]) == 10
    assert out["pairwise_summary"]["included_in_summary"].dtype == bool

    with pytest.raises(
        ValueError,
        match="stable_threshold",
    ):
        gp3.summarise_gazepoint_scanpath_cluster_stability(
            x=bootstrap,
            stable_threshold=2,
        )

    with pytest.raises(
        ValueError,
        match="missing",
    ):
        gp3.summarise_gazepoint_scanpath_cluster_stability(x={"reference_fits": {}})


def test_scanpath_additional_validation_coverage():
    with pytest.raises(TypeError, match="either path_a or data"):
        gp3.compute_gazepoint_scanpath_similarity(
            ["A"],
            data=fixture(),
            aoi_col="AOI",
            group_cols=["sequence_id"],
        )

    with pytest.raises(ValueError, match="Missing columns"):
        gp3.compute_gazepoint_scanpath_similarity(
            data=fixture().drop(columns=["AOI"]),
            aoi_col="AOI",
            group_cols=["sequence_id"],
        )

    with pytest.raises(ValueError, match="max_sequences"):
        gp3.compute_gazepoint_scanpath_similarity(
            data=fixture(),
            aoi_col="AOI",
            group_cols=["sequence_id"],
            max_sequences=1,
        )

    assert np.isnan(
        gp3.compute_gazepoint_scanpath_similarity(
            [],
            [],
            method="coordinate",
        )
    )


def test_scanpath_distance_matrix_validation_coverage():
    with pytest.raises(ValueError, match="square"):
        gp3.cluster_gazepoint_scanpaths(
            x=np.zeros((3, 2)),
            k=2,
            method="hierarchical",
        )

    with pytest.raises(ValueError, match="finite"):
        gp3.cluster_gazepoint_scanpaths(
            x=np.array(
                [
                    [0.0, np.nan, 1.0],
                    [np.nan, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                ]
            ),
            k=2,
            method="hierarchical",
        )

    with pytest.raises(ValueError, match="non-negative"):
        gp3.cluster_gazepoint_scanpaths(
            x=np.array(
                [
                    [0.0, -1.0, 1.0],
                    [-1.0, 0.0, 1.0],
                    [1.0, 1.0, 0.0],
                ]
            ),
            k=2,
            method="hierarchical",
        )

    with pytest.raises(ValueError, match="diagonal"):
        gp3.cluster_gazepoint_scanpaths(
            x=np.array(
                [
                    [1.0, 0.5, 0.5],
                    [0.5, 0.0, 0.5],
                    [0.5, 0.5, 0.0],
                ]
            ),
            k=2,
            method="hierarchical",
        )


def test_scanpath_cluster_additional_modes_and_guards():
    square = pd.DataFrame(
        [
            [0.0, 0.1, 0.8, 0.9],
            [0.1, 0.0, 0.9, 0.8],
            [0.8, 0.9, 0.0, 0.1],
            [0.9, 0.8, 0.1, 0.0],
        ],
        index=["A", "B", "C", "D"],
        columns=["A", "B", "C", "D"],
    )

    result = gp3.cluster_gazepoint_scanpaths(
        x=square,
        k=2,
        method="hierarchical",
    )
    assert result["distance_source"] == "distance_matrix"
    assert result["assignments"]["cluster"].nunique() == 2

    with pytest.raises(ValueError, match="At least three"):
        gp3.cluster_gazepoint_scanpaths(
            x=np.array(
                [
                    [0.0, 0.5],
                    [0.5, 0.0],
                ]
            ),
            k=1,
            method="hierarchical",
        )

    with pytest.raises(ValueError, match="method"):
        gp3.cluster_gazepoint_scanpaths(
            x=square,
            k=2,
            method="unknown",
        )


def test_scanpath_pairwise_incomplete_guard():
    pairs = pd.DataFrame(
        {
            "sequence_a": ["A", "A", "B", "C"],
            "sequence_b": ["A", "B", "B", "C"],
            "normalized_distance": [0.0, 0.5, 0.0, 0.0],
        }
    )

    with pytest.raises(ValueError, match="every sequence pair"):
        gp3.cluster_gazepoint_scanpaths(
            x=pairs,
            k=2,
            method="hierarchical",
        )


def test_scanpath_bootstrap_additional_guards():
    with pytest.raises(TypeError, match="legacy clustering kwargs"):
        gp3.bootstrap_gazepoint_scanpath_clusters(
            x=fixture(),
            k=2,
            n_boot=2,
            aoi_col="AOI",
            group_cols=["sequence_id"],
            unsupported=True,
        )

    with pytest.raises(ValueError, match="n_boot"):
        gp3.bootstrap_gazepoint_scanpath_clusters(
            x=fixture(),
            k=2,
            n_boot=0,
            aoi_col="AOI",
            group_cols=["sequence_id"],
        )

    with pytest.raises(ValueError, match="smaller than"):
        gp3.bootstrap_gazepoint_scanpath_clusters(
            x=fixture(),
            k=5,
            n_boot=2,
            aoi_col="AOI",
            group_cols=["sequence_id"],
        )


def test_scanpath_stability_additional_guards():
    with pytest.raises(ValueError, match="R-compatible bootstrap"):
        gp3.summarise_gazepoint_scanpath_cluster_stability(x=["not", "a", "bootstrap"])

    bootstrap = gp3.bootstrap_gazepoint_scanpath_clusters(
        x=fixture(),
        k=2,
        n_boot=3,
        sample_fraction=1.0,
        seed=7,
        aoi_col="AOI",
        group_cols=["sequence_id"],
        time_col="TIME",
    )

    with pytest.raises(ValueError, match="min_pair_coverage"):
        gp3.summarise_gazepoint_scanpath_cluster_stability(
            x=bootstrap,
            min_pair_coverage=-0.1,
        )
