import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3


def vertices():
    return pd.DataFrame(
        {
            "aoi_name": [
                "A",
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
                "B",
            ],
            "vertex_x": [
                0,
                1,
                1,
                0,
                0.5,
                1.5,
                1.5,
                0.5,
            ],
            "vertex_y": [
                0,
                0,
                1,
                1,
                0,
                0,
                1,
                1,
            ],
            "order": [
                1,
                2,
                3,
                4,
                1,
                2,
                3,
                4,
            ],
        }
    )


def gaze():
    return pd.DataFrame(
        {
            "FPOGX": [
                0.25,
                0.75,
                1.25,
                2.0,
                np.nan,
                0.0,
            ],
            "FPOGY": [
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
                0.5,
            ],
        }
    )


def test_polygon_legacy_interface():
    data = pd.DataFrame(
        {
            "x": [0.5, 2.0],
            "y": [0.5, 2.0],
        }
    )

    out = gp3.add_gazepoint_polygon_aoi(
        data,
        {
            "inside": [
                (0, 0),
                (1, 0),
                (1, 1),
                (0, 1),
            ]
        },
    )

    assert out["aoi_current"].tolist() == [
        "inside",
        "outside",
    ]


def test_polygon_r_first_overlap():
    out = gp3.add_gazepoint_polygon_aoi(
        master_df=gaze(),
        vertices=vertices(),
    )

    values = out["aoi_current"].tolist()

    assert values[:4] == [
        "A",
        "A",
        "B",
        "outside",
    ]

    assert pd.isna(values[4])

    assert out["aoi_overlap_count"].tolist() == [1, 2, 1, 0, 0, 1]

    assert "gazepoint_polygon_aoi_definitions" in out.attrs


def test_polygon_r_last_overlap():
    out = gp3.add_gazepoint_polygon_aoi(
        master_df=gaze(),
        vertices=vertices(),
        overlap="last",
    )

    assert (
        out.loc[
            1,
            "aoi_current",
        ]
        == "B"
    )


def test_polygon_r_overlap_error():
    with pytest.raises(
        ValueError,
        match="overlapping AOIs",
    ):
        gp3.add_gazepoint_polygon_aoi(
            master_df=gaze(),
            vertices=vertices(),
            overlap="error",
        )


def test_polygon_boundary_inside_outside():
    inside = gp3.add_gazepoint_polygon_aoi(
        master_df=gaze(),
        vertices=vertices(),
        boundary="inside",
    )

    outside = gp3.add_gazepoint_polygon_aoi(
        master_df=gaze(),
        vertices=vertices(),
        boundary="outside",
    )

    assert (
        inside.loc[
            5,
            "aoi_current",
        ]
        == "A"
    )

    assert (
        outside.loc[
            5,
            "aoi_current",
        ]
        == "outside"
    )


def test_polygon_logical_and_both():
    logical = gp3.add_gazepoint_polygon_aoi(
        master_df=gaze(),
        vertices=vertices(),
        output="logical",
        include_overlap_count=False,
    )

    assert {
        "aoi_A",
        "aoi_B",
    }.issubset(logical.columns)

    assert "aoi_current" not in logical.columns

    assert "aoi_overlap_count" not in logical.columns

    both = gp3.add_gazepoint_polygon_aoi(
        master_df=gaze(),
        vertices=vertices(),
        output="both",
        prefix="hit_",
        label_col="derived",
    )

    assert {
        "hit_A",
        "hit_B",
        "derived",
    }.issubset(both.columns)


def test_polygon_vertex_order():
    shuffled = vertices().sample(
        frac=1,
        random_state=2,
    )

    out = gp3.add_gazepoint_polygon_aoi(
        master_df=gaze(),
        vertices=shuffled,
        vertex_order_col="order",
    )

    assert (
        out.loc[
            0,
            "aoi_current",
        ]
        == "A"
    )


def test_polygon_validation():
    bad = vertices()
    bad.loc[
        0,
        "vertex_x",
    ] = np.nan

    with pytest.raises(
        ValueError,
        match="non-finite",
    ):
        gp3.add_gazepoint_polygon_aoi(
            master_df=gaze(),
            vertices=bad,
        )

    tiny = pd.DataFrame(
        {
            "aoi_name": ["A", "A", "A"],
            "vertex_x": [0, 0, 1],
            "vertex_y": [0, 0, 1],
        }
    )

    with pytest.raises(
        ValueError,
        match="three unique",
    ):
        gp3.add_gazepoint_polygon_aoi(
            master_df=gaze(),
            vertices=tiny,
        )

    with pytest.raises(
        ValueError,
        match="missing required",
    ):
        gp3.add_gazepoint_polygon_aoi(
            master_df=gaze().drop(columns=["FPOGX"]),
            vertices=vertices(),
        )


def test_polygon_bad_arguments():
    for key, value in [
        ("output", "bad"),
        ("overlap", "bad"),
        ("boundary", "bad"),
    ]:
        kwargs = {
            "master_df": gaze(),
            "vertices": vertices(),
            key: value,
        }

        with pytest.raises(ValueError):
            gp3.add_gazepoint_polygon_aoi(**kwargs)


def test_network_legacy_matrix():
    matrix = pd.DataFrame(
        [
            [0, 2],
            [1, 0],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )

    out = gp3.compute_gazepoint_transition_network_metrics(matrix)

    assert {
        "node",
        "pagerank",
    }.issubset(out.columns)


def test_network_direct_transitions():
    data = pd.DataFrame(
        {
            "from": [
                "A",
                "A",
                "A",
                "B",
            ],
            "to": [
                "B",
                "B",
                "A",
                "A",
            ],
        }
    )

    out = gp3.compute_gazepoint_transition_network_metrics(
        data=data,
        from_col="from",
        to_col="to",
    )

    assert out["network_status"] == "ok"

    graph = out["graph_summary"].iloc[0]

    assert graph["n_states"] == 2

    assert graph["n_edges"] == 3

    assert graph["total_transitions"] == 4

    assert graph["self_loops"] == 1

    edges = out["edge_summary"]

    ab = edges[edges["from_state"].eq("A") & edges["to_state"].eq("B")]

    assert ab.iloc[0]["count"] == 2


def test_network_without_self_loops():
    data = pd.DataFrame(
        {
            "from": ["A", "A", "B"],
            "to": ["A", "B", "A"],
        }
    )

    out = gp3.compute_gazepoint_transition_network_metrics(
        data=data,
        from_col="from",
        to_col="to",
        include_self_loops=False,
    )

    assert out["graph_summary"].iloc[0]["self_loops"] == 0

    assert out["graph_summary"].iloc[0]["total_transitions"] == 2


def test_network_long_sequence_order_and_missing_bridge():
    data = pd.DataFrame(
        {
            "id": [
                1,
                1,
                1,
                1,
                2,
                2,
                2,
            ],
            "time": [
                3,
                1,
                2,
                4,
                1,
                2,
                3,
            ],
            "AOI": [
                "C",
                "A",
                None,
                "D",
                "X",
                "",
                "Y",
            ],
        }
    )

    out = gp3.compute_gazepoint_transition_network_metrics(
        data=data,
        aoi_col="AOI",
        group_cols=["id"],
        time_col="time",
    )

    edges = {
        (
            row.from_state,
            row.to_state,
        )
        for row in out["edge_summary"].itertuples()
    }

    assert ("A", "C") in edges

    assert ("C", "D") in edges

    assert ("X", "Y") in edges


def test_network_state_summary():
    data = pd.DataFrame(
        {
            "from": ["A", "A", "B", "C"],
            "to": ["B", "C", "C", "A"],
        }
    )

    out = gp3.compute_gazepoint_transition_network_metrics(
        data=data,
        from_col="from",
        to_col="to",
    )

    states = out["state_summary"].set_index("state")

    assert (
        states.loc[
            "A",
            "out_degree",
        ]
        == 2
    )

    assert (
        states.loc[
            "C",
            "in_degree",
        ]
        == 2
    )


def test_network_empty():
    data = pd.DataFrame({"AOI": ["A"]})

    out = gp3.compute_gazepoint_transition_network_metrics(
        data=data,
        aoi_col="AOI",
    )

    assert out["network_status"] == "empty"

    assert out["graph_summary"].iloc[0]["n_states"] == 0


def test_network_validation():
    with pytest.raises(
        ValueError,
        match="aoi_col",
    ):
        gp3.compute_gazepoint_transition_network_metrics(data=pd.DataFrame({"x": [1]}))

    with pytest.raises(
        ValueError,
        match="missing required",
    ):
        gp3.compute_gazepoint_transition_network_metrics(
            data=pd.DataFrame({"AOI": ["A"]}),
            aoi_col="AOI",
            time_col="TIME",
        )

    with pytest.raises(
        ValueError,
        match="group_cols",
    ):
        gp3.compute_gazepoint_transition_network_metrics(
            data=pd.DataFrame({"AOI": ["A"]}),
            aoi_col="AOI",
            group_cols=[],
        )
