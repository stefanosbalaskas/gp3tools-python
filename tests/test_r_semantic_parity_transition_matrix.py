import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3


def test_legacy_transition_matrix_preserved():
    out = gp3.compute_transition_matrix(["A", "B", "A"])

    assert out.index.tolist() == [
        "A",
        "B",
    ]

    assert out.columns.tolist() == [
        "A",
        "B",
    ]

    assert out.loc["A", "B"] == 1
    assert out.loc["B", "A"] == 1


def test_legacy_transition_matrix_normalize_preserved():
    out = gp3.compute_transition_matrix(
        ["A", "B", "A", "C"],
        normalize=True,
    )

    assert np.isclose(
        out.loc["A", "B"],
        0.5,
    )

    assert np.isclose(
        out.loc["A", "C"],
        0.5,
    )


def test_r_transition_matrix_collapses_repeats():
    data = pd.DataFrame(
        {
            "MEDIA_ID": [
                1,
                1,
                1,
                1,
                1,
            ],
            "AOI": [
                "A",
                "A",
                "B",
                "B",
                "A",
            ],
            "TIME": [
                1,
                2,
                3,
                4,
                5,
            ],
        }
    )

    out = gp3.compute_transition_matrix(data)

    assert out[["from", "to", "n"]].to_dict("records") == [
        {
            "from": "A",
            "to": "B",
            "n": 1,
        },
        {
            "from": "B",
            "to": "A",
            "n": 1,
        },
    ]

    assert out["prob"].tolist() == [
        1.0,
        1.0,
    ]


def test_r_transition_matrix_without_collapse():
    data = pd.DataFrame(
        {
            "MEDIA_ID": [1, 1, 1],
            "AOI": ["A", "A", "B"],
            "TIME": [1, 2, 3],
        }
    )

    out = gp3.compute_transition_matrix(
        data=data,
        collapse_repeats=False,
    )

    observed = {
        (
            row["from"],
            row["to"],
        ): (
            row["n"],
            row["prob"],
        )
        for row in out.to_dict("records")
    }

    assert observed[("A", "A")][0] == 1

    assert observed[("A", "B")][0] == 1

    assert np.isclose(
        observed[("A", "A")][1],
        0.5,
    )

    assert np.isclose(
        observed[("A", "B")][1],
        0.5,
    )


def test_r_transition_matrix_multiple_groups():
    data = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S1",
                "S2",
                "S2",
                "S2",
            ],
            "trial": [
                1,
                1,
                1,
                1,
                1,
                1,
            ],
            "AOI": [
                "A",
                "B",
                "C",
                "A",
                "C",
                "B",
            ],
            "TIME": [
                1,
                2,
                3,
                1,
                2,
                3,
            ],
        }
    )

    out = gp3.compute_transition_matrix(
        data=data,
        group_cols=[
            "subject",
            "trial",
        ],
    )

    assert set(out.columns) == {
        "subject",
        "trial",
        "from",
        "to",
        "n",
        "prob",
    }

    assert len(out) == 4

    assert (
        out.groupby(
            [
                "subject",
                "trial",
                "from",
            ]
        )["prob"]
        .sum()
        .eq(1.0)
        .all()
    )


def test_r_transition_matrix_standardises_time_name():
    data = pd.DataFrame(
        {
            "MEDIA_ID": [1, 1],
            "AOI": ["A", "B"],
            "TIME(ms)": [2, 1],
            "Unnamed: 4": [
                np.nan,
                np.nan,
            ],
        }
    )

    out = gp3.compute_transition_matrix(data=data)

    assert out.iloc[0]["from"] == "B"

    assert out.iloc[0]["to"] == "A"


def test_r_transition_matrix_filters_missing_aoi():
    data = pd.DataFrame(
        {
            "MEDIA_ID": [
                1,
                1,
                1,
                1,
            ],
            "AOI": [
                "A",
                None,
                "",
                "B",
            ],
            "TIME": [
                1,
                2,
                3,
                4,
            ],
        }
    )

    out = gp3.compute_transition_matrix(data=data)

    assert len(out) == 1
    assert out.iloc[0]["from"] == "A"
    assert out.iloc[0]["to"] == "B"


def test_r_transition_matrix_missing_columns():
    data = pd.DataFrame(
        {
            "MEDIA_ID": [1],
            "AOI": ["A"],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing columns: TIME",
    ):
        gp3.compute_transition_matrix(data=data)


def test_transition_matrix_rejects_both_interfaces():
    data = pd.DataFrame(
        {
            "MEDIA_ID": [1],
            "AOI": ["A"],
            "TIME": [1],
        }
    )

    with pytest.raises(
        TypeError,
        match="either sequence or data",
    ):
        gp3.compute_transition_matrix(
            ["A", "B"],
            data=data,
        )
