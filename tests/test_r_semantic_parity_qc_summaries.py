import pandas as pd
import pytest

import gp3tools as gp3


def test_legacy_collect_qc_summaries_preserved():
    data = gp3.load_example_master()

    out = gp3.collect_gazepoint_qc_summaries(data)

    assert set(out) == {
        "master",
        "tracking",
        "missingness",
        "screen",
    }


def test_r_qc_bundle_single_dataframe():
    overview = pd.DataFrame(
        {
            "status": [
                "pass",
                "pass",
            ],
            "message": [
                "all good",
                "all good",
            ],
        }
    )

    out = gp3.collect_gazepoint_qc_summaries(objects=overview)

    assert set(out) == {
        "overview",
        "object_summary",
        "status_counts",
        "overview_rows",
        "settings",
    }

    summary = out["object_summary"].iloc[0]

    assert summary["object_name"] == "object_1"

    assert bool(summary["overview_available"])

    assert summary["n_overview_rows"] == 2

    assert summary["qc_status"] == "pass"

    assert summary["qc_message"] == "all good"

    assert out["overview"].iloc[0]["qc_bundle_status"] == "pass"


def test_r_qc_bundle_named_objects_and_worst_status():
    objects = {
        "clean": {
            "overview": pd.DataFrame(
                {
                    "decision": ["ready"],
                    "note": ["usable"],
                }
            )
        },
        "problem": {
            "overview": pd.DataFrame(
                {
                    "status": ["failed"],
                    "reason": ["tracking loss"],
                }
            )
        },
    }

    out = gp3.collect_gazepoint_qc_summaries(
        objects=objects,
        name="bundle",
    )

    summary = out["object_summary"]

    assert summary["object_name"].tolist() == [
        "clean",
        "problem",
    ]

    assert summary["qc_status"].tolist() == [
        "pass",
        "fail",
    ]

    overall = out["overview"].iloc[0]

    assert overall["object_name"] == "bundle"

    assert overall["n_objects"] == 2

    assert overall["n_pass"] == 1

    assert overall["n_fail"] == 1

    assert overall["qc_bundle_status"] == "fail"


def test_r_qc_boolean_status_rules():
    objects = {
        "warn": pd.DataFrame({"review_flag": [True]}),
        "fail": pd.DataFrame({"ready": [False]}),
    }

    out = gp3.collect_gazepoint_qc_summaries(objects=objects)

    assert out["object_summary"]["qc_status"].tolist() == [
        "warn",
        "fail",
    ]


def test_r_qc_unknown_object():
    out = gp3.collect_gazepoint_qc_summaries(objects=[123])

    summary = out["object_summary"].iloc[0]

    assert not bool(summary["overview_available"])

    assert summary["qc_status"] == "unknown"

    assert summary["qc_message"] == "Object had no interpretable overview data frame."


def test_r_qc_status_counts_have_fixed_levels():
    objects = {
        "pass": pd.DataFrame({"status": ["ok"]}),
        "warn": pd.DataFrame({"status": ["warning"]}),
        "unknown": pd.DataFrame({"value": [1]}),
    }

    out = gp3.collect_gazepoint_qc_summaries(objects=objects)

    counts = out["status_counts"]

    assert counts["qc_status"].tolist() == [
        "pass",
        "warn",
        "fail",
        "info",
        "unknown",
    ]

    assert counts["n_objects"].tolist() == [
        1,
        1,
        0,
        0,
        1,
    ]

    assert out["overview"].iloc[0]["qc_bundle_status"] == "warn"


def test_r_qc_overview_rows_metadata_and_union():
    objects = {
        "first": pd.DataFrame(
            {
                "status": ["pass"],
                "alpha": [1],
            }
        ),
        "second": pd.DataFrame(
            {
                "decision": ["ready"],
                "beta": [2],
            }
        ),
    }

    out = gp3.collect_gazepoint_qc_summaries(objects=objects)

    rows = out["overview_rows"]

    assert rows.columns[:3].tolist() == [
        ".gp3_qc_object_name",
        ".gp3_qc_object_index",
        ".gp3_qc_row",
    ]

    assert rows[".gp3_qc_object_name"].tolist() == [
        "first",
        "second",
    ]

    assert {
        "status",
        "alpha",
        "decision",
        "beta",
    }.issubset(rows.columns)


def test_r_qc_can_omit_overview_rows():
    out = gp3.collect_gazepoint_qc_summaries(
        objects=pd.DataFrame({"status": ["pass"]}),
        include_overview_rows=False,
    )

    assert out["overview_rows"].empty

    settings = dict(
        zip(
            out["settings"]["setting"],
            out["settings"]["value"],
            strict=True,
        )
    )

    assert settings["include_overview_rows"] == "FALSE"


def test_r_qc_object_names_override():
    out = gp3.collect_gazepoint_qc_summaries(
        objects=[
            pd.DataFrame({"status": ["pass"]}),
            pd.DataFrame({"status": ["warn"]}),
        ],
        object_names=[
            "one",
            "two",
        ],
    )

    assert out["object_summary"]["object_name"].tolist() == [
        "one",
        "two",
    ]


def test_r_qc_message_uses_first_three_unique_values():
    overview = pd.DataFrame(
        {
            "status": [
                "warn",
                "warn",
                "warn",
                "warn",
            ],
            "reason": [
                "A",
                "B",
                "A",
                "C",
            ],
            "note": [
                "D",
                "E",
                "F",
                "G",
            ],
        }
    )

    out = gp3.collect_gazepoint_qc_summaries(objects=overview)

    assert out["object_summary"].iloc[0]["qc_message"] == "A | B | C"


def test_r_qc_validation():
    with pytest.raises(
        ValueError,
        match="at least one object",
    ):
        gp3.collect_gazepoint_qc_summaries(objects=[])

    with pytest.raises(
        ValueError,
        match="object_names",
    ):
        gp3.collect_gazepoint_qc_summaries(
            objects=[pd.DataFrame({"status": ["pass"]})],
            object_names=[
                "one",
                "two",
            ],
        )

    with pytest.raises(
        ValueError,
        match="include_overview_rows",
    ):
        gp3.collect_gazepoint_qc_summaries(
            objects=pd.DataFrame({"status": ["pass"]}),
            include_overview_rows="yes",
        )
