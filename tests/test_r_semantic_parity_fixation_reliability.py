import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3


def _count_fixture():
    rows = []

    for subject_index, subject in enumerate(
        ["S1", "S2", "S3"],
        start=1,
    ):
        for trial in range(1, 5):
            n = subject_index * trial

            for _ in range(n):
                rows.append(
                    {
                        "subject": subject,
                        "trial": trial,
                    }
                )

    return pd.DataFrame(rows)


def test_legacy_fixation_reliability_preserved():
    data = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S2",
                "S2",
            ],
            "duration_ms": [
                100.0,
                200.0,
                300.0,
                500.0,
            ],
        }
    )

    out = gp3.audit_gazepoint_fixation_reliability(data)

    assert out["subject"].tolist() == [
        "S1",
        "S2",
    ]

    assert out["n"].tolist() == [
        2,
        2,
    ]


def test_r_fixation_count_split_half_reliability():
    out = gp3.audit_gazepoint_fixation_reliability(
        _count_fixture(),
        subject_col="subject",
        trial_col="trial",
        metric="fixation_count",
        min_trials=4,
    )

    row = out.iloc[0]

    assert row["metric"] == "fixation_count"
    assert row["reliability_status"] == "ok"
    assert row["n_subjects_total"] == 3
    assert row["n_subjects_used"] == 3
    assert row["n_trials"] == 12

    assert np.isclose(
        row["split_half_r"],
        1.0,
    )

    assert np.isclose(
        row["spearman_brown"],
        1.0,
    )


def test_r_duration_metric():
    rows = []

    for subject_index, subject in enumerate(
        ["S1", "S2", "S3"],
        start=1,
    ):
        for trial in range(1, 5):
            value = subject_index * trial * 10.0

            rows.extend(
                [
                    {
                        "subject": subject,
                        "trial": trial,
                        "duration": value,
                    },
                    {
                        "subject": subject,
                        "trial": trial,
                        "duration": value,
                    },
                ]
            )

    data = pd.DataFrame(rows)

    out = gp3.audit_gazepoint_fixation_reliability(
        data,
        subject_col="subject",
        trial_col="trial",
        metric="mean_fixation_duration",
        duration_col="duration",
    )

    row = out.iloc[0]

    assert row["reliability_status"] == "ok"
    assert np.isclose(
        row["split_half_r"],
        1.0,
    )


def test_r_aoi_dwell_prop_metric():
    rows = []

    for subject in [
        "S1",
        "S2",
        "S3",
    ]:
        for trial in range(1, 5):
            rows.extend(
                [
                    {
                        "subject": subject,
                        "trial": trial,
                        "aoi": "target",
                    },
                    {
                        "subject": subject,
                        "trial": trial,
                        "aoi": "other",
                    },
                ]
            )

    out = gp3.audit_gazepoint_fixation_reliability(
        pd.DataFrame(rows),
        subject_col="subject",
        trial_col="trial",
        metric="aoi_dwell_prop",
        aoi_col="aoi",
        target_aoi="target",
    )

    assert out.iloc[0]["reliability_status"] == "no_variance"


def test_r_transition_count_metric():
    rows = []

    sequence = [
        "A",
        "A",
        "B",
        "B",
        "C",
    ]

    for subject in [
        "S1",
        "S2",
        "S3",
    ]:
        for trial in range(1, 5):
            for time, aoi in enumerate(sequence):
                rows.append(
                    {
                        "subject": subject,
                        "trial": trial,
                        "time": time,
                        "aoi": aoi,
                    }
                )

    out = gp3.audit_gazepoint_fixation_reliability(
        pd.DataFrame(rows),
        subject_col="subject",
        trial_col="trial",
        metric="transition_count",
        aoi_col="aoi",
        time_col="time",
    )

    assert out.iloc[0]["reliability_status"] == "no_variance"


def test_r_entropy_metric():
    rows = []

    for subject in [
        "S1",
        "S2",
        "S3",
    ]:
        for trial in range(1, 5):
            for time, aoi in enumerate(["A", "A", "B", "B"]):
                rows.append(
                    {
                        "subject": subject,
                        "trial": trial,
                        "time": time,
                        "aoi": aoi,
                    }
                )

    out = gp3.audit_gazepoint_fixation_reliability(
        pd.DataFrame(rows),
        subject_col="subject",
        trial_col="trial",
        metric="entropy_score",
        aoi_col="aoi",
        time_col="time",
    )

    assert out.iloc[0]["reliability_status"] == "no_variance"


def test_r_fixation_reliability_too_few_subjects():
    data = _count_fixture().query("subject != 'S3'")

    out = gp3.audit_gazepoint_fixation_reliability(
        data,
        subject_col="subject",
        trial_col="trial",
        min_trials=4,
    )

    row = out.iloc[0]

    assert row["reliability_status"] == "too_few_subjects"

    assert row["n_subjects_used"] == 2


def test_r_fixation_reliability_grouping():
    data = _count_fixture()

    data = pd.concat(
        [
            data.assign(condition="A"),
            data.assign(condition="B"),
        ],
        ignore_index=True,
    )

    out = gp3.audit_gazepoint_fixation_reliability(
        data,
        subject_col="subject",
        trial_col="trial",
        group_cols=["condition"],
    )

    assert set(out["condition"]) == {
        "A",
        "B",
    }

    assert len(out) == 2


def test_r_random_split_is_reproducible():
    data = _count_fixture()

    first = gp3.audit_gazepoint_fixation_reliability(
        data,
        subject_col="subject",
        trial_col="trial",
        split_method="random",
        seed=123,
        correlation_method="spearman",
    )

    second = gp3.audit_gazepoint_fixation_reliability(
        data,
        subject_col="subject",
        trial_col="trial",
        split_method="random",
        seed=123,
        correlation_method="spearman",
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )


def test_r_fixation_reliability_required_arguments():
    data = pd.DataFrame(
        {
            "subject": ["S1"],
            "trial": [1],
            "aoi": ["A"],
        }
    )

    with pytest.raises(
        ValueError,
        match="duration_col",
    ):
        gp3.audit_gazepoint_fixation_reliability(
            data,
            subject_col="subject",
            trial_col="trial",
            metric="mean_fixation_duration",
        )

    with pytest.raises(
        ValueError,
        match="target_aoi",
    ):
        gp3.audit_gazepoint_fixation_reliability(
            data,
            subject_col="subject",
            trial_col="trial",
            metric="aoi_dwell_prop",
            aoi_col="aoi",
        )
