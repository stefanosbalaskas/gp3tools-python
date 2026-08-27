from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import gp3tools as gp3

ORACLE = Path(__file__).parent / "oracles" / "r_v2_3_0_behavioral_r3b.csv"


REPORT_ORACLE = Path(__file__).parent / "oracles" / "r_v2_3_0_behavioral_r3b_report_current.json"


def _rows(function):
    with ORACLE.open(
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        return [row for row in csv.DictReader(handle) if row["function"] == function]


def _lookup(function):
    return {row["path"]: row for row in _rows(function)}


def _parse(row):
    value = row["value"]

    kind = row["type"]

    if value == "<NA>":
        return np.nan

    if value == "<NULL>":
        return None

    if kind == "integer":
        return int(float(value))

    if kind in {
        "double",
        "numeric",
    }:
        return float(value)

    if kind == "logical":
        return value == "TRUE"

    return value


def _oracle_frame(
    function,
    prefix,
):
    lookup = _lookup(function)

    columns = lookup[f"{prefix}/@columns"]["value"].split("|")

    nrow = int(float(lookup[f"{prefix}/@nrow"]["value"]))

    rows = []

    for index in range(
        1,
        nrow + 1,
    ):
        record = {}

        for column in columns:
            record[column] = _parse(lookup[(f"{prefix}/row[{index}]/{column}")])

        rows.append(record)

    return pd.DataFrame(
        rows,
        columns=columns,
    )


def _assert_frame(
    actual,
    expected,
):
    assert list(actual.columns) == list(expected.columns)

    assert actual.shape == expected.shape

    def missing(value):
        try:
            result = pd.isna(value)

            if isinstance(
                result,
                (
                    bool,
                    np.bool_,
                ),
            ):
                return bool(result)

        except Exception:
            pass

        return False

    for row_index in range(len(expected)):
        for column in expected.columns:
            left = expected.iloc[row_index][column]

            right = actual.iloc[row_index][column]

            if missing(left) and missing(right):
                continue

            numeric_types = (
                int,
                float,
                np.integer,
                np.floating,
            )

            if (
                isinstance(
                    left,
                    numeric_types,
                )
                and isinstance(
                    right,
                    numeric_types,
                )
                and not isinstance(
                    left,
                    (
                        bool,
                        np.bool_,
                    ),
                )
                and not isinstance(
                    right,
                    (
                        bool,
                        np.bool_,
                    ),
                )
            ):
                assert np.isclose(
                    float(left),
                    float(right),
                    rtol=1e-10,
                    atol=1e-12,
                    equal_nan=True,
                ), (
                    row_index,
                    column,
                    left,
                    right,
                )

            else:
                assert left == right, (
                    row_index,
                    column,
                    left,
                    right,
                )


def _report_results():
    return {
        "all_gaze": pd.DataFrame(
            {
                "row": [
                    "a",
                    "b",
                ]
            }
        ),
        "all_fix": pd.DataFrame(
            {
                "row": [
                    "a",
                    "b",
                ]
            }
        ),
        "sampling": pd.DataFrame(
            {
                "recording": [
                    "S1",
                    "S2",
                ],
                "rate": [
                    "60",
                    "58",
                ],
            }
        ),
        "quality": pd.DataFrame(
            {
                "recording": [
                    "S1",
                    "S2",
                ],
                "status": [
                    "ok",
                    "ok",
                ],
            }
        ),
        "flagged_quality": pd.DataFrame(
            {
                "recording": [
                    "S1",
                    "S2",
                ],
                "review_required": [
                    False,
                    False,
                ],
            }
        ),
        "aoi_table": pd.DataFrame(
            {
                "AOI": [
                    "target",
                    "distractor",
                ],
                "count": [
                    "10",
                    "5",
                ],
            }
        ),
    }


def test_r3b_report_current_r230_byte_parity(
    tmp_path,
):
    expected = json.loads(REPORT_ORACLE.read_text(encoding="utf-8"))

    output = tmp_path / "canonical_current_report.html"

    result = gp3.create_gazepoint_report(
        _report_results(),
        output_file=output,
        title="R3-B Current Canonical Report",
        overwrite=True,
        max_rows=30,
        save_plots=False,
        plot_dir=None,
    )

    assert isinstance(
        result,
        pd.DataFrame,
    )

    assert list(result.columns) == [
        "report",
        "plot_dir",
        "n_flagged",
    ]

    assert (
        result.loc[
            0,
            "n_flagged",
        ]
        == 0
    )

    assert (
        Path(
            result.loc[
                0,
                "plot_dir",
            ]
        ).name
        == expected["expected_plot_dir_basename"]
    )

    text = (
        output.read_text(encoding="utf-8")
        .replace(
            "\r\n",
            "\n",
        )
        .replace(
            "\r",
            "\n",
        )
    )

    assert len(text.splitlines()) == expected["report_lines"]

    assert hashlib.md5(text.encode("utf-8")).hexdigest() == expected["normalized_md5"]


def test_r3b_divergence_frozen_oracle():
    function = "estimate_gazepoint_divergence_point"

    observed = _oracle_frame(
        function,
        "result/observed_curve",
    )

    records = []

    for _, row in observed.iterrows():
        for subject in range(
            1,
            int(row["n"]) + 1,
        ):
            records.append(
                {
                    "subject": f"S{subject}",
                    "trial": 1,
                    "condition": row["condition"],
                    "time": float(row["time"]),
                    "outcome": float(row["estimate"]),
                }
            )

    result = gp3.estimate_gazepoint_divergence_point(
        pd.DataFrame(records),
        value_col="outcome",
        time_col="time",
        condition_col="condition",
        outcome_col="outcome",
        participant_col="subject",
        trial_col="trial",
        comparison=[
            "control",
            "treatment",
        ],
        bootstrap_unit="participant",
        summary_function="mean",
        n_boot=40,
        ci=0.8,
        consecutive_points=2,
        null_value=0,
        min_abs_difference=0,
        direction="two_sided",
        seed=2401,
        keep_bootstrap=False,
        name="gazepoint_divergence_point",
    )

    for key in (
        "overview",
        "divergence_point",
        "observed_curve",
        "difference_summary",
        "bootstrap_onsets",
        "settings",
    ):
        _assert_frame(
            result[key],
            _oracle_frame(
                function,
                f"result/{key}",
            ),
        )


def test_r3b_aoi_multiverse_frozen_oracle():
    function = "run_gazepoint_aoi_multiverse"

    data = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
            ],
            "condition": [
                "control",
                "control",
            ],
            "trial": [
                1,
                1,
            ],
            "time": [
                0.0,
                0.1,
            ],
            "aoi": [
                "target",
                "distractor",
            ],
        }
    )

    result = gp3.run_gazepoint_aoi_multiverse(
        data,
        multiverse=(gp3.create_gazepoint_preprocessing_multiverse()),
        windows={
            "early": [
                0.0,
                0.2,
            ],
            "late": [
                0.3,
                0.5,
            ],
        },
        time_col="time",
        aoi_col="aoi",
        subject_col="subject",
        condition_col="condition",
        group_cols=[
            "subject",
            "condition",
            "trial",
        ],
        target_aoi_values=[
            "target",
        ],
        distractor_aoi_values=[
            "distractor",
        ],
        success_col="n_target_samples",
        outcome_label="target",
        keep_outputs=False,
        stop_on_error=False,
    )

    for key in (
        "overview",
        "branch_results",
        "settings",
    ):
        _assert_frame(
            result[key],
            _oracle_frame(
                function,
                f"result/{key}",
            ),
        )


def test_r3b_pupil_multiverse_frozen_oracle():
    function = "run_gazepoint_pupil_multiverse"

    data = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
            ],
            "condition": [
                "control",
                "control",
            ],
            "trial": [
                1,
                1,
            ],
            "time": [
                0.0,
                0.1,
            ],
            "pupil": [
                3.0,
                3.1,
            ],
        }
    )

    result = gp3.run_gazepoint_pupil_multiverse(
        data,
        multiverse=(gp3.create_gazepoint_preprocessing_multiverse()),
        pupil_col="pupil",
        time_col="time",
        group_cols=[
            "subject",
            "condition",
            "trial",
        ],
        summarise_windows=False,
        windows=None,
        keep_outputs=False,
        stop_on_error=False,
    )

    for key in (
        "overview",
        "branch_results",
        "settings",
    ):
        _assert_frame(
            result[key],
            _oracle_frame(
                function,
                f"result/{key}",
            ),
        )


def test_r3b_workflow_frozen_oracle(
    tmp_path,
):
    function = "run_gazepoint_workflow"

    expected = {
        key: _oracle_frame(
            function,
            f"result/{key}",
        )
        for key in (
            "all_gaze",
            "all_fix",
            "sampling",
            "quality",
            "flagged_quality",
            "aoi_table",
        )
    }

    input_dir = tmp_path / "input"

    input_dir.mkdir()

    expected["all_gaze"].to_csv(
        input_dir / "canonical_all_gaze.csv",
        index=False,
    )

    expected["all_fix"].to_csv(
        input_dir / "canonical_fixations.csv",
        index=False,
    )

    result = gp3.run_gazepoint_workflow(
        export_dir=input_dir,
        all_gaze_pattern="all_gaze",
        fixation_pattern="fixations",
        check_file_pairs=False,
        group_cols=[
            "USER",
            "MEDIA_ID",
        ],
        user_col="USER",
        sample_rate=60,
        min_gaze_valid_pct=70,
        min_pupil_valid_pct=70,
        expected_hz=60,
        hz_tolerance=5,
        min_duration_sec=None,
        output_dir=None,
        prefix="gazepoint",
        overwrite=True,
        save_plots=False,
        create_report=False,
    )

    for key, frame in expected.items():
        _assert_frame(
            result[key],
            frame,
        )
