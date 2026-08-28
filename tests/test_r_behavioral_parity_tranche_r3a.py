from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import gp3tools

ORACLE = Path(__file__).parent / "oracles" / "r_v2_3_0_behavioral_r3a.csv"


TARGETS = [
    (
        "analyse_gazepoint_binocular_sensitivity",
        "reconstructed_grouped",
    ),
    (
        "create_gazepoint_master",
        "canonical_gaze",
    ),
    (
        "detect_gazepoint_fixations_ivt",
        "grouped_pixels_ms",
    ),
    (
        "summarise_gazepoint_binocular_reporting",
        "reconstructed_reporting",
    ),
    (
        "summarise_gazepoint_event_detector_agreement",
        "standardized_events",
    ),
    (
        "summarize_gazepoint_face_windows",
        "grouped_windows",
    ),
    (
        "summarize_gazepoint_face_reactivity",
        "face_window_summary",
    ),
]


def _scalar(
    value: Any,
) -> tuple[str, str]:
    if value is None:
        return (
            "null",
            "<NULL>",
        )

    try:
        if pd.isna(value):
            return (
                "na",
                "<NA>",
            )
    except (
        TypeError,
        ValueError,
    ):
        pass

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):
        return (
            "bool",
            ("true" if bool(value) else "false"),
        )

    if isinstance(
        value,
        (
            int,
            float,
            np.integer,
            np.floating,
        ),
    ):
        numeric = float(value)

        if math.isnan(numeric):
            return (
                "nan",
                "<NaN>",
            )

        if math.isinf(numeric):
            return (
                "numeric",
                ("Inf" if numeric > 0 else "-Inf"),
            )

        return (
            "numeric",
            format(
                numeric,
                ".17g",
            ),
        )

    return (
        "string",
        str(value),
    )


def _add(
    rows: list[
        tuple[
            str,
            str,
            str,
        ]
    ],
    path: str,
    kind: str,
    value: str,
) -> None:
    rows.append(
        (
            path,
            kind,
            value,
        )
    )


def _flatten(
    value: Any,
    path: str = "$",
) -> list[
    tuple[
        str,
        str,
        str,
    ]
]:
    rows: list[
        tuple[
            str,
            str,
            str,
        ]
    ] = []

    _flatten_into(
        value,
        rows,
        path,
    )

    return sorted(rows)


def _flatten_into(
    value: Any,
    rows: list[
        tuple[
            str,
            str,
            str,
        ]
    ],
    path: str,
) -> None:
    if value is None:
        _add(
            rows,
            path,
            "null",
            "<NULL>",
        )

        return

    if isinstance(
        value,
        pd.DataFrame,
    ):
        _add(
            rows,
            f"{path}.@kind",
            "meta",
            "table",
        )

        _add(
            rows,
            f"{path}.@nrow",
            "numeric",
            str(len(value)),
        )

        _add(
            rows,
            f"{path}.@ncol",
            "numeric",
            str(len(value.columns)),
        )

        _add(
            rows,
            f"{path}.@columns",
            "meta",
            "|".join(
                map(
                    str,
                    value.columns,
                )
            ),
        )

        for column in value.columns:
            series = value[column]

            for index, current in enumerate(
                series.tolist(),
                start=1,
            ):
                kind, rendered = _scalar(current)

                _add(
                    rows,
                    (f"{path}.{column}[{index}]"),
                    kind,
                    rendered,
                )

        attrs = getattr(
            value,
            "attrs",
            {},
        )

        if isinstance(
            attrs,
            Mapping,
        ):
            for name in sorted(attrs):
                if not str(name).startswith("gp3_"):
                    continue

                _flatten_into(
                    attrs[name],
                    rows,
                    (f"{path}.@attr.{name}"),
                )

        return

    if isinstance(
        value,
        Mapping,
    ):
        keys = list(value.keys())

        _add(
            rows,
            f"{path}.@kind",
            "meta",
            "mapping",
        )

        _add(
            rows,
            f"{path}.@length",
            "numeric",
            str(len(keys)),
        )

        _add(
            rows,
            f"{path}.@keys",
            "meta",
            "|".join(
                map(
                    str,
                    keys,
                )
            ),
        )

        for key in keys:
            _flatten_into(
                value[key],
                rows,
                (f"{path}.{key}"),
            )

        return

    if isinstance(
        value,
        pd.Series,
    ):
        value = value.tolist()

    if isinstance(
        value,
        np.ndarray,
    ):
        value = value.tolist()

    if isinstance(
        value,
        Sequence,
    ) and not isinstance(
        value,
        (
            str,
            bytes,
            bytearray,
        ),
    ):
        _add(
            rows,
            f"{path}.@kind",
            "meta",
            "sequence",
        )

        _add(
            rows,
            f"{path}.@length",
            "numeric",
            str(len(value)),
        )

        for index, current in enumerate(
            value,
            start=1,
        ):
            _flatten_into(
                current,
                rows,
                (f"{path}[{index}]"),
            )

        return

    kind, rendered = _scalar(value)

    _add(
        rows,
        path,
        kind,
        rendered,
    )


def _binocular_input() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": (["S1"] * 6 + ["S2"] * 6),
            "trial": ([1] * 6 + [2] * 6),
            "time": (
                list(
                    range(
                        0,
                        600,
                        100,
                    )
                )
                * 2
            ),
            "left": [
                3.10,
                3.20,
                np.nan,
                3.35,
                3.40,
                3.50,
                2.90,
                3.00,
                3.05,
                np.nan,
                3.20,
                3.25,
            ],
            "right": [
                3.00,
                3.15,
                3.25,
                np.nan,
                3.45,
                3.55,
                2.85,
                np.nan,
                3.10,
                3.15,
                3.22,
                3.30,
            ],
        }
    )


def _reconstructed() -> pd.DataFrame:
    return gp3tools.reconstruct_gazepoint_binocular_pupil(
        _binocular_input(),
        left_col="left",
        right_col="right",
        time_col="time",
        group_cols=[
            "subject",
            "trial",
        ],
        min_pairs=2,
        min_unique=2,
    )


def _master_input() -> pd.DataFrame:
    binocular = _binocular_input()

    return pd.DataFrame(
        {
            "USER": (["S1"] * 6 + ["S2"] * 6),
            "USER_FILE": (["S1.csv"] * 6 + ["S2.csv"] * 6),
            "MEDIA_ID": (["stimA"] * 6 + ["stimB"] * 6),
            "MEDIA_NAME": (["Stimulus A"] * 6 + ["Stimulus B"] * 6),
            "TIME": (
                [
                    0.0,
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                ]
                * 2
            ),
            "BPOGX": [
                0.20,
                0.21,
                0.22,
                0.40,
                0.41,
                0.42,
                0.30,
                0.31,
                0.32,
                0.50,
                0.51,
                0.52,
            ],
            "BPOGY": [
                0.30,
                0.31,
                0.32,
                0.50,
                0.51,
                0.52,
                0.40,
                0.41,
                0.42,
                0.60,
                0.61,
                0.62,
            ],
            "BPOGV": 1,
            "LPD": binocular["left"].to_numpy(),
            "LPDV": (binocular["left"].notna().astype(int).to_numpy()),
            "RPD": binocular["right"].to_numpy(),
            "RPDV": (binocular["right"].notna().astype(int).to_numpy()),
        }
    )


def _ivt_input() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": (["S1"] * 8 + ["S2"] * 8),
            "trial": [1] * 16,
            "time": (
                list(
                    range(
                        0,
                        800,
                        100,
                    )
                )
                * 2
            ),
            "x": [
                100,
                101,
                102,
                103,
                180,
                181,
                182,
                183,
                200,
                201,
                202,
                203,
                270,
                271,
                272,
                273,
            ],
            "y": [
                100,
                100,
                101,
                101,
                160,
                160,
                161,
                161,
                200,
                200,
                201,
                201,
                260,
                260,
                261,
                261,
            ],
        }
    )


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S1",
                "S1",
                "S2",
                "S2",
                "S2",
                "S2",
            ],
            "trial": [1] * 8,
            "detector": [
                "A",
                "A",
                "B",
                "B",
                "A",
                "A",
                "B",
                "B",
            ],
            "family": ["manual"] * 8,
            "threshold": [np.nan] * 8,
            "event_id": [
                1,
                2,
                1,
                2,
                1,
                2,
                1,
                2,
            ],
            "start_time": [
                0,
                500,
                20,
                530,
                0,
                600,
                40,
                900,
            ],
            "end_time": [
                200,
                700,
                210,
                720,
                220,
                800,
                250,
                1050,
            ],
            "duration_ms": [
                200,
                200,
                190,
                190,
                220,
                200,
                210,
                150,
            ],
            "mean_x": [
                0.20,
                0.40,
                0.21,
                0.41,
                0.30,
                0.50,
                0.31,
                0.55,
            ],
            "mean_y": [
                0.30,
                0.50,
                0.31,
                0.51,
                0.40,
                0.60,
                0.41,
                0.65,
            ],
            "n_samples": [3] * 8,
            "source_status": ["ok"] * 8,
        }
    )


def _face_input() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": (["S1"] * 8 + ["S2"] * 8),
            "trial": (["T1"] * 8 + ["T2"] * 8),
            "condition": (["A"] * 8 + ["B"] * 8),
            "time": (
                [
                    -300,
                    -200,
                    -100,
                    0,
                    100,
                    200,
                    300,
                    400,
                ]
                * 2
            ),
            "AU01": [
                0.10,
                0.12,
                0.11,
                0.20,
                0.25,
                0.30,
                0.28,
                0.26,
                0.15,
                0.14,
                0.16,
                0.24,
                0.29,
                0.33,
                0.31,
                0.30,
            ],
            "AU12": [
                0.05,
                0.04,
                0.06,
                0.12,
                0.18,
                0.20,
                0.19,
                0.17,
                0.08,
                0.07,
                0.09,
                0.16,
                0.21,
                0.24,
                0.23,
                0.22,
            ],
            "valid": [True] * 16,
            "confidence": [0.95] * 16,
        }
    )


def _windows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S2",
                "S2",
            ],
            "trial": [
                "T1",
                "T1",
                "T2",
                "T2",
            ],
            "condition": [
                "A",
                "A",
                "B",
                "B",
            ],
            "window_id": [
                "baseline",
                "response",
                "baseline",
                "response",
            ],
            "window_label": [
                "Baseline",
                "Response",
                "Baseline",
                "Response",
            ],
            "window_start": [
                -300,
                0,
                -300,
                0,
            ],
            "window_end": [
                -100,
                400,
                -100,
                400,
            ],
        }
    )


def _face_windows() -> pd.DataFrame:
    return gp3tools.summarize_gazepoint_face_windows(
        _face_input(),
        windows=_windows(),
        time_col="time",
        window_start_col="window_start",
        window_end_col="window_end",
        group_cols=[
            "subject",
            "trial",
            "condition",
        ],
        window_id_col="window_id",
        window_label_col="window_label",
        measure_cols=[
            "AU01",
            "AU12",
        ],
        validity_col="valid",
        confidence_col="confidence",
        require_valid=True,
        include_empty_windows=True,
    )


def _execute(
    function_name: str,
) -> Any:
    if function_name == "analyse_gazepoint_binocular_sensitivity":
        return gp3tools.analyse_gazepoint_binocular_sensitivity(
            _reconstructed(),
            left_col="left",
            right_col="right",
            group_cols=[
                "subject",
                "trial",
            ],
        )

    if function_name == "create_gazepoint_master":
        return gp3tools.create_gazepoint_master(_master_input())

    if function_name == "detect_gazepoint_fixations_ivt":
        return gp3tools.detect_gazepoint_fixations_ivt(
            _ivt_input(),
            x_col="x",
            y_col="y",
            time_col="time",
            group_cols=[
                "subject",
                "trial",
            ],
            velocity_threshold=1,
            min_duration_ms=100,
            distance_scale=1,
            time_scale=1,
        )

    if function_name == "summarise_gazepoint_binocular_reporting":
        return gp3tools.summarise_gazepoint_binocular_reporting(
            _reconstructed(),
            by="subject",
            prefix="gp3_binocular",
        )

    if function_name == "summarise_gazepoint_event_detector_agreement":
        return gp3tools.summarise_gazepoint_event_detector_agreement(
            _events(),
            min_overlap=0.5,
        )

    if function_name == "summarize_gazepoint_face_windows":
        return _face_windows()

    if function_name == "summarize_gazepoint_face_reactivity":
        return gp3tools.summarize_gazepoint_face_reactivity(
            _face_windows(),
            baseline_window="baseline",
            response_window="response",
            group_cols=[
                "subject",
                "trial",
                "condition",
            ],
            window_col="face_window_id",
            measure_cols=[
                "AU01",
                "AU12",
            ],
            statistic="mean",
        )

    raise AssertionError(f"Unhandled target: {function_name}")


def _expected_rows(
    function_name: str,
    fixture_id: str,
) -> list[
    tuple[
        str,
        str,
        str,
    ]
]:
    oracle = pd.read_csv(
        ORACLE,
        dtype=str,
        keep_default_na=False,
    )

    block = oracle[
        (oracle["function_name"] == function_name) & (oracle["fixture_id"] == fixture_id)
    ]

    if block.empty:
        raise AssertionError(f"Oracle block missing for {function_name}/{fixture_id}")

    return sorted(
        zip(
            block["path"],
            block["kind"],
            block["value"],
            strict=True,
        )
    )


def _compare(
    expected: list[
        tuple[
            str,
            str,
            str,
        ]
    ],
    actual: list[
        tuple[
            str,
            str,
            str,
        ]
    ],
) -> None:
    expected_map = {
        path: (
            kind,
            value,
        )
        for (
            path,
            kind,
            value,
        ) in expected
    }

    actual_map = {
        path: (
            kind,
            value,
        )
        for (
            path,
            kind,
            value,
        ) in actual
    }

    missing = sorted(set(expected_map) - set(actual_map))

    extra = sorted(set(actual_map) - set(expected_map))

    differences = []

    for path in sorted(set(expected_map) & set(actual_map)):
        expected_kind, expected_value = expected_map[path]

        actual_kind, actual_value = actual_map[path]

        if expected_kind == "numeric" and actual_kind == "numeric":
            if expected_value in {
                "Inf",
                "-Inf",
            }:
                if expected_value != actual_value:
                    differences.append(
                        (
                            path,
                            expected_kind,
                            expected_value,
                            actual_kind,
                            actual_value,
                        )
                    )

                continue

            try:
                expected_number = float(expected_value)

                actual_number = float(actual_value)
            except ValueError:
                differences.append(
                    (
                        path,
                        expected_kind,
                        expected_value,
                        actual_kind,
                        actual_value,
                    )
                )

                continue

            if not math.isclose(
                expected_number,
                actual_number,
                rel_tol=1e-8,
                abs_tol=1e-10,
            ):
                differences.append(
                    (
                        path,
                        expected_kind,
                        expected_value,
                        actual_kind,
                        actual_value,
                    )
                )

            continue

        if expected_kind != actual_kind or expected_value != actual_value:
            differences.append(
                (
                    path,
                    expected_kind,
                    expected_value,
                    actual_kind,
                    actual_value,
                )
            )

    if missing or extra or differences:
        lines = []

        if missing:
            lines.append("MISSING PATHS:")

            lines.extend(f"  {path}" for path in missing[:25])

        if extra:
            lines.append("EXTRA PATHS:")

            lines.extend(f"  {path}" for path in extra[:25])

        if differences:
            lines.append("VALUE/KIND DIFFERENCES:")

            for (
                path,
                expected_kind,
                expected_value,
                actual_kind,
                actual_value,
            ) in differences[:40]:
                lines.append(
                    f"  {path}: "
                    f"R=({expected_kind}, "
                    f"{expected_value!r}) "
                    f"PY=({actual_kind}, "
                    f"{actual_value!r})"
                )

        lines.append("")

        lines.append(
            f"Counts: missing={len(missing)}, extra={len(extra)}, different={len(differences)}"
        )

        raise AssertionError("\n".join(lines))


@pytest.mark.parametrize(
    (
        "function_name",
        "fixture_id",
    ),
    TARGETS,
)
def test_r3a_canonical_r_parity(
    function_name: str,
    fixture_id: str,
) -> None:
    expected = _expected_rows(
        function_name,
        fixture_id,
    )

    actual = _flatten(_execute(function_name))

    _compare(
        expected,
        actual,
    )
