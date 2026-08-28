from __future__ import annotations

import numpy as np
import pandas as pd

import gp3tools as gp3
from gp3tools._utils import COLUMN_CANDIDATES, infer_column


def _lpmm_rpmm_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TIME": [
                0.000,
                0.016,
                0.032,
                0.048,
            ],
            "LPMM": [
                3.0,
                3.2,
                np.nan,
                3.6,
            ],
            "RPMM": [
                3.2,
                3.4,
                3.5,
                np.nan,
            ],
        }
    )


def test_native_lpmm_rpmm_alias_registry() -> None:
    assert "LPMM" in COLUMN_CANDIDATES["left_pupil"]

    assert "RPMM" in COLUMN_CANDIDATES["right_pupil"]

    assert "LPMM" in COLUMN_CANDIDATES["pupil"]

    assert "RPMM" in COLUMN_CANDIDATES["pupil"]


def test_infer_native_lpmm_rpmm_columns() -> None:
    data = _lpmm_rpmm_data()

    assert (
        infer_column(
            data,
            "left_pupil",
            required=True,
        )
        == "LPMM"
    )

    assert (
        infer_column(
            data,
            "right_pupil",
            required=True,
        )
        == "RPMM"
    )


def test_mean_gazepoint_pupil_accepts_lpmm_rpmm() -> None:
    data = _lpmm_rpmm_data()

    out = gp3.mean_gazepoint_pupil(data)

    assert "pupil_mean" in out.columns

    assert np.allclose(
        out["pupil_mean"].to_numpy(dtype=float),
        np.array(
            [
                3.1,
                3.3,
                3.5,
                3.6,
            ]
        ),
        equal_nan=True,
    )


def test_mean_gazepoint_pupil_require_both_lpmm_rpmm() -> None:
    data = _lpmm_rpmm_data()

    out = gp3.mean_gazepoint_pupil(
        data,
        require_both=True,
    )

    expected = np.array(
        [
            3.1,
            3.3,
            np.nan,
            np.nan,
        ]
    )

    assert np.allclose(
        out["pupil_mean"].to_numpy(dtype=float),
        expected,
        equal_nan=True,
    )


def test_combine_gazepoint_eyes_accepts_lpmm_rpmm() -> None:
    data = _lpmm_rpmm_data()

    out = gp3.combine_gazepoint_eyes(data)

    assert "pupil_combined" in out.columns

    assert out["pupil_eye_source"].tolist() == [
        "both",
        "both",
        "right",
        "left",
    ]

    assert np.allclose(
        out["pupil_combined"].to_numpy(dtype=float),
        np.array(
            [
                3.1,
                3.3,
                3.5,
                3.6,
            ]
        ),
        equal_nan=True,
    )


def test_existing_lpd_rpd_precedence_is_preserved() -> None:
    data = pd.DataFrame(
        {
            "LPD": [
                10.0,
            ],
            "RPD": [
                12.0,
            ],
            "LPMM": [
                3.0,
            ],
            "RPMM": [
                3.2,
            ],
        }
    )

    # Adding native millimetre aliases must not change the historical
    # precedence when the older LPD/RPD fields are also present.
    assert (
        infer_column(
            data,
            "left_pupil",
            required=True,
        )
        == "LPD"
    )

    assert (
        infer_column(
            data,
            "right_pupil",
            required=True,
        )
        == "RPD"
    )
