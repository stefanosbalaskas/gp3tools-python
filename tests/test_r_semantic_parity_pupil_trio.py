import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3


def pupil_fixture():
    return pd.DataFrame(
        {
            "USER_ID": [
                "A",
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
                "B",
            ],
            "TIME": [
                0.0,
                0.1,
                0.2,
                0.3,
                0.0,
                0.1,
                0.2,
                0.3,
            ],
            "LPupil": [
                2.0,
                3.0,
                4.0,
                5.0,
                10.0,
                11.0,
                12.0,
                13.0,
            ],
            "RPupil": [
                4.0,
                6.0,
                8.0,
                10.0,
                20.0,
                22.0,
                24.0,
                26.0,
            ],
            "FPOGX": [
                0.0,
                1.0,
                2.0,
                3.0,
                10.0,
                11.0,
                12.0,
                13.0,
            ],
            "FPOGY": [
                1.0,
                2.0,
                3.0,
                4.0,
                20.0,
                21.0,
                22.0,
                23.0,
            ],
            "marker": list("abcdefgh"),
        }
    )


def test_downsample_legacy_python_interface():
    data = pd.DataFrame(
        {
            "TIME": [
                0.00,
                0.01,
                0.04,
            ],
            "pupil": [
                2.0,
                4.0,
                8.0,
            ],
        }
    )

    out = gp3.downsample_gazepoint_pupil(
        data,
        time_col="TIME",
        pupil_col="pupil",
        target_hz=25,
    )

    assert isinstance(
        out,
        pd.DataFrame,
    )

    assert {
        "TIME",
        "pupil",
    }.issubset(out.columns)


def test_downsample_r_mean_and_metadata():
    out = gp3.downsample_gazepoint_pupil(
        master_df=pupil_fixture(),
        factor=2,
        pupil_cols=[
            "LPupil",
            "RPupil",
        ],
    )

    assert len(out) == 4

    assert np.allclose(
        out["LPupil"],
        [
            2.5,
            4.5,
            10.5,
            12.5,
        ],
    )

    assert np.allclose(
        out["RPupil"],
        [
            5.0,
            9.0,
            21.0,
            25.0,
        ],
    )

    assert np.allclose(
        out["TIME"],
        [
            0.05,
            0.25,
            0.05,
            0.25,
        ],
    )

    assert out["n_samples_aggregated"].eq(2).all()

    assert out["downsample_factor"].eq(2).all()

    assert "downsample_bin" not in out.columns

    assert out.attrs["gazepoint_downsampling"]["method"] == "mean"


def test_downsample_r_first_and_keep_bin():
    out = gp3.downsample_gazepoint_pupil(
        master_df=pupil_fixture(),
        factor=3,
        pupil_cols=["LPupil"],
        method="first",
        keep_bin=True,
    )

    assert out["LPupil"].tolist() == [
        2.0,
        5.0,
        10.0,
        13.0,
    ]

    assert out["marker"].tolist() == [
        "a",
        "d",
        "e",
        "h",
    ]

    assert out["downsample_bin"].tolist() == [
        1,
        2,
        1,
        2,
    ]


def test_downsample_r_detects_first_pupil_column():
    data = pupil_fixture()

    out = gp3.downsample_gazepoint_pupil(
        master_df=data,
        factor=2,
    )

    assert out.attrs["gazepoint_downsampling"]["pupil_cols"] == ["LPupil"]


def test_downsample_r_timestamp_sorting():
    data = pupil_fixture().iloc[:4].copy()

    data["TIME"] = [
        3,
        1,
        4,
        2,
    ]

    out = gp3.downsample_gazepoint_pupil(
        master_df=data,
        factor=2,
        pupil_cols=["LPupil"],
        method="first",
        keep_bin=True,
    )

    assert out["TIME"].tolist() == [
        1,
        3,
    ]

    assert out["LPupil"].tolist() == [
        3.0,
        2.0,
    ]


def test_downsample_r_null_timestamp():
    out = gp3.downsample_gazepoint_pupil(
        pupil_fixture(),
        factor=2,
        pupil_cols=["LPupil"],
        ts_col=None,
    )

    assert len(out) == 4


@pytest.mark.parametrize(
    "factor",
    [
        0,
        1.5,
        np.inf,
        True,
    ],
)
def test_downsample_r_factor_validation(
    factor,
):
    with pytest.raises(
        ValueError,
        match="factor",
    ):
        gp3.downsample_gazepoint_pupil(
            master_df=pupil_fixture(),
            factor=factor,
            pupil_cols=["LPupil"],
        )


def test_downsample_r_missing_pupil_detection():
    data = pupil_fixture().drop(
        columns=[
            "LPupil",
            "RPupil",
        ]
    )

    with pytest.raises(
        ValueError,
        match="No pupil column",
    ):
        gp3.downsample_gazepoint_pupil(
            master_df=data,
        )


def test_regress_legacy_python_interface():
    out = gp3.regress_gazepoint_pupils(
        pupil_fixture(),
        left_col="LPupil",
        right_col="RPupil",
    )

    assert isinstance(out, dict)
    assert out["n"] == 8
    assert np.isclose(
        out["slope"],
        2.0,
    )


def test_regress_r_fallback():
    out = gp3.regress_gazepoint_pupils(
        master_df=pupil_fixture(),
        min_complete=10,
    )

    expected = pupil_fixture()[["LPupil", "RPupil"]].mean(axis=1)

    assert np.allclose(
        out["pupil_regressed"],
        expected,
    )

    assert out["pupil_regression_method"].eq("binocular_mean_fallback").all()

    assert out["pupil_regression_n"].eq(4).all()


@pytest.mark.parametrize(
    "direction",
    [
        "right_on_left",
        "left_on_right",
        "bidirectional",
    ],
)
def test_regress_r_fitted_directions(
    direction,
):
    out = gp3.regress_gazepoint_pupils(
        master_df=pupil_fixture(),
        direction=direction,
        min_complete=2,
    )

    assert out["pupil_regression_method"].eq(direction).all()

    assert np.nanmax(np.abs(out["pupil_regression_residual"])) < 1e-10


def test_regress_r_missing_eye_prediction():
    data = pupil_fixture().iloc[:4].copy()

    data.loc[
        data.index[0],
        "RPupil",
    ] = np.nan

    out = gp3.regress_gazepoint_pupils(
        master_df=data,
        direction="right_on_left",
        min_complete=2,
    )

    assert np.isfinite(
        out.loc[
            data.index[0],
            "pupil_regressed",
        ]
    )


def test_regress_r_group_specific_fits():
    data = pupil_fixture().copy()

    data.loc[
        data["USER_ID"].eq("B"),
        "RPupil",
    ] = (
        3
        * data.loc[
            data["USER_ID"].eq("B"),
            "LPupil",
        ]
    )

    out = gp3.regress_gazepoint_pupils(
        master_df=data,
        direction="right_on_left",
        min_complete=2,
    )

    assert np.allclose(
        out.loc[
            data["USER_ID"].eq("A"),
            "pupil_regressed",
        ],
        2
        * data.loc[
            data["USER_ID"].eq("A"),
            "LPupil",
        ],
    )

    assert np.allclose(
        out.loc[
            data["USER_ID"].eq("B"),
            "pupil_regressed",
        ],
        3
        * data.loc[
            data["USER_ID"].eq("B"),
            "LPupil",
        ],
    )


@pytest.mark.parametrize(
    "min_complete",
    [
        0,
        1,
        2.5,
        np.inf,
        True,
    ],
)
def test_regress_r_min_complete_validation(
    min_complete,
):
    with pytest.raises(
        ValueError,
        match="min_complete",
    ):
        gp3.regress_gazepoint_pupils(
            master_df=pupil_fixture(),
            min_complete=min_complete,
        )


def test_regress_r_missing_columns():
    with pytest.raises(
        ValueError,
        match="missing required",
    ):
        gp3.regress_gazepoint_pupils(
            master_df=pupil_fixture().drop(columns=["LPupil"]),
            min_complete=2,
        )


def test_smooth_legacy_python_interface():
    data = pd.DataFrame(
        {
            "x": [
                1.0,
                2.0,
                3.0,
            ]
        }
    )

    out = gp3.smooth_gazepoint_coordinate(
        data,
        column="x",
        window=3,
    )

    assert "x_smoothed" in out


def test_smooth_r_mean_exact_window():
    data = pupil_fixture().iloc[:4].copy()

    out = gp3.smooth_gazepoint_coordinate(
        all_gaze=data,
        method="mean",
        window=3,
    )

    assert np.allclose(
        out["FPOGX_smooth"],
        [
            0.5,
            1.0,
            2.0,
            2.5,
        ],
    )

    assert np.allclose(
        out["FPOGY_smooth"],
        [
            1.5,
            2.0,
            3.0,
            3.5,
        ],
    )


def test_smooth_r_even_window_matches_r_widths():
    data = pupil_fixture().iloc[:4].copy()

    out = gp3.smooth_gazepoint_coordinate(
        all_gaze=data,
        method="mean",
        window=4,
    )

    assert np.allclose(
        out["FPOGX_smooth"],
        [
            1.0,
            1.5,
            2.0,
            2.5,
        ],
    )


def test_smooth_r_group_boundaries():
    out = gp3.smooth_gazepoint_coordinate(
        all_gaze=pupil_fixture(),
        method="mean",
        window=3,
    )

    assert np.isclose(
        out.loc[
            3,
            "FPOGX_smooth",
        ],
        2.5,
    )

    assert np.isclose(
        out.loc[
            4,
            "FPOGX_smooth",
        ],
        10.5,
    )


def test_smooth_r_preserve_missing():
    data = pupil_fixture().iloc[:4].copy()

    data.loc[
        1,
        "FPOGX",
    ] = np.nan

    preserved = gp3.smooth_gazepoint_coordinate(
        all_gaze=data,
        method="mean",
        window=3,
        preserve_missing=True,
    )

    filled = gp3.smooth_gazepoint_coordinate(
        all_gaze=data,
        method="mean",
        window=3,
        preserve_missing=False,
    )

    assert np.isnan(
        preserved.loc[
            1,
            "FPOGX_smooth",
        ]
    )

    assert np.isfinite(
        filled.loc[
            1,
            "FPOGX_smooth",
        ]
    )


def test_smooth_r_min_valid():
    data = pupil_fixture().iloc[:4].copy()

    data.loc[
        [0, 1],
        "FPOGX",
    ] = np.nan

    out = gp3.smooth_gazepoint_coordinate(
        all_gaze=data,
        method="mean",
        window=3,
        min_valid=2,
        preserve_missing=False,
    )

    assert np.isnan(
        out.loc[
            0,
            "FPOGX_smooth",
        ]
    )


@pytest.mark.parametrize(
    "window",
    [
        0,
        2.5,
        np.inf,
        True,
    ],
)
def test_smooth_r_window_validation(
    window,
):
    with pytest.raises(
        ValueError,
        match="window",
    ):
        gp3.smooth_gazepoint_coordinate(
            all_gaze=pupil_fixture(),
            window=window,
        )


@pytest.mark.parametrize(
    "min_valid",
    [
        0,
        1.5,
        np.inf,
        True,
    ],
)
def test_smooth_r_min_valid_validation(
    min_valid,
):
    with pytest.raises(
        ValueError,
        match="min_valid",
    ):
        gp3.smooth_gazepoint_coordinate(
            all_gaze=pupil_fixture(),
            min_valid=min_valid,
        )


def test_smooth_r_missing_columns():
    with pytest.raises(
        ValueError,
        match="missing required",
    ):
        gp3.smooth_gazepoint_coordinate(all_gaze=pupil_fixture().drop(columns=["FPOGY"]))
