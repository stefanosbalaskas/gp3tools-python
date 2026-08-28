import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3


def test_legacy_scanpath_geometry_preserved():
    data = pd.DataFrame(
        {
            "x": [0.0, 1.0],
            "y": [0.0, 0.0],
            "subject": ["S1", "S1"],
        }
    )

    out = gp3.compute_gazepoint_scanpath_geometry(
        data,
        group_cols=["subject"],
    )

    assert "path_length" in out
    assert out.iloc[0]["path_length"] == 1.0


def test_r_scanpath_geometry_exact_metrics():
    data = pd.DataFrame(
        {
            "x": [1.0, 0.0, 1.0],
            "y": [1.0, 0.0, 0.0],
            "subject_id": ["S1"] * 3,
            "trial_id": [1] * 3,
            "time_value": [3, 1, 2],
            "condition_id": ["A"] * 3,
        }
    )

    out = gp3.compute_gazepoint_scanpath_geometry(
        data,
        x="x",
        y="y",
        subject="subject_id",
        trial="trial_id",
        time="time_value",
        condition="condition_id",
    )

    row = out.iloc[0]

    assert row["subject"] == "S1"
    assert row["trial"] == 1
    assert row["condition"] == "A"
    assert row["n_points"] == 3

    assert np.isclose(
        row["scanpath_length"],
        2.0,
    )

    assert np.isclose(
        row["straight_line_distance"],
        np.sqrt(2.0),
    )

    assert np.isclose(
        row["scanpath_efficiency"],
        np.sqrt(2.0) / 2.0,
    )

    assert np.isclose(
        row["convex_hull_area"],
        0.5,
    )

    expected_dispersion = np.mean(
        [
            np.sqrt(5) / 3,
            np.sqrt(2) / 3,
            np.sqrt(5) / 3,
        ]
    )

    assert np.isclose(
        row["spatial_dispersion"],
        expected_dispersion,
    )


def test_r_scanpath_geometry_single_point_na_metrics():
    data = pd.DataFrame(
        {
            "x": [1.0],
            "y": [2.0],
            "subject": ["S1"],
            "trial": [1],
        }
    )

    out = gp3.compute_gazepoint_scanpath_geometry(
        data,
        x="x",
        y="y",
        subject="subject",
        trial="trial",
    )

    row = out.iloc[0]

    assert row["n_points"] == 1
    assert np.isnan(row["scanpath_length"])
    assert np.isnan(row["straight_line_distance"])
    assert np.isnan(row["scanpath_efficiency"])
    assert np.isnan(row["convex_hull_area"])


def test_legacy_aoi_summary_preserved():
    data = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S1"],
            "AOI": ["A", "A", "B"],
        }
    )

    out = gp3.summarise_gazepoint_aoi(
        data,
        aoi_col="AOI",
        group_cols=["subject"],
    )

    assert set(out.columns) == {
        "subject",
        "AOI",
        "n_samples",
        "proportion",
    }

    assert out["n_samples"].sum() == 3


def test_r_aoi_summary_samples_and_fixations():
    gaze = pd.DataFrame(
        {
            "USER_FILE": [
                "Participant001",
                "Participant001",
                "Participant001",
            ],
            "MEDIA_ID": [10, 10, 10],
            "MEDIA_NAME": ["Stim"] * 3,
            "AOI": ["A", "A", "B"],
            "TIME": [0.2, 0.4, 0.5],
        }
    )

    fix = pd.DataFrame(
        {
            "USER_FILE": [
                "Participant001",
                "Participant001",
            ],
            "MEDIA_ID": [10, 10],
            "MEDIA_NAME": ["Stim", "Stim"],
            "AOI": ["A", "A"],
            "FPOGD": [0.1, 0.3],
            "FPOGS": [0.25, 0.45],
        }
    )

    out = gp3.summarise_gazepoint_aoi(
        gaze_data=gaze,
        fixation_data=fix,
        sample_rate=10,
    )

    a = out.loc[out["AOI"].eq("A")].iloc[0]

    assert a["USER_ID"] == 1
    assert a["sample_ttff_sec"] == 0.2
    assert a["sample_count"] == 2
    assert a["sample_time_viewed_sec"] == 0.2
    assert a["fixation_count"] == 2

    assert np.isclose(
        a["fixation_duration_sum_sec"],
        0.4,
    )

    assert np.isclose(
        a["fixation_duration_mean_ms"],
        200.0,
    )

    assert np.isclose(
        a["fixation_ttff_sec"],
        0.25,
    )


def test_r_aoi_summary_supports_two_positional_frames():
    gaze = pd.DataFrame(
        {
            "USER_FILE": ["U7"],
            "MEDIA_ID": [1],
            "MEDIA_NAME": ["Stim"],
            "AOI": ["A"],
            "TIME": [0.1],
        }
    )

    fix = pd.DataFrame(
        {
            "USER_FILE": ["U7"],
            "MEDIA_ID": [1],
            "MEDIA_NAME": ["Stim"],
            "AOI": ["B"],
            "FPOGD": [0.2],
            "FPOGS": [0.3],
        }
    )

    out = gp3.summarise_gazepoint_aoi(
        gaze,
        fix,
    )

    assert out["AOI"].tolist() == [
        "A",
        "B",
    ]

    assert pd.isna(
        out.loc[
            out["AOI"].eq("A"),
            "fixation_count",
        ].iloc[0]
    )

    assert pd.isna(
        out.loc[
            out["AOI"].eq("B"),
            "sample_count",
        ].iloc[0]
    )


def test_legacy_pupil_summary_preserved():
    data = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
            ],
            "pupil": [
                3.0,
                5.0,
            ],
        }
    )

    out = gp3.summarise_gazepoint_pupil(
        data,
        pupil_col="pupil",
        group_cols=["subject"],
    )

    assert "n_samples" in out
    assert "n_valid" in out
    assert out.iloc[0]["mean_pupil"] == 4.0


def test_r_pupil_summary_detects_columns_and_groups():
    master = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S1",
                "S2",
            ],
            "media_id": [
                "M1",
                "M1",
                "M1",
                "M1",
            ],
            "time_ms": [
                0.0,
                10.0,
                20.0,
                0.0,
            ],
            "mean_pupil": [
                2.0,
                4.0,
                np.nan,
                8.0,
            ],
        }
    )

    out = gp3.summarise_gazepoint_pupil(master=master)

    s1 = out.loc[out["subject"].eq("S1")].iloc[0]

    assert s1["n_rows"] == 3
    assert s1["n_pupil_samples"] == 2
    assert s1["n_missing_pupil"] == 1

    assert np.isclose(
        s1["missing_pupil_pct"],
        100 / 3,
    )

    assert np.isclose(
        s1["valid_pupil_pct"],
        200 / 3,
    )

    assert s1["mean_pupil"] == 3.0
    assert s1["median_pupil"] == 3.0

    assert np.isclose(
        s1["sd_pupil"],
        np.sqrt(2.0),
    )

    assert s1["time_min_ms"] == 0.0
    assert s1["time_max_ms"] == 20.0
    assert s1["time_span_ms"] == 20.0
    assert s1["pupil_column"] == "mean_pupil"
    assert s1["time_column"] == "time_ms"


def test_r_pupil_summary_missing_and_plausibility():
    master = pd.DataFrame(
        {
            "subject": ["S1"] * 6,
            "media_id": ["M1"] * 6,
            "time_ms": [
                0,
                1,
                2,
                3,
                4,
                5,
            ],
            "pupil": [
                1.0,
                2.0,
                3.0,
                4.0,
                100.0,
                5.0,
            ],
            "missing_pupil": [
                False,
                False,
                False,
                False,
                False,
                True,
            ],
        }
    )

    out = gp3.summarise_gazepoint_pupil(
        master=master,
        min_pupil=1,
        max_pupil=10,
    )

    row = out.iloc[0]

    assert row["n_rows"] == 6
    assert row["n_pupil_samples"] == 5
    assert row["n_missing_pupil"] == 1
    assert row["n_above_plausible"] == 1
    assert row["n_below_plausible"] == 0
    assert row["n_implausible"] == 1

    assert np.isclose(
        row["implausible_pct"],
        100 / 6,
    )

    assert row["n_iqr_outliers"] == 0


def test_r_pupil_summary_can_be_ungrouped():
    master = pd.DataFrame(
        {
            "participant": [
                "S1",
                "S2",
            ],
            "MEDIA_ID": [
                1,
                1,
            ],
            "time": [
                0,
                1,
            ],
            "pupil_raw": [
                2.0,
                4.0,
            ],
        }
    )

    out = gp3.summarise_gazepoint_pupil(
        master=master,
        group_cols=[],
    )

    assert len(out) == 1
    assert out.iloc[0]["n_rows"] == 2
    assert out.iloc[0]["mean_pupil"] == 3.0


def test_r_pupil_summary_validates_group_cols():
    master = pd.DataFrame(
        {
            "subject": ["S1"],
            "media_id": ["M1"],
            "time_ms": [0],
            "pupil": [3.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="group_cols can only",
    ):
        gp3.summarise_gazepoint_pupil(
            master=master,
            group_cols=["condition"],
        )


def test_r_scanpath_geometry_requires_complete_column_arguments():
    data = pd.DataFrame(
        {
            "x": [1.0],
            "y": [2.0],
            "subject": ["S1"],
        }
    )

    with pytest.raises(
        ValueError,
        match="trial",
    ):
        gp3.compute_gazepoint_scanpath_geometry(
            data,
            x="x",
            y="y",
            subject="subject",
        )


def test_r_scanpath_geometry_reports_missing_columns():
    data = pd.DataFrame(
        {
            "x": [1.0],
            "subject": ["S1"],
            "trial": [1],
        }
    )

    with pytest.raises(
        ValueError,
        match="Missing columns",
    ):
        gp3.compute_gazepoint_scanpath_geometry(
            data,
            x="x",
            y="y",
            subject="subject",
            trial="trial",
        )


def test_r_aoi_summary_requires_fixation_data():
    gaze = pd.DataFrame(
        {
            "USER_FILE": ["U1"],
            "MEDIA_ID": [1],
            "MEDIA_NAME": ["Stim"],
            "AOI": ["A"],
            "TIME": [0.1],
        }
    )

    with pytest.raises(
        TypeError,
        match="fixation_data",
    ):
        gp3.summarise_gazepoint_aoi(gaze_data=gaze)


def test_r_aoi_summary_reports_missing_gaze_columns():
    gaze = pd.DataFrame(
        {
            "USER_FILE": ["U1"],
            "MEDIA_ID": [1],
        }
    )

    fix = pd.DataFrame(
        {
            "USER_FILE": ["U1"],
            "MEDIA_ID": [1],
            "MEDIA_NAME": ["Stim"],
            "AOI": ["A"],
            "FPOGD": [0.2],
            "FPOGS": [0.3],
        }
    )

    with pytest.raises(
        ValueError,
        match="gaze_data",
    ):
        gp3.summarise_gazepoint_aoi(
            gaze_data=gaze,
            fixation_data=fix,
        )


def test_r_pupil_summary_rejects_invalid_plausible_range():
    master = pd.DataFrame(
        {
            "subject": ["S1"],
            "media_id": ["M1"],
            "time_ms": [0],
            "pupil": [3.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="max_pupil",
    ):
        gp3.summarise_gazepoint_pupil(
            master=master,
            min_pupil=5,
            max_pupil=4,
        )


def test_r_pupil_summary_requires_subject_column():
    master = pd.DataFrame(
        {
            "media_id": ["M1"],
            "time_ms": [0],
            "pupil": [3.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="subject",
    ):
        gp3.summarise_gazepoint_pupil(master=master)


def test_r_pupil_summary_rejects_missing_explicit_missingness_column():
    master = pd.DataFrame(
        {
            "subject": ["S1"],
            "media_id": ["M1"],
            "time_ms": [0],
            "pupil": [3.0],
        }
    )

    with pytest.raises(
        ValueError,
        match="missing_pupil_col",
    ):
        gp3.summarise_gazepoint_pupil(
            master=master,
            missing_pupil_col="does_not_exist",
        )
