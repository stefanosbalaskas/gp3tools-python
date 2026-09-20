from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gp3tools.aoi as aoi_mod
import gp3tools.events as events_mod
import gp3tools.face as face_mod
import gp3tools.io as io_mod
import gp3tools.plotting as plotting_mod
import gp3tools.qc as qc_mod


def _close_figures() -> None:
    plotting_mod.plt.close("all")


def test_io_name_standardisation_and_empty_columns() -> None:
    names = io_mod.standardise_gazepoint_names(
        [
            None,
            np.nan,
            " TIME(ms) ",
            "TIMETICK(Hz)",
            "  FPOGX  ",
            "",
        ]
    )

    assert names == [
        "EMPTY_TRAILING",
        "EMPTY_TRAILING",
        "TIME",
        "TIMETICK",
        "FPOGX",
        "EMPTY_TRAILING",
    ]

    frame = pd.DataFrame(
        [
            [1.0, np.nan, "", None, "keep"],
            [2.0, np.nan, " ", None, "also keep"],
        ],
        columns=[
            "TIME(ms)",
            "Unnamed: 1",
            "..2",
            "EMPTY_TRAILING",
            "value",
        ],
    )

    result = io_mod.standardise_gazepoint_names(frame)

    assert list(result.columns) == [
        "TIME",
        "value",
    ]

    assert io_mod._column_is_empty(
        pd.Series(
            [np.nan, np.nan],
            dtype=float,
        )
    )

    assert io_mod._column_is_empty(
        pd.Series(
            ["", "  "],
            dtype=object,
        )
    )

    assert not io_mod._column_is_empty(
        pd.Series(
            ["", "x"],
            dtype=object,
        )
    )

    assert not io_mod._column_is_empty(
        pd.Series(
            [1.0, np.nan],
            dtype=float,
        )
    )


def test_io_export_classification_paths(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.csv"

    with pytest.raises(
        FileNotFoundError
    ):
        io_mod.classify_gazepoint_export(
            missing
        )

    summary = (
        tmp_path
        / "Data_Summary_export_demo.csv"
    )
    summary.write_text(
        "x,y\n1,2\n",
        encoding="utf-8",
    )

    fixation = tmp_path / "fixations.csv"
    fixation.write_text(
        "x,y\n1,2\n",
        encoding="utf-8",
    )

    all_gaze = tmp_path / "all_gaze.csv"
    all_gaze.write_text(
        "x,y\n1,2\n",
        encoding="utf-8",
    )

    user = tmp_path / "user.csv"
    user.write_text(
        "x,y\n1,2\n",
        encoding="utf-8",
    )

    header_summary = (
        tmp_path
        / "generic_summary.csv"
    )
    header_summary.write_text(
        "Gazepoint Analysis,Version\n",
        encoding="utf-8",
    )

    gaze_table = (
        tmp_path
        / "generic_gaze.csv"
    )
    gaze_table.write_text(
        "FPOGX,FPOGY\n",
        encoding="utf-8",
    )

    unknown = tmp_path / "generic.csv"
    unknown.write_text(
        "alpha,beta\n",
        encoding="utf-8",
    )

    assert (
        io_mod.classify_gazepoint_export(
            summary
        )
        == "summary"
    )

    assert (
        io_mod.classify_gazepoint_export(
            fixation
        )
        == "fixations"
    )

    assert (
        io_mod.classify_gazepoint_export(
            all_gaze
        )
        == "all_gaze"
    )

    assert (
        io_mod.classify_gazepoint_export(
            user
        )
        == "all_gaze"
    )

    assert (
        io_mod.classify_gazepoint_export(
            header_summary
        )
        == "summary"
    )

    assert (
        io_mod.classify_gazepoint_export(
            gaze_table
        )
        == "gaze_table"
    )

    assert (
        io_mod.classify_gazepoint_export(
            unknown
        )
        == "unknown"
    )


def test_io_read_gazepoint_contracts(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError
    ):
        io_mod.read_gazepoint(
            tmp_path / "missing.csv"
        )

    summary = (
        tmp_path
        / "Data_Summary_export.csv"
    )
    summary.write_text(
        "x,y\n1,2\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="summary",
    ):
        io_mod.read_gazepoint(
            summary
        )

    gaze = tmp_path / "user.csv"

    gaze.write_text(
        "TIME(ms),FPOGX,Unnamed: 2\n"
        "0,0.1,\n"
        "1,0.2,\n",
        encoding="utf-8",
    )

    result = io_mod.read_gazepoint(
        gaze
    )

    assert list(result.columns) == [
        "TIME",
        "FPOGX",
    ]

    assert (
        result.attrs[
            "gp3_file_type"
        ]
        == "all_gaze"
    )

    assert (
        result.attrs[
            "gp3_source_file"
        ]
        == "user.csv"
    )

    raw = io_mod.read_gazepoint(
        gaze,
        standardise_names=False,
        drop_empty_cols=False,
    )

    assert "TIME(ms)" in raw
    assert "Unnamed: 2" in raw


def test_io_folder_contracts(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "absent"

    with pytest.raises(
        FileNotFoundError
    ):
        io_mod.read_gazepoint_folder(
            missing
        )

    empty = tmp_path / "empty"
    empty.mkdir()

    with pytest.raises(
        FileNotFoundError,
        match="No files matching",
    ):
        io_mod.read_gazepoint_folder(
            empty
        )

    summaries = tmp_path / "summaries"
    summaries.mkdir()

    (
        summaries
        / "Data_Summary_export.csv"
    ).write_text(
        "x,y\n1,2\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="none were row-level",
    ):
        io_mod.read_gazepoint_folder(
            summaries
        )

    root = tmp_path / "exports"
    nested = root / "nested"
    nested.mkdir(
        parents=True
    )

    (
        root
        / "all_gaze_A.csv"
    ).write_text(
        "TIME,FPOGX\n"
        "0,0.1\n",
        encoding="utf-8",
    )

    (
        nested
        / "fix_B.csv"
    ).write_text(
        "TIME,FPOGX\n"
        "1,0.2\n",
        encoding="utf-8",
    )

    top = io_mod.read_gazepoint_folder(
        root,
        source_col="SOURCE",
    )

    assert len(top) == 1
    assert top["SOURCE"].tolist() == [
        "all_gaze_A.csv"
    ]

    recursive = (
        io_mod.read_gazepoint_folder(
            root,
            source_col="SOURCE",
            recursive=True,
        )
    )

    assert len(recursive) == 2

    assert set(
        recursive["SOURCE"]
    ) == {
        "all_gaze_A.csv",
        "fix_B.csv",
    }


def test_io_summary_parser_and_raw_fallback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    path = tmp_path / "summary.csv"

    path.write_text(
        "Gazepoint Analysis,5.0\n"
        "USER,VALUE\n"
        "S1,1\n"
        "S2,2\n"
        "\n"
        "other,data\n",
        encoding="utf-8",
    )

    parsed = (
        io_mod.read_gazepoint_summary(
            path
        )
    )

    assert (
        parsed["source_file"]
        == "summary.csv"
    )

    assert (
        parsed["metadata"][
            "Gazepoint Analysis"
        ]
        == "5.0"
    )

    assert parsed["tables"]
    assert len(parsed["tables"][0]) == 2

    def fail_read_csv(*args, **kwargs):
        raise RuntimeError(
            "synthetic parser failure"
        )

    monkeypatch.setattr(
        io_mod.pd,
        "read_csv",
        fail_read_csv,
    )

    fallback = (
        io_mod.read_gazepoint_summary(
            path
        )
    )

    assert list(
        fallback["raw"].columns
    ) == ["raw_line"]


def test_io_face_export_and_column_inspection(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        FileNotFoundError
    ):
        io_mod.read_gazepoint_face_export(
            tmp_path / "missing.tsv"
        )

    tsv = tmp_path / "face.tsv"

    tsv.write_text(
        "timestamp\tconfidence\n"
        "0.0\t0.9\n",
        encoding="utf-8",
    )

    face = (
        io_mod.read_gazepoint_face_export(
            tsv
        )
    )

    assert face.shape == (1, 2)

    assert (
        face.attrs[
            "gp3_source_file"
        ]
        == "face.tsv"
    )

    csv = tmp_path / "user.csv"

    csv.write_text(
        "TIME,LPD\n"
        "0,3.1\n"
        "1,3.2\n",
        encoding="utf-8",
    )

    inspected_path = (
        io_mod.inspect_gazepoint_columns(
            csv
        )
    )

    assert set(
        inspected_path["column"]
    ) == {
        "TIME",
        "LPD",
    }

    with pytest.raises(
        TypeError
    ):
        io_mod.inspect_gazepoint_columns(
            pd.DataFrame(
                {"x": [1]}
            ),
            x=pd.DataFrame(
                {"x": [1]}
            ),
        )

    with pytest.raises(
        ValueError
    ):
        io_mod.inspect_gazepoint_columns(
            [1, 2]
        )

    frame = pd.DataFrame(
        {
            "MEDIA_ID": pd.Series(
                [1, 2],
                dtype="int64",
            ),
            "TIME": pd.Series(
                [0.0, 1.0],
                dtype="float64",
            ),
            "LPD": pd.Series(
                [3.0, np.nan],
                dtype="float64",
            ),
            "flag": pd.Series(
                [True, False],
                dtype="bool",
            ),
            "when": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-02",
                ]
            ),
            "cat": pd.Series(
                ["a", "b"],
                dtype="category",
            ),
            "label": pd.Series(
                ["x", "y"],
                dtype="string",
            ),
        }
    )

    inspected = (
        io_mod.inspect_gazepoint_columns(
            frame
        )
    )

    dtype_map = dict(
        zip(
            inspected["column"],
            inspected["dtype"],
            strict=True,
        )
    )

    assert (
        dtype_map["MEDIA_ID"]
        == "integer"
    )

    assert dtype_map["TIME"] == "numeric"
    assert dtype_map["flag"] == "logical"

    assert (
        dtype_map["when"]
        == "POSIXct/POSIXt"
    )

    assert dtype_map["cat"] == "factor"

    group_map = dict(
        zip(
            inspected["column"],
            inspected[
                "semantic_group"
            ],
            strict=True,
        )
    )

    assert (
        group_map["MEDIA_ID"]
        == "identification"
    )

    assert group_map["TIME"] == "time"

    assert (
        group_map["LPD"]
        == "left_eye_pupil"
    )

    assert group_map["label"] == "other"


def test_plot_figax_xy_and_time_series() -> None:
    frame = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "pupil": [3.0, 3.1, 3.2],
            "group": ["A", "A", "B"],
        }
    )

    fig, ax = plotting_mod._figax()

    same_fig, same_ax = (
        plotting_mod._figax(ax)
    )

    assert same_fig is fig
    assert same_ax is ax

    data, x, y = plotting_mod._xy(
        frame
    )

    assert data is frame
    assert x == "time"
    assert y == "pupil"

    fallback = pd.DataFrame(
        {
            "a": [1.0, 2.0],
            "b": [3.0, 4.0],
        }
    )

    _, fallback_x, fallback_y = (
        plotting_mod._xy(
            fallback
        )
    )

    assert fallback_x == "a"
    assert fallback_y == "b"

    grouped = (
        plotting_mod.plot_gazepoint_time_series(
            frame,
            group_col="group",
        )
    )

    assert len(grouped.axes[0].lines) == 2

    ungrouped = (
        plotting_mod.plot_gazepoint_time_series(
            frame,
            group_col="missing",
        )
    )

    assert len(ungrouped.axes[0].lines) == 1

    _close_figures()


def test_plot_pupil_and_basic_wrappers() -> None:
    frame = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "pupil": [3.0, 3.2, 3.1],
            "other": [1.0, 2.0, 3.0],
            "pupil_flag": [
                "ok",
                "bad",
                "ok",
            ],
        }
    )

    assert (
        plotting_mod.plot_gazepoint_pupil_timecourse(
            frame,
            y_col="pupil",
        )
        is not None
    )

    assert (
        plotting_mod.plot_gazepoint_pupil_timecourse(
            frame,
            pupil_col="pupil",
            y_col="other",
        )
        is not None
    )

    assert (
        plotting_mod.plot_gazepoint_pupil_preprocessing(
            frame
        )
        is not None
    )

    assert (
        plotting_mod.plot_gazepoint_pupil_status(
            frame
        )
        is not None
    )

    assert (
        plotting_mod.plot_gazepoint_qc_overview(
            frame
        )
        is not None
    )

    assert (
        plotting_mod.plot_gazepoint_phase_timeline(
            frame
        )
        is not None
    )

    _close_figures()


def test_plot_sampling_and_tracking_quality(
    monkeypatch,
) -> None:
    frame = pd.DataFrame(
        {
            "time": [0.0, 1.0],
            "value": [1.0, 2.0],
        }
    )

    def fake_sampling(
        data,
        **kwargs,
    ):
        return pd.DataFrame(
            {
                "sampling_hz": [
                    60.0,
                    59.5,
                ]
            }
        )

    monkeypatch.setattr(
        qc_mod,
        "check_sampling_rate",
        fake_sampling,
    )

    sampling = (
        plotting_mod.plot_sampling_rate(
            frame,
            expected_hz=60,
        )
    )

    assert sampling.axes[0].get_title() == (
        "Sampling rate"
    )

    def fake_quality_prop(
        data,
        **kwargs,
    ):
        return pd.DataFrame(
            {
                "valid_prop": [
                    0.9,
                    0.8,
                ]
            }
        )

    monkeypatch.setattr(
        qc_mod,
        "summarise_tracking_quality",
        fake_quality_prop,
    )

    quality = (
        plotting_mod.plot_tracking_quality(
            frame
        )
    )

    assert quality is not None

    def fake_quality_numeric(
        data,
        **kwargs,
    ):
        return pd.DataFrame(
            {
                "score": [
                    1.0,
                    2.0,
                ]
            }
        )

    monkeypatch.setattr(
        qc_mod,
        "summarise_tracking_quality",
        fake_quality_numeric,
    )

    fallback = (
        plotting_mod.plot_tracking_quality(
            frame
        )
    )

    assert fallback is not None

    _close_figures()


def test_plot_missingness_heatmap_and_export(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        {
            "x": [
                0.1,
                0.2,
                np.nan,
            ],
            "y": [
                0.3,
                0.4,
                0.5,
            ],
        }
    )

    assert (
        plotting_mod.plot_gazepoint_missingness_profile(
            frame
        )
        is not None
    )

    with pytest.warns(
        RuntimeWarning,
        match="non-finite gaze coordinates",
    ):
        heatmap = (
            plotting_mod.plot_gazepoint_heatmap(
                frame,
                x_col="x",
                y_col="y",
            )
        )

    assert heatmap is not None

    finite_frame = (
        frame.dropna(
            subset=["x", "y"]
        )
        .reset_index(drop=True)
    )

    assert (
        plotting_mod.plot_gazepoint_heatmap(
            finite_frame,
            x_col="x",
            y_col="y",
        )
        is not None
    )

    with pytest.raises(
        ValueError,
        match="No finite gaze coordinate pairs",
    ):
        plotting_mod.plot_gazepoint_heatmap(
            pd.DataFrame(
                {
                    "x": [
                        np.nan,
                    ],
                    "y": [
                        np.nan,
                    ],
                }
            ),
            x_col="x",
            y_col="y",
        )

    assert (
        plotting_mod.plot_gazepoint_heatmap_overlay(
            frame,
            x_col="x",
            y_col="y",
        )
        is not None
    )

    assert (
        plotting_mod.plot_gazepoint_aoi_verification(
            frame,
            x_col="x",
            y_col="y",
        )
        is not None
    )

    assert (
        plotting_mod.plot_gazepoint_stimulus_layout_qc(
            frame,
            x_col="x",
            y_col="y",
        )
        is not None
    )

    path = (
        tmp_path
        / "nested"
        / "heatmap.png"
    )

    exported = (
        plotting_mod.export_gazepoint_heatmap_png(
            frame,
            path=path,
            x_col="x",
            y_col="y",
        )
    )

    assert exported == path
    assert path.exists()

    _close_figures()


def test_plot_aoi_timeline_and_transition_paths(
    monkeypatch,
) -> None:
    timeline = pd.DataFrame(
        {
            "time": [0.0, 1.0, 2.0],
            "aoi": ["A", "B", "A"],
        }
    )

    assert (
        plotting_mod.plot_gazepoint_aoi_timeline(
            timeline,
            aoi_col="aoi",
            time_col="time",
        )
        is not None
    )

    matrix = pd.DataFrame(
        [
            [0.0, 1.0],
            [2.0, 0.0],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )

    matrix.index.name = "from"

    assert (
        plotting_mod.plot_gazepoint_aoi_transition_matrix(
            matrix
        )
        is not None
    )

    def fake_transition(
        data,
        **kwargs,
    ):
        return matrix

    monkeypatch.setattr(
        aoi_mod,
        "compute_gazepoint_aoi_transition_matrix",
        fake_transition,
    )

    raw = pd.DataFrame(
        {
            "aoi": ["A", "B"],
        }
    )

    assert (
        plotting_mod.plot_gazepoint_aoi_transition_matrix(
            raw
        )
        is not None
    )

    assert (
        plotting_mod.plot_transition_heatmap(
            matrix
        )
        is not None
    )

    _close_figures()


def test_plot_scanpath_paths() -> None:
    frame = pd.DataFrame(
        {
            "x": [
                0.1,
                0.2,
                0.3,
            ],
            "y": [
                0.3,
                0.2,
                0.1,
            ],
            "subject": [
                "S1",
                "S1",
                "S2",
            ],
        }
    )

    assert (
        plotting_mod.plot_gazepoint_scanpath(
            frame,
            x_col="x",
            y_col="y",
        )
        is not None
    )

    assert (
        plotting_mod.plot_gazepoint_scanpaths(
            frame,
            group_col="subject",
        )
        is not None
    )

    no_group = frame[
        [
            "x",
            "y",
        ]
    ].copy()

    assert (
        plotting_mod.plot_gazepoint_scanpaths(
            no_group,
            x_col="x",
            y_col="y",
        )
        is not None
    )

    clustered = frame.assign(
        cluster=[
            1,
            1,
            2,
        ]
    )

    assert (
        plotting_mod.plot_gazepoint_scanpath_clusters(
            clustered
        )
        is not None
    )

    stability = pd.DataFrame(
        {
            "n_clusters": [
                2,
                3,
                4,
            ],
            "stability": [
                0.7,
                0.8,
                0.75,
            ],
        }
    )

    assert (
        plotting_mod.plot_gazepoint_scanpath_cluster_stability(
            stability
        )
        is not None
    )

    fallback = pd.DataFrame(
        {
            "value": [
                0.5,
                0.6,
            ]
        }
    )

    assert (
        plotting_mod.plot_gazepoint_scanpath_cluster_stability(
            fallback
        )
        is not None
    )

    _close_figures()


def test_plot_event_detector_paths(
    monkeypatch,
) -> None:
    agreement = pd.DataFrame(
        {
            "agreement": [
                "same",
                "same",
                "different",
            ]
        }
    )

    assert (
        plotting_mod.plot_gazepoint_event_detector_agreement(
            agreement
        )
        is not None
    )

    def fake_compare(
        data,
        **kwargs,
    ):
        return agreement

    monkeypatch.setattr(
        events_mod,
        "compare_gazepoint_event_detectors",
        fake_compare,
    )

    assert (
        plotting_mod.plot_gazepoint_event_detector_agreement(
            pd.DataFrame(
                {"x": [1]}
            )
        )
        is not None
    )

    benchmark = pd.DataFrame(
        {
            "detector": [
                "A",
                "A",
                "B",
            ],
            "elapsed_seconds": [
                0.1,
                0.2,
                0.3,
            ],
        }
    )

    assert (
        plotting_mod.plot_gazepoint_event_detector_benchmark(
            benchmark
        )
        is not None
    )

    _close_figures()


def test_plot_binocular_and_model_predictions() -> None:
    binocular = pd.DataFrame(
        {
            "LPMM": [
                3.0,
                3.1,
            ],
            "RPMM": [
                3.2,
                3.3,
            ],
        }
    )

    assert (
        plotting_mod.plot_gazepoint_binocular_diagnostics(
            binocular
        )
        is not None
    )

    class Model:
        fittedvalues = np.array(
            [
                1.0,
                2.0,
            ]
        )

        resid = np.array(
            [
                -0.1,
                0.1,
            ]
        )

        def predict(
            self,
            data,
        ):
            return np.repeat(
                3.0,
                len(data),
            )

    model = Model()

    with pytest.raises(
        ValueError,
        match="model is required",
    ):
        plotting_mod.plot_gazepoint_model_predictions()

    assert (
        plotting_mod.plot_gazepoint_model_predictions(
            model
        )
        is not None
    )

    assert (
        plotting_mod.plot_gazepoint_model_predictions(
            model,
            data=pd.DataFrame(
                {
                    "x": [
                        1,
                        2,
                        3,
                    ]
                }
            ),
        )
        is not None
    )

    assert (
        plotting_mod.plot_gazepoint_gca(
            model
        )
        is not None
    )

    series = pd.DataFrame(
        {
            "time": [
                0.0,
                1.0,
            ],
            "pupil": [
                3.0,
                3.2,
            ],
        }
    )

    assert (
        plotting_mod.plot_gazepoint_gca(
            series
        )
        is not None
    )

    assert (
        plotting_mod.plot_gazepoint_aoi_gamm(
            series
        )
        is not None
    )

    assert (
        plotting_mod.plot_gazepoint_time_varying_effect(
            series
        )
        is not None
    )

    _close_figures()


def test_plot_model_residual_contracts() -> None:
    with pytest.raises(
        ValueError,
        match="type must",
    ):
        plotting_mod.plot_gazepoint_model_residuals(
            data=pd.DataFrame(
                {
                    "fitted": [1.0],
                    "residual": [0.1],
                }
            ),
            type="invalid",
        )

    with pytest.raises(
        ValueError,
        match="Could not identify",
    ):
        plotting_mod.plot_gazepoint_model_residuals(
            data=pd.DataFrame(
                {"x": [1.0]}
            )
        )

    with pytest.raises(
        ValueError,
        match="Supply either",
    ):
        plotting_mod.plot_gazepoint_model_residuals()

    with pytest.raises(
        ValueError,
        match="No finite",
    ):
        plotting_mod.plot_gazepoint_model_residuals(
            data=pd.DataFrame(
                {
                    "fitted": [np.nan],
                    "residual": [np.nan],
                }
            )
        )

    residual_data = pd.DataFrame(
        {
            "fitted": [
                1.0,
                2.0,
                np.nan,
            ],
            "residual": [
                -0.1,
                0.2,
                0.5,
            ],
        }
    )

    assert (
        plotting_mod.plot_gazepoint_model_residuals(
            data=residual_data,
            type="residuals_fitted",
        )
        is not None
    )

    assert (
        plotting_mod.plot_gazepoint_model_residuals(
            data=residual_data.iloc[:2],
            type="qq",
            title="QQ",
        )
        is not None
    )

    assert (
        plotting_mod.plot_gazepoint_model_residuals(
            data=residual_data.iloc[:1],
            type="qq",
        )
        is not None
    )

    class Model:
        resid = np.array(
            [
                -0.2,
                0.2,
            ]
        )

        fittedvalues = np.array(
            [
                1.0,
                2.0,
            ]
        )

    assert (
        plotting_mod.plot_gazepoint_model_residuals(
            model=Model()
        )
        is not None
    )

    _close_figures()


def test_plot_cluster_and_multiverse_paths() -> None:
    observed = pd.DataFrame(
        {
            "time": [
                0.0,
                1.0,
            ],
            "difference": [
                -1.0,
                1.0,
            ],
        }
    )

    assert (
        plotting_mod.plot_gazepoint_cluster_results(
            {
                "observed": observed,
            }
        )
        is not None
    )

    alternative = pd.DataFrame(
        {
            "time": [
                0.0,
                1.0,
            ],
            "estimate": [
                0.1,
                0.2,
            ],
        }
    )

    assert (
        plotting_mod.plot_gazepoint_cluster_results(
            alternative
        )
        is not None
    )

    assert (
        plotting_mod.plot_gazepoint_cluster_permutation(
            {
                "observed": observed,
            }
        )
        is not None
    )

    assert (
        plotting_mod.plot_gazepoint_cluster_null_distribution(
            {
                "null_distribution": [
                    -1.0,
                    0.0,
                    1.0,
                ]
            }
        )
        is not None
    )

    assert (
        plotting_mod.plot_gazepoint_cluster_null_distribution(
            np.array(
                [
                    -1.0,
                    0.0,
                    1.0,
                ]
            )
        )
        is not None
    )

    assert (
        plotting_mod.plot_gazepoint_multiverse_results(
            pd.DataFrame(
                {
                    "mean_pupil": [
                        3.0,
                        3.1,
                    ]
                }
            )
        )
        is not None
    )

    assert (
        plotting_mod.plot_gazepoint_multiverse_results(
            pd.DataFrame(
                {
                    "estimate": [
                        1.0,
                        2.0,
                    ]
                }
            )
        )
        is not None
    )

    assert (
        plotting_mod.plot_gazepoint_multiverse_results(
            pd.DataFrame(
                {
                    "label": [
                        "a",
                        "b",
                    ]
                }
            )
        )
        is not None
    )

    _close_figures()


def test_plot_face_quality_supports_structured_and_legacy_audits() -> None:
    face = pd.DataFrame(
        {
            "frame": [
                1,
                2,
                3,
            ],
            "timestamp": [
                0.0,
                0.1,
                0.2,
            ],
            "confidence": [
                0.95,
                0.50,
                np.nan,
            ],
            "success": [
                1,
                1,
                0,
            ],
            "AU01_r": [
                0.1,
                0.2,
                0.3,
            ],
        }
    )

    structured = (
        plotting_mod.plot_gazepoint_face_quality(
            face
        )
    )

    heights = [
        patch.get_height()
        for patch in structured.axes[0].patches
    ]

    assert heights == pytest.approx(
        [
            1.0,
            1.0,
        ]
    )

    legacy = (
        plotting_mod.plot_gazepoint_face_quality(
            face,
            confidence_col="confidence",
            threshold=0.80,
        )
    )

    legacy_heights = [
        patch.get_height()
        for patch in legacy.axes[0].patches
    ]

    assert legacy_heights == pytest.approx(
        [
            2.0,
            1.0,
        ]
    )

    _close_figures()


def test_plot_face_quality_rejects_malformed_audit(
    monkeypatch,
) -> None:
    def bad_audit(
        data,
        **kwargs,
    ):
        return {
            "overview": pd.DataFrame(),
        }

    monkeypatch.setattr(
        face_mod,
        "audit_gazepoint_face_quality",
        bad_audit,
    )

    with pytest.raises(
        ValueError,
        match="usable overview",
    ):
        plotting_mod.plot_gazepoint_face_quality(
            pd.DataFrame(
                {"x": [1]}
            )
        )

    def missing_valid(
        data,
        **kwargs,
    ):
        return {
            "overview": pd.DataFrame(
                {
                    "n_rows": [
                        1,
                    ]
                }
            )
        }

    monkeypatch.setattr(
        face_mod,
        "audit_gazepoint_face_quality",
        missing_valid,
    )

    with pytest.raises(
        ValueError,
        match="missing n_valid",
    ):
        plotting_mod.plot_gazepoint_face_quality(
            pd.DataFrame(
                {"x": [1]}
            )
        )

    def bad_legacy(
        data,
        **kwargs,
    ):
        return pd.DataFrame()

    monkeypatch.setattr(
        face_mod,
        "audit_gazepoint_face_quality",
        bad_legacy,
    )

    with pytest.raises(
        ValueError,
        match="usable result",
    ):
        plotting_mod.plot_gazepoint_face_quality(
            pd.DataFrame(
                {"x": [1]}
            )
        )

    _close_figures()


# TRANCHE_2A_RESIDUAL_BRANCH_CLOSURE


def test_io_residual_branch_contracts(
    tmp_path: Path,
) -> None:
    # Scalar string route in standardise_gazepoint_names().
    assert (
        io_mod.standardise_gazepoint_names(
            " TIME(ms) "
        )
        == "TIME"
    )

    assert (
        io_mod.standardise_gazepoint_names(
            "TIMETICK(Hz)"
        )
        == "TIMETICK"
    )

    # Missing summary input has its own deterministic error route.
    with pytest.raises(
        FileNotFoundError
    ):
        io_mod.read_gazepoint_summary(
            tmp_path / "missing-summary.csv"
        )

    # A detected table header with no body exercises the
    # zero-iteration and empty-body paths.
    empty_table = (
        tmp_path
        / "empty-table-summary.csv"
    )

    empty_table.write_text(
        "AOI,VALUE\n",
        encoding="utf-8",
    )

    parsed_empty = (
        io_mod.read_gazepoint_summary(
            empty_table
        )
    )

    assert (
        parsed_empty["tables"]
        == []
    )

    # A malformed body width must terminate that candidate
    # table rather than manufacturing/coercing a row.
    malformed_table = (
        tmp_path
        / "malformed-table-summary.csv"
    )

    malformed_table.write_text(
        "USER,VALUE\n"
        "only-one-field\n",
        encoding="utf-8",
    )

    parsed_malformed = (
        io_mod.read_gazepoint_summary(
            malformed_table
        )
    )

    assert (
        parsed_malformed["tables"]
        == []
    )

    # Non-TSV facial export: separator must not be silently
    # changed to tab.
    face_csv = (
        tmp_path
        / "face.csv"
    )

    face_csv.write_text(
        "timestamp,confidence\n"
        "0.0,0.95\n",
        encoding="utf-8",
    )

    face_result = (
        io_mod.read_gazepoint_face_export(
            face_csv
        )
    )

    assert list(
        face_result.columns
    ) == [
        "timestamp",
        "confidence",
    ]

    assert (
        face_result.attrs[
            "gp3_source_file"
        ]
        == "face.csv"
    )

    # Cover path-via-string and R-compatible x alias path route.
    gaze_csv = (
        tmp_path
        / "user.csv"
    )

    gaze_csv.write_text(
        "TIME,LPD\n"
        "0,3.1\n",
        encoding="utf-8",
    )

    inspected_string = (
        io_mod.inspect_gazepoint_columns(
            str(gaze_csv)
        )
    )

    assert set(
        inspected_string["column"]
    ) == {
        "TIME",
        "LPD",
    }

    inspected_alias = (
        io_mod.inspect_gazepoint_columns(
            x=str(gaze_csv)
        )
    )

    assert set(
        inspected_alias["column"]
    ) == {
        "TIME",
        "LPD",
    }


def test_plot_residual_explicit_column_and_model_fallback_paths() -> None:
    # Existing tests cover automatic fitted/residual discovery.
    # Explicit names cover the complementary branch.
    explicit = pd.DataFrame(
        {
            "fit_custom": [
                1.0,
                2.0,
            ],
            "res_custom": [
                -0.2,
                0.3,
            ],
        }
    )

    explicit_fig = (
        plotting_mod.plot_gazepoint_model_residuals(
            data=explicit,
            fitted_col="fit_custom",
            residual_col="res_custom",
        )
    )

    assert explicit_fig is not None

    # Exercise model residual route when fittedvalues is absent.
    class ResidualOnlyModel:
        resid = np.array(
            [
                -0.1,
                0.2,
            ]
        )

    residual_fig = (
        plotting_mod.plot_gazepoint_model_residuals(
            model=ResidualOnlyModel()
        )
    )

    assert residual_fig is not None

    # Prediction path without explicit newdata uses fittedvalues.
    class FittedModel:
        fittedvalues = np.array(
            [
                1.0,
                1.5,
            ]
        )

        def predict(
            self,
            data,
        ):
            return np.asarray(
                data["x"],
                dtype=float,
            )

    fitted_fig = (
        plotting_mod.plot_gazepoint_model_predictions(
            FittedModel()
        )
    )

    assert fitted_fig is not None

    _close_figures()


def test_plot_face_quality_structured_fallback_paths(
    monkeypatch,
) -> None:
    # Structured audit without face_confidence but with
    # aggregate n_invalid.
    def structured_n_invalid(
        data,
        **kwargs,
    ):
        return {
            "overview": pd.DataFrame(
                {
                    "n_valid": [
                        2,
                    ],
                    "n_invalid": [
                        1,
                    ],
                }
            ),
            "data": pd.DataFrame(
                {
                    "other": [
                        1,
                        2,
                        3,
                    ]
                }
            ),
            "settings": {
                "confidence_threshold": 0.8,
            },
        }

    monkeypatch.setattr(
        face_mod,
        "audit_gazepoint_face_quality",
        structured_n_invalid,
    )

    fig = (
        plotting_mod.plot_gazepoint_face_quality(
            pd.DataFrame(
                {
                    "x": [
                        1,
                        2,
                        3,
                    ]
                }
            )
        )
    )

    heights = [
        patch.get_height()
        for patch in fig.axes[0].patches
    ]

    assert heights == pytest.approx(
        [
            2.0,
            1.0,
        ]
    )

    # Structured audit lacking both row-level confidence and
    # aggregate n_invalid has an explicit zero fallback.
    def structured_zero_fallback(
        data,
        **kwargs,
    ):
        return {
            "overview": pd.DataFrame(
                {
                    "n_valid": [
                        2,
                    ]
                }
            ),
            "data": pd.DataFrame(
                {
                    "other": [
                        1,
                        2,
                    ]
                }
            ),
            "settings": {},
        }

    monkeypatch.setattr(
        face_mod,
        "audit_gazepoint_face_quality",
        structured_zero_fallback,
    )

    zero_fig = (
        plotting_mod.plot_gazepoint_face_quality(
            pd.DataFrame(
                {
                    "x": [
                        1,
                        2,
                    ]
                }
            )
        )
    )

    zero_heights = [
        patch.get_height()
        for patch in zero_fig.axes[0].patches
    ]

    assert zero_heights == pytest.approx(
        [
            2.0,
            0.0,
        ]
    )

    _close_figures()


def test_plot_face_quality_legacy_nonfinite_proportion_fallback(
    monkeypatch,
) -> None:
    # Complement the finite legacy-proportion route.
    def legacy_nonfinite(
        data,
        **kwargs,
    ):
        return pd.DataFrame(
            {
                "n_valid": [
                    2,
                ],
                "prop_below_threshold": [
                    np.nan,
                ],
                "n": [
                    3,
                ],
            }
        )

    monkeypatch.setattr(
        face_mod,
        "audit_gazepoint_face_quality",
        legacy_nonfinite,
    )

    fig = (
        plotting_mod.plot_gazepoint_face_quality(
            pd.DataFrame(
                {
                    "x": [
                        1,
                        2,
                        3,
                    ]
                }
            )
        )
    )

    heights = [
        patch.get_height()
        for patch in fig.axes[0].patches
    ]

    assert heights == pytest.approx(
        [
            2.0,
            0.0,
        ]
    )

    _close_figures()


def test_io_native_column_inspection_rejects_conflicting_names() -> None:
    """The native implementation retains its own alias-conflict invariant.

    The public r_aliases() wrapper normally detects this conflict first.
    functools.wraps intentionally exposes __wrapped__, so the native
    defensive contract remains independently testable.
    """
    frame = pd.DataFrame(
        {
            "TIME": [
                0.0,
            ]
        }
    )

    native = (
        io_mod.inspect_gazepoint_columns.__wrapped__
    )

    with pytest.raises(
        TypeError,
        match="received both 'data' and",
    ):
        native(
            data=frame,
            x=frame,
        )
