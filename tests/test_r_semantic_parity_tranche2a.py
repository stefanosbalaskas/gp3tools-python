from pathlib import Path

import pandas as pd

import gp3tools as gp3


def test_r_screen_coordinate_harmonization():
    data = pd.DataFrame(
        {
            "x": [0.0, 100.0, 200.0],
            "y": [0.0, 50.0, 100.0],
        }
    )

    out = gp3.harmonize_gazepoint_screen_coordinates(
        data,
        x_col="x",
        y_col="y",
        from_width=200,
        from_height=100,
        to_width=1000,
        to_height=500,
    )

    assert out["gaze_x_harmonized"].tolist() == [
        0.0,
        500.0,
        1000.0,
    ]

    assert out["gaze_y_harmonized"].tolist() == [
        0.0,
        250.0,
        500.0,
    ]

    meta = out.attrs["gp3_screen_harmonization"]

    assert meta["x_scale"] == 5.0
    assert meta["y_scale"] == 5.0


def test_r_screen_coordinate_harmonization_drop_original():
    data = pd.DataFrame(
        {
            "x": [100.0],
            "y": [50.0],
        }
    )

    out = gp3.harmonize_gazepoint_screen_coordinates(
        data,
        x_col="x",
        y_col="y",
        from_width=200,
        from_height=100,
        to_width=1000,
        to_height=500,
        keep_original=False,
    )

    assert "x" not in out
    assert "y" not in out
    assert "gaze_x_harmonized" in out
    assert "gaze_y_harmonized" in out


def test_python_screen_coordinate_interface_is_preserved():
    data = pd.DataFrame(
        {
            "x": [0.0, 100.0],
            "y": [0.0, 50.0],
        }
    )

    out = gp3.harmonize_gazepoint_screen_coordinates(
        data,
        x_col="x",
        y_col="y",
        width=100,
        height=50,
    )

    assert out["x_norm"].tolist() == [
        0.0,
        1.0,
    ]

    assert out["y_norm"].tolist() == [
        0.0,
        1.0,
    ]


def test_r_task_phase_windows():
    data = pd.DataFrame(
        {
            "TIME": [
                -1.0,
                0.0,
                0.5,
                1.0,
                1.5,
                2.0,
            ]
        }
    )

    windows = pd.DataFrame(
        {
            "phase": ["early", "late"],
            "start": [0.0, 1.0],
            "end": [1.0, 2.0],
        }
    )

    out = gp3.segment_gazepoint_task_phases(
        data,
        time_col="TIME",
        phase_windows=windows,
    )

    assert out["task_phase"].tolist() == [
        "outside",
        "early",
        "early",
        "late",
        "late",
        "outside",
    ]

    assert out[".gp3_phase_assigned"].tolist() == [
        False,
        True,
        True,
        True,
        True,
        False,
    ]


def test_r_task_phase_endpoint_policy():
    data = pd.DataFrame({"TIME": [0.0, 1.0]})

    windows = pd.DataFrame(
        {
            "phase": ["A"],
            "start": [0.0],
            "end": [1.0],
        }
    )

    out = gp3.segment_gazepoint_task_phases(
        data,
        time_col="TIME",
        phase_windows=windows,
        include_lower=False,
        include_upper=True,
    )

    assert out["task_phase"].tolist() == [
        "outside",
        "A",
    ]


def test_r_event_detector_benchmark_levels():
    benchmark = {
        "detector_metrics": pd.DataFrame(
            {
                "detector": ["B", "A", "C"],
                "f1": [0.7, 0.9, None],
            }
        ),
        "sequence_metrics": pd.DataFrame({"sequence": ["S1"]}),
        "matches": pd.DataFrame({"match": [1]}),
        "errors": pd.DataFrame({"error": ["x"]}),
    }

    out = gp3.summarise_gazepoint_event_detector_benchmark(
        x=benchmark,
        level="detector",
    )

    assert out["detector"].tolist() == [
        "A",
        "B",
        "C",
    ]

    seq = gp3.summarise_gazepoint_event_detector_benchmark(
        x=benchmark,
        level="sequence",
    )

    assert seq["sequence"].tolist() == ["S1"]


def test_python_event_benchmark_interface_is_preserved():
    data = pd.DataFrame(
        {
            "detector": ["A", "A", "B"],
            "elapsed_seconds": [
                1.0,
                3.0,
                2.0,
            ],
        }
    )

    out = gp3.summarise_gazepoint_event_detector_benchmark(data)

    assert set(out["detector"]) == {"A", "B"}
    assert "mean_seconds" in out


def test_r_coordinate_coverage():
    data = pd.DataFrame(
        {
            "x": [
                0.0,
                25.0,
                75.0,
                100.0,
                110.0,
            ],
            "y": [
                0.0,
                25.0,
                75.0,
                100.0,
                50.0,
            ],
        }
    )

    out = gp3.summarize_gazepoint_coordinate_coverage(
        data,
        x_col="x",
        y_col="y",
        screen_width=100,
        screen_height=100,
        grid_n_x=2,
        grid_n_y=2,
    )

    row = out.iloc[0]

    assert row["group_id"] == "all"
    assert row["n_rows"] == 5
    assert row["n_finite_coordinates"] == 5
    assert row["n_inside_screen"] == 4
    assert row["total_grid_cells"] == 4
    assert row["occupied_grid_cells"] == 2


def test_r_naming_audit_writer(tmp_path):
    pairs = pd.DataFrame(
        {
            "original": ["A"],
            "standardized": ["a"],
        }
    )

    output = tmp_path / "nested" / "audit.csv"

    result = gp3.write_gazepoint_naming_audit(
        x={"pairs": pairs},
        output_file=output,
    )

    assert result.exists()

    written = pd.read_csv(result)
    pd.testing.assert_frame_equal(
        written,
        pairs,
    )


def test_python_naming_audit_writer_is_preserved(tmp_path):
    output = tmp_path / "audit.csv"

    result = gp3.write_gazepoint_naming_audit(
        output,
        names=["A", "B"],
    )

    assert result.exists()


def test_r_performance_benchmark_writer(tmp_path):
    benchmark = {
        "trials": pd.DataFrame({"trial": [1]}),
        "summary": pd.DataFrame({"operation": ["read"]}),
        "regression": {
            "checks": pd.DataFrame({"check": ["elapsed"]}),
            "evaluated": pd.DataFrame({"operation": ["read"]}),
        },
    }

    files = gp3.write_gazepoint_performance_benchmark(
        x=benchmark,
        output_dir=tmp_path,
        prefix="bench",
    )

    assert set(files) == {
        "trials",
        "summary",
        "checks",
        "evaluated",
    }

    for path in files.values():
        assert Path(path).exists()


def test_python_performance_benchmark_writer_is_preserved(
    tmp_path,
):
    data = pd.DataFrame({"elapsed_seconds": [1.0]})

    output = tmp_path / "performance.csv"

    result = gp3.write_gazepoint_performance_benchmark(
        data,
        path=output,
    )

    assert result == output
    assert output.exists()
