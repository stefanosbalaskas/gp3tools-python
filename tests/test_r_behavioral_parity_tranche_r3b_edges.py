from pathlib import Path

import pandas as pd
import pytest

import gp3tools as gp3


def _valid_report_results():
    return {
        "all_gaze": pd.DataFrame(
            {
                "x": [
                    "a",
                ]
            }
        ),
        "all_fix": pd.DataFrame(
            {
                "x": [
                    "a",
                ]
            }
        ),
        "sampling": pd.DataFrame(
            {
                "rate": [
                    "60",
                ]
            }
        ),
        "quality": pd.DataFrame(
            {
                "status": [
                    "ok",
                ]
            }
        ),
        "flagged_quality": pd.DataFrame(
            {
                "review_required": [
                    False,
                ]
            }
        ),
        "aoi_table": pd.DataFrame(
            {
                "AOI": [
                    "target",
                ]
            }
        ),
    }


def test_r3b_report_validation_and_overwrite(
    tmp_path,
):
    output = tmp_path / "report.html"

    with pytest.raises(ValueError):
        gp3.create_gazepoint_report(
            {"quality": pd.DataFrame()},
            output_file=output,
            overwrite=True,
            save_plots=False,
        )

    result = gp3.create_gazepoint_report(
        _valid_report_results(),
        output_file=output,
        title="<R3&B>",
        overwrite=True,
        save_plots=False,
    )

    assert Path(
        result.loc[
            0,
            "report",
        ]
    ).exists()

    html = output.read_text(encoding="utf-8")

    assert "&lt;R3&amp;B&gt;" in html

    with pytest.raises(FileExistsError):
        gp3.create_gazepoint_report(
            _valid_report_results(),
            output_file=output,
            overwrite=False,
            save_plots=False,
        )


def test_r3b_report_legacy_metadata_path(
    tmp_path,
):
    output = tmp_path / "legacy.html"

    result = gp3.create_gazepoint_report(
        {
            "Results": pd.DataFrame(
                {
                    "x": [
                        1,
                    ]
                }
            )
        },
        output_file=output,
        metadata={"legacy": True},
    )

    assert output.exists()

    assert result is not None


def test_r3b_multiverse_failure_statuses():
    multiverse = gp3.create_gazepoint_preprocessing_multiverse()

    aoi = gp3.run_gazepoint_aoi_multiverse(
        pd.DataFrame(
            {
                "subject": [
                    "S1",
                ],
                "condition": [
                    "A",
                ],
                "trial": [
                    1,
                ],
                "time": [
                    0.0,
                ],
                "aoi": [
                    "target",
                ],
            }
        ),
        multiverse=multiverse,
        windows={
            "early": [
                0.0,
                0.2,
            ]
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
        keep_outputs=False,
        stop_on_error=False,
    )

    assert (
        aoi["overview"].loc[
            0,
            "multiverse_status",
        ]
        == "completed_with_errors"
    )

    pupil = gp3.run_gazepoint_pupil_multiverse(
        pd.DataFrame(
            {
                "subject": [
                    "S1",
                ],
                "condition": [
                    "A",
                ],
                "trial": [
                    1,
                ],
                "time": [
                    0.0,
                ],
                "pupil": [
                    3.0,
                ],
            }
        ),
        multiverse=multiverse,
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

    assert (
        pupil["overview"].loc[
            0,
            "multiverse_status",
        ]
        == "completed_with_errors"
    )


def test_r3b_workflow_missing_directory():
    with pytest.raises(FileNotFoundError):
        gp3.run_gazepoint_workflow(
            export_dir=("__definitely_missing_r3b__"),
            check_file_pairs=False,
            create_report=False,
            save_plots=False,
        )


# === R3B LIVE BRANCH COVERAGE PASS 3 ===


def _r3b_direct():
    import gp3tools._behavioral_r3b as module

    return module


def _r3b_multiverse():
    return gp3.create_gazepoint_preprocessing_multiverse()


def _r3b_aoi_data():
    return pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
                "S2",
                "S2",
            ],
            "condition": [
                "A",
                "A",
                "A",
                "A",
            ],
            "trial": [
                1,
                1,
                1,
                1,
            ],
            "time": [
                0.05,
                0.10,
                0.05,
                0.10,
            ],
            "aoi": [
                "target",
                "distractor",
                "target",
                "other",
            ],
        }
    )


def _r3b_divergence_data(
    difference=1.0,
):
    rows = []

    for subject in (
        "S1",
        "S2",
        "S3",
    ):
        for time in (
            0.0,
            0.1,
            0.2,
        ):
            rows.append(
                {
                    "subject": subject,
                    "trial": 1,
                    "condition": "A",
                    "time": time,
                    "outcome": 0.0,
                }
            )

            rows.append(
                {
                    "subject": subject,
                    "trial": 1,
                    "condition": "B",
                    "time": time,
                    "outcome": float(difference),
                }
            )

    return pd.DataFrame(rows)


def _write_r3b_gaze(
    path,
    *,
    include_aoi=True,
    one_sample=False,
    include_pupils=True,
):
    frame = pd.DataFrame(
        {
            "TIME": (
                [
                    0.0,
                ]
                if one_sample
                else [
                    0.0,
                    0.016,
                ]
            ),
            "MEDIA_ID": (
                [
                    1,
                ]
                if one_sample
                else [
                    1,
                    1,
                ]
            ),
            "MEDIA_NAME": (
                [
                    "stimulus",
                ]
                if one_sample
                else [
                    "stimulus",
                    "stimulus",
                ]
            ),
            "FPOGV": (
                [
                    1,
                ]
                if one_sample
                else [
                    1,
                    1,
                ]
            ),
        }
    )

    if include_pupils:
        frame["LPV"] = 1

        frame["RPV"] = 1

    if include_aoi:
        frame["AOI"] = "target"

    frame.to_csv(
        path,
        index=False,
    )


def _write_r3b_fixations(
    path,
):
    pd.DataFrame(
        {
            "MEDIA_ID": [
                1,
            ],
            "MEDIA_NAME": [
                "stimulus",
            ],
            "AOI": [
                "target",
            ],
            "FPOGD": [
                0.1,
            ],
            "FPOGS": [
                0.05,
            ],
        }
    ).to_csv(
        path,
        index=False,
    )


def _write_r3b_empty_fixations(
    path,
):
    pd.DataFrame(
        columns=[
            "MEDIA_ID",
            "MEDIA_NAME",
            "AOI",
            "FPOGD",
            "FPOGS",
        ]
    ).to_csv(
        path,
        index=False,
    )


def test_r3b_report_direct_validation_rendering_and_plots(
    tmp_path,
    monkeypatch,
):
    r3b = _r3b_direct()

    output = tmp_path / "diagnostic.html"

    with pytest.raises(ValueError):
        r3b.create_gazepoint_report(
            [],
            output,
            save_plots=False,
        )

    with pytest.raises(ValueError):
        r3b.create_gazepoint_report(
            {"sampling": pd.DataFrame()},
            output,
            save_plots=False,
        )

    malformed = {
        "sampling": pd.DataFrame(),
        "quality": "not a frame",
        "flagged_quality": pd.DataFrame(),
        "aoi_table": pd.DataFrame(),
    }

    with pytest.raises(ValueError):
        r3b.create_gazepoint_report(
            malformed,
            output,
            save_plots=False,
        )

    valid = {
        "sampling": pd.DataFrame(
            {
                "recording": [
                    "S1",
                    "S2",
                ],
                "rate": [
                    60.0,
                    58.5,
                ],
                "accepted": [
                    True,
                    False,
                ],
            }
        ),
        "quality": pd.DataFrame(
            {
                "recording": [
                    "S1",
                    "S2",
                ],
                "score": [
                    98.25,
                    float("nan"),
                ],
            }
        ),
        "flagged_quality": pd.DataFrame(
            {
                "recording": [
                    "<S1>",
                    "S2",
                ],
                "reason": [
                    "A&B",
                    "none",
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
                    10,
                    5,
                ],
            }
        ),
    }

    monkeypatch.setattr(
        gp3,
        "save_gazepoint_plots",
        lambda **kwargs: pd.DataFrame(
            {
                "plot": [
                    "sampling <rate>",
                ],
                "file": [
                    str(tmp_path / "sampling.png"),
                ],
            }
        ),
    )

    result = r3b.create_gazepoint_report(
        valid,
        output,
        title="A&B <Report>",
        overwrite=True,
        max_rows=1,
        save_plots=True,
        plot_dir=(tmp_path / "plots"),
    )

    assert pd.isna(
        result.loc[
            0,
            "n_flagged",
        ]
    )

    html = output.read_text(encoding="utf-8")

    assert "A&amp;B &lt;Report&gt;" in html

    assert "Showing first 1 of 2 rows" in html

    assert "Diagnostic plots" in html

    assert "sampling &lt;rate&gt;" in html

    assert "<p>No rows available.</p>" not in html

    with pytest.raises(FileExistsError):
        r3b.create_gazepoint_report(
            valid,
            output,
            overwrite=False,
            save_plots=False,
        )

    with pytest.raises(TypeError):
        r3b.create_gazepoint_report(
            valid,
            tmp_path / "metadata.html",
            metadata={
                "legacy": True,
            },
            save_plots=False,
        )


def test_r3b_report_empty_and_flagged_table_paths(
    tmp_path,
):
    r3b = _r3b_direct()

    results = {
        "all_gaze": pd.DataFrame(
            {
                "x": [
                    1,
                ]
            }
        ),
        "all_fix": pd.DataFrame(
            {
                "x": [
                    1,
                ]
            }
        ),
        "sampling": pd.DataFrame(
            {
                "recording": [
                    "S1",
                ],
                "rate": [
                    60,
                ],
            }
        ),
        "quality": pd.DataFrame(
            {
                "recording": [
                    "S1",
                ],
                "valid": [
                    100,
                ],
            }
        ),
        "flagged_quality": pd.DataFrame(
            {
                "recording": [
                    "S1",
                ],
                "review_required": [
                    True,
                ],
            }
        ),
        "aoi_table": pd.DataFrame(
            columns=[
                "AOI",
                "count",
            ]
        ),
    }

    output = tmp_path / "flagged.html"

    result = r3b.create_gazepoint_report(
        results,
        output,
        save_plots=False,
        max_rows=30,
    )

    assert (
        result.loc[
            0,
            "n_flagged",
        ]
        == 1
    )

    html = output.read_text(encoding="utf-8")

    assert "No rows available." in html

    assert "TRUE" in html


def test_r3b_divergence_validation_row_bootstrap_and_retention():
    r3b = _r3b_direct()

    data = _r3b_divergence_data(1.0)

    with pytest.raises(ValueError):
        r3b.estimate_gazepoint_divergence_point(
            data,
            outcome_col="outcome",
            time_col="time",
            condition_col="condition",
            participant_col=None,
            comparison=[
                "A",
                "B",
            ],
            bootstrap_unit="participant",
            n_boot=3,
        )

    with pytest.raises(ValueError):
        r3b.estimate_gazepoint_divergence_point(
            data,
            outcome_col="outcome",
            time_col="time",
            condition_col="condition",
            participant_col="subject",
            comparison=[
                "A",
            ],
            bootstrap_unit="participant",
            n_boot=3,
        )

    result = r3b.estimate_gazepoint_divergence_point(
        data,
        outcome_col="outcome",
        time_col="time",
        condition_col="condition",
        participant_col=None,
        trial_col="trial",
        comparison=[
            "A",
            "B",
        ],
        bootstrap_unit="row",
        summary_function="mean",
        n_boot=4,
        ci=0.8,
        consecutive_points=1,
        min_abs_difference=0,
        direction="greater",
        seed=12,
        keep_bootstrap=True,
        name="edge_row_bootstrap",
    )

    assert isinstance(
        result["bootstrap_differences"],
        pd.DataFrame,
    )

    assert len(result["bootstrap_differences"]) > 0

    assert (
        result["divergence_point"].loc[
            0,
            "observed_direction",
        ]
        == "positive"
    )


def test_r3b_divergence_negative_and_no_divergence_paths():
    r3b = _r3b_direct()

    negative = r3b.estimate_gazepoint_divergence_point(
        _r3b_divergence_data(-1.0),
        outcome_col="outcome",
        time_col="time",
        condition_col="condition",
        participant_col="subject",
        trial_col="trial",
        comparison=[
            "A",
            "B",
        ],
        bootstrap_unit="participant",
        n_boot=4,
        ci=0.8,
        consecutive_points=1,
        min_abs_difference=0,
        direction="less",
        seed=23,
        keep_bootstrap=False,
        name="negative",
    )

    assert (
        negative["divergence_point"].loc[
            0,
            "observed_direction",
        ]
        == "negative"
    )

    none = r3b.estimate_gazepoint_divergence_point(
        _r3b_divergence_data(0.1),
        outcome_col="outcome",
        time_col="time",
        condition_col="condition",
        participant_col="subject",
        trial_col="trial",
        comparison=[
            "A",
            "B",
        ],
        bootstrap_unit="participant",
        n_boot=3,
        ci=0.8,
        consecutive_points=2,
        min_abs_difference=100,
        direction="two_sided",
        seed=31,
        keep_bootstrap=False,
        name="none",
    )

    assert (
        none["divergence_point"].loc[
            0,
            "detector_status",
        ]
        == "no_divergence"
    )

    assert pd.isna(
        none["divergence_point"].loc[
            0,
            "observed_difference_at_onset",
        ]
    )


def test_r3b_aoi_multiverse_selection_success_empty_and_stop():
    r3b = _r3b_direct()

    data = _r3b_aoi_data()

    multiverse = _r3b_multiverse()

    completed = r3b.run_gazepoint_aoi_multiverse(
        data,
        multiverse=multiverse,
        branch_ids=[
            "mv_aoi_1",
        ],
        windows=[
            0.0,
            0.2,
        ],
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

    assert (
        completed["branch_results"].loc[
            0,
            "branch_status",
        ]
        == "completed"
    )

    assert (
        completed["overview"].loc[
            0,
            "multiverse_status",
        ]
        == "completed"
    )

    empty = r3b.run_gazepoint_aoi_multiverse(
        data,
        multiverse=multiverse,
        branch_ids=[],
        windows=None,
        time_col="time",
        aoi_col="aoi",
        subject_col="subject",
        condition_col="condition",
        group_cols=[
            "subject",
            "condition",
            "trial",
        ],
        target_aoi_values="target",
        distractor_aoi_values=None,
        keep_outputs=False,
        stop_on_error=False,
    )

    assert (
        empty["overview"].loc[
            0,
            "multiverse_status",
        ]
        == "no_branches_requested"
    )

    with pytest.raises(ValueError):
        r3b.run_gazepoint_aoi_multiverse(
            data,
            multiverse=multiverse,
            branch_ids=[
                "mv_aoi_1",
            ],
            windows={
                "early": [
                    0.0,
                    0.2,
                ]
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
            keep_outputs=False,
            stop_on_error=True,
        )


def test_r3b_pupil_multiverse_selection_success_empty_and_stop():
    r3b = _r3b_direct()

    multiverse = _r3b_multiverse()

    data = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
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
                3.2,
            ],
        }
    )

    completed = r3b.run_gazepoint_pupil_multiverse(
        data,
        multiverse=multiverse,
        branch_ids=[
            "mv_pupil_1",
        ],
        pupil_col="pupil",
        time_col="time",
        group_cols=[
            "subject",
            "trial",
        ],
        summarise_windows=False,
        windows=None,
        keep_outputs=False,
        stop_on_error=False,
    )

    assert (
        completed["branch_results"].loc[
            0,
            "branch_status",
        ]
        == "completed"
    )

    assert (
        completed["overview"].loc[
            0,
            "multiverse_status",
        ]
        == "completed"
    )

    empty = r3b.run_gazepoint_pupil_multiverse(
        data,
        multiverse=multiverse,
        branch_ids=[],
        pupil_col="pupil",
        time_col="time",
        group_cols=[
            "subject",
            "trial",
        ],
        summarise_windows=False,
        windows=None,
        keep_outputs=False,
        stop_on_error=False,
    )

    assert (
        empty["overview"].loc[
            0,
            "multiverse_status",
        ]
        == "no_branches_requested"
    )

    with pytest.raises(ValueError):
        r3b.run_gazepoint_pupil_multiverse(
            data.assign(condition="A"),
            multiverse=multiverse,
            branch_ids=[
                "mv_pupil_1",
            ],
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
            stop_on_error=True,
        )


def test_r3b_workflow_single_sample_no_fix_and_missing_metrics(
    tmp_path,
):
    r3b = _r3b_direct()

    input_dir = tmp_path / "input"

    input_dir.mkdir()

    _write_r3b_gaze(
        input_dir / "one_all_gaze.csv",
        one_sample=True,
        include_aoi=True,
        include_pupils=False,
    )

    _write_r3b_empty_fixations(
        input_dir / "empty_fixations.csv",
    )

    result = r3b.run_gazepoint_workflow(
        export_dir=input_dir,
        all_gaze_pattern="all_gaze",
        fixation_pattern="fixations",
        check_file_pairs=False,
        group_cols=[
            "USER",
            "MEDIA_ID",
        ],
        user_col="USER",
        sample_rate=None,
        min_gaze_valid_pct=70,
        min_pupil_valid_pct=70,
        expected_hz=None,
        hz_tolerance=5,
        min_duration_sec=1,
        output_dir=None,
        save_plots=False,
        create_report=False,
    )

    sampling = result["sampling"]

    assert pd.isna(
        sampling.loc[
            0,
            "mean_interval_ms",
        ]
    )

    assert pd.isna(
        sampling.loc[
            0,
            "estimated_hz",
        ]
    )

    assert not bool(result["flagged_quality"].loc[0, "flag_sampling_rate"])

    assert result["all_fix"].empty

    assert pd.isna(
        result["aoi_table"].loc[
            0,
            "fixation_count",
        ]
    )


def test_r3b_workflow_fixation_only_aoi_join_and_missing_files(
    tmp_path,
):
    r3b = _r3b_direct()

    empty_dir = tmp_path / "empty"

    empty_dir.mkdir()

    with pytest.raises(ValueError):
        r3b.run_gazepoint_workflow(
            export_dir=empty_dir,
            all_gaze_pattern="all_gaze",
            fixation_pattern=None,
            check_file_pairs=False,
            group_cols=[
                "USER",
                "MEDIA_ID",
            ],
            user_col="USER",
            create_report=False,
            save_plots=False,
        )

    input_dir = tmp_path / "fix_only_join"

    input_dir.mkdir()

    _write_r3b_gaze(
        input_dir / "case_all_gaze.csv",
        include_aoi=False,
    )

    _write_r3b_fixations(
        input_dir / "case_fixations.csv",
    )

    result = r3b.run_gazepoint_workflow(
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
        output_dir=None,
        save_plots=False,
        create_report=False,
    )

    assert len(result["aoi_table"]) == 1

    assert (
        result["aoi_table"].loc[
            0,
            "fixation_count",
        ]
        == 1
    )

    assert pd.isna(
        result["aoi_table"].loc[
            0,
            "sample_count",
        ]
    )


def test_r3b_workflow_output_plot_and_report_paths(
    tmp_path,
    monkeypatch,
):
    r3b = _r3b_direct()

    input_dir = tmp_path / "input"

    input_dir.mkdir()

    _write_r3b_gaze(
        input_dir / "case_all_gaze.csv",
        include_aoi=True,
    )

    _write_r3b_empty_fixations(
        input_dir / "empty_fixations.csv",
    )

    calls = {
        "write": 0,
        "plots": 0,
        "report": 0,
    }

    def fake_write(
        **kwargs,
    ):
        calls["write"] += 1

        return pd.DataFrame(
            {
                "table": [
                    "sampling",
                ],
                "file": [
                    "sampling.csv",
                ],
            }
        )

    def fake_plots(
        **kwargs,
    ):
        calls["plots"] += 1

        return pd.DataFrame(
            {
                "plot": [
                    "sampling",
                ],
                "file": [
                    str(tmp_path / "sampling.png"),
                ],
            }
        )

    def fake_report(
        *args,
        **kwargs,
    ):
        calls["report"] += 1

        return pd.DataFrame(
            {
                "report": [
                    str(kwargs["output_file"]),
                ],
                "plot_dir": [
                    str(tmp_path / "report_files"),
                ],
                "n_flagged": [
                    0,
                ],
            }
        )

    monkeypatch.setattr(
        gp3,
        "write_gazepoint_outputs",
        fake_write,
    )

    monkeypatch.setattr(
        gp3,
        "save_gazepoint_plots",
        fake_plots,
    )

    monkeypatch.setattr(
        r3b,
        "create_gazepoint_report",
        fake_report,
    )

    output_dir = tmp_path / "output"

    result = r3b.run_gazepoint_workflow(
        export_dir=input_dir,
        all_gaze_pattern="all_gaze",
        fixation_pattern="fixations",
        check_file_pairs=False,
        group_cols=[
            "USER",
            "MEDIA_ID",
        ],
        user_col="USER",
        output_dir=output_dir,
        prefix="edge",
        save_plots=True,
        plot_output_dir=None,
        create_report=True,
        report_file=None,
    )

    assert calls == {
        "write": 1,
        "plots": 1,
        "report": 1,
    }

    assert result["written_files"] is not None

    assert result["written_plots"] is not None

    assert result["written_report"] is not None


def test_r3b_workflow_output_requirements(
    tmp_path,
):
    r3b = _r3b_direct()

    input_dir = tmp_path / "input"

    input_dir.mkdir()

    _write_r3b_gaze(
        input_dir / "case_all_gaze.csv",
    )

    _write_r3b_empty_fixations(
        input_dir / "empty_fixations.csv",
    )

    with pytest.raises(ValueError):
        r3b.run_gazepoint_workflow(
            export_dir=input_dir,
            all_gaze_pattern="all_gaze",
            fixation_pattern="fixations",
            check_file_pairs=False,
            group_cols=[
                "USER",
                "MEDIA_ID",
            ],
            user_col="USER",
            output_dir=None,
            save_plots=True,
            plot_output_dir=None,
            create_report=False,
        )

    with pytest.raises(ValueError):
        r3b.run_gazepoint_workflow(
            export_dir=input_dir,
            all_gaze_pattern="all_gaze",
            fixation_pattern="fixations",
            check_file_pairs=False,
            group_cols=[
                "USER",
                "MEDIA_ID",
            ],
            user_col="USER",
            output_dir=None,
            save_plots=False,
            create_report=True,
            report_file=None,
        )
