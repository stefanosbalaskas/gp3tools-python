from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3


def test_io_classification_summary_face_and_inspection(tmp_path: Path):
    gaze = tmp_path / "User 9_all_gaze.csv"
    gaze.write_text("TIME(test),FPOGX,FPOGY,Unnamed: 3\n0,0.1,0.2,\n", encoding="utf-8")
    fix = tmp_path / "User 9_fixations.csv"
    fix.write_text("TIME,FPOGX,FPOGY\n0,0.1,0.2\n", encoding="utf-8")
    summary = tmp_path / "Data_Summary_export_demo.csv"
    summary.write_text("Gazepoint Analysis,7.2\nAOI,Count\nleft,3\n\n", encoding="utf-8")
    unknown = tmp_path / "misc.csv"
    unknown.write_text("foo,bar\n1,2\n", encoding="utf-8")

    assert gp3.classify_gazepoint_export(gaze) == "all_gaze"
    assert gp3.classify_gazepoint_export(fix) == "fixations"
    assert gp3.classify_gazepoint_export(summary) == "summary"
    assert gp3.classify_gazepoint_export(unknown) == "unknown"
    with pytest.raises(FileNotFoundError):
        gp3.classify_gazepoint_export(tmp_path / "missing.csv")

    df = gp3.read_gazepoint(gaze)
    assert list(df.columns) == ["TIME", "FPOGX", "FPOGY"]
    assert df.attrs["gp3_source_file"] == gaze.name
    raw = gp3.read_gazepoint(gaze, standardise_names=False, drop_empty_cols=False)
    assert "TIME(test)" in raw.columns
    with pytest.raises(ValueError):
        gp3.read_gazepoint(summary)

    parsed = gp3.read_gazepoint_summary(summary)
    assert parsed["source_file"] == summary.name
    assert "raw" in parsed and isinstance(parsed["tables"], list)

    face_tsv = tmp_path / "face.tsv"
    face_tsv.write_text("time\tFace Confidence\n0\t0.9\n", encoding="utf-8")
    face = gp3.read_gazepoint_face_export(face_tsv)
    assert face.attrs["gp3_source_file"] == face_tsv.name
    assert "Face Confidence" in face

    inspect = gp3.inspect_gazepoint_columns(df)
    assert {"column", "dtype", "n_missing", "missing_prop", "n_unique"} <= set(inspect)


def test_folder_reader_recursive_and_errors(tmp_path: Path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "User 1_all_gaze.csv").write_text("TIME,FPOGX,FPOGY\n0,0.1,0.2\n", encoding="utf-8")
    (nested / "User 2_all_gaze.csv").write_text("TIME,FPOGX,FPOGY\n0,0.3,0.4\n", encoding="utf-8")
    (tmp_path / "Data_Summary_export.csv").write_text("Gazepoint Analysis,7.2\n", encoding="utf-8")

    one = gp3.read_gazepoint_folder(tmp_path, pattern=r"all_gaze\.csv$")
    assert len(one) == 1
    both = gp3.read_gazepoint_folder(
        tmp_path, pattern=r"all_gaze\.csv$", recursive=True, source_col="source"
    )
    assert len(both) == 2 and "source" in both
    with pytest.raises(FileNotFoundError):
        gp3.read_gazepoint_folder(tmp_path, pattern=r"nomatch$")
    with pytest.raises(FileNotFoundError):
        gp3.read_gazepoint_folder(tmp_path / "absent")
    with pytest.raises(ValueError):
        gp3.read_gazepoint_folder(tmp_path, pattern=r"Summary.*\.csv$")


def test_master_qc_and_coordinate_helpers(tmp_path: Path):
    raw = pd.DataFrame(
        {
            "Participant": ["S1"] * 4,
            "TIME": [0.0, 1 / 60, 2 / 60, 3 / 60],
            "FPOGX": [0.2, 0.3, 1.2, np.nan],
            "FPOGY": [0.3, 0.4, 0.5, np.nan],
            "pupil": [3.0, 3.1, np.nan, 3.2],
            "condition": ["A"] * 4,
            "MEDIA_ID": [1] * 4,
            "TRACKLOSS": [0, 0, 1, 1],
        }
    )
    master = gp3.create_gazepoint_master(raw)
    assert master.attrs["gp3_class"] == "gazepoint_master"
    assert "sample_index" in master

    valid = gp3.validate_gazepoint_master(master, required=("subject", "time"))
    assert isinstance(valid["valid"], bool)
    invalid = gp3.validate_gazepoint_master(pd.DataFrame({"x": [1]}), required=("subject", "time"))
    assert not invalid["valid"]

    audit = gp3.audit_gazepoint_master(master)
    assert isinstance(audit, dict)
    sr = gp3.check_sampling_rate(master, time_col="time")
    assert sr.iloc[0]["sampling_hz"] == pytest.approx(60)

    tracking = gp3.summarise_tracking_quality(
        master, validity_col="TRACKLOSS", x_col="x", y_col="y"
    )
    assert len(tracking) == 1
    flagged = gp3.flag_tracking_quality(
        master, validity_col="TRACKLOSS", x_col="x", y_col="y", min_usable_prop=0.8
    )
    assert len(flagged) >= 1
    dropped = gp3.clean_gazepoint_by_trackloss(master, validity_col="TRACKLOSS", drop=True)
    assert len(dropped) < len(master)
    kept = gp3.clean_gazepoint_by_trackloss(master, validity_col="TRACKLOSS", drop=False)
    assert len(kept) == len(master)

    miss = gp3.summarise_gazepoint_missingness(master, columns=["x", "pupil"])
    assert set(miss["column"]) == {"x", "pupil"}
    bounds = gp3.audit_gazepoint_screen_bounds(master, x_col="x", y_col="y")
    assert bounds["n_outside"].iloc[0] >= 1
    coverage = gp3.summarise_gazepoint_coordinate_coverage(master, x_col="x", y_col="y")
    assert coverage["n_samples"].iloc[0] == 4

    pixel = pd.DataFrame({"x": [0, 960, 1920], "y": [0, 540, 1080]})
    norm = gp3.harmonize_gazepoint_screen_coordinates(
        pixel, x_col="x", y_col="y", width=1920, height=1080
    )
    assert norm["x_norm"].tolist() == pytest.approx([0, 0.5, 1.0])

    phases = gp3.segment_gazepoint_task_phases(
        master, time_col="time", boundaries=[0, 0.03, 0.06], labels=["early", "late"]
    )
    phase_summary = gp3.summarise_gazepoint_phase_coverage(phases, group_cols=["subject"])
    assert "n_samples" in phase_summary

    qc = gp3.collect_gazepoint_qc_summaries(master)
    status = gp3.summarise_gazepoint_qc_status(qc)
    assert set(status["component"]) >= {"master", "tracking", "missingness"}

    readiness = gp3.check_gazepoint_real_data_readiness(master)
    assert "checks" in readiness
    exclusions = gp3.recommend_gazepoint_exclusions(
        master,
        participant_col="subject",
        trial_col="trial_global",
        validity_col="TRACKLOSS",
        x_col="x",
        y_col="y",
        pupil_col="pupil",
        min_trial_samples=1,
    )
    assert {"trial_recommendations", "participant_recommendations", "exclusions"} <= set(exclusions)

    flow_data = master.assign(exclude_a=[False, True, False, False], exclude_b=False)
    flow = gp3.audit_gazepoint_exclusion_flow(flow_data)
    assert flow.iloc[-1]["n"] <= len(master)
    post = gp3.audit_gazepoint_post_exclusion_balance(
        flow_data.assign(excluded=flow_data.exclude_a), group_cols=("condition",)
    )
    assert "n_retained" in post
    balance = gp3.audit_gazepoint_design_balance(master, group_cols=("subject", "condition"))
    assert "n" in balance

    naming = gp3.audit_gazepoint_naming_consistency(["summarise_x", "summarize_x"])
    assert naming["summary"]["n_names"].iloc[0] == 2
    policy = gp3.gp3tools_naming_policy()
    assert "canonical" in policy
    out = gp3.write_gazepoint_naming_audit(tmp_path / "naming.csv", ["summarise_x"])
    assert out.exists()


def test_file_pair_audit(tmp_path: Path):
    for name in ["User 1_all_gaze.csv", "User 1_fixations.csv", "User 2_all_gaze.csv"]:
        (tmp_path / name).write_text("x\n1\n", encoding="utf-8")
    pairs = gp3.check_gazepoint_file_pairs(tmp_path)
    assert pairs.loc[pairs.user.eq("1"), "paired"].iloc[0]
    assert not pairs.loc[pairs.user.eq("2"), "paired"].iloc[0]


def test_simulation_and_misc_helpers():
    sim = gp3.simulate_gazepoint_data(
        n_subjects=2, n_trials=2, samples_per_trial=20, random_state=1
    )
    assert len(sim) == 80
    pupil = gp3.simulate_gazepoint_pupil_data(
        n_subjects=2, n_trials=2, samples_per_trial=20, random_state=1
    )
    assert len(pupil) == 80
    tc = gp3.simulate_gazepoint_cluster_timecourse_data(
        n_subjects=4, n_time=10, effect_window=(3, 6), random_state=1
    )
    assert {"subject", "condition", "time_bin", "value"} <= set(tc)

    heat = gp3.prepare_gazepoint_heatmap_data(sim, x_col="FPOGX", y_col="FPOGY", bins=8)
    assert len(heat) > 0
    rec = gp3.recalibrate_gazepoint_gaze(sim, x_col="FPOGX", y_col="FPOGY", method="offset")
    assert {"x_recalibrated", "y_recalibrated"} <= set(rec)
    rec2 = gp3.recalibrate_gazepoint_gaze(sim, x_col="FPOGX", y_col="FPOGY", method="scale")
    assert len(rec2) == len(sim)

    scores = pd.DataFrame({"uncertainty": [0.1, 0.8, 0.5], "score": [1, 3, 2]})
    filt = gp3.filter_gazepoint_cnn_uncertainty(scores, threshold=0.5)
    assert "cnn_uncertainty_pass" in filt
    chosen = gp3.select_gazepoint_adaptive_trial(
        scores, score_col="score", strategy="highest_uncertainty"
    )
    assert len(chosen) >= 1
