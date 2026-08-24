from pathlib import Path

import numpy as np
import pandas as pd

import gp3tools as gp3


def test_events_full_family(tmp_path: Path):
    samples = gp3.simulate_gazepoint_fixations(
        n_fixations=12, samples_per_fixation=8, random_state=4
    )
    vel = gp3.detect_gazepoint_fixations_velocity(
        samples, velocity_threshold=2.0, min_duration_ms=10
    )
    ivt = gp3.detect_gazepoint_fixations_ivt(samples, velocity_threshold=2.0, min_duration_ms=10)
    hmm = gp3.classify_gazepoint_events_hmm(samples)
    assert {"fixation", "fixation_id"} <= set(vel)
    assert len(ivt) == len(samples)
    assert "event_state" in hmm

    sacc = gp3.compute_gazepoint_saccade_metrics(hmm, event_col="event_state")
    assert isinstance(sacc, pd.DataFrame)
    fix_summary = gp3.summarise_fixations(ivt)
    assert isinstance(fix_summary, pd.DataFrame)

    if len(fix_summary):
        fix_summary["subject"] = [f"S{i % 3}" for i in range(len(fix_summary))]
        fix_summary["trial_global"] = [f"T{i % 2}" for i in range(len(fix_summary))]
        trials = gp3.summarise_gazepoint_fixation_trials(
            fix_summary, trial_col="trial_global", subject_col="subject"
        )
        assert isinstance(trials, pd.DataFrame)
        rel = gp3.audit_gazepoint_fixation_reliability(fix_summary, subject_col="subject")
        assert isinstance(rel, pd.DataFrame)

    agreement = gp3.compare_gazepoint_event_detectors(
        samples, velocity_threshold=2.0, min_duration_ms=10
    )
    assert "agreement" in agreement
    agree_summary = gp3.summarise_gazepoint_event_detector_agreement(agreement)
    assert len(agree_summary) >= 1
    bench = gp3.benchmark_gazepoint_event_detectors(
        samples, repeats=2, velocity_threshold=2.0, min_duration_ms=10
    )
    assert len(bench) >= 2
    bench_summary = gp3.summarise_gazepoint_event_detector_benchmark(bench)
    assert len(bench_summary) >= 1
    review_path = tmp_path / "review.csv"
    review = gp3.create_gazepoint_event_review_template(ivt, path=review_path)
    assert review_path.exists() and isinstance(review, pd.DataFrame)


def test_face_workflows_and_aliases():
    gaze = gp3.load_example_master().head(100).copy()
    face = pd.DataFrame(
        {
            "Timestamp": gaze["TIME"].to_numpy(),
            "Face Confidence": np.linspace(0.7, 1.0, len(gaze)),
            "Smile": np.linspace(0, 1, len(gaze)),
            "Brow Raise": np.linspace(1, 0, len(gaze)),
        }
    )
    std = gp3.standardize_gazepoint_face_columns(face)
    assert any("confidence" in c.lower() for c in std.columns)
    quality = gp3.audit_gazepoint_face_quality(std)
    assert len(quality) >= 1
    assert len(gp3.summarize_gazepoint_face_quality(std)) >= 1
    assert len(gp3.summarise_gazepoint_face_quality(std)) >= 1

    synced = gp3.sync_gazepoint_face_data(
        gaze, std, gaze_time_col="TIME", face_time_col="timestamp"
    )
    assert len(synced) == len(gaze)
    sync_audit = gp3.audit_gazepoint_face_sync(
        gaze, std, gaze_time_col="TIME", face_time_col="timestamp"
    )
    assert isinstance(sync_audit, pd.DataFrame)
    event_audit = gp3.audit_gazepoint_event_sync(
        gaze, std, gaze_time_col="TIME", face_time_col="timestamp"
    )
    assert isinstance(event_audit, pd.DataFrame)

    windows = gp3.summarize_gazepoint_face_windows(synced, group_cols=["condition"])
    assert isinstance(windows, pd.DataFrame)
    assert isinstance(
        gp3.summarise_gazepoint_face_windows(synced, group_cols=["condition"]), pd.DataFrame
    )
    react = gp3.summarize_gazepoint_face_reactivity(synced, group_cols=["condition"])
    assert isinstance(react, pd.DataFrame)
    assert isinstance(
        gp3.summarise_gazepoint_face_reactivity(synced, group_cols=["condition"]), pd.DataFrame
    )
    mm = gp3.prepare_gazepoint_multimodal_data(
        gaze, std, gaze_time_col="TIME", face_time_col="timestamp"
    )
    assert len(mm) == len(gaze)
    checklist = gp3.create_gazepoint_face_reporting_checklist(std)
    assert len(checklist) >= 1


def test_interop_adapters_and_exports(tmp_path: Path):
    df = gp3.load_example_master().head(100).copy()
    etr = gp3.prepare_gazepoint_eyetrackingr_data(
        df,
        time_col="TIME",
        x_col="FPOGX",
        y_col="FPOGY",
        participant_col="subject",
        trial_col="trial_global",
    )
    assert {"Participant", "Time", "X", "Y"} <= set(etr)
    pup = gp3.prepare_gazepoint_pupillometryr_data(
        df, participant_col="subject", time_col="TIME", pupil_col="pupil"
    )
    assert len(pup) == len(df)
    assert len(gp3.prepare_gazepoint_gazer_data(df)) == len(df)
    assert len(gp3.prepare_gazepoint_eyetools_data(df)) == len(df)
    bio = gp3.prepare_gazepoint_gpbiometrics_bridge(df, participant_col="subject", time_col="TIME")
    assert len(bio) == len(df)

    hddm_df = df.assign(response=np.arange(len(df)) % 2, rt=0.5)
    hddm = gp3.prepare_gazepoint_hddm_export(
        hddm_df,
        response_col="response",
        rt_col="rt",
        subject_col="subject",
        condition_col="condition",
    )
    assert isinstance(hddm, pd.DataFrame)
    script_path = tmp_path / "fit_hddm.py"
    script = gp3.create_gazepoint_hddm_fit_script(script_path)
    assert "HDDM" in script and script_path.exists()

    for fun, name in [
        (gp3.export_gazepoint_mne_cluster_input, "mne.csv"),
        (gp3.export_gazepoint_permuco_cluster_input, "permuco.csv"),
        (gp3.export_gazepoint_permutes_cluster_input, "permutes.csv"),
    ]:
        path = tmp_path / name
        result = fun(df, path=path)
        assert path.exists() and result is not None

    bids = gp3.export_gazepoint_to_bids(df, tmp_path / "bids", subject_col="subject", task="demo")
    assert bids["files"]
    assert (tmp_path / "bids" / "dataset_description.json").exists()

    assert isinstance(gp3.run_gazepoint_gazer_crosscheck(df), dict)
    assert isinstance(gp3.run_gazepoint_eyetools_fixation_detection(df), pd.DataFrame)
    assert isinstance(gp3.run_gazepoint_gpbiometrics_workflow(df), dict)
