from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3


def test_example_datasets_load():
    master = gp3.load_example_master()
    assert len(master) > 100
    assert {"subject", "TIME", "FPOGX", "FPOGY", "pupil", "aoi_current"} <= set(master)
    assert len(gp3.load_example_fixations()) > 0


def test_read_and_standardise(tmp_path: Path):
    p = tmp_path / "User 1_all_gaze.csv"
    p.write_text("TIME(test),FPOGX,FPOGY,\\n0,0.4,0.5,\\n", encoding="utf-8")
    df = gp3.read_gazepoint(p)
    assert "TIME" in df
    assert df.attrs["gp3_file_type"] == "all_gaze"


def test_sampling_rate_about_60_hz():
    master = gp3.load_example_master()
    out = gp3.check_sampling_rate(master, time_col="TIME", group_cols=["subject", "trial_global"])
    assert np.nanmedian(out["sampling_hz"]) == pytest.approx(60, rel=0.02)


def test_pupil_workflow_and_binocular():
    master = gp3.load_example_master().copy()
    master.loc[10:12, "pupil"] = np.nan
    clean = gp3.interpolate_gazepoint_pupil(
        master, pupil_col="pupil", time_col="TIME", group_cols=["subject", "trial_global"]
    )
    assert clean["pupil_interpolated"].notna().sum() >= master["pupil"].notna().sum()
    recon = gp3.reconstruct_gazepoint_binocular_pupil(master, left_col="LPMM", right_col="RPMM")
    assert "pupil_combined" in recon
    diag = gp3.diagnose_gazepoint_binocular_pupil(master, left_col="LPMM", right_col="RPMM")
    assert len(diag) >= 1


def test_aoi_and_sequence_workflows():
    master = gp3.load_example_master()
    matrix = gp3.compute_gazepoint_aoi_transition_matrix(master, aoi_col="aoi_current")
    assert matrix.shape[0] >= 2
    entropy = gp3.compute_gazepoint_aoi_entropy(
        master, aoi_col="aoi_current", group_cols=["subject"]
    )
    assert len(entropy) == master["subject"].nunique()
    seq = gp3.prepare_gazepoint_aoi_sequences(
        master, aoi_col="aoi_current", group_cols=["subject", "trial_global"]
    )
    assert len(seq) > 0


def test_event_and_fixation_workflows():
    samples = gp3.simulate_gazepoint_fixations(n_fixations=10, samples_per_fixation=10)
    detected = gp3.detect_gazepoint_fixations_ivt(
        samples, velocity_threshold=2.0, min_duration_ms=30
    )
    assert {"fixation", "fixation_id"} <= set(detected)
    agreement = gp3.compare_gazepoint_event_detectors(
        samples, velocity_threshold=2.0, min_duration_ms=30
    )
    assert "agreement" in agreement


def test_face_sync_and_interop():
    master = gp3.load_example_master().head(50).copy()
    face = pd.DataFrame(
        {
            "time": master["TIME"].to_numpy(),
            "Face Confidence": 0.95,
            "smile": np.linspace(0, 1, len(master)),
        }
    )
    synced = gp3.sync_gazepoint_face_data(master, face, gaze_time_col="TIME", face_time_col="time")
    assert len(synced) == len(master)
    adapter = gp3.prepare_gazepoint_eyetrackingr_data(
        master, time_col="TIME", x_col="FPOGX", y_col="FPOGY"
    )
    assert {"Participant", "Time", "X", "Y"} <= set(adapter)


def test_cluster_and_model_adaptations():
    tc = gp3.simulate_gazepoint_cluster_timecourse_data(
        n_subjects=8, n_time=30, effect_window=(10, 20)
    )
    cluster = gp3.run_gazepoint_cluster_permutation(tc, n_permutations=19)
    assert {"clusters", "observed", "null_distribution"} <= set(cluster)
    master = gp3.load_example_master()
    windows = gp3.summarise_gazepoint_pupil_windows(
        master,
        pupil_col="pupil",
        time_col="TIME",
        windows={"all": (0, 2)},
        group_cols=["subject", "condition"],
    )
    model = gp3.fit_gazepoint_pupil_window_lmm(
        windows, formula="mean_pupil ~ condition", subject_col="subject"
    )
    tidy = gp3.tidy_gazepoint_model_summary(model)
    assert {"term", "estimate"} <= set(tidy)


def test_reporting_and_plots(tmp_path: Path):
    master = gp3.load_example_master().head(200)
    report = gp3.create_gazepoint_report({"master": master.head()}, tmp_path / "report.html")
    assert report.exists()
    fig = gp3.plot_gazepoint_heatmap(master, x_col="FPOGX", y_col="FPOGY")
    out = tmp_path / "heatmap.png"
    fig.savefig(out)
    assert out.stat().st_size > 0
