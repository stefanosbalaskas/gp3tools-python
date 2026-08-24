import numpy as np
import pandas as pd

import gp3tools as gp3


def test_pupil_processing_family():
    df = gp3.load_example_master().head(180).copy()
    df.loc[df.index[5:8], "pupil"] = np.nan
    df.loc[df.index[20], "pupil"] = 20.0
    df.loc[df.index[30], "LPMM"] = np.nan
    df.loc[df.index[31], "RPMM"] = np.nan

    mean = gp3.mean_gazepoint_pupil(df, left_col="LPMM", right_col="RPMM")
    assert "pupil_mean" in mean
    combined = gp3.combine_gazepoint_eyes(
        df, left_col="LPMM", right_col="RPMM", policy="available_eye"
    )
    assert "pupil_combined" in combined
    bilateral = gp3.construct_gazepoint_combined_pupil(
        df, left_col="LPMM", right_col="RPMM", policy="bilateral_mean"
    )
    assert "pupil_combined" in bilateral

    flags = gp3.flag_gazepoint_pupil(
        df, pupil_col="pupil", physiological_min=1, physiological_max=9
    )
    assert "above_max" in set(flags["pupil_flag"].astype(str))
    hampel = gp3.flag_gazepoint_pupil_hampel(df, pupil_col="pupil", window=7)
    assert "pupil_hampel_flag" in hampel
    artifact = gp3.flag_gazepoint_pupil_artifacts(df, pupil_col="pupil", time_col="TIME")
    assert "pupil_artifact" in artifact

    interp = gp3.interpolate_gazepoint_pupil(
        df, pupil_col="pupil", time_col="TIME", method="linear"
    )
    assert "pupil_interpolated" in interp
    pchip = gp3.interpolate_gazepoint_pupil_pchip(df, pupil_col="pupil", time_col="TIME")
    assert len(pchip) == len(df)
    smooth = gp3.smooth_gazepoint_pupil(df, pupil_col="pupil", window=5)
    assert any(c.endswith("_smoothed") for c in smooth.columns)
    coords = gp3.smooth_gazepoint_coordinate(df, column="FPOGX", window=5)
    assert any(c.endswith("_smoothed") for c in coords.columns)

    blinks = gp3.detect_gazepoint_blinks(
        df, pupil_col="pupil", time_col="TIME", min_duration_ms=1, max_duration_ms=1000
    )
    assert "blink" in blinks
    interp_blinks = gp3.interpolate_gazepoint_blinks(blinks, pupil_col="pupil", time_col="TIME")
    assert len(interp_blinks) == len(blinks)
    down = gp3.downsample_gazepoint_pupil(df, time_col="TIME", pupil_col="pupil", target_hz=30)
    assert len(down) <= len(df)

    summary = gp3.summarise_gazepoint_pupil(df, pupil_col="pupil", group_cols=["condition"])
    assert "mean_pupil" in summary
    windows = gp3.summarise_gazepoint_pupil_windows(
        df, pupil_col="pupil", time_col="TIME", windows={"first": (0, 1)}, group_cols=["condition"]
    )
    assert "window" in windows
    trial = gp3.summarise_gazepoint_pupil_trial_features(
        df, pupil_col="pupil", trial_col="trial_global"
    )
    assert len(trial) >= 1


def test_pupil_baseline_drift_gap_and_reliability():
    df = gp3.load_example_master().head(240).copy()
    centered = df.copy()
    centered["time_ms"] = centered["TIME"] * 1000 - 200
    corrected = gp3.baseline_correct_gazepoint_pupil(
        centered, pupil_col="pupil", time_col="time_ms", baseline=(-200, 0)
    )
    assert len(corrected) == len(centered)
    pct = gp3.baseline_correct_gazepoint_pupil(
        centered, pupil_col="pupil", time_col="time_ms", baseline=(-200, 0), mode="percent"
    )
    assert len(pct) == len(centered)
    base = gp3.audit_gazepoint_pupil_baseline(
        centered, pupil_col="pupil", time_col="time_ms", baseline=(-200, 0)
    )
    assert len(base) >= 1
    drift = gp3.audit_gazepoint_pupil_drift(df, pupil_col="pupil", time_col="TIME")
    assert "slope" in drift

    gapdf = df.copy()
    gapdf.loc[gapdf.index[5:12], "pupil"] = np.nan
    gaps = gp3.audit_gazepoint_pupil_gaps(gapdf, pupil_col="pupil", time_col="TIME")
    assert "n_gaps" in gaps
    imbalance = gp3.audit_gazepoint_pupil_imbalance(
        df, pupil_col="pupil", condition_col="condition"
    )
    assert len(imbalance) >= 1
    overlap = gp3.audit_gazepoint_pupil_overlap_risk(
        df, trial_duration_ms=3000, event_gap_ms=1000, trial_col="trial_global", time_col="TIME"
    )
    assert len(overlap) >= 1
    lum = df.assign(luminance=np.linspace(0, 1, len(df)))
    luminance = gp3.audit_gazepoint_stimulus_luminance(
        lum, luminance_col="luminance", pupil_col="pupil"
    )
    assert len(luminance) >= 1


def test_binocular_calibration_reconstruction_validation():
    df = gp3.load_example_master().head(360).copy()
    diag = gp3.diagnose_gazepoint_binocular_pupil(
        df, left_col="LPMM", right_col="RPMM", group_cols=["condition"]
    )
    assert "correlation" in diag
    reg = gp3.regress_gazepoint_pupils(
        df, left_col="LPMM", right_col="RPMM", direction="right_from_left"
    )
    assert "slope" in reg
    cal = gp3.fit_gazepoint_binocular_calibration(df, left_col="LPMM", right_col="RPMM")
    assert isinstance(cal, dict)

    damaged = df.copy()
    damaged.loc[damaged.index[::10], "LPMM"] = np.nan
    damaged.loc[damaged.index[5::11], "RPMM"] = np.nan
    for method in ["available_eye", "linear_regression", "none"]:
        recon = gp3.reconstruct_gazepoint_binocular_pupil(
            damaged, left_col="LPMM", right_col="RPMM", method=method
        )
        assert "pupil_combined" in recon

    recon = gp3.reconstruct_gazepoint_binocular_pupil(damaged, left_col="LPMM", right_col="RPMM")
    audit = gp3.audit_gazepoint_binocular_reconstruction(
        recon, observed_left="LPMM", observed_right="RPMM"
    )
    assert len(audit) >= 1
    valid = gp3.validate_gazepoint_binocular_reconstruction(
        df, left_col="LPMM", right_col="RPMM", fraction=0.1
    )
    assert len(valid) >= 1
    stress = gp3.stress_test_gazepoint_binocular_reconstruction(
        df, fractions=(0.05, 0.1), left_col="LPMM", right_col="RPMM"
    )
    assert len(stress) >= 2
    sensitivity = gp3.analyse_gazepoint_binocular_sensitivity(
        df, methods=("available_eye", "linear_regression"), left_col="LPMM", right_col="RPMM"
    )
    assert len(sensitivity) >= 2
    report = gp3.summarise_gazepoint_binocular_reporting(df, left_col="LPMM", right_col="RPMM")
    assert isinstance(report, dict)


def test_preprocessing_registry_multiverse_and_pipeline():
    reg = gp3.create_gazepoint_preprocessing_registry()
    assert len(reg) == 15
    multi = gp3.create_gazepoint_preprocessing_multiverse(
        interpolation_method=["linear", "pchip"], smoothing_window=[3, 5]
    )
    assert len(multi) == 4
    df = gp3.load_example_master().head(180).copy()
    pre = gp3.preprocess_gazepoint_signals(
        df, pupil_col="pupil", time_col="TIME", baseline=(0, 0.2)
    )
    assert len(pre) == len(df)


def test_aoi_geometry_assignment_and_audits():
    data = pd.DataFrame({"x": [0.1, 0.5, 0.9, 1.2], "y": [0.5, 0.5, 0.5, 0.5]})
    geometry = pd.DataFrame(
        {
            "aoi": ["left", "center", "right"],
            "xmin": [0.0, 0.4, 0.6],
            "xmax": [0.4, 0.6, 1.0],
            "ymin": [0.0, 0.0, 0.0],
            "ymax": [1.0, 1.0, 1.0],
        }
    )
    assigned = gp3.add_gazepoint_aoi(data, x_col="x", y_col="y", aoi_geometry=geometry)
    assert assigned["aoi_current"].tolist()[:3] == ["left", "center", "right"]
    assert assigned["aoi_current"].iloc[3] == "outside"
    audit = gp3.audit_gazepoint_aoi_geometry(geometry)
    assert isinstance(audit, dict)
    overlap = gp3.audit_gazepoint_aoi_overlap(geometry)
    assert isinstance(overlap, pd.DataFrame)
    screen = gp3.audit_gazepoint_aoi_screen_coverage(geometry)
    assert len(screen) >= 1
    margins = gp3.audit_gazepoint_aoi_margin_sensitivity(
        data, geometry, margins=(-0.01, 0, 0.01), x_col="x", y_col="y"
    )
    assert len(margins) == 3

    polygons = {
        "left": [(0, 0), (0.4, 0), (0.4, 1), (0, 1)],
        "right": [(0.6, 0), (1, 0), (1, 1), (0.6, 1)],
    }
    poly = gp3.add_gazepoint_polygon_aoi(data, polygons, x_col="x", y_col="y")
    assert "aoi_current" in poly


def test_aoi_sequences_transitions_entropy_complexity():
    df = gp3.load_example_master().head(360).copy()
    samples = gp3.summarise_aoi_samples(df, aoi_col="aoi_current", group_cols=["condition"])
    assert len(samples) >= 1
    coding = gp3.audit_gazepoint_aoi_coding_matrix(
        df, aoi_col="aoi_current", group_cols=["condition"]
    )
    assert len(coding) >= 1
    seq = gp3.prepare_gazepoint_aoi_sequences(
        df, aoi_col="aoi_current", group_cols=["subject", "trial_global"]
    )
    assert "sequence" in seq
    entries = gp3.summarise_gazepoint_aoi_entries(
        df, aoi_col="aoi_current", group_cols=["trial_global"], time_col="TIME"
    )
    assert len(entries) >= 1
    transitions = gp3.summarise_gazepoint_aoi_transitions(
        df, aoi_col="aoi_current", group_cols=["trial_global"], time_col="TIME"
    )
    assert set(transitions.columns) >= {"from_aoi", "to_aoi"}

    s = ["A", "A", "B", "C", "B"]
    mat = gp3.compute_transition_matrix(s)
    norm = gp3.compute_transition_matrix(s, normalize=True)
    assert mat.to_numpy().sum() == 4
    assert np.all(norm.sum(axis=1).dropna() <= 1.000001)
    mat2 = gp3.compute_gazepoint_aoi_transition_matrix(sequence=s)
    assert mat2.shape == mat.shape
    entropy = gp3.compute_gazepoint_aoi_entropy(sequence=s)
    assert "entropy" in entropy
    metrics = gp3.compute_gazepoint_aoi_sequence_metrics(sequence=s)
    assert len(metrics) == 1
    complex_ = gp3.compute_gazepoint_sequence_complexity(sequence=s)
    assert len(complex_) == 1
    assert gp3.compute_gazepoint_sequence_distance(s, s) == 0
    assert 0 <= gp3.compute_gazepoint_sequence_distance(s, ["A", "C"]) <= 1
    recurrence = gp3.compute_gazepoint_sequence_recurrence(s, lag=1)
    assert "recurrence" in recurrence


def test_scanpath_markov_window_and_clustering():
    df = gp3.load_example_master().head(360).copy()
    geom = gp3.compute_gazepoint_scanpath_geometry(
        df, x_col="FPOGX", y_col="FPOGY", group_cols=["subject", "trial_global"]
    )
    assert "path_length" in geom
    sim = gp3.compute_gazepoint_scanpath_similarity(["A", "B"], ["A", "C"], method="sequence")
    assert 0 <= sim <= 1
    mark = gp3.summarise_gazepoint_markovchain(sequence=["A", "B", "A", "B", "C"])
    assert "transition_matrix" in mark
    assert gp3.create_gazepoint_markovchain_object(sequence=["A", "B"]) is not None
    assert gp3.summarise_gazepoint_semimarkov(sequence=["A", "B"]) is not None

    windows = gp3.summarise_gazepoint_aoi_windows(
        df,
        aoi_col="aoi_current",
        time_col="TIME",
        windows={"early": (0, 1)},
        group_cols=["condition"],
    )
    assert "window" in windows
    denom = gp3.audit_gazepoint_aoi_window_denominators(
        pd.DataFrame({"success": [1, 2], "total": [3, 4]})
    )
    assert len(denom) >= 1
    elog = gp3.transform_gazepoint_aoi_empirical_logit(
        pd.DataFrame({"success": [1, 2], "total": [3, 4]})
    )
    assert "empirical_logit" in elog

    clustered = gp3.cluster_gazepoint_scanpaths(
        df, aoi_col="aoi_current", group_cols=["subject", "trial_global"], n_clusters=2
    )
    assert "cluster" in clustered
    reps = gp3.extract_gazepoint_representative_scanpaths(clustered)
    assert len(reps) >= 1
    selection = gp3.select_gazepoint_scanpath_clusters(
        df, aoi_col="aoi_current", group_cols=["subject", "trial_global"], max_clusters=3
    )
    assert len(selection) >= 1
    stability = gp3.bootstrap_gazepoint_scanpath_clusters(
        df, n_boot=3, aoi_col="aoi_current", group_cols=["subject", "trial_global"], n_clusters=2
    )
    assert len(stability) >= 1
    summary = gp3.summarise_gazepoint_scanpath_cluster_stability(stability)
    assert len(summary) >= 1
