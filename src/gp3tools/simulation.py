"""Synthetic Gazepoint data generators for examples and validation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_gazepoint_data(
    n_subjects: int = 6,
    n_trials: int = 8,
    samples_per_trial: int = 120,
    sampling_rate: float = 60.0,
    random_state: int = 123,
    *,
    trial_duration_ms=None,
    sampling_rate_hz=None,
    conditions=None,
    aoi_labels=None,
    effect_size=None,
    target_aoi=None,
    seed=None,
    include_fixations=None,
):
    """Simulate Gazepoint data using legacy DataFrame or R v2.3.0 structured output."""
    r_mode = any(
        value is not None
        for value in (
            trial_duration_ms,
            sampling_rate_hz,
            conditions,
            aoi_labels,
            effect_size,
            target_aoi,
            seed,
            include_fixations,
        )
    )
    if not r_mode:
        rng = np.random.default_rng(random_state)
        rows = []
        for subject_index in range(1, n_subjects + 1):
            for trial_index in range(1, n_trials + 1):
                condition = "A" if trial_index % 2 else "B"
                time = np.arange(samples_per_trial) / sampling_rate
                x = np.clip(
                    0.5
                    + 0.18 * np.sin(time * 2 * np.pi / 2)
                    + rng.normal(0, 0.035, samples_per_trial),
                    0,
                    1,
                )
                y = np.clip(
                    0.5
                    + 0.12 * np.cos(time * 2 * np.pi / 2.5)
                    + rng.normal(0, 0.035, samples_per_trial),
                    0,
                    1,
                )
                pupil = (
                    3.5
                    + 0.18 * (condition == "B")
                    + 0.25 * np.exp(-(((time - 0.8) / 0.35) ** 2))
                    + rng.normal(0, 0.06, samples_per_trial)
                )
                aoi = np.where(x < 0.4, "left", np.where(x > 0.6, "right", "center"))
                for index in range(samples_per_trial):
                    rows.append(
                        {
                            "subject": f"S{subject_index:02d}",
                            "trial_global": f"S{subject_index:02d}::T{trial_index:02d}",
                            "condition": condition,
                            "TIME": float(time[index]),
                            "FPOGX": float(x[index]),
                            "FPOGY": float(y[index]),
                            "LPMM": float(pupil[index] + rng.normal(0, 0.025)),
                            "RPMM": float(pupil[index] + rng.normal(0, 0.025)),
                            "pupil": float(pupil[index]),
                            "aoi_current": str(aoi[index]),
                            "MEDIA_ID": trial_index,
                        }
                    )
        return pd.DataFrame(rows)

    n_subjects = 12 if n_subjects == 6 else n_subjects
    trial_duration_ms = 2000 if trial_duration_ms is None else trial_duration_ms
    sampling_rate_hz = 60 if sampling_rate_hz is None else sampling_rate_hz
    conditions = ["control", "treatment"] if conditions is None else list(conditions)
    aoi_labels = ["target", "other"] if aoi_labels is None else list(aoi_labels)
    effect_size = 0.5 if effect_size is None else effect_size
    include_fixations = True if include_fixations is None else bool(include_fixations)
    target_aoi = aoi_labels[0] if target_aoi is None else target_aoi

    for value, name in ((n_subjects, "n_subjects"), (n_trials, "n_trials")):
        if int(value) < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not np.isfinite(trial_duration_ms) or trial_duration_ms <= 0:
        raise ValueError("trial_duration_ms must be positive")
    if not np.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    if not conditions or any(not str(value) for value in conditions):
        raise ValueError("conditions must contain non-empty labels")
    if len(aoi_labels) < 2 or any(not str(value) for value in aoi_labels):
        raise ValueError("aoi_labels must contain at least two non-empty labels")
    if target_aoi not in aoi_labels:
        raise ValueError("target_aoi must be one of aoi_labels")

    rng = np.random.default_rng(seed)
    sample_interval = 1000 / float(sampling_rate_hz)
    time_values = np.arange(0, float(trial_duration_ms) + sample_interval * 0.5, sample_interval)
    base_probability = np.full(len(aoi_labels), (1 - 0.45) / (len(aoi_labels) - 1))
    target_index = aoi_labels.index(target_aoi)
    base_probability[target_index] = 0.45

    rows = []
    for subject_index in range(1, int(n_subjects) + 1):
        subject_id = f"S{subject_index:03d}"
        subject_shift = rng.normal(0, 0.35)
        for trial_index in range(1, int(n_trials) + 1):
            condition = conditions[(trial_index - 1) % len(conditions)]
            condition_effect = 0 if condition == conditions[0] else float(effect_size)
            clipped = np.clip(base_probability, 0.001, 0.999)
            logits = np.log(clipped / (1 - clipped))
            logits[target_index] += condition_effect + subject_shift
            probabilities = np.exp(logits)
            probabilities = probabilities / probabilities.sum()
            aoi = rng.choice(aoi_labels, len(time_values), replace=True, p=probabilities)
            x_centers = dict(zip(aoi_labels, np.linspace(0.25, 0.75, len(aoi_labels)), strict=True))
            y_centers = dict(zip(aoi_labels, np.linspace(0.75, 0.25, len(aoi_labels)), strict=True))
            x = np.array([rng.normal(x_centers[value], 0.04) for value in aoi])
            y = np.array([rng.normal(y_centers[value], 0.04) for value in aoi])
            pupil = (
                3
                + 0.10 * (aoi == target_aoi)
                + 0.05 * (condition != conditions[0])
                + rng.normal(0, 0.05, len(time_values))
            )
            for index, time_ms in enumerate(time_values):
                rows.append(
                    {
                        "subject_id": subject_id,
                        "trial_id": trial_index,
                        "condition": condition,
                        "time_ms": time_ms,
                        "aoi": str(aoi[index]),
                        "x": float(np.clip(x[index], 0, 1)),
                        "y": float(np.clip(y[index], 0, 1)),
                        "pupil": float(pupil[index]),
                        "valid": True,
                    }
                )
    all_gaze = pd.DataFrame(rows)

    window_rows = []
    fixation_rows = []
    for (subject_id, trial_id), block in all_gaze.groupby(
        ["subject_id", "trial_id"], sort=True, dropna=True
    ):
        window_rows.append(
            {
                "subject_id": subject_id,
                "trial_id": trial_id,
                "condition": block["condition"].iloc[0],
                "target_aoi": target_aoi,
                "n_samples": len(block),
                "target_samples": int(block["aoi"].eq(target_aoi).sum()),
                "target_prop": float(block["aoi"].eq(target_aoi).mean()),
                "mean_pupil": float(block["pupil"].mean()),
            }
        )
        if include_fixations:
            run_id = block["aoi"].ne(block["aoi"].shift()).cumsum()
            for fixation_index, (_, run) in enumerate(block.groupby(run_id, sort=False), start=1):
                fixation_rows.append(
                    {
                        "subject_id": subject_id,
                        "trial_id": trial_id,
                        "condition": run["condition"].iloc[0],
                        "fixation_index": fixation_index,
                        "aoi": run["aoi"].iloc[0],
                        "start_time_ms": float(run["time_ms"].min()),
                        "end_time_ms": float(run["time_ms"].max()),
                        "duration_ms": float(
                            run["time_ms"].max() - run["time_ms"].min() + sample_interval
                        ),
                        "x": float(run["x"].mean()),
                        "y": float(run["y"].mean()),
                    }
                )

    return {
        "all_gaze": all_gaze,
        "aoi_windows": pd.DataFrame(window_rows),
        "fixations": pd.DataFrame(fixation_rows) if include_fixations else None,
        "metadata": {
            "n_subjects": int(n_subjects),
            "n_trials": int(n_trials),
            "trial_duration_ms": trial_duration_ms,
            "sampling_rate_hz": sampling_rate_hz,
            "conditions": conditions,
            "aoi_labels": aoi_labels,
            "target_aoi": target_aoi,
            "effect_size": effect_size,
        },
        "_gp3_class": "gp3_simulated_data",
    }


def simulate_gazepoint_pupil_data(
    n_subjects: int = 12,
    n_trials: int = 12,
    samples_per_trial: int = 180,
    random_state: int = 123,
    **kwargs,
) -> pd.DataFrame:
    return simulate_gazepoint_data(
        n_subjects=n_subjects,
        n_trials=n_trials,
        samples_per_trial=samples_per_trial,
        random_state=random_state,
        **kwargs,
    )


def simulate_gazepoint_cluster_timecourse_data(
    n_subjects: int = 16,
    n_time: int = 80,
    effect_window=(30, 50),
    effect_size: float = 0.35,
    random_state: int = 123,
    *,
    n_time_bins=None,
    conditions=None,
    effect_start=None,
    effect_end=None,
    subject_sd=None,
    noise_sd=None,
    seed=None,
) -> pd.DataFrame:
    """Simulate legacy timecourses or the R v2.3.0 cluster fixture."""
    r_mode = any(
        value is not None
        for value in (n_time_bins, conditions, effect_start, effect_end, subject_sd, noise_sd, seed)
    )
    if not r_mode:
        rng = np.random.default_rng(random_state)
        rows = []
        for s in range(n_subjects):
            subj = rng.normal(0, 0.15)
            for cond in [0, 1]:
                for ti in range(n_time):
                    effect = (
                        effect_size * cond if effect_window[0] <= ti <= effect_window[1] else 0.0
                    )
                    rows.append(
                        {
                            "subject": f"S{s + 1:02d}",
                            "condition": "B" if cond else "A",
                            "time_bin": ti,
                            "value": subj + effect + rng.normal(0, 0.25),
                        }
                    )
        return pd.DataFrame(rows)

    n_time_bins = 60 if n_time_bins is None else int(n_time_bins)
    conditions = ["control", "treatment"] if conditions is None else list(conditions)
    effect_start = 25 if effect_start is None else effect_start
    effect_end = 40 if effect_end is None else effect_end
    subject_sd = 0.3 if subject_sd is None else float(subject_sd)
    noise_sd = 0.4 if noise_sd is None else float(noise_sd)
    if int(n_subjects) < 2 or n_time_bins < 2:
        raise ValueError("n_subjects and n_time_bins must be at least 2")
    if len(conditions) != 2 or any(not str(value) for value in conditions):
        raise ValueError("conditions must be a character vector of length two")

    rng = np.random.default_rng(seed)
    subjects = [f"S{i:03d}" for i in range(1, int(n_subjects) + 1)]
    subject_shift = dict(zip(subjects, rng.normal(0, subject_sd, len(subjects)), strict=True))
    rows = []
    # R expand.grid varies subject fastest, then condition, then time_bin.
    for time_bin in range(1, n_time_bins + 1):
        for condition in conditions:
            for subject in subjects:
                baseline = 0.15 * np.sin(time_bin / n_time_bins * 2 * np.pi)
                treatment = condition == conditions[1]
                in_window = effect_start <= time_bin <= effect_end
                rows.append(
                    {
                        "subject": subject,
                        "condition": condition,
                        "time_bin": time_bin,
                        "outcome": baseline
                        + subject_shift[subject]
                        + (effect_size if treatment and in_window else 0)
                        + rng.normal(0, noise_sd),
                    }
                )
    return pd.DataFrame(rows)
