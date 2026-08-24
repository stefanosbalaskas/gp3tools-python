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
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    np.array(["left", "right", "center"])
    for s in range(1, n_subjects + 1):
        for tr in range(1, n_trials + 1):
            condition = "A" if tr % 2 else "B"
            t = np.arange(samples_per_trial) / sampling_rate
            x = np.clip(
                0.5 + 0.18 * np.sin(t * 2 * np.pi / 2) + rng.normal(0, 0.035, samples_per_trial),
                0,
                1,
            )
            y = np.clip(
                0.5 + 0.12 * np.cos(t * 2 * np.pi / 2.5) + rng.normal(0, 0.035, samples_per_trial),
                0,
                1,
            )
            p = (
                3.5
                + 0.18 * (condition == "B")
                + 0.25 * np.exp(-(((t - 0.8) / 0.35) ** 2))
                + rng.normal(0, 0.06, samples_per_trial)
            )
            aoi = np.where(x < 0.4, "left", np.where(x > 0.6, "right", "center"))
            for i in range(samples_per_trial):
                rows.append(
                    {
                        "subject": f"S{s:02d}",
                        "trial_global": f"S{s:02d}::T{tr:02d}",
                        "condition": condition,
                        "TIME": float(t[i]),
                        "FPOGX": float(x[i]),
                        "FPOGY": float(y[i]),
                        "LPMM": float(p[i] + rng.normal(0, 0.025)),
                        "RPMM": float(p[i] + rng.normal(0, 0.025)),
                        "pupil": float(p[i]),
                        "aoi_current": str(aoi[i]),
                        "MEDIA_ID": tr,
                    }
                )
    return pd.DataFrame(rows)


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
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    rows = []
    for s in range(n_subjects):
        subj = rng.normal(0, 0.15)
        for cond in [0, 1]:
            for ti in range(n_time):
                effect = effect_size * cond if effect_window[0] <= ti <= effect_window[1] else 0.0
                rows.append(
                    {
                        "subject": f"S{s + 1:02d}",
                        "condition": "B" if cond else "A",
                        "time_bin": ti,
                        "value": subj + effect + rng.normal(0, 0.25),
                    }
                )
    return pd.DataFrame(rows)
