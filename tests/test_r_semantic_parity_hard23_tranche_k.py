import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3


def test_drift_r_structure_and_warning():
    df = pd.DataFrame(
        {
            "subject": ["S1"] * 4 + ["S2"] * 4,
            "trial": [1, 2, 3, 4] * 2,
            "condition": ["A", "A", "B", "B"] * 2,
            "time": [0, 1000, 2000, 3000] * 2,
            "pupil": [1, 2, 3, 4, 4, 3, 2, 1],
            "excluded_trial": [False] * 8,
        }
    )
    out = gp3.audit_gazepoint_pupil_drift(
        df,
        pupil_col="pupil",
        time_col="time",
        group_cols=["subject"],
        order_col="trial",
        condition_col="condition",
        exclude_col="excluded_trial",
        max_abs_slope_per_min=10,
    )
    assert out["_gp3_class"] == "gp3_pupil_drift_audit"
    assert set(["by_group", "by_subject", "by_condition", "condition_balance", "summary"]).issubset(
        out
    )
    assert len(out["by_group"]) == 2
    assert out["by_group"]["drift_warning"].all()
    slopes = out["by_group"].set_index("subject")["pupil_time_slope_per_min"]
    assert slopes["S1"] == pytest.approx(60.0)
    assert slopes["S2"] == pytest.approx(-60.0)


def test_drift_legacy_dataframe_still_available():
    df = pd.DataFrame({"time": [0, 1, 2, 3], "pupil": [1, 2, 3, 4]})
    out = gp3.audit_gazepoint_pupil_drift(df, pupil_col="pupil", time_col="time")
    assert isinstance(out, pd.DataFrame)
    assert "slope" in out


def test_overlap_r_structure_and_no_events_status():
    df = pd.DataFrame(
        {
            "subject": ["S1"] * 3,
            "trial_global": [1] * 3,
            "time": [0, 500, 1000],
            "stimulus_onset_time": [0, 0, 0],
            "target_onset_time": [600, 600, 600],
            "response_time": [1500, 1500, 1500],
        }
    )
    out = gp3.audit_gazepoint_pupil_overlap_risk(
        df,
        group_cols=["subject"],
        trial_col="trial_global",
        time_col="time",
        event_time_cols=["stimulus_onset_time", "target_onset_time", "response_time"],
        window_start_ms=0,
        window_end_ms=1000,
        min_event_gap_ms=1000,
    )
    assert out["_gp3_class"] == "gp3_pupil_overlap_risk_audit"
    assert out["summary"].loc[0, "n_events"] == 3
    assert out["summary"].loc[0, "n_overlap_risk_trials"] == 1
    assert out["summary"].loc[0, "overlap_assessment_status"] == "possible_overlap_risk"
    assert (out["event_gaps"]["event_gap_status"] == "overlap_and_short_gap").sum() == 2

    no = df.copy()
    no[["stimulus_onset_time", "target_onset_time", "response_time"]] = np.nan
    empty = gp3.audit_gazepoint_pupil_overlap_risk(
        no,
        group_cols=["subject"],
        event_time_cols=["stimulus_onset_time", "target_onset_time", "response_time"],
        window_start_ms=0,
        window_end_ms=1000,
    )
    assert empty["summary"].loc[0, "overlap_assessment_status"] == "no_usable_event_times"


def test_overlap_legacy_dataframe_still_available():
    df = pd.DataFrame({"trial_global": [1, 1, 1], "time": [0, 1, 2]})
    out = gp3.audit_gazepoint_pupil_overlap_risk(df, trial_col="trial_global", time_col="time")
    assert isinstance(out, pd.DataFrame)
    assert "overlap_risk" in out


def test_reliability_r_structure_ready():
    rows = []
    for i, subject in enumerate(["S1", "S2", "S3", "S4"], start=1):
        for trial in range(1, 5):
            rows.append({"subject": subject, "trial": trial, "pupil_auc": i * trial})
    df = pd.DataFrame(rows)
    out = gp3.audit_gazepoint_pupil_reliability(
        df,
        outcome_cols=["pupil_auc"],
        participant_col="subject",
        trial_col="trial",
        min_trials_per_split=2,
    )
    assert out["_gp3_class"] == "gp3_pupil_reliability_audit"
    rel = out["reliability_summary"]
    assert len(rel) == 1
    assert rel.loc[0, "n_complete_pairs"] == 4
    assert rel.loc[0, "split_half_correlation"] == pytest.approx(1.0)
    assert rel.loc[0, "spearman_brown_reliability"] == pytest.approx(1.0)
    assert rel.loc[0, "reliability_status"] == "ready"


def test_reliability_r_by_group_and_predefined_split():
    df = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S2", "S2", "S3", "S3"],
            "condition": ["A"] * 6,
            "half": ["x", "y"] * 3,
            "pupil_auc": [1, 2, 2, 4, 3, 6],
        }
    )
    out = gp3.audit_gazepoint_pupil_reliability(
        df,
        outcome_cols=["pupil_auc"],
        participant_col="subject",
        split_col="half",
        by_cols=["condition"],
        min_trials_per_split=1,
    )
    assert out["overview"].loc[0, "split_method"] == "predefined_split_col"
    assert out["reliability_summary"].loc[0, "reliability_status"] == "ready"


def test_reliability_legacy_dataframe_still_available():
    df = pd.DataFrame({"subject": ["S1"] * 4 + ["S2"] * 4, "pupil": [1, 2, 1, 2, 2, 4, 2, 4]})
    out = gp3.audit_gazepoint_pupil_reliability(df, pupil_col="pupil", subject_col="subject")
    assert isinstance(out, pd.DataFrame)
    assert out.loc[0, "split_half_correlation"] == pytest.approx(1.0)


def test_drift_r_exclusion_and_insufficient_status():
    df = pd.DataFrame(
        {
            "subject": ["S1"] * 4,
            "trial": [1, 2, 3, 4],
            "condition": ["A"] * 4,
            "time": [0, 1000, 2000, 3000],
            "pupil": [1.0, np.nan, np.nan, 4.0],
            "excluded_trial": [False, True, False, False],
        }
    )
    out = gp3.audit_gazepoint_pupil_drift(
        df,
        pupil_col="pupil",
        time_col="time",
        order_col="trial",
        condition_col="condition",
        exclude_col="excluded_trial",
        min_valid_samples=3,
    )
    assert out["summary"].loc[0, "n_rows"] == 3
    assert out["by_group"].loc[0, "drift_status"] == "insufficient_valid_samples"
    assert out["condition_balance"].loc[0, "condition_balance_reason"] == "ok"


def test_overlap_r_exclusion_and_ok_status():
    df = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S1", "S1"],
            "trial_global": [1, 1, 2, 2],
            "time": [0, 1000, 0, 1000],
            "stimulus_onset_time": [0, 0, 0, 0],
            "target_onset_time": [2000, 2000, 2000, 2000],
            "response_time": [4000, 4000, 4000, 4000],
            "excluded_trial": [False, False, True, True],
        }
    )
    out = gp3.audit_gazepoint_pupil_overlap_risk(
        df,
        group_cols=["subject"],
        event_time_cols=["stimulus_onset_time", "target_onset_time", "response_time"],
        window_start_ms=0,
        window_end_ms=500,
        min_event_gap_ms=1000,
        exclude_col="excluded_trial",
    )
    assert out["summary"].loc[0, "n_trials"] == 1
    assert out["summary"].loc[0, "overlap_assessment_status"] == "ok"
    assert not out["by_trial"].loc[0, "overlap_risk_warning"]


def test_reliability_r_first_second_median_spearman():
    rows = []
    for i, subject in enumerate(["S1", "S2", "S3", "S4"], start=1):
        for trial in range(1, 7):
            rows.append(
                {
                    "subject": subject,
                    "trial": trial,
                    "condition": "A" if i % 2 else "B",
                    "pupil_auc": float(i * trial),
                }
            )
    out = gp3.audit_gazepoint_pupil_reliability(
        pd.DataFrame(rows),
        outcome_cols=["pupil_auc"],
        participant_col="subject",
        trial_col="trial",
        split_method="first_second",
        aggregate_function="median",
        correlation_method="spearman",
        min_trials_per_split=2,
    )
    rel = out["reliability_summary"]
    assert rel.loc[0, "split1_label"] == "first"
    assert rel.loc[0, "split2_label"] == "second"
    assert rel.loc[0, "split_half_correlation"] == pytest.approx(1.0)
    assert out["overview"].loc[0, "aggregate_function"] == "median"


def test_tranche_k_r_validation_paths():
    base = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S2", "S2"],
            "trial": [1, 2, 1, 2],
            "condition": ["A", "A", "A", "A"],
            "time": [0, 1000, 0, 1000],
            "pupil": [1.0, 2.0, 2.0, 4.0],
        }
    )
    with pytest.raises(ValueError, match="unique"):
        gp3.audit_gazepoint_pupil_drift(
            base,
            group_cols=["subject", "subject"],
            order_col="trial",
            condition_col="condition",
        )
    with pytest.raises(KeyError, match="Missing required columns"):
        gp3.audit_gazepoint_pupil_drift(
            base,
            group_cols=["subject"],
            order_col="missing",
            condition_col="condition",
        )
    with pytest.raises(ValueError, match="greater"):
        gp3.audit_gazepoint_pupil_overlap_risk(
            base.assign(
                trial_global=[1, 1, 1, 1],
                stimulus_onset_time=0,
                target_onset_time=1000,
                response_time=2000,
            ),
            group_cols=["subject"],
            event_time_cols=["stimulus_onset_time", "target_onset_time", "response_time"],
            window_start_ms=1000,
            window_end_ms=100,
        )
    with pytest.raises(ValueError, match="split_method"):
        gp3.audit_gazepoint_pupil_reliability(
            base,
            outcome_cols=["pupil"],
            participant_col="subject",
            split_method="bad",
        )
    with pytest.raises(ValueError, match="exactly two"):
        gp3.audit_gazepoint_pupil_reliability(
            base.assign(half=["a", "b", "c", "a"]),
            outcome_cols=["pupil"],
            participant_col="subject",
            split_col="half",
            min_trials_per_split=1,
        )


def test_reliability_more_validation_paths():
    df = pd.DataFrame(
        {"subject": ["S1", "S1", "S2", "S2"], "trial": [1, 2, 1, 2], "pupil": [1.0, 2.0, 2.0, 4.0]}
    )
    with pytest.raises(ValueError, match="aggregate_function"):
        gp3.audit_gazepoint_pupil_reliability(
            df, outcome_cols=["pupil"], participant_col="subject", aggregate_function="bad"
        )
    with pytest.raises(ValueError, match="correlation_method"):
        gp3.audit_gazepoint_pupil_reliability(
            df, outcome_cols=["pupil"], participant_col="subject", correlation_method="bad"
        )
    with pytest.raises(ValueError, match="positive integer"):
        gp3.audit_gazepoint_pupil_reliability(
            df, outcome_cols=["pupil"], participant_col="subject", min_trials_per_split=0
        )
    with pytest.raises(KeyError, match="outcome_cols"):
        gp3.audit_gazepoint_pupil_reliability(
            df, outcome_cols=["missing"], participant_col="subject"
        )
    with pytest.raises(KeyError, match="participant_col"):
        gp3.audit_gazepoint_pupil_reliability(
            df.drop(columns="subject"), outcome_cols=["pupil"], participant_col="missing"
        )


def test_reliability_autodetect_and_string_trial_order():
    rows = []
    for i, subject in enumerate(["S1", "S2", "S3"], start=1):
        for trial in range(1, 5):
            rows.append(
                {
                    "subject": subject,
                    "trial_id": f"trial_{trial}",
                    "pupil_auc": float(i * trial),
                }
            )
    out = gp3.audit_gazepoint_pupil_reliability(
        pd.DataFrame(rows),
        participant_col="subject",
        trial_col="trial_id",
        split_method="first_second",
        min_trials_per_split=2,
    )
    assert out["overview"].loc[0, "n_outcomes"] >= 1
    assert out["split_data"]["trial_order"].notna().all()
    assert out["reliability_summary"].loc[0, "reliability_status"] == "ready"


def test_reliability_constant_split_status():
    rows = []
    for subject in ["S1", "S2", "S3"]:
        for trial in range(1, 5):
            rows.append({"subject": subject, "trial": trial, "pupil_auc": 1.0})
    out = gp3.audit_gazepoint_pupil_reliability(
        pd.DataFrame(rows),
        outcome_cols=["pupil_auc"],
        participant_col="subject",
        trial_col="trial",
        min_trials_per_split=2,
    )
    assert out["reliability_summary"].loc[0, "reliability_status"] == "constant_split_values"
    assert np.isnan(out["reliability_summary"].loc[0, "split_half_correlation"])


def test_overlap_more_validation_paths():
    base = pd.DataFrame(
        {
            "subject": ["S1"],
            "trial_global": [1],
            "time": [0],
            "stimulus_onset_time": [0],
            "target_onset_time": [1000],
            "response_time": [2000],
        }
    )
    with pytest.raises(ValueError, match="unique"):
        gp3.audit_gazepoint_pupil_overlap_risk(
            base,
            group_cols=["subject"],
            event_time_cols=["stimulus_onset_time", "stimulus_onset_time"],
            window_start_ms=0,
            window_end_ms=1000,
        )
    with pytest.raises(KeyError, match="Missing required columns"):
        gp3.audit_gazepoint_pupil_overlap_risk(
            base,
            group_cols=["subject"],
            event_time_cols=["missing_event"],
            window_start_ms=0,
            window_end_ms=1000,
        )


def test_tranche_k_additional_validation_coverage():
    df = pd.DataFrame({"subject": ["S1", "S2"], "trial": [1, 1], "pupil": [1.0, 2.0]})
    with pytest.raises(KeyError, match="Missing by_cols"):
        gp3.audit_gazepoint_pupil_reliability(
            df,
            outcome_cols=["pupil"],
            participant_col="subject",
            by_cols=["missing"],
        )
    with pytest.raises(KeyError, match="trial_col"):
        gp3.audit_gazepoint_pupil_reliability(
            df,
            outcome_cols=["pupil"],
            participant_col="subject",
            trial_col="missing",
        )
    with pytest.raises(ValueError, match="finite numeric"):
        gp3.audit_gazepoint_pupil_drift(
            df.assign(condition="A", time=[0, 1]),
            order_col="trial",
            condition_col="condition",
            max_abs_slope_per_min=np.inf,
        )
