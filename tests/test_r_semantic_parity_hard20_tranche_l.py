import pandas as pd
import pytest

import gp3tools as gp3


def _raw_master_input():
    return pd.DataFrame(
        {
            "USER_FILE": ["S01_all_gaze.csv"] * 4,
            "MEDIA_ID": [1] * 4,
            "MEDIA_NAME": ["stim.png"] * 4,
            "TIME": [0.0, 0.1, 0.2, 0.3],
            "CNT": [1, 2, 3, 4],
            "BPOGX": [0.25, 0.50, 1.20, 0.75],
            "BPOGY": [0.25, 0.50, 0.50, 0.75],
            "BPOGV": [1, 1, 1, 0],
            "LPMM": [3.0, 3.1, 3.2, 3.3],
            "RPMM": [3.2, 3.3, 3.4, 3.5],
            "LPMMV": [1, 1, 1, 1],
            "RPMMV": [1, 1, 0, 1],
            "AOI": ["target", "", "target", "target"],
            "USER": ["TRIAL_START", "", "TARGET_ONSET", "TRIAL_END"],
        }
    )


def test_r_as_master_schema_scaling_and_events():
    out = gp3.as_gazepoint_master(
        _raw_master_input(),
        screen_width_px=1000,
        screen_height_px=500,
        coordinate_unit="auto",
        event_latency_offset_ms=10,
    )
    assert out.loc[0, "subject"] == "S01"
    assert out.loc[0, "trial_global"] == "S01_MEDIA_1"
    assert out.loc[0, "time_ms"] == pytest.approx(10)
    assert out.loc[0, "x"] == pytest.approx(250)
    assert out.loc[0, "y"] == pytest.approx(125)
    assert out.loc[2, "gaze_offscreen"]
    assert out.loc[2, "aoi_current"] == "offscreen"
    assert out.loc[3, "aoi_current"] == "missing"
    assert out.loc[2, "mean_pupil"] == pytest.approx(3.2)
    assert out.loc[0, "event_type"] == "trial_start"
    assert out.loc[2, "event_type"] == "target_onset"
    assert out.loc[0, "pupil_unit"] == "diameter_mm"
    assert out.attrs["gp3_class"] == "gazepoint_master"


def test_r_as_master_rejects_missing_time_and_bad_unit():
    with pytest.raises(KeyError):
        gp3.as_gazepoint_master(pd.DataFrame({"x": [1]}), coordinate_unit="pixels")
    with pytest.raises(ValueError):
        gp3.as_gazepoint_master(_raw_master_input(), coordinate_unit="degrees")


def test_legacy_as_master_still_renames_roles():
    df = pd.DataFrame({"Participant": ["S1"], "TIME": [0.1], "FPOGX": [0.2], "FPOGY": [0.3]})
    out = gp3.as_gazepoint_master(df)
    assert out.attrs["gp3_class"] == "gazepoint_master"
    assert "time" in out.columns


def test_r_validate_master_structured_outputs_and_thresholds():
    master = gp3.as_gazepoint_master(
        _raw_master_input(), screen_width_px=1000, screen_height_px=500
    )
    out = gp3.validate_gazepoint_master(
        master,
        min_valid_sample_pct=70,
        max_missing_gaze_pct=30,
        max_missing_pupil_pct=100,
        max_offscreen_gaze_pct=50,
    )
    assert set(out) >= {
        "summary",
        "checks",
        "failed_checks",
        "warning_checks",
        "column_map",
        "valid",
    }
    assert out["summary"].loc[0, "n_checks"] == 15
    assert out["summary"].loc[0, "valid_sample_pct"] == pytest.approx(75)
    assert set(out["checks"]["check_id"]) == {f"C{i:03d}" for i in range(1, 16)}
    assert bool(out["valid"])


def test_r_validate_master_fail_on_error():
    bad = pd.DataFrame({"subject": ["S1"]})
    with pytest.raises(ValueError):
        gp3.validate_gazepoint_master(bad, min_valid_sample_pct=75, fail_on_error=True)


def test_legacy_validate_master_required_path_unchanged():
    out = gp3.validate_gazepoint_master(
        pd.DataFrame({"subject": ["S1"], "time": [0]}), required=("subject", "time")
    )
    assert out["valid"]
    assert set(out["checks"].columns) == {"check", "passed", "detail"}


def _readiness_data():
    return pd.DataFrame(
        {
            "subject": ["S1", "S1", "S2", "S2"],
            "trial_global": ["t1", "t1", "t2", "t2"],
            "time": [0.0, 0.1, 0.0, 0.1],
            "condition": ["A", "A", "B", "B"],
            "pupil": [3.0, 3.1, 3.2, 3.3],
            "x": [0.2, 0.3, 0.4, 0.5],
            "y": [0.3, 0.4, 0.5, 0.6],
            "valid": [1, 1, 1, 1],
        }
    )


def test_r_real_data_readiness_gate_structure():
    out = gp3.check_gazepoint_real_data_readiness(
        _readiness_data(), analysis_type="pupil", min_participants=2, min_trials=2
    )
    assert set(out) >= {
        "overview",
        "gate_decision",
        "checks",
        "detected_columns",
        "data_summary",
        "condition_summary",
        "settings",
    }
    assert out["overview"].loc[0, "analysis_type"] == "pupil"
    assert bool(out["gate_decision"].loc[0, "ready_for_real_data_analysis"])
    assert {"minimum_participants", "minimum_trials", "pupil_missingness"} <= set(
        out["checks"]["check_id"]
    )


def test_r_real_data_readiness_detects_missing_pupil_for_pupil_analysis():
    data = _readiness_data().drop(columns="pupil")
    out = gp3.check_gazepoint_real_data_readiness(data, analysis_type="pupil")
    assert out["overview"].loc[0, "readiness_status"] == "fail"
    assert not bool(out["overview"].loc[0, "ready_for_real_data_analysis"])


def test_legacy_real_data_readiness_path_unchanged():
    out = gp3.check_gazepoint_real_data_readiness(_readiness_data())
    assert set(out) == {"ready", "checks"}


def test_r_reporting_checklist_structured_pupil_mode():
    gate = gp3.check_gazepoint_real_data_readiness(_readiness_data(), analysis_type="pupil")
    out = gp3.create_gazepoint_reporting_checklist(
        _readiness_data(),
        objects={"readiness": gate},
        analysis_type="pupil",
        study_title="Demo",
        include_optional=False,
    )
    assert set(out) >= {
        "overview",
        "checklist",
        "section_summary",
        "object_summary",
        "data_summary",
        "text_summary",
        "settings",
    }
    assert out["overview"].loc[0, "analysis_type"] == "pupil"
    ids = set(out["checklist"]["item_id"])
    assert "pupil_preprocessing_reported" in ids
    assert "stimulus_luminance_reported" in ids
    assert "advanced_sequence_or_transition_methods" not in ids
    readiness_row = (
        out["checklist"].loc[out["checklist"]["item_id"].eq("real_data_readiness_gate")].iloc[0]
    )
    assert readiness_row["status"] in {"pass", "warn"}


def test_r_reporting_checklist_combined_adds_aoi_and_pupil_sections():
    out = gp3.create_gazepoint_reporting_checklist(
        _readiness_data(), analysis_type="combined", study_title="Combined"
    )
    areas = set(out["checklist"]["reporting_area"])
    assert {"aoi_reporting", "pupil_reporting"} <= areas


def test_legacy_reporting_checklist_remains_dataframe():
    out = gp3.create_gazepoint_reporting_checklist(_readiness_data())
    assert isinstance(out, pd.DataFrame)
    assert {"item", "reported"} <= set(out.columns)
