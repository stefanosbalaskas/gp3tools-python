import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3


def _standard_face():
    return pd.DataFrame(
        {
            "participant_id": ["P1"] * 4 + ["P2"] * 4,
            "face_file": ["a.csv"] * 4 + ["b.csv"] * 4,
            "face_frame": [1, 2, 2, 4, 1, 2, 3, 4],
            "face_time_sec": [0.0, 0.03, 0.06, 0.20, 0.0, 0.03, 0.06, 0.09],
            "face_confidence": [0.95, 0.90, 0.70, np.nan, 0.95, 0.96, 0.97, 0.98],
            "face_success": [True] * 8,
            "face_valid": [True, True, False, False, True, True, True, True],
        }
    )


def test_face_quality_r_structure_group_metrics_and_issue_precedence():
    out = gp3.audit_gazepoint_face_quality(
        _standard_face(),
        group_cols=["participant_id", "face_file", "missing_column"],
        min_valid_percent=60,
        warning_valid_percent=90,
        max_time_gap_sec=0.10,
        max_duplicate_frame_percent=10,
        standardize=False,
    )
    assert out["_gp3_class"] == "gp3_face_quality_audit"
    assert list(out["settings"]["group_cols"]) == ["participant_id", "face_file"]
    assert len(out["group_summary"]) == 2
    assert set(out["group_summary"]["face_quality_status"]) == {"fail", "pass"}
    p1 = out["group_summary"].loc[out["group_summary"]["participant_id"] == "P1"].iloc[0]
    assert p1["n_duplicate_frames"] == 1
    assert p1["max_time_gap_sec"] == pytest.approx(0.14)
    assert p1["face_quality_status"] == "fail"
    issues = out["issue_summary"].set_index("issue")
    assert issues.loc["valid_percent_below_minimum", "n_groups_affected"] == 1
    assert issues.loc["large_time_gaps", "n_groups_affected"] == 1
    assert out["overview"].iloc[0]["n_groups"] == 2


def test_face_quality_unknown_and_overall_only():
    frame = _standard_face().copy()
    frame["face_valid"] = pd.Series([pd.NA] * len(frame), dtype="boolean")
    frame["face_confidence"] = np.nan
    frame["face_success"] = pd.Series([pd.NA] * len(frame), dtype="boolean")
    out = gp3.audit_gazepoint_face_quality(frame, group_cols=None, standardize=False)
    assert len(out["group_summary"]) == 1
    assert out["overview"].iloc[0]["face_quality_status"] == "unknown"
    assert out["issue_summary"].set_index("issue").loc["large_time_gaps", "status"] == "not_checked"


def test_face_quality_summary_and_legacy_path():
    audit = gp3.audit_gazepoint_face_quality(_standard_face(), standardize=False)
    summary = gp3.summarize_gazepoint_face_quality(audit)
    assert isinstance(summary, pd.DataFrame)
    assert summary.attrs["_gp3_class"] == "gp3_face_quality_summary"
    legacy = gp3.audit_gazepoint_face_quality(
        pd.DataFrame({"confidence": [0.9, 0.4]}),
        confidence_col="face_confidence",
        threshold=0.8,
    )
    assert isinstance(legacy, pd.DataFrame)
    assert "prop_below_threshold" in legacy


def test_face_reporting_checklist_r_rows_and_cautions():
    face = gp3.standardize_gazepoint_face_columns(_standard_face(), keep_original_columns=True)
    quality = gp3.audit_gazepoint_face_quality(face, standardize=False)
    windows = pd.DataFrame({"n_used": [5, 0], "face_window_label": ["base", "resp"]})
    checklist = gp3.create_gazepoint_face_reporting_checklist(
        face_data=face,
        quality_audit=quality,
        window_summary=windows,
    )
    assert checklist.attrs["_gp3_class"] == "gp3_face_reporting_checklist"
    assert len(checklist) == 13
    assert set(checklist.columns) == {"section", "item", "status", "evidence", "recommendation"}
    assert "review" in set(checklist["status"])
    no_cautions = gp3.create_gazepoint_face_reporting_checklist(
        face_data=face,
        include_interpretation_cautions=False,
    )
    assert len(no_cautions) == 11


def test_face_qc_report_list_and_markdown():
    face = _standard_face()
    quality = gp3.audit_gazepoint_face_quality(face, standardize=False)
    checklist = gp3.create_gazepoint_face_reporting_checklist(face_data=face, quality_audit=quality)
    structured = gp3.report_gazepoint_face_qc(
        face_data=face,
        quality_audit=quality,
        checklist=checklist,
        output="list",
    )
    assert structured["_gp3_class"] == "gp3_face_qc_report_list"
    assert isinstance(structured["quality_overview"], pd.DataFrame)
    markdown = gp3.report_gazepoint_face_qc(
        face_data=face,
        quality_audit=quality,
        checklist=checklist,
        output="markdown",
    )
    assert isinstance(markdown, str)
    assert "# External facial-behaviour QC report" in markdown
    assert "## Interpretation cautions" in markdown


def test_face_qc_report_validation():
    with pytest.raises(ValueError):
        gp3.report_gazepoint_face_qc(output="html")
    with pytest.raises(TypeError):
        gp3.report_gazepoint_face_qc(checklist={"not": "a frame"})


def test_face_quality_standardize_path_and_warn_status():
    raw = pd.DataFrame(
        {
            "frame": [1, 2, 2, 4],
            "timestamp": [0.0, 0.03, 0.06, 0.20],
            "confidence": [0.95, 0.90, 0.85, 0.82],
            "success": [1, 1, 1, 1],
        }
    )
    out = gp3.audit_gazepoint_face_quality(
        raw,
        group_cols=None,
        confidence_threshold=0.8,
        min_valid_percent=70,
        warning_valid_percent=85,
        max_time_gap_sec=0.10,
        max_duplicate_frame_percent=10,
        standardize=True,
    )
    assert out["overview"].iloc[0]["face_quality_status"] == "warn"
    assert out["overview"].iloc[0]["n_nonpositive_time_steps"] == 0
    assert out["data"].attrs["_gp3_class"] == "gp3_face_data"


def test_face_reporting_sync_and_model_evidence_paths():
    face = _standard_face()
    sync = {
        "overview": pd.DataFrame(
            [{"n_rows": 8, "matched_percent": 97.5, "face_sync_audit_status": "warn"}]
        ),
        "issue_summary": pd.DataFrame([{"issue": "outside_tolerance", "n_groups_affected": 1}]),
        "_gp3_class": "gp3_face_sync_audit",
    }
    model = {
        "settings": {
            "outcome": "AU12_r",
            "predictors": ["condition"],
            "covariates": ["age"],
            "random_effects": "1|participant_id",
            "n_rows_input": 100,
            "n_rows_model": 96,
        },
        "_gp3_class": "gp3_face_window_lmm",
    }
    reactivity = pd.DataFrame(
        {
            "participant_id": ["P1"],
            "measure": ["AU12_r"],
            "reactivity": [0.2],
            "absolute_reactivity": [0.2],
        }
    )
    checklist = gp3.create_gazepoint_face_reporting_checklist(
        face_data=face,
        sync_audit=sync,
        reactivity_summary=reactivity,
        multimodal_model=model,
    )
    sync_row = checklist.loc[checklist["item"] == "Synchronisation status is acceptable"].iloc[0]
    assert sync_row["status"] == "warn"
    structured = gp3.report_gazepoint_face_qc(
        face_data=face,
        sync_audit=sync,
        reactivity_summary=reactivity,
        multimodal_model=model,
        output="list",
        include_cautions=False,
    )
    assert len(structured["sync_overview"]) == 1
    assert structured["cautions"] == []
    assert structured["model_summary"].iloc[0]["n_rows_model"] == 96
