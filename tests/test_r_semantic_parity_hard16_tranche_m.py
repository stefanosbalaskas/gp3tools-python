import inspect

import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3


def _flow_data():
    return pd.DataFrame(
        {
            "subject": ["S1", "S1", "S1", "S2", "S2", "S2"],
            "condition": ["A", "A", "B", "A", "B", "B"],
            "trial": [1, 1, 2, 1, 2, 3],
            "status": ["included", "included", "excluded", "included", "review", "included"],
            "reason": [None, None, "blink", None, "trackloss", None],
        }
    )


def test_call_surface_accepts_r_arguments():
    sig = inspect.signature(gp3.audit_gazepoint_exclusion_flow)
    assert {
        "subject_col",
        "condition_col",
        "unit_cols",
        "include_col",
        "exclude_col",
        "status_col",
        "reason_col",
        "included_values",
        "excluded_values",
        "min_retained_prop",
        "max_condition_exclusion_ratio",
    } <= set(sig.parameters)
    sig2 = inspect.signature(gp3.audit_gazepoint_post_exclusion_balance)
    assert {
        "subject_col",
        "condition_col",
        "unit_cols",
        "retained_col",
        "include_col",
        "exclude_col",
        "status_col",
        "expected_conditions",
        "included_values",
        "excluded_values",
        "min_retained_units_per_condition",
        "min_retained_units_per_subject_condition",
        "max_condition_count_ratio",
        "max_subject_condition_ratio",
        "require_all_conditions_per_subject",
    } <= set(sig2.parameters)


def test_exclusion_flow_r_structured_output_and_reasons():
    out = gp3.audit_gazepoint_exclusion_flow(
        _flow_data(),
        subject_col="subject",
        condition_col="condition",
        unit_cols=["trial"],
        status_col="status",
        reason_col="reason",
        min_retained_prop=0.5,
    )
    assert set(out) == {
        "overview",
        "unit_flow",
        "reason_summary",
        "condition_summary",
        "subject_summary",
        "flagged_units",
        "settings",
    }
    assert len(out["unit_flow"]) == 5
    assert set(out["unit_flow"]["exclusion_flow_status"]) == {"retained", "excluded"}
    assert {"blink", "trackloss"} <= set(out["reason_summary"]["exclusion_reason"])
    assert out["overview"].loc[0, "n_retained_units"] == 3
    assert out["overview"].loc[0, "n_excluded_units"] == 2
    assert out["overview"].loc[0, "exclusion_flow_status"] in {"ok", "review"}


def test_exclusion_flow_conflicts_precedence_and_default_reasons():
    df = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S2"],
            "condition": ["A", "A", "A"],
            "trial": [1, 1, 2],
            "keep": [1, 1, 1],
            "drop": [0, 1, np.nan],
        }
    )
    out = gp3.audit_gazepoint_exclusion_flow(
        df,
        subject_col="subject",
        condition_col="condition",
        unit_cols="trial",
        include_col="keep",
        exclude_col="drop",
    )
    unit = out["unit_flow"]
    assert "conflicting_flags" in set(unit["exclusion_flow_status"])
    assert "conflicting_flags" in set(out["flagged_units"]["exclusion_reason"])


def test_exclusion_flow_aliases_and_condition_ratio_zero_case():
    df = pd.DataFrame(
        {
            "USER_FILE": ["S1", "S2"],
            "MEDIA_ID": [1, 2],
            "condition": ["A", "B"],
            "flag": [True, True],
        }
    )
    out = gp3.audit_gazepoint_exclusion_flow(
        df,
        subject_col="USER_FILE",
        condition_col="condition",
        unit_cols="MEDIA_ID",
        include_col="flag",
    )
    assert out["overview"].loc[0, "condition_exclusion_ratio"] == pytest.approx(1.0)
    assert out["overview"].loc[0, "exclusion_flow_status"] == "ok"


def test_exclusion_flow_validations():
    df = pd.DataFrame({"subject": ["S1"], "bad": [2]})
    with pytest.raises(ValueError):
        gp3.audit_gazepoint_exclusion_flow(df, subject_col="subject")
    with pytest.raises(ValueError):
        gp3.audit_gazepoint_exclusion_flow(df, subject_col="subject", include_col="bad")
    with pytest.raises(ValueError):
        gp3.audit_gazepoint_exclusion_flow(pd.DataFrame(), subject_col="subject", include_col="x")
    with pytest.raises(ValueError):
        gp3.audit_gazepoint_exclusion_flow(
            pd.DataFrame({"subject": ["S1"], "keep": [1]}),
            subject_col="subject",
            include_col="keep",
            min_retained_prop=2,
        )


def test_exclusion_flow_legacy_path_unchanged():
    df = pd.DataFrame({"excluded_a": [False, True, False], "flag_b": [False, False, True]})
    out = gp3.audit_gazepoint_exclusion_flow(df)
    assert list(out["stage"]) == ["input", "excluded_a", "flag_b"]
    assert list(out["n"]) == [3, 2, 1]


def test_post_exclusion_r_structured_balanced_all_retained():
    df = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S2", "S2"],
            "condition": ["A", "B", "A", "B"],
            "trial": [1, 2, 1, 2],
        }
    )
    out = gp3.audit_gazepoint_post_exclusion_balance(
        df,
        subject_col="subject",
        condition_col="condition",
        unit_cols="trial",
    )
    assert set(out) == {
        "overview",
        "unit_flow",
        "cell_summary",
        "condition_summary",
        "subject_summary",
        "flagged_cells",
        "flagged_subjects",
        "settings",
    }
    assert out["overview"].loc[0, "n_retained_units"] == 4
    assert out["overview"].loc[0, "post_exclusion_balance_status"] == "ok"
    assert out["flagged_cells"].empty


def test_post_exclusion_expected_missing_condition_and_imbalance():
    df = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S2"],
            "condition": ["A", "B", "A"],
            "trial": [1, 2, 1],
            "retained": [True, False, True],
        }
    )
    out = gp3.audit_gazepoint_post_exclusion_balance(
        df,
        subject_col="subject",
        condition_col="condition",
        unit_cols="trial",
        retained_col="retained",
        expected_conditions=["A", "B"],
        max_condition_count_ratio=1.5,
    )
    assert not out["flagged_cells"].empty
    assert not out["flagged_subjects"].empty
    assert out["overview"].loc[0, "post_exclusion_balance_status"] == "review"
    assert out["overview"].loc[0, "condition_ratio_status"] == "condition_count_imbalance"


def test_post_exclusion_flag_precedence_and_status_values():
    df = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S1", "S1"],
            "condition": ["A", "A", "B", "B"],
            "trial": [1, 2, 3, 4],
            "include": [True, True, True, True],
            "exclude": [False, True, False, False],
            "status": ["included", "included", "invalid", "included"],
        }
    )
    out = gp3.audit_gazepoint_post_exclusion_balance(
        df,
        subject_col="subject",
        condition_col="condition",
        unit_cols="trial",
        include_col="include",
        exclude_col="exclude",
        status_col="status",
    )
    units = out["unit_flow"].sort_values("trial")
    assert list(units["retained"]) == [True, True, False, True]


def test_post_exclusion_low_counts_and_ratio_statuses():
    df = pd.DataFrame(
        {
            "subject": ["S1", "S1", "S1", "S2", "S2", "S2"],
            "condition": ["A", "A", "B", "A", "A", "B"],
            "trial": [1, 2, 3, 1, 2, 3],
            "retained": [True, True, True, True, True, True],
        }
    )
    out = gp3.audit_gazepoint_post_exclusion_balance(
        df,
        subject_col="subject",
        condition_col="condition",
        unit_cols="trial",
        retained_col="retained",
        min_retained_units_per_condition=3,
        min_retained_units_per_subject_condition=2,
        max_subject_condition_ratio=1.5,
    )
    assert (out["condition_summary"]["post_exclusion_condition_status"] != "ok").any()
    assert (out["subject_summary"]["post_exclusion_subject_status"] != "ok").any()


def test_post_exclusion_alias_and_validation_paths():
    df = pd.DataFrame({"USER_FILE": ["S1"], "MEDIA_ID": [1], "condition": ["A"], "retained": [1]})
    out = gp3.audit_gazepoint_post_exclusion_balance(
        df,
        subject_col="USER_FILE",
        condition_col="condition",
        unit_cols="MEDIA_ID",
        retained_col="retained",
        require_all_conditions_per_subject=False,
    )
    assert out["overview"].loc[0, "n_subjects"] == 1
    with pytest.raises(ValueError):
        gp3.audit_gazepoint_post_exclusion_balance(
            df,
            subject_col="USER_FILE",
            condition_col="condition",
            retained_col="retained",
            min_retained_units_per_condition=0,
        )
    with pytest.raises(KeyError):
        gp3.audit_gazepoint_post_exclusion_balance(
            df, subject_col="missing", condition_col="condition", retained_col="retained"
        )


def test_post_exclusion_legacy_path_unchanged():
    df = pd.DataFrame({"condition": ["A", "A", "B"], "excluded": [False, True, False]})
    out = gp3.audit_gazepoint_post_exclusion_balance(df)
    assert dict(zip(out["condition"], out["n_retained"], strict=False)) == {"A": 1, "B": 1}
