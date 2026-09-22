from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3
import gp3tools._exports as exports_mod
import gp3tools.face as face_mod
import gp3tools.io as io_mod

# ---------------------------------------------------------------------------
# gp3tools.__init__
# ---------------------------------------------------------------------------


class _RejectStatusAttribute:
    def __setattr__(self, name, value):
        if name == "_gp3tools_status":
            raise RuntimeError("synthetic immutable export")
        object.__setattr__(self, name, value)


def test_init_bootstrap_handles_status_assignment_failure_and_bridge() -> None:
    """Exercise bootstrap branches without reloading the live package."""
    original_exports = exports_mod.R_EXPORTS

    native_name = "_coverage_native_export"
    bridge_name = "_coverage_bridge_export"
    velocity_name = "detect_gazepoint_fixations_velocity"

    had_native = hasattr(io_mod, native_name)
    previous_native = getattr(io_mod, native_name, None)

    try:
        exports_mod.R_EXPORTS = (
            native_name,
            bridge_name,
            velocity_name,
        )

        setattr(
            io_mod,
            native_name,
            _RejectStatusAttribute(),
        )

        source_path = gp3.__file__

        with open(source_path, encoding="utf-8") as handle:
            source = handle.read()

        namespace = {
            "__name__": "gp3tools",
            "__package__": "gp3tools",
            "__file__": source_path,
            "__builtins__": __builtins__,
        }

        exec(
            compile(
                source,
                source_path,
                "exec",
            ),
            namespace,
        )

        assert native_name in namespace
        assert bridge_name in namespace
        assert namespace["_NATIVE_SOURCE"][native_name] == "io"
        assert bridge_name not in namespace["_NATIVE_SOURCE"]
        assert callable(namespace[bridge_name])

    finally:
        exports_mod.R_EXPORTS = original_exports

        if had_native:
            setattr(
                io_mod,
                native_name,
                previous_native,
            )
        else:
            try:
                delattr(io_mod, native_name)
            except AttributeError:
                pass


def test_velocity_public_explicit_return_mode_branch(
    monkeypatch,
) -> None:
    def fake_native(*args, **kwargs):
        return {
            "events": pd.DataFrame(
                {
                    "x": [1],
                }
            )
        }

    monkeypatch.setattr(
        gp3,
        "_gp3_velocity_native",
        fake_native,
    )

    result = gp3._gp3_velocity_public(
        return_mode="both",
    )

    assert result["_gp3_class"] == "gp3_velocity_fixation_result"


def test_api_status_survives_signature_failure(
    monkeypatch,
) -> None:
    original = gp3.inspect.signature
    calls = {"n": 0}

    def flaky(obj):
        if calls["n"] == 0:
            calls["n"] += 1
            raise ValueError("synthetic signature failure")

        return original(obj)

    monkeypatch.setattr(
        gp3.inspect,
        "signature",
        flaky,
    )

    status = gp3.api_status()

    assert len(status) == len(gp3.R_EXPORTS)
    assert status["signature"].eq("").any()


# ---------------------------------------------------------------------------
# Face source / schema standardisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("frame", "expected"),
    [
        (
            pd.DataFrame(
                {
                    "blendshape_smile": [0.2],
                }
            ),
            "mediapipe",
        ),
        (
            pd.DataFrame(
                {
                    "AU01": [0.2],
                    "happy": [0.8],
                }
            ),
            "pyfeat",
        ),
        (
            pd.DataFrame(
                {
                    "valence": [0.2],
                    "quality": [0.9],
                }
            ),
            "facereader",
        ),
    ],
)
def test_face_auto_source_detection(
    frame,
    expected,
) -> None:
    result = face_mod.standardize_gazepoint_face_columns(frame)

    assert result.attrs["gp3_face_standardization"]["detected_source"] == expected


def test_face_standardization_validation_and_blank_columns(
    tmp_path,
) -> None:
    with pytest.raises(
        ValueError,
        match="source must be one of",
    ):
        face_mod.standardize_gazepoint_face_columns(
            pd.DataFrame({"x": [1]}),
            source="invalid",
        )

    with pytest.raises(
        ValueError,
        match="readable CSV path",
    ):
        face_mod.standardize_gazepoint_face_columns(tmp_path / "missing.csv")

    frame = pd.DataFrame(
        [[1, 2]],
        columns=[
            None,
            "",
        ],
    )

    result = face_mod.standardize_gazepoint_face_columns(frame)

    assert {
        "unnamed_face_column_1",
        "unnamed_face_column_2",
    } <= set(result.columns)


def test_face_source_metadata_single_mixed_and_empty() -> None:
    single = face_mod.standardize_gazepoint_face_columns(
        pd.DataFrame(
            {
                "gp3_face_source": ["custom"],
            }
        )
    )

    assert single.attrs["gp3_face_standardization"]["detected_source"] == "custom"

    mixed = face_mod.standardize_gazepoint_face_columns(
        pd.DataFrame(
            {
                "gp3_face_source": [
                    "source_a",
                    "source_b",
                ],
            }
        )
    )

    assert mixed.attrs["gp3_face_standardization"]["detected_source"] == "mixed"

    empty = face_mod.standardize_gazepoint_face_columns(
        pd.DataFrame(
            {
                "gp3_face_source": [pd.NA],
                "blendshape_smile": [0.5],
            }
        )
    )

    assert empty.attrs["gp3_face_standardization"]["detected_source"] == "mediapipe"


def test_face_explicit_column_and_boolean_paths() -> None:
    frame = pd.DataFrame(
        {
            "pid": [
                "P1",
                "P2",
            ],
            "conf": [
                0.95,
                0.95,
            ],
            "ok": pd.Series(
                [
                    True,
                    False,
                ],
                dtype=bool,
            ),
        }
    )

    result = face_mod.standardize_gazepoint_face_columns(
        frame,
        participant_id_col="pid",
        confidence_col="conf",
        success_col="ok",
    )

    assert result["participant_id"].astype(str).tolist() == [
        "P1",
        "P2",
    ]

    assert result["face_valid"].tolist() == [
        True,
        False,
    ]

    with pytest.raises(
        ValueError,
        match="Column not found",
    ):
        face_mod.standardize_gazepoint_face_columns(
            frame,
            participant_id_col="missing",
        )


# ---------------------------------------------------------------------------
# Face quality audit
# ---------------------------------------------------------------------------


def test_face_quality_legacy_autodetect_and_missing_named_confidence() -> None:
    autodetected = face_mod.audit_gazepoint_face_quality(
        pd.DataFrame(
            {
                "confidence": [
                    0.9,
                    0.5,
                ]
            }
        ),
        threshold=0.8,
    )

    assert autodetected["n_valid"].iloc[0] == 2

    fallback = face_mod.audit_gazepoint_face_quality(
        pd.DataFrame(
            {
                "x": [
                    1,
                    2,
                ]
            }
        ),
        confidence_col="not_present",
    )

    assert fallback["n_valid"].iloc[0] == 2
    assert np.isnan(fallback["mean_confidence"].iloc[0])


def test_face_quality_path_empty_single_group_and_time_gap(
    tmp_path,
) -> None:
    csv_path = tmp_path / "face.csv"

    csv_path.write_text(
        "frame,timestamp,confidence,success,AU01_r\n1,0.0,0.95,1,0.1\n2,0.1,0.96,1,0.2\n",
        encoding="utf-8",
    )

    result = face_mod.audit_gazepoint_face_quality(
        csv_path,
        group_cols=None,
    )

    assert result["_gp3_class"] == "gp3_face_quality_audit"

    with pytest.raises(
        ValueError,
        match="at least one row",
    ):
        face_mod.audit_gazepoint_face_quality(pd.DataFrame())

    raw = pd.DataFrame(
        {
            "participant_id": [
                "P1",
                "P1",
            ],
            "frame": [
                1,
                2,
            ],
            "timestamp": [
                0.0,
                2.0,
            ],
            "confidence": [
                0.95,
                0.95,
            ],
            "success": [
                1,
                1,
            ],
        }
    )

    audited = face_mod.audit_gazepoint_face_quality(
        raw,
        group_cols="participant_id",
        max_time_gap_sec=0.5,
        max_duplicate_frame_percent=100,
    )

    assert audited["group_summary"]["face_quality_status"].iloc[0] == "warn"

    assert audited["group_summary"]["participant_id"].iloc[0] == "P1"


# ---------------------------------------------------------------------------
# Synchronisation
# ---------------------------------------------------------------------------


def _standard_face(
    group="A",
    frame=1,
    time=0.0,
):
    return pd.DataFrame(
        {
            "group": [group],
            "face_frame": pd.Series(
                [frame],
                dtype="Int64",
            ),
            "face_time_sec": [time],
            "face_confidence": [0.95],
            "face_valid": pd.Series(
                [True],
                dtype="boolean",
            ),
        }
    )


def test_face_sync_validation_mapping_and_tolerance() -> None:
    gaze = pd.DataFrame(
        {
            "time": [0.0],
            "group": ["A"],
        }
    )

    face = _standard_face()

    with pytest.raises(
        ValueError,
        match="method must",
    ):
        face_mod.sync_gazepoint_face_data(
            gaze,
            face,
            method="invalid",
            standardize_face=False,
        )

    with pytest.raises(
        ValueError,
        match="non-negative finite",
    ):
        face_mod.sync_gazepoint_face_data(
            gaze,
            face,
            tolerance_sec=-1,
            standardize_face=False,
        )

    milliseconds = face_mod.sync_gazepoint_face_data(
        gaze,
        face,
        gaze_time_col="time",
        face_time_col="face_time_sec",
        tolerance_ms=25,
        standardize_face=False,
    )

    assert milliseconds.attrs["gp3_face_sync_settings"]["tolerance_sec"] == pytest.approx(0.025)

    for by in (
        "group",
        ["group"],
    ):
        synced = face_mod.sync_gazepoint_face_data(
            gaze,
            face,
            gaze_time_col="time",
            face_time_col="face_time_sec",
            by=by,
            standardize_face=False,
        )

        assert synced["face_sync_status"].iloc[0] == "matched"

    with pytest.raises(
        ValueError,
        match="Gazepoint grouping",
    ):
        face_mod.sync_gazepoint_face_data(
            gaze,
            face,
            by="missing",
            standardize_face=False,
        )

    with pytest.raises(
        ValueError,
        match="Facial-data grouping",
    ):
        face_mod.sync_gazepoint_face_data(
            gaze,
            face,
            by={
                "group": "missing_face",
            },
            standardize_face=False,
        )


def test_face_sync_column_detection_and_errors() -> None:
    face = _standard_face()

    with pytest.raises(
        ValueError,
        match="gaze_time_col was not found",
    ):
        face_mod.sync_gazepoint_face_data(
            pd.DataFrame(
                {
                    "time": [0.0],
                }
            ),
            face,
            gaze_time_col="missing",
            standardize_face=False,
        )

    detected = face_mod.sync_gazepoint_face_data(
        pd.DataFrame(
            {
                "timestamp": [0.0],
            }
        ),
        face,
        standardize_face=False,
    )

    assert detected["face_sync_status"].iloc[0] == "matched"

    with pytest.raises(
        ValueError,
        match="could not be detected automatically",
    ):
        face_mod.sync_gazepoint_face_data(
            pd.DataFrame(
                {
                    "x": [1],
                }
            ),
            face,
            standardize_face=False,
        )


def test_face_sync_unmatched_missing_time_and_frame() -> None:
    unmatched = face_mod.sync_gazepoint_face_data(
        pd.DataFrame(
            {
                "time": [0.0],
                "group": ["A"],
            }
        ),
        _standard_face(
            group="B",
        ),
        gaze_time_col="time",
        face_time_col="face_time_sec",
        by="group",
        standardize_face=False,
    )

    assert unmatched["face_sync_status"].iloc[0] == "unmatched"

    missing_time = face_mod.sync_gazepoint_face_data(
        pd.DataFrame(
            {
                "time": [np.nan],
                "group": ["A"],
            }
        ),
        _standard_face(),
        gaze_time_col="time",
        face_time_col="face_time_sec",
        by="group",
        standardize_face=False,
    )

    assert missing_time["face_sync_status"].iloc[0] == "missing_gaze_time"

    frames = face_mod.sync_gazepoint_face_data(
        pd.DataFrame(
            {
                "frame": pd.Series(
                    [
                        pd.NA,
                        99,
                    ],
                    dtype="Int64",
                ),
            }
        ),
        _standard_face(
            frame=1,
        ),
        method="frame_exact",
        standardize_face=False,
    )

    assert frames["face_sync_status"].tolist() == [
        "missing_gaze_frame",
        "unmatched",
    ]


# ---------------------------------------------------------------------------
# Compatibility audit routes
# ---------------------------------------------------------------------------


def test_face_sync_audit_legacy_summary_and_keyword_error() -> None:
    with pytest.raises(
        TypeError,
        match="Unexpected keyword",
    ):
        face_mod.audit_gazepoint_face_sync(
            data=pd.DataFrame(
                {
                    "x": [1],
                }
            ),
            unexpected=True,
        )

    with_face = face_mod.audit_gazepoint_face_sync(
        data=pd.DataFrame(
            {
                "score_face": [
                    1.0,
                    np.nan,
                ]
            }
        )
    )

    assert with_face["n_matched"].iloc[0] == 1

    without_face = face_mod.audit_gazepoint_face_sync(
        data=pd.DataFrame(
            {
                "x": [
                    1,
                    2,
                ]
            }
        )
    )

    assert without_face["n_matched"].iloc[0] == 2


def test_event_sync_unknown_keyword_contract() -> None:
    with pytest.raises(
        TypeError,
        match="Unexpected keyword",
    ):
        face_mod.audit_gazepoint_event_sync(
            data=pd.DataFrame(
                {
                    "time": [0],
                }
            ),
            unexpected=True,
        )


# ---------------------------------------------------------------------------
# Window / multimodal compatibility
# ---------------------------------------------------------------------------


def test_face_window_summary_without_groups() -> None:
    result = face_mod.summarize_gazepoint_face_windows(
        pd.DataFrame(
            {
                "metric": [
                    1.0,
                    3.0,
                ]
            }
        ),
        group_cols=None,
        value_cols=["metric"],
    )

    assert result["metric"].iloc[0] == pytest.approx(2.0)


def test_prepare_multimodal_historical_and_unknown_kwargs() -> None:
    gaze = pd.DataFrame(
        {
            "x": [
                1,
                2,
            ]
        }
    )

    direct = face_mod.prepare_gazepoint_multimodal_data(gaze=gaze)

    pd.testing.assert_frame_equal(
        direct,
        gaze,
    )

    direct_kwargs = face_mod.prepare_gazepoint_multimodal_data(
        gaze=gaze,
        legacy_option=True,
    )

    pd.testing.assert_frame_equal(
        direct_kwargs,
        gaze,
    )

    with pytest.raises(
        TypeError,
        match="Unexpected keyword",
    ):
        face_mod.prepare_gazepoint_multimodal_data(
            face_windows=pd.DataFrame(
                {
                    "participant": [
                        "P1",
                    ]
                }
            ),
            unexpected=True,
        )


# ---------------------------------------------------------------------------
# Reporting checklist
# ---------------------------------------------------------------------------


def _audit(
    column,
    value,
    issue_summary=None,
    gp3_class=None,
):
    result = {
        "overview": pd.DataFrame(
            {
                column: [value],
            }
        )
    }

    if issue_summary is not None:
        result["issue_summary"] = issue_summary

    if gp3_class is not None:
        result["_gp3_class"] = gp3_class

    return result


def test_reporting_checklist_non_dataframe_and_status_routes() -> None:
    quality = _audit(
        "face_quality_status",
        "pass",
        issue_summary=pd.DataFrame(
            {
                "issue": ["x"],
            }
        ),
        gp3_class="gp3_face_quality_audit",
    )

    sync = _audit(
        "face_sync_audit_status",
        "fail",
        gp3_class="gp3_face_sync_audit",
    )

    checklist = face_mod.create_gazepoint_face_reporting_checklist(
        face_data=object(),
        quality_audit=quality,
        sync_audit=sync,
        window_summary=pd.DataFrame(),
        reactivity_summary=object(),
        multimodal_model=object(),
        include_interpretation_cautions=False,
    )

    statuses = set(checklist["status"])

    assert "pass" in statuses
    assert "fail" in statuses
    assert "not_available" in statuses


def test_reporting_checklist_unknown_review_and_zero_issue_routes() -> None:
    quality = _audit(
        "face_quality_status",
        None,
        issue_summary=pd.DataFrame(
            {
                "issue": ["none"],
                "n_groups_affected": [0],
            }
        ),
    )

    sync = _audit(
        "face_sync_audit_status",
        "custom_status",
    )

    checklist = face_mod.create_gazepoint_face_reporting_checklist(
        quality_audit=quality,
        sync_audit=sync,
        window_summary=pd.DataFrame(
            {
                "metric": [1.0],
            }
        ),
        reactivity_summary=pd.DataFrame(
            {
                "measure": ["AU01"],
            }
        ),
        multimodal_model={
            "not_settings": {
                "x": 1,
            }
        },
        include_interpretation_cautions=False,
    )

    statuses = set(checklist["status"])

    assert "unknown" in statuses
    assert "review" in statuses
    assert "pass" in statuses


def test_reporting_checklist_affected_issue_and_nan_window() -> None:
    quality = _audit(
        "face_quality_status",
        "not_available",
        issue_summary=pd.DataFrame(
            {
                "issue": ["missing"],
                "n_groups_affected": [2],
            }
        ),
    )

    sync = _audit(
        "face_sync_audit_status",
        "ok",
    )

    checklist = face_mod.create_gazepoint_face_reporting_checklist(
        quality_audit=quality,
        sync_audit=sync,
        window_summary=pd.DataFrame(
            {
                "n_used": [np.nan],
            }
        ),
        include_interpretation_cautions=False,
    )

    statuses = set(checklist["status"])

    assert "not_available" in statuses
    assert "review" in statuses
    assert "pass" in statuses


def test_reporting_checklist_missing_audit_status_column() -> None:
    quality = {
        "overview": pd.DataFrame(
            {
                "n_rows": [1],
            }
        ),
        "issue_summary": pd.DataFrame(
            {
                "issue": ["x"],
                "n_groups_affected": [0],
            }
        ),
    }

    checklist = face_mod.create_gazepoint_face_reporting_checklist(
        quality_audit=quality,
        include_interpretation_cautions=False,
    )

    row = checklist.loc[checklist["item"].eq("Face-data quality status is acceptable")].iloc[0]

    assert row["status"] == "unknown"

    assert "missing the status column" in row["evidence"]


def test_velocity_public_legacy_return_fallback_and_no_class(
    monkeypatch,
) -> None:
    """Cover legacy return alias and the no-decoration result branch."""

    def fake_native(*args, **kwargs):
        return {
            "events": pd.DataFrame(
                {
                    "x": [1],
                }
            )
        }

    monkeypatch.setattr(
        gp3,
        "_gp3_velocity_native",
        fake_native,
    )

    result = gp3._gp3_velocity_public(
        **{
            "return": "events",
        }
    )

    assert "_gp3_class" not in result
    assert result["events"]["x"].tolist() == [1]
