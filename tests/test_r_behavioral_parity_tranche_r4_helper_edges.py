from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gp3tools import _behavioral_r4 as r4


def test_r4_scalar_helpers_cover_missing_and_numeric_cases() -> None:
    assert np.isnan(
        r4._safe_min(
            [
                np.nan,
                np.inf,
            ]
        )
    )

    assert np.isnan(
        r4._safe_max(
            [
                np.nan,
                -np.inf,
            ]
        )
    )

    assert np.isnan(
        r4._safe_mean(
            [
                np.nan,
            ]
        )
    )

    assert np.isnan(
        r4._safe_median(
            [
                np.nan,
            ]
        )
    )

    assert np.isnan(
        r4._safe_sd(
            [
                1.0,
            ]
        )
    )

    assert np.isclose(
        r4._safe_sd(
            [
                1.0,
                3.0,
            ]
        ),
        np.sqrt(2.0),
    )

    assert np.isnan(
        r4._prop_true_pct(
            [
                None,
                np.nan,
            ]
        )
    )

    assert np.isclose(
        r4._prop_true_pct(
            [
                True,
                False,
                True,
            ]
        ),
        100.0 * 2.0 / 3.0,
    )


def test_r4_character_and_nullable_helpers() -> None:
    assert r4._r_character(True) == "TRUE"

    assert r4._r_character(False) == "FALSE"

    assert r4._r_character(3) == "3"

    assert r4._r_character(3.5) == "3.5"

    assert r4._r_character(4.0) == "4"

    assert pd.isna(r4._r_character(np.nan))

    assert pd.isna(r4._collapse_nullable(None))

    assert r4._collapse_nullable("x") == "x"

    assert r4._collapse_nullable(3) == "3"

    assert (
        r4._collapse_nullable(
            [
                "a",
                "b",
            ]
        )
        == "a, b"
    )

    assert pd.isna(r4._collapse_nullable([]))


def test_r4_trapezoid_helper_edge_cases() -> None:
    assert np.isnan(
        r4._trapz_r(
            np.array(
                [
                    np.nan,
                ]
            ),
            np.array(
                [
                    1.0,
                ]
            ),
        )
    )

    value = r4._trapz_r(
        np.array(
            [
                2.0,
                0.0,
                1.0,
            ]
        ),
        np.array(
            [
                0.0,
                0.0,
                1.0,
            ]
        ),
    )

    assert np.isclose(
        value,
        1.0,
    )


def test_r4_pupil_helper_window_validation_and_empty_response() -> None:
    data = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S1",
            ],
            "trial": [
                1,
                1,
            ],
            "time": [
                -0.1,
                0.0,
            ],
            "pupil": [
                np.nan,
                np.nan,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="baseline_window",
    ):
        r4._pupil_response(
            data,
            pupil="pupil",
            time="time",
            subject="subject",
            trial="trial",
            baseline_window=[
                0.0,
            ],
            response_window=[
                0.0,
                1.0,
            ],
            condition=None,
            interpolated=None,
        )

    with pytest.raises(
        ValueError,
        match="response_window",
    ):
        r4._pupil_response(
            data,
            pupil="pupil",
            time="time",
            subject="subject",
            trial="trial",
            baseline_window=[
                -1.0,
                0.0,
            ],
            response_window=[
                0.0,
            ],
            condition=None,
            interpolated=None,
        )

    out = r4._pupil_response(
        data,
        pupil="pupil",
        time="time",
        subject="subject",
        trial="trial",
        baseline_window=[
            -0.1,
            0.0,
        ],
        response_window=[
            10.0,
            11.0,
        ],
        condition=None,
        interpolated=None,
    )

    assert np.isnan(
        out.loc[
            0,
            "baseline_mean",
        ]
    )

    assert np.isnan(
        out.loc[
            0,
            "peak_dilation",
        ]
    )

    assert np.isnan(
        out.loc[
            0,
            "latency_to_peak",
        ]
    )

    assert np.isnan(
        out.loc[
            0,
            "missing_percent",
        ]
    )


def test_r4_master_detection_false_and_alias_paths() -> None:
    assert not r4._canonical_master(object())

    incomplete = pd.DataFrame(
        {
            "subject": [
                "S1",
            ]
        }
    )

    assert not r4._canonical_master(incomplete)

    aliased = pd.DataFrame(
        {
            "participant": [
                "S1",
            ],
            "MEDIA_ID": [
                "M1",
            ],
            "time_orig_ms": [
                0.0,
            ],
            "gaze_x": [
                0.5,
            ],
            "gaze_y": [
                0.5,
            ],
            "valid_sample": [
                True,
            ],
            "missing_gaze": [
                False,
            ],
            "missing_pupil": [
                False,
            ],
            "gaze_offscreen": [
                False,
            ],
            "pupil_raw": [
                3.0,
            ],
            "AOI": [
                "target",
            ],
            "aoi_count": [
                1,
            ],
        }
    )

    mapping = r4._detect_master_columns(aliased)

    assert mapping["subject"] == "participant"

    assert mapping["media_id"] == "MEDIA_ID"

    assert mapping["time_ms"] == "time_orig_ms"

    assert r4._canonical_master(aliased)


def test_r4_qc_helper_discovery_and_priorities() -> None:
    direct = pd.DataFrame(
        {
            "object_name": [
                "a",
            ],
            "qc_status": [
                "pass",
            ],
        }
    )

    found = r4._find_object_summary(direct)

    assert found is not None

    nested = {
        "x": {
            "object_summary": direct,
        }
    }

    found2 = r4._find_object_summary(nested)

    assert found2 is not None

    assert (
        r4._find_object_summary(
            pd.DataFrame(
                {
                    "x": [
                        1,
                    ]
                }
            )
        )
        is None
    )

    assert r4._status_priority("fail") > r4._status_priority("warn")

    assert r4._status_priority("unrecognised") == 1


def test_r4_qc_derived_object_summary_variants() -> None:
    bundle = {
        "sampling": pd.DataFrame(
            {
                "qc_status": [
                    "pass",
                    "warn",
                ]
            }
        ),
        "geometry": {
            "qc_status": "fail",
        },
        "nested": {
            "overview": pd.DataFrame(
                {
                    "geometry_status": [
                        "info",
                    ]
                }
            )
        },
        "empty": pd.DataFrame(
            {
                "x": [
                    1,
                ]
            }
        ),
    }

    derived = r4._derive_object_summary(bundle)

    assert derived is not None

    statuses = dict(
        zip(
            derived["object_name"],
            derived["qc_status"],
            strict=False,
        )
    )

    assert statuses["sampling"] == "warn"

    assert statuses["geometry"] == "fail"

    assert statuses["nested"] == "info"

    assert statuses["empty"] == "unknown"

    assert r4._derive_object_summary([]) is None

    assert r4._derive_object_summary({}) is None


def test_r4_qc_summary_overall_status_branches() -> None:
    pass_only = pd.DataFrame(
        {
            "object_name": [
                "a",
            ],
            "qc_status": [
                "pass",
            ],
        }
    )

    out = r4._qc_status_summary(
        None,
        pass_only,
    )

    assert out is not None

    assert (
        out["overview"].loc[
            0,
            "qc_overview_status",
        ]
        == "pass"
    )

    warn = pass_only.copy()

    warn.loc[
        0,
        "qc_status",
    ] = "warn"

    out_warn = r4._qc_status_summary(
        None,
        warn,
    )

    assert out_warn is not None

    assert (
        out_warn["overview"].loc[
            0,
            "qc_overview_status",
        ]
        == "warn"
    )

    info = pass_only.copy()

    info.loc[
        0,
        "qc_status",
    ] = "unknown"

    out_info = r4._qc_status_summary(
        None,
        info,
    )

    assert out_info is not None

    assert (
        out_info["overview"].loc[
            0,
            "qc_overview_status",
        ]
        == "info"
    )

    assert (
        r4._qc_status_summary(
            None,
            object(),
        )
        is None
    )


def test_r4_workflow_count_helpers() -> None:
    assert (
        r4._nrow_safe(
            pd.DataFrame(
                {
                    "x": [
                        1,
                        2,
                    ]
                }
            )
        )
        == 2
    )

    assert r4._nrow_safe([]) is None

    assert r4._n_entries_safe(None) == 0

    assert (
        r4._n_entries_safe(
            pd.DataFrame(
                {
                    "x": [
                        1,
                        2,
                    ]
                }
            )
        )
        == 2
    )

    assert (
        r4._n_entries_safe(
            [
                "a",
                "b",
            ]
        )
        == 2
    )

    assert r4._n_entries_safe(7) == 1


def test_r4_aoi_resolution_and_name_helpers() -> None:
    missing = pd.DataFrame(
        {
            "foo": [
                1,
            ]
        }
    )

    assert r4._resolve_aoi_columns(missing) is None

    assert r4._make_name("1 bad name") == "X1.bad.name"

    master = pd.DataFrame(
        {
            "FPOGX": [
                0.2,
            ],
            "FPOGY": [
                0.2,
            ],
        }
    )

    empty = pd.DataFrame(
        {
            "AOI": pd.Series(dtype=str),
            "x_min": pd.Series(dtype=float),
            "x_max": pd.Series(dtype=float),
            "y_min": pd.Series(dtype=float),
            "y_max": pd.Series(dtype=float),
        }
    )

    with pytest.raises(
        ValueError,
        match="at least one AOI",
    ):
        r4._static_aoi(
            master,
            empty,
            x_col="FPOGX",
            y_col="FPOGY",
            aoi_name=None,
            output="logical",
            prefix="aoi_",
            label_col="aoi_current",
            outside_label="outside",
            overlap="first",
            include_overlap_count=True,
        )

    invalid = pd.DataFrame(
        {
            "AOI": [
                "a",
            ],
            "x_min": [
                np.nan,
            ],
            "x_max": [
                0.5,
            ],
            "y_min": [
                0.1,
            ],
            "y_max": [
                0.5,
            ],
        }
    )

    with pytest.raises(
        ValueError,
        match="finite",
    ):
        r4._static_aoi(
            master,
            invalid,
            x_col="FPOGX",
            y_col="FPOGY",
            aoi_name=None,
            output="logical",
            prefix="aoi_",
            label_col="aoi_current",
            outside_label="outside",
            overlap="first",
            include_overlap_count=True,
        )


def test_r4_geometry_alias_and_resolver_helpers() -> None:
    source = pd.DataFrame(
        {
            "AOI": [
                "a",
            ],
            "MEDIA_ID": [
                "M1",
            ],
        }
    )

    aliased = r4._geometry_aliases(source)

    assert "aoi" in aliased
    assert "media_id" in aliased

    assert (
        r4._resolve_column(
            None,
            aliased,
            ("AOI",),
            required=True,
        )
        == "aoi"
    )

    assert (
        r4._resolve_column(
            None,
            aliased,
            ("not_here",),
            required=False,
        )
        is None
    )

    with pytest.raises(
        ValueError,
        match="Required AOI",
    ):
        r4._resolve_column(
            None,
            aliased,
            ("not_here",),
            required=True,
        )

    with pytest.raises(
        ValueError,
        match="Column not found",
    ):
        r4._resolve_column(
            "not_here",
            aliased,
            (),
            required=True,
        )


def _geometry_call(
    data: pd.DataFrame,
    **kwargs,
):
    defaults = {
        "aoi_col": None,
        "stimulus_col": None,
        "x_min_col": None,
        "y_min_col": None,
        "x_max_col": None,
        "y_max_col": None,
        "x_col": None,
        "y_col": None,
        "width_col": None,
        "height_col": None,
        "screen_x_range": (
            0,
            1,
        ),
        "screen_y_range": (
            0,
            1,
        ),
        "min_width": 0,
        "min_height": 0,
        "min_area": 0,
        "max_area_prop": 1,
        "require_within_screen": True,
    }

    defaults.update(kwargs)

    return r4._geometry_audit(
        data,
        **defaults,
    )


def test_r4_geometry_status_branches() -> None:
    invalid_coordinate = pd.DataFrame(
        {
            "AOI": [
                "bad",
            ],
            "x_min": [
                np.nan,
            ],
            "x_max": [
                0.5,
            ],
            "y_min": [
                0.1,
            ],
            "y_max": [
                0.5,
            ],
        }
    )

    out = _geometry_call(invalid_coordinate)

    assert (
        out["geometry_summary"].loc[
            0,
            "aoi_geometry_status",
        ]
        == "invalid_coordinate"
    )

    zero_width = pd.DataFrame(
        {
            "AOI": [
                "zero",
            ],
            "x_min": [
                0.5,
            ],
            "x_max": [
                0.5,
            ],
            "y_min": [
                0.1,
            ],
            "y_max": [
                0.5,
            ],
        }
    )

    out_zero = _geometry_call(zero_width)

    assert (
        out_zero["geometry_summary"].loc[
            0,
            "aoi_geometry_status",
        ]
        == "invalid_dimension"
    )

    outside = pd.DataFrame(
        {
            "AOI": [
                "outside",
            ],
            "x_min": [
                -0.2,
            ],
            "x_max": [
                0.2,
            ],
            "y_min": [
                0.1,
            ],
            "y_max": [
                0.5,
            ],
        }
    )

    out_outside = _geometry_call(outside)

    assert (
        out_outside["geometry_summary"].loc[
            0,
            "aoi_geometry_status",
        ]
        == "outside_screen"
    )

    out_allowed = _geometry_call(
        outside,
        require_within_screen=False,
    )

    assert (
        out_allowed["geometry_summary"].loc[
            0,
            "aoi_geometry_status",
        ]
        == "ok"
    )


def test_r4_geometry_small_large_and_missing_geometry_errors() -> None:
    small = pd.DataFrame(
        {
            "AOI": [
                "small",
            ],
            "x_min": [
                0.1,
            ],
            "x_max": [
                0.15,
            ],
            "y_min": [
                0.1,
            ],
            "y_max": [
                0.15,
            ],
        }
    )

    out_small = _geometry_call(
        small,
        min_width=0.1,
    )

    assert (
        out_small["geometry_summary"].loc[
            0,
            "aoi_geometry_status",
        ]
        == "too_small"
    )

    large = pd.DataFrame(
        {
            "AOI": [
                "large",
            ],
            "x_min": [
                0.0,
            ],
            "x_max": [
                0.9,
            ],
            "y_min": [
                0.0,
            ],
            "y_max": [
                0.9,
            ],
        }
    )

    out_large = _geometry_call(
        large,
        max_area_prop=0.5,
    )

    assert (
        out_large["geometry_summary"].loc[
            0,
            "aoi_geometry_status",
        ]
        == "too_large"
    )

    no_geometry = pd.DataFrame(
        {
            "AOI": [
                "a",
            ]
        }
    )

    with pytest.raises(
        ValueError,
        match="requires either",
    ):
        _geometry_call(no_geometry)

    with pytest.raises(
        ValueError,
        match="at least one row",
    ):
        _geometry_call(
            pd.DataFrame(
                {
                    "AOI": pd.Series(dtype=str),
                    "x_min": pd.Series(dtype=float),
                    "x_max": pd.Series(dtype=float),
                    "y_min": pd.Series(dtype=float),
                    "y_max": pd.Series(dtype=float),
                }
            )
        )


def test_r4_detector_normalizer_branches() -> None:
    assert (
        r4._detector_result(
            "not-a-mapping",
            result_class="demo|list",
            events_class="events|data.frame",
        )
        == "not-a-mapping"
    )

    result = {
        "events": pd.DataFrame(
            {
                "x": [
                    1,
                ]
            }
        ),
        "samples": pd.DataFrame(
            {
                "y": [
                    2,
                ]
            }
        ),
        "_gp3_class": "legacy",
    }

    out = r4._detector_result(
        result,
        result_class="demo|list",
        events_class=("demo_events|tbl_df|tbl|data.frame"),
    )

    assert list(out) == [
        "events",
        "samples",
    ]

    assert out.r_class == "demo|list"

    assert out["events"].attrs["r_class"] == "demo_events|tbl_df|tbl|data.frame"


def test_r4_coding_matrix_normalizer_branches() -> None:
    sentinel = "x"

    assert r4._coding_matrix_result(sentinel) == sentinel

    partial = {"overview": pd.DataFrame()}

    assert r4._coding_matrix_result(partial) is partial

    names = [
        "overview",
        "geometry_summary",
        "sample_coding",
        "coding_matrix",
        "observed_summary",
        "derived_summary",
        "flagged_samples",
        "settings",
    ]

    bundle = {name: pd.DataFrame() for name in names}

    bundle["coding_matrix"] = pd.DataFrame(
        {
            "observed": [
                "a",
                "b",
            ],
            "derived": [
                "a",
                "b",
            ],
            "n_samples": [
                2,
                1,
            ],
            "sample_prop": [
                0.0,
                0.0,
            ],
        }
    )

    out = r4._coding_matrix_result(bundle)

    assert out.r_class == ("gp3_aoi_coding_matrix_audit|list")

    assert np.allclose(
        out["coding_matrix"]["sample_prop"],
        [
            2.0 / 3.0,
            1.0 / 3.0,
        ],
    )

    for name in names:
        assert out[name].attrs["r_class"] == "tbl_df|tbl|data.frame"


def test_r4_legacy_detector_bridge_canonical_vs_legacy() -> None:
    def fake(
        data=None,
        *,
        all_gaze=None,
        return_mode=None,
        **kwargs,
    ):
        return {
            "events": pd.DataFrame(),
            "samples": pd.DataFrame(),
        }

    wrapped = r4.legacy_detector_result_bridge(
        fake,
        result_class="demo_result",
    )

    legacy = wrapped(
        data=pd.DataFrame(),
        return_mode="both",
    )

    assert legacy["_gp3_class"] == "demo_result"

    canonical = wrapped(
        all_gaze=pd.DataFrame(),
        return_mode="both",
    )

    assert "_gp3_class" not in canonical

    events_only = wrapped(
        data=pd.DataFrame(),
        return_mode="events",
    )

    assert "_gp3_class" not in events_only


def test_r4_geometry_validation_bridge_all_scalar_checks() -> None:
    def fake(
        *args,
        **kwargs,
    ):
        return "ok"

    wrapped = r4.geometry_validation_bridge(fake)

    assert (
        wrapped(
            max_area_prop=0.5,
            min_width=0,
            min_height=0,
            min_area=0,
            screen_x_range=(
                0,
                1,
            ),
            screen_y_range=(
                0,
                1,
            ),
            require_within_screen=True,
        )
        == "ok"
    )

    for name in (
        "min_width",
        "min_height",
        "min_area",
    ):
        with pytest.raises(
            ValueError,
            match=name,
        ):
            wrapped(
                **{
                    name: -1,
                }
            )

        with pytest.raises(
            ValueError,
            match=name,
        ):
            wrapped(
                **{
                    name: "bad",
                }
            )

    for bad in (
        -0.1,
        1.1,
        np.inf,
        "bad",
    ):
        with pytest.raises(
            ValueError,
            match="max_area_prop",
        ):
            wrapped(max_area_prop=bad)

    with pytest.raises(
        ValueError,
        match="screen_x_range",
    ):
        wrapped(
            screen_x_range=(
                1,
                0,
            )
        )

    with pytest.raises(
        ValueError,
        match="screen_y_range",
    ):
        wrapped(
            screen_y_range=(
                0,
                np.nan,
            )
        )

    with pytest.raises(
        ValueError,
        match="require_within_screen",
    ):
        wrapped(require_within_screen=1)
