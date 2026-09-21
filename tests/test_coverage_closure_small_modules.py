from __future__ import annotations

import builtins
import contextlib
import inspect
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import gp3tools as gp3
import gp3tools._compat as compat
import gp3tools._r4_dual_contract as dual
import gp3tools._rbridge as rbridge
import gp3tools._utils as utils
import gp3tools.interop as interop
import gp3tools.misc as misc
from gp3tools.datasets import load_example_data

# ---------------------------------------------------------------------------
# datasets.py
# ---------------------------------------------------------------------------


def test_dataset_loader_direct_csv_and_unknown_name():
    master = load_example_data("gazepoint_example_master.csv")

    assert master.shape == (
        1440,
        11,
    )

    with pytest.raises(
        KeyError,
        match="Unknown example dataset",
    ):
        load_example_data("definitely_not_a_dataset")


# ---------------------------------------------------------------------------
# _utils.py
# ---------------------------------------------------------------------------


def test_ensure_dataframe_all_supported_inputs_and_copy_contract():
    frame = pd.DataFrame(
        {
            "x": [
                1,
                2,
            ]
        }
    )

    copied = utils.ensure_dataframe(frame)

    same = utils.ensure_dataframe(
        frame,
        copy=False,
    )

    assert copied.equals(frame)
    assert copied is not frame
    assert same is frame

    series = pd.Series(
        [
            1,
            2,
        ],
        name="x",
    )

    series_copy = utils.ensure_dataframe(series)

    series_view = utils.ensure_dataframe(
        series,
        copy=False,
    )

    assert list(series_copy.columns) == ["x"]

    assert list(series_view.columns) == ["x"]

    assert utils.ensure_dataframe({"x": [1]}).shape == (
        1,
        1,
    )

    assert utils.ensure_dataframe([{"x": 1}]).shape == (
        1,
        1,
    )

    assert utils.ensure_dataframe(({"x": 1},)).shape == (
        1,
        1,
    )

    with pytest.raises(
        TypeError,
        match="Expected a pandas DataFrame",
    ):
        utils.ensure_dataframe(object())


def test_column_inference_all_contract_edges():
    frame = pd.DataFrame(
        {
            "USER": ["S1"],
            "TIME": [0.0],
            "custom": [1],
        }
    )

    assert (
        utils.infer_column(
            frame,
            "subject",
        )
        == "USER"
    )

    assert (
        utils.infer_column(
            frame,
            "time",
            explicit="TIME",
        )
        == "TIME"
    )

    assert (
        utils.infer_column(
            frame,
            "time",
            explicit="missing",
        )
        is None
    )

    with pytest.raises(
        KeyError,
        match="was not found",
    ):
        utils.infer_column(
            frame,
            "time",
            explicit="missing",
            required=True,
        )

    assert (
        utils.infer_column(
            frame,
            "custom",
        )
        == "custom"
    )

    assert (
        utils.infer_column(
            frame,
            "missing_role",
        )
        is None
    )

    with pytest.raises(
        KeyError,
        match="Could not infer",
    ):
        utils.infer_column(
            frame,
            "missing_role",
            required=True,
        )

    inferred = utils.infer_columns(
        frame,
        [
            "subject",
            "time",
            "missing_role",
        ],
    )

    assert inferred == {
        "subject": "USER",
        "time": "TIME",
        "missing_role": None,
    }


def test_group_column_normalization_contract():
    frame = pd.DataFrame(
        {
            "subject": ["S1"],
            "trial": [1],
        }
    )

    assert (
        utils.normalize_group_cols(
            frame,
            None,
        )
        == []
    )

    assert utils.normalize_group_cols(
        frame,
        "subject",
    ) == ["subject"]

    assert utils.normalize_group_cols(
        frame,
        [
            "subject",
            "trial",
        ],
    ) == [
        "subject",
        "trial",
    ]

    with pytest.raises(
        KeyError,
        match="Grouping columns not found",
    ):
        utils.normalize_group_cols(
            frame,
            ["missing"],
        )


def test_numeric_and_boolean_helpers_cover_all_types():
    numeric = utils.finite_numeric(
        pd.Series(
            [
                "1",
                "bad",
                np.inf,
                -np.inf,
                2,
            ]
        )
    )

    assert numeric.iloc[0] == 1
    assert pd.isna(numeric.iloc[1])
    assert pd.isna(numeric.iloc[2])
    assert pd.isna(numeric.iloc[3])
    assert numeric.iloc[4] == 2

    boolean = utils.as_bool(
        pd.Series(
            [
                True,
                False,
                None,
            ],
            dtype="boolean",
        )
    )

    assert boolean.tolist() == [
        True,
        False,
        False,
    ]

    numeric_bool = utils.as_bool(
        pd.Series(
            [
                0,
                1,
                np.nan,
                -1,
            ]
        )
    )

    assert numeric_bool.tolist() == [
        False,
        True,
        False,
        True,
    ]

    string_bool = utils.as_bool(
        pd.Series(
            [
                " TRUE ",
                "yes",
                "valid",
                "bad",
                None,
            ]
        )
    )

    assert string_bool.tolist() == [
        True,
        True,
        True,
        False,
        False,
    ]

    inverted = utils.as_bool(
        pd.Series(
            [
                0,
                1,
            ]
        ),
        invert_trackloss=True,
    )

    assert inverted.tolist() == [
        True,
        False,
    ]


def test_robust_mad_empty_and_finite_values():
    assert np.isnan(
        utils.robust_mad(
            [
                np.nan,
                np.inf,
            ]
        )
    )

    assert utils.robust_mad(
        [
            1,
            2,
            3,
        ]
    ) == pytest.approx(1.0)


def test_group_iter_ungrouped_single_and_multi_key():
    frame = pd.DataFrame(
        {
            "subject": [
                "S1",
                "S2",
            ],
            "trial": [
                1,
                2,
            ],
            "x": [
                1,
                2,
            ],
        }
    )

    ungrouped = list(
        utils.group_iter(
            frame,
            [],
        )
    )

    assert len(ungrouped) == 1

    assert ungrouped[0][0] == ()

    one = list(
        utils.group_iter(
            frame,
            ["subject"],
        )
    )

    assert all(
        isinstance(
            key,
            tuple,
        )
        for key, _ in one
    )

    two = list(
        utils.group_iter(
            frame,
            [
                "subject",
                "trial",
            ],
        )
    )

    assert all(len(key) == 2 for key, _ in two)


def test_utility_output_helpers(tmp_path):
    frame = pd.DataFrame({"x": [1]})

    out = utils.attach_attrs(
        frame,
        method="test",
        version=1,
    )

    assert out is frame
    assert out.attrs["method"] == "test"
    assert out.attrs["version"] == 1

    resolved = utils.safe_path(tmp_path / ".." / tmp_path.name)

    assert resolved == tmp_path.resolve()

    result = utils.result_table(
        a=1,
        b="x",
    )

    assert result.to_dict("records") == [
        {
            "a": 1,
            "b": "x",
        }
    ]


def test_ordered_unique_hashable_and_unhashable_values():
    values = [
        "a",
        "a",
        1,
        1,
        [
            1,
            2,
        ],
        [
            1,
            2,
        ],
        {"x": 1},
        {"x": 1},
    ]

    result = utils.ordered_unique(values)

    assert result[0:2] == [
        "a",
        1,
    ]

    assert result[2] == [
        1,
        2,
    ]

    assert result[3] == {"x": 1}

    assert len(result) == 4


def test_collapse_consecutive_contract():
    assert utils.collapse_consecutive([]) == []

    assert utils.collapse_consecutive(
        [
            "A",
            "A",
            "B",
            "B",
            "A",
        ]
    ) == [
        "A",
        "B",
        "A",
    ]


def test_time_to_seconds_all_heuristic_paths():
    empty = utils.time_to_seconds(
        pd.Series(
            [
                np.nan,
                np.inf,
            ]
        )
    )

    assert empty.isna().iloc[0]
    assert np.isinf(empty.iloc[1])

    seconds = utils.time_to_seconds(
        pd.Series(
            [
                0.0,
                0.016,
                0.032,
            ]
        )
    )

    np.testing.assert_allclose(
        seconds,
        [
            0.0,
            0.016,
            0.032,
        ],
    )

    milliseconds = utils.time_to_seconds(
        pd.Series(
            [
                0.0,
                16.0,
                32.0,
            ]
        )
    )

    np.testing.assert_allclose(
        milliseconds,
        [
            0.0,
            0.016,
            0.032,
        ],
    )

    repeated = utils.time_to_seconds(
        pd.Series(
            [
                5.0,
                5.0,
                5.0,
            ]
        )
    )

    np.testing.assert_allclose(
        repeated,
        [
            5.0,
            5.0,
            5.0,
        ],
    )


def test_require_optional_success_and_failure():
    module = utils.require_optional(
        "math",
        "testing",
    )

    assert module.sqrt(9) == 3

    with pytest.raises(
        ImportError,
        match="requires optional dependency",
    ):
        utils.require_optional(
            "gp3tools_definitely_missing_dependency",
            "Coverage test",
        )


# ---------------------------------------------------------------------------
# _compat.py
# ---------------------------------------------------------------------------


def test_r_bridge_wrapper_metadata_and_dispatch(monkeypatch):
    calls = []

    def fake_call(
        name,
        *args,
        **kwargs,
    ):
        calls.append(
            (
                name,
                args,
                kwargs,
            )
        )

        return "ok"

    monkeypatch.setattr(
        compat,
        "call_r_function",
        fake_call,
    )

    wrapper = compat.make_r_bridge_wrapper("example_r_function")

    assert wrapper.__name__ == "example_r_function"

    assert wrapper.__qualname__ == "example_r_function"

    assert wrapper._gp3tools_status == "r-bridge"

    assert "Compatibility wrapper" in wrapper.__doc__

    assert (
        wrapper(
            1,
            value=2,
        )
        == "ok"
    )

    assert calls == [
        (
            "example_r_function",
            (1,),
            {
                "value": 2,
            },
        )
    ]


def test_r_aliases_alias_absent_present_conflict_and_signature():
    def original(
        data=None,
        value=1,
        **kwargs,
    ):
        return (
            data,
            value,
            kwargs,
        )

    wrapped = compat.r_aliases(
        original,
        x="data",
        r_value="value",
    )

    assert (
        wrapped(
            data="native",
        )[0]
        == "native"
    )

    result = wrapped(
        x="r-alias",
        r_value=3,
        other=4,
    )

    assert result == (
        "r-alias",
        3,
        {
            "other": 4,
        },
    )

    parameters = inspect.signature(wrapped).parameters

    assert "x" in parameters
    assert "r_value" in parameters

    assert wrapped.__r_aliases__ == {
        "x": "data",
        "r_value": "value",
    }

    with pytest.raises(
        TypeError,
        match="received both",
    ):
        wrapped(
            "positional",
            x="alias",
        )

    with pytest.raises(
        TypeError,
        match="received both",
    ):
        wrapped(
            data="native",
            x="alias",
        )


def test_r_alias_existing_parameter_not_duplicated():
    def original(
        value=None,
    ):
        return value

    wrapped = compat.r_aliases(
        original,
        value="value",
    )

    signature = inspect.signature(wrapped)

    assert list(signature.parameters).count("value") == 1


# ---------------------------------------------------------------------------
# _rbridge.py
# ---------------------------------------------------------------------------


class _FakeConverter:
    def __add__(
        self,
        other,
    ):
        return self


class _FakeConversion:
    def __init__(
        self,
        *,
        fail_back_conversion=False,
    ):
        self.fail_back_conversion = fail_back_conversion

    @contextlib.contextmanager
    def localconverter(
        self,
        converter,
    ):
        yield

    def py2rpy(
        self,
        value,
    ):
        return (
            "R",
            value,
        )

    def rpy2py(
        self,
        value,
    ):
        if self.fail_back_conversion:
            raise RuntimeError("deliberate conversion failure")

        return (
            "PY",
            value,
        )


def _install_fake_rpy2(
    monkeypatch,
    *,
    package,
    fail_back_conversion=False,
    importr_error=None,
):
    top = types.ModuleType("rpy2")

    top.__path__ = []

    robjects = types.ModuleType("rpy2.robjects")

    packages = types.ModuleType("rpy2.robjects.packages")

    conversion = _FakeConversion(fail_back_conversion=(fail_back_conversion))

    robjects.conversion = conversion
    robjects.default_converter = _FakeConverter()

    robjects.pandas2ri = types.SimpleNamespace(converter=_FakeConverter())

    if importr_error is None:
        packages.importr = lambda name: package
    else:

        def fail_importr(
            name,
        ):
            raise importr_error

        packages.importr = fail_importr

    monkeypatch.setitem(
        sys.modules,
        "rpy2",
        top,
    )

    monkeypatch.setitem(
        sys.modules,
        "rpy2.robjects",
        robjects,
    )

    monkeypatch.setitem(
        sys.modules,
        "rpy2.robjects.packages",
        packages,
    )

    return conversion


def test_rbridge_missing_rpy2_is_explicit(monkeypatch):
    real_import = builtins.__import__

    def blocked_import(
        name,
        globals=None,
        locals=None,
        fromlist=(),
        level=0,
    ):
        if name.startswith("rpy2"):
            raise ImportError("rpy2 unavailable")

        return real_import(
            name,
            globals,
            locals,
            fromlist,
            level,
        )

    monkeypatch.setattr(
        builtins,
        "__import__",
        blocked_import,
    )

    with pytest.raises(
        rbridge.BackendUnavailableError,
        match="optional `rbridge` extra",
    ):
        rbridge.call_r_function(
            "example_function",
        )


def test_rbridge_missing_r_package_is_explicit(monkeypatch):
    _install_fake_rpy2(
        monkeypatch,
        package=None,
        importr_error=RuntimeError("R package unavailable"),
    )

    with pytest.raises(
        rbridge.BackendUnavailableError,
        match="could not be loaded",
    ):
        rbridge.call_r_function(
            "example_function",
        )


def test_rbridge_missing_export_is_explicit(monkeypatch):
    package = types.SimpleNamespace()

    _install_fake_rpy2(
        monkeypatch,
        package=package,
    )

    with pytest.raises(
        rbridge.BackendUnavailableError,
        match="does not expose",
    ):
        rbridge.call_r_function(
            "not_exported",
        )


def test_rbridge_success_converts_arguments_and_keyword_names(monkeypatch):
    captured = {}

    def bridged(
        *args,
        **kwargs,
    ):
        captured["args"] = args

        captured["kwargs"] = kwargs

        return "r-result"

    package = types.SimpleNamespace(target=bridged)

    _install_fake_rpy2(
        monkeypatch,
        package=package,
    )

    result = rbridge.call_r_function(
        "target",
        1,
        sample_value=2,
    )

    assert captured["args"] == (
        (
            "R",
            1,
        ),
    )

    assert captured["kwargs"] == {
        "sample.value": (
            "R",
            2,
        )
    }

    assert result == (
        "PY",
        "r-result",
    )


def test_rbridge_failed_back_conversion_returns_raw_result(monkeypatch):
    raw = object()

    package = types.SimpleNamespace(target=lambda: raw)

    _install_fake_rpy2(
        monkeypatch,
        package=package,
        fail_back_conversion=True,
    )

    assert rbridge.call_r_function("target") is raw


# ---------------------------------------------------------------------------
# interop.py
# ---------------------------------------------------------------------------


def test_hddm_script_without_path_has_no_write_side_effect():
    script = interop.create_gazepoint_hddm_fit_script()

    assert "import hddm" in script
    assert "model.sample" in script


def test_bids_export_without_subject_column_uses_default_subject(tmp_path):
    frame = pd.DataFrame(
        {
            "time": [
                0.0,
                0.1,
            ],
            "x": [
                0.1,
                0.2,
            ],
        }
    )

    result = interop.export_gazepoint_to_bids(
        frame,
        tmp_path,
        task="coverage",
    )

    assert len(result["files"]) == 1

    exported = Path(result["files"][0])

    assert "sub-01" in str(exported)

    assert exported.exists()

    assert (tmp_path / "dataset_description.json").exists()


# ---------------------------------------------------------------------------
# simulation.py
# ---------------------------------------------------------------------------


def test_cluster_simulation_validation_edges():
    with pytest.raises(
        ValueError,
        match="at least 2",
    ):
        gp3.simulate_gazepoint_cluster_timecourse_data(
            n_subjects=1,
            n_time_bins=10,
        )

    with pytest.raises(
        ValueError,
        match="character vector of length two",
    ):
        gp3.simulate_gazepoint_cluster_timecourse_data(
            n_subjects=3,
            n_time_bins=10,
            conditions=[
                "only-one",
            ],
        )

    with pytest.raises(
        ValueError,
        match="character vector of length two",
    ):
        gp3.simulate_gazepoint_cluster_timecourse_data(
            n_subjects=3,
            n_time_bins=10,
            conditions=[
                "control",
                "",
            ],
        )


# ---------------------------------------------------------------------------
# misc.py
# ---------------------------------------------------------------------------


def test_recalibration_legacy_unknown_keyword_and_invalid_method():
    frame = pd.DataFrame(
        {
            "x": [
                0.1,
                0.2,
            ],
            "y": [
                0.3,
                0.4,
            ],
        }
    )

    with pytest.raises(
        TypeError,
        match="Unexpected keyword",
    ):
        misc.recalibrate_gazepoint_gaze(
            frame,
            unexpected=True,
        )

    with pytest.raises(
        ValueError,
        match="method must be",
    ):
        misc.recalibrate_gazepoint_gaze(
            frame,
            x_col="x",
            y_col="y",
            method="not-a-method",
        )


def test_recalibration_r_route_unknown_keyword_and_offset_mapping(
    monkeypatch,
):
    frame = pd.DataFrame(
        {
            "x": [
                0.1,
                0.2,
            ],
            "y": [
                0.3,
                0.4,
            ],
            "tx": [
                0.2,
                0.3,
            ],
            "ty": [
                0.4,
                0.5,
            ],
        }
    )

    with pytest.raises(
        TypeError,
        match="Unexpected keyword",
    ):
        misc.recalibrate_gazepoint_gaze(
            frame,
            x_col="x",
            y_col="y",
            target_x_col="tx",
            target_y_col="ty",
            unexpected=True,
        )

    import gp3tools._behavioral_r2 as r2

    captured = {}

    def fake_recalibrate(
        data,
        **kwargs,
    ):
        captured.update(kwargs)

        return data.copy()

    monkeypatch.setattr(
        r2,
        "recalibrate_gaze",
        fake_recalibrate,
    )

    out = misc.recalibrate_gazepoint_gaze(
        frame,
        x_col="x",
        y_col="y",
        target_x_col="tx",
        target_y_col="ty",
        method="offset",
    )

    assert len(out) == len(frame)

    assert captured["method"] == "median_shift"


def test_adaptive_trial_data_alias_and_no_score_column():
    frame = pd.DataFrame(
        {
            "candidate": [
                "A",
                "B",
            ]
        }
    )

    result = misc.select_gazepoint_adaptive_trial(
        data=frame,
    )

    assert len(result) == 1

    assert result.iloc[0]["candidate"] == "A"

    empty = misc.select_gazepoint_adaptive_trial(data=frame.iloc[0:0])

    assert empty.empty


# ---------------------------------------------------------------------------
# _r4_dual_contract.py
# ---------------------------------------------------------------------------


def test_r4_argument_and_geometry_helpers():
    assert (
        dual._argument(
            (),
            {"value": 1},
            "value",
        )
        == 1
    )

    assert (
        dual._argument(
            (2,),
            {},
            "value",
        )
        == 2
    )

    assert (
        dual._argument(
            (),
            {},
            "value",
        )
        is None
    )

    geometry = pd.DataFrame(
        {
            "AOI": ["target"],
            "x_min": [0],
        }
    )

    assert (
        dual._aoi_geometry(
            (),
            {"aoi_defs": geometry},
        )
        is geometry
    )

    assert (
        dual._aoi_geometry(
            (
                None,
                geometry,
            ),
            {},
        )
        is geometry
    )

    assert (
        dual._aoi_geometry(
            (),
            {},
        )
        is None
    )


def test_r4_static_aoi_dispatch_edges():
    assert dual._r4_static_aoi(
        (),
        {"aoi_name": "target"},
    )

    canonical = pd.DataFrame({"AOI": ["target"]})

    assert dual._r4_static_aoi(
        (
            None,
            canonical,
        ),
        {},
    )

    legacy = pd.DataFrame(
        {
            "aoi": ["target"],
            "xmin": [0],
            "xmax": [1],
            "ymin": [0],
            "ymax": [1],
        }
    )

    assert not dual._r4_static_aoi(
        (
            None,
            legacy,
        ),
        {},
    )

    assert not dual._r4_static_aoi(
        (),
        {},
    )


def test_r4_geometry_audit_dispatch_edges():
    assert dual._r4_geometry_audit(
        (),
        {"aoi_col": "AOI"},
    )

    canonical = pd.DataFrame({"NAME": ["target"]})

    assert dual._r4_geometry_audit(
        (),
        {"data": canonical},
    )

    legacy = pd.DataFrame(
        {
            "aoi": ["target"],
            "xmin": [0],
            "xmax": [1],
            "ymin": [0],
            "ymax": [1],
        }
    )

    assert not dual._r4_geometry_audit(
        (legacy,),
        {},
    )

    assert not dual._r4_geometry_audit(
        (),
        {},
    )


def test_r4_qc_and_workflow_dispatch_edges():
    qc_frame = pd.DataFrame(
        {
            "object_name": ["sampling"],
            "qc_status": ["ok"],
        }
    )

    assert dual._r4_qc(
        (qc_frame,),
        {},
    )

    assert not dual._r4_qc(
        (pd.DataFrame({"x": [1]}),),
        {},
    )

    canonical = {
        "sampling": 1,
        "quality": 2,
        "flagged_quality": 3,
        "aoi_table": 4,
    }

    assert dual._r4_qc(
        (canonical,),
        {},
    )

    alternative = {
        "tracking_quality": 1,
        "sampling_rate": 2,
        "flags": 3,
        "aoi": 4,
    }

    assert dual._r4_qc(
        (alternative,),
        {},
    )

    assert not dual._r4_qc(
        ({"x": 1},),
        {},
    )

    assert not dual._r4_qc(
        (object(),),
        {},
    )

    workflow = {
        "all_gaze": 1,
        "all_fix": 2,
        "sampling": 3,
        "quality": 4,
        "flagged_quality": 5,
        "aoi_table": 6,
    }

    assert dual._r4_workflow(
        (workflow,),
        {},
    )

    assert not dual._r4_workflow(
        ({"all_gaze": 1},),
        {},
    )


def test_use_r4_all_dispatch_names():
    assert dual._use_r4(
        "add_gazepoint_aoi",
        (),
        {"aoi_name": "target"},
    )

    assert dual._use_r4(
        "audit_gazepoint_aoi_geometry",
        (),
        {"aoi_col": "AOI"},
    )

    qc = {
        "sampling": 1,
        "quality": 2,
        "flagged_quality": 3,
        "aoi_table": 4,
    }

    assert dual._use_r4(
        "summarise_gazepoint_qc_status",
        (qc,),
        {},
    )

    assert dual._use_r4(
        "summarize_gazepoint_qc_status",
        (qc,),
        {},
    )

    workflow = {
        "all_gaze": 1,
        "all_fix": 2,
        "sampling": 3,
        "quality": 4,
        "flagged_quality": 5,
        "aoi_table": 6,
    }

    assert dual._use_r4(
        "summarise_gazepoint_workflow",
        (workflow,),
        {},
    )

    assert dual._use_r4(
        "unrelated_function",
        (),
        {},
    )


def test_legacy_callable_and_dual_contract_dispatch():
    def legacy(
        value,
    ):
        return (
            "legacy",
            value,
        )

    legacy.__name__ = "example"

    legacy.__module__ = "gp3tools.aoi"

    def canonical(
        value,
    ):
        return (
            "canonical",
            value,
        )

    canonical.__name__ = "example"

    canonical.__module__ = "gp3tools._behavioral_r4"

    canonical.__wrapped__ = legacy

    recovered = dual._legacy_callable(
        canonical,
        "example",
    )

    assert recovered is legacy

    wrapped = dual.r4_dual_contract(
        canonical,
        name="unrelated_function",
    )

    assert wrapped(1) == (
        "canonical",
        1,
    )

    assert (
        dual.r4_dual_contract(
            wrapped,
            name="unrelated_function",
        )
        is wrapped
    )


def test_legacy_callable_failure_is_explicit():
    def standalone():
        return None

    with pytest.raises(
        RuntimeError,
        match="Could not recover",
    ):
        dual._legacy_callable(
            standalone,
            "missing",
        )


def test_dual_contract_legacy_fallback():
    def legacy(
        value,
    ):
        return (
            "legacy",
            value,
        )

    legacy.__name__ = "add_gazepoint_aoi"

    legacy.__module__ = "gp3tools.aoi"

    def canonical(
        value,
    ):
        return (
            "canonical",
            value,
        )

    canonical.__name__ = "add_gazepoint_aoi"

    canonical.__module__ = "gp3tools._behavioral_r4"

    canonical.__wrapped__ = legacy

    wrapped = dual.r4_dual_contract(
        canonical,
        name="add_gazepoint_aoi",
    )

    legacy_geometry = pd.DataFrame(
        {
            "aoi": ["A"],
            "xmin": [0],
            "xmax": [1],
            "ymin": [0],
            "ymax": [1],
        }
    )

    result = wrapped(legacy_geometry)

    assert result == (
        "legacy",
        legacy_geometry,
    )


# ---------------------------------------------------------------------------
# Tranche 1 residual branch closure
# ---------------------------------------------------------------------------


def _make_empty_closure_function(name):
    value = object()

    def function():
        return value

    def clear_cell():
        nonlocal value
        del value

    clear_cell()

    function.__name__ = name

    return function


def test_group_iter_normalizes_scalar_group_key(monkeypatch):
    frame = pd.DataFrame(
        {
            "subject": [
                "S1",
            ],
            "x": [
                1,
            ],
        }
    )

    class FakeGrouped:
        def __iter__(self):
            return iter(
                [
                    (
                        "S1",
                        frame,
                    )
                ]
            )

    def fake_groupby(
        self,
        by,
        dropna=False,
        sort=False,
    ):
        assert by == ["subject"]

        assert dropna is False
        assert sort is False

        return FakeGrouped()

    monkeypatch.setattr(
        pd.DataFrame,
        "groupby",
        fake_groupby,
    )

    result = list(
        utils.group_iter(
            frame,
            ["subject"],
        )
    )

    assert len(result) == 1

    key, grouped = result[0]

    assert key == ("S1",)

    assert grouped is frame


def test_r4_legacy_callable_handles_empty_closure_on_input_function():
    canonical = _make_empty_closure_function("canonical")

    with pytest.raises(
        RuntimeError,
        match="Could not recover",
    ):
        dual._legacy_callable(
            canonical,
            "missing",
        )


def test_r4_legacy_callable_handles_empty_closure_on_candidate():
    candidate = _make_empty_closure_function("other_name")

    candidate.__module__ = "gp3tools._behavioral_r4"

    def canonical():
        return None

    canonical.__wrapped__ = candidate

    recovered = dual._legacy_callable(
        canonical,
        "wanted_name",
    )

    assert recovered is candidate


def test_r4_static_aoi_unknown_geometry_shape_falls_back():
    geometry = pd.DataFrame(
        {
            "unexpected": [
                1,
            ]
        }
    )

    assert not dual._r4_static_aoi(
        (
            None,
            geometry,
        ),
        {},
    )


def test_r4_geometry_audit_unknown_geometry_shape_falls_back():
    geometry = pd.DataFrame(
        {
            "unexpected": [
                1,
            ]
        }
    )

    assert not dual._r4_geometry_audit(
        (geometry,),
        {},
    )
