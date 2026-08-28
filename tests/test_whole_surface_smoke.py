from __future__ import annotations

import inspect

import gp3tools as gp3


def test_frozen_public_namespace_contract() -> None:
    assert len(gp3.__all__) == 285
    assert len(set(gp3.__all__)) == 285
    assert len(gp3.R_EXPORTS) == 278
    assert len(set(gp3.R_EXPORTS)) == 278


def test_every_r_export_resolves_to_callable() -> None:
    missing = [name for name in gp3.R_EXPORTS if not callable(getattr(gp3, name, None))]
    assert missing == []


def test_every_r_export_has_introspectable_call_surface() -> None:
    failures = {}
    for name in gp3.R_EXPORTS:
        try:
            signature = inspect.signature(getattr(gp3, name))
        except (TypeError, ValueError) as exc:
            failures[name] = repr(exc)
            continue
        assert signature is not None
    assert failures == {}


def test_api_status_covers_exactly_frozen_exports_when_available() -> None:
    status_fn = getattr(gp3, "api_status", None)
    if status_fn is None:
        return
    status = status_fn()
    assert len(status) == 278
    assert set(status["r_export"]) == set(gp3.R_EXPORTS)
