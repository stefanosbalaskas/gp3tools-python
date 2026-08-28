"""R4 canonical/legacy dual-contract dispatch.

The R4 canonical layer intentionally returns R-like structures.  Historical
Python aliases remain supported.  Selection is based on semantic input shape,
not merely whether an R spelling was used.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

import pandas as pd

_R4_REQUIRED_WORKFLOW = {
    "all_gaze",
    "all_fix",
    "sampling",
    "quality",
    "flagged_quality",
    "aoi_table",
}

_R4_QC_BUNDLE = {
    "sampling",
    "quality",
    "flagged_quality",
    "aoi_table",
}

_R4_QC_BUNDLE_ALT = {
    "tracking_quality",
    "sampling_rate",
    "flags",
    "aoi",
}


def _legacy_callable(fn, name: str):
    """Recover the pre-R4 callable captured by the canonical wrapper."""
    seen: set[int] = set()
    queue = [getattr(fn, "__wrapped__", None)]
    closure = getattr(fn, "__closure__", None) or ()
    for cell in closure:
        try:
            queue.append(cell.cell_contents)
        except ValueError:
            pass

    fallback = None
    while queue:
        candidate = queue.pop(0)
        if not callable(candidate) or id(candidate) in seen or candidate is fn:
            continue
        seen.add(id(candidate))
        if fallback is None:
            fallback = candidate
        module = str(getattr(candidate, "__module__", ""))
        candidate_name = str(getattr(candidate, "__name__", ""))
        if candidate_name == name and not module.endswith("._behavioral_r4"):
            return candidate
        wrapped = getattr(candidate, "__wrapped__", None)
        if wrapped is not None:
            queue.append(wrapped)
        for cell in getattr(candidate, "__closure__", None) or ():
            try:
                queue.append(cell.cell_contents)
            except ValueError:
                pass

    if fallback is not None:
        return fallback
    raise RuntimeError(f"Could not recover pre-R4 callable for {name}")


def _argument(args: tuple[Any, ...], kwargs: dict[str, Any], key: str, position: int = 0):
    if key in kwargs:
        return kwargs[key]
    return args[position] if len(args) > position else None


def _aoi_geometry(args: tuple[Any, ...], kwargs: dict[str, Any]):
    for key in ("aoi_defs", "aoi_geometry"):
        value = kwargs.get(key)
        if isinstance(value, pd.DataFrame):
            return value
    if len(args) > 1 and isinstance(args[1], pd.DataFrame):
        return args[1]
    return None


def _r4_static_aoi(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    # Explicit R-only behavior controls are unambiguous.
    if any(
        key in kwargs
        for key in (
            "aoi_name",
            "output",
            "prefix",
            "label_col",
            "overlap",
            "include_overlap_count",
        )
    ):
        return True
    geometry = _aoi_geometry(args, kwargs)
    if isinstance(geometry, pd.DataFrame):
        columns = set(map(str, geometry.columns))
        # Canonical frozen R4 fixtures use AOI / x_min-y_min style geometry.
        # Legacy compatibility fixtures use lower-case aoi + xmin/xmax/ymin/ymax.
        if "AOI" in columns or "AOI_ID" in columns or "NAME" in columns:
            return True
        if {"aoi", "xmin", "xmax", "ymin", "ymax"}.issubset(columns):
            return False
    return False


def _r4_geometry_audit(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    explicit = {
        "aoi_col",
        "stimulus_col",
        "x_min_col",
        "y_min_col",
        "x_max_col",
        "y_max_col",
        "x_col",
        "y_col",
        "width_col",
        "height_col",
        "screen_x_range",
        "screen_y_range",
        "min_width",
        "min_height",
        "min_area",
        "max_area_prop",
        "require_within_screen",
    }
    if explicit.intersection(kwargs):
        return True
    geometry = kwargs.get("data")
    if geometry is None:
        geometry = args[0] if args else None
    if isinstance(geometry, pd.DataFrame):
        columns = set(map(str, geometry.columns))
        if "AOI" in columns or "AOI_ID" in columns or "NAME" in columns:
            return True
        if {"aoi", "xmin", "xmax", "ymin", "ymax"}.issubset(columns):
            return False
    return False


def _r4_qc(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    bundle = _argument(args, kwargs, "qc_bundle")
    if isinstance(bundle, pd.DataFrame):
        return {"object_name", "qc_status"}.issubset(bundle.columns)
    if isinstance(bundle, dict):
        keys = set(bundle)
        return _R4_QC_BUNDLE.issubset(keys) or _R4_QC_BUNDLE_ALT.issubset(keys)
    return False


def _r4_workflow(args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    results = _argument(args, kwargs, "results")
    return isinstance(results, dict) and _R4_REQUIRED_WORKFLOW.issubset(results)


def _use_r4(name: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
    if name == "add_gazepoint_aoi":
        return _r4_static_aoi(args, kwargs)
    if name == "audit_gazepoint_aoi_geometry":
        return _r4_geometry_audit(args, kwargs)
    if name in {"summarise_gazepoint_qc_status", "summarize_gazepoint_qc_status"}:
        return _r4_qc(args, kwargs)
    if name == "summarise_gazepoint_workflow":
        return _r4_workflow(args, kwargs)
    return True


def r4_dual_contract(r4_callable, *, name: str):
    """Wrap an R4 callable with semantic legacy fallback."""
    if getattr(r4_callable, "_gp3_r4_dual_contract", False):
        return r4_callable
    legacy = _legacy_callable(r4_callable, name)

    @wraps(r4_callable)
    def dispatched(*args, **kwargs):
        if _use_r4(name, args, kwargs):
            return r4_callable(*args, **kwargs)
        return legacy(*args, **kwargs)

    dispatched._gp3_r4_dual_contract = True
    dispatched._gp3_r4_legacy = legacy
    dispatched._gp3_r4_canonical = r4_callable
    return dispatched
