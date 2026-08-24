"""Compatibility wrappers for API names awaiting validated native Python backends."""

from __future__ import annotations

from ._rbridge import call_r_function


def make_r_bridge_wrapper(name: str):
    def wrapper(*args, **kwargs):
        return call_r_function(name, *args, **kwargs)

    wrapper.__name__ = name
    wrapper.__qualname__ = name
    wrapper.__doc__ = (
        f"Compatibility wrapper for R gp3tools `{name}()`. "
        "This API name is frozen in the Python alpha build; when no validated "
        "native backend is available, calls delegate through optional rpy2."
    )
    wrapper._gp3tools_status = "r-bridge"
    return wrapper
