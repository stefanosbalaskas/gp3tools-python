"""Optional bridge to the R gp3tools package for backend-specific compatibility."""

from __future__ import annotations

from typing import Any


class BackendUnavailableError(RuntimeError):
    """Raised when a compatibility function needs an unavailable optional backend."""


def call_r_function(name: str, *args, **kwargs) -> Any:
    try:
        from rpy2.robjects import conversion, default_converter, pandas2ri
        from rpy2.robjects.packages import importr
    except Exception as exc:
        raise BackendUnavailableError(
            f"`{name}()` is exposed for R gp3tools v2.3.0 API compatibility but does not yet "
            "have a validated native Python implementation in this alpha build. Install the "
            "optional `rbridge` extra, R, and the R package gp3tools to delegate this call. "
            f"Original import error: {exc}"
        ) from exc
    try:
        pkg = importr("gp3tools")
    except Exception as exc:
        raise BackendUnavailableError(
            f"R package `gp3tools` is required to bridge `{name}()`, but it could not be loaded: {exc}"
        ) from exc
    fn = getattr(pkg, name, None)
    if fn is None:
        raise BackendUnavailableError(
            f"R gp3tools does not expose `{name}` in the installed version."
        )
    converter = default_converter + pandas2ri.converter
    with conversion.localconverter(converter):
        r_args = [conversion.py2rpy(x) for x in args]
        r_kwargs = {k.replace("_", "."): conversion.py2rpy(v) for k, v in kwargs.items()}
        result = fn(*r_args, **r_kwargs)
        try:
            return conversion.rpy2py(result)
        except Exception:
            return result
