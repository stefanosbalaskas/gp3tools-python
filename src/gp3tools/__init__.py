"""gp3tools for Python: reproducible Gazepoint GP3 analysis.

This alpha migration exposes the complete public function-name surface from the
R gp3tools v2.3.0 NAMESPACE. Core workflows have native Python implementations;
backend-specific functions without a validated native implementation use an
explicit optional R bridge rather than silently changing scientific semantics.
"""

from __future__ import annotations

import inspect

import pandas as pd

from ._compat import make_r_bridge_wrapper
from ._exports import R_EXPORTS
from ._rbridge import BackendUnavailableError
from .datasets import load_example_data, load_example_fixations, load_example_master

__version__ = "0.1.0a1"

_MODULE_NAMES = (
    "io",
    "qc",
    "pupil",
    "aoi",
    "events",
    "face",
    "simulation",
    "stats",
    "interop",
    "plotting",
    "reporting",
    "misc",
)

_NATIVE_SOURCE = {}
for _module_name in _MODULE_NAMES:
    _module = __import__(f"{__name__}.{_module_name}", fromlist=["*"])
    for _name in R_EXPORTS:
        if _name in globals():
            continue
        if hasattr(_module, _name):
            _obj = getattr(_module, _name)
            globals()[_name] = _obj
            _NATIVE_SOURCE[_name] = _module_name
            try:
                _obj._gp3tools_status = "native"
            except Exception:
                pass

for _name in R_EXPORTS:
    if _name not in globals():
        globals()[_name] = make_r_bridge_wrapper(_name)


def api_status() -> pd.DataFrame:
    """Return the frozen R-to-Python API implementation manifest."""
    rows = []
    for name in R_EXPORTS:
        obj = globals()[name]
        module = _NATIVE_SOURCE.get(name, "_compat")
        adapted = {
            "stats": "native-adapted",
            "interop": "native-adapter",
        }
        status = adapted.get(module, "native") if name in _NATIVE_SOURCE else "r-bridge"
        if name in {"classify_gazepoint_events_hmm", "launch_gazepoint_qc_dashboard"}:
            status = "native-adapted"
        try:
            signature = str(inspect.signature(obj))
        except Exception:
            signature = ""
        rows.append(
            {
                "r_export": name,
                "python_name": name,
                "status": status,
                "module": module,
                "signature": signature,
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    *R_EXPORTS,
    "R_EXPORTS",
    "BackendUnavailableError",
    "api_status",
    "load_example_data",
    "load_example_master",
    "load_example_fixations",
    "__version__",
]
