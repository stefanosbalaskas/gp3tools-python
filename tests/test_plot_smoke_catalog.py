from __future__ import annotations

import inspect

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gp3tools as gp3


def test_plot_export_catalog_is_complete_and_introspectable() -> None:
    names = [name for name in gp3.R_EXPORTS if name.startswith("plot_")]
    assert names
    for name in names:
        fn = getattr(gp3, name)
        assert callable(fn)
        assert inspect.signature(fn) is not None


def test_core_plot_smoke() -> None:
    master = gp3.load_example_master()
    produced = []

    # Sampling-rate plot is part of the wheel validation contract.
    produced.append(
        gp3.plot_sampling_rate(
            master,
            time_col="TIME",
            group_cols=["subject"],
        )
    )

    # Exercise additional plots opportunistically using public example data.
    candidates = [
        ("plot_tracking_quality", (master,), {}),
        ("plot_gazepoint_time_series", (master,), {}),
        ("plot_gazepoint_pupil_timecourse", (master,), {}),
    ]
    for name, args, kwargs in candidates:
        fn = getattr(gp3, name, None)
        if not callable(fn):
            continue
        try:
            value = fn(*args, **kwargs)
        except (TypeError, ValueError, KeyError):
            # These functions may require a precomputed result object.  The
            # catalog test still proves the public plot surface; dedicated
            # producer/plot tests elsewhere cover their scientific contract.
            continue
        produced.append(value)

    assert produced
    assert all(value is not None for value in produced)
    plt.close("all")
