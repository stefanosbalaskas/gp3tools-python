import numpy as np
import pandas as pd

import gp3tools as gp3


def _summary(
    operation="import",
    total_rows=1_000_000,
    n_files=1,
    elapsed=100.0,
    heap=100.0,
    success=3,
    trials=3,
):
    return pd.DataFrame(
        {
            "operation": [operation],
            "total_rows": [total_rows],
            "n_files": [n_files],
            "median_elapsed_s": [elapsed],
            "median_heap_delta_mb": [heap],
            "n_success": [success],
            "n_trials": [trials],
        }
    )


def test_r_performance_limits_exact_values():
    out = gp3.gp3tools_performance_limits()

    assert out["operation"].tolist() == [
        "generate",
        "import",
        "master",
        "sampling",
        "quality",
    ]

    assert out["max_seconds_per_million_rows"].tolist() == [
        90.0,
        240.0,
        240.0,
        180.0,
        180.0,
    ]

    assert out["max_heap_delta_mb_per_million_rows"].tolist() == [
        1200.0,
        1800.0,
        1800.0,
        1200.0,
        1200.0,
    ]

    assert out["max_scaling_exponent"].tolist() == [1.6] * 5


def test_r_performance_regression_without_baseline():
    out = gp3.check_gazepoint_performance_regression(x=_summary())

    assert set(out) == {
        "overall",
        "checks",
        "evaluated",
        "limits",
        "baseline_used",
        "elapsed_ratio_limit",
        "memory_ratio_limit",
    }

    assert out["baseline_used"] is False

    overall = out["overall"].iloc[0]

    assert bool(overall["pass"])
    assert overall["n_checks"] == 6
    assert overall["n_pass"] == 6
    assert overall["n_fail"] == 0

    assert out["checks"]["check"].tolist() == [
        "operation_completed",
        "elapsed_absolute_limit",
        "memory_absolute_limit",
        "scaling_exponent_limit",
        "elapsed_baseline_ratio",
        "memory_baseline_ratio",
    ]


def test_r_performance_absolute_elapsed_failure():
    out = gp3.check_gazepoint_performance_regression(
        x=_summary(
            operation="generate",
            elapsed=100.0,
        )
    )

    failed = out["checks"].query("status == 'fail'")

    assert failed["check"].tolist() == ["elapsed_absolute_limit"]

    assert not bool(out["overall"].iloc[0]["pass"])


def test_r_performance_operation_failure():
    out = gp3.check_gazepoint_performance_regression(
        x=_summary(
            success=2,
            trials=3,
        )
    )

    failed = out["checks"].query("status == 'fail'")

    assert "operation_completed" in set(failed["check"])


def test_r_performance_baseline_ratios():
    current = _summary(
        elapsed=150.0,
        heap=300.0,
    )

    baseline = _summary(
        elapsed=100.0,
        heap=100.0,
    )

    out = gp3.check_gazepoint_performance_regression(
        x=current,
        baseline=baseline,
        elapsed_ratio_limit=1.4,
        memory_ratio_limit=2.5,
    )

    evaluated = out["evaluated"].iloc[0]

    assert np.isclose(
        evaluated["elapsed_ratio"],
        1.5,
    )

    assert np.isclose(
        evaluated["memory_ratio"],
        3.0,
    )

    failed = set(out["checks"].query("status == 'fail'")["check"])

    assert "elapsed_baseline_ratio" in failed
    assert "memory_baseline_ratio" in failed
    assert out["baseline_used"] is True


def test_r_performance_scaling_exponent():
    data = pd.DataFrame(
        {
            "operation": [
                "custom",
                "custom",
            ],
            "total_rows": [
                100_000,
                1_000_000,
            ],
            "n_files": [1, 1],
            "median_elapsed_s": [
                1.0,
                10.0,
            ],
            "median_heap_delta_mb": [
                1.0,
                10.0,
            ],
            "n_success": [3, 3],
            "n_trials": [3, 3],
        }
    )

    out = gp3.check_gazepoint_performance_regression(x=data)

    exponent = out["evaluated"]["scaling_exponent"]

    assert np.allclose(
        exponent,
        [1.0, 1.0],
        atol=1e-12,
    )


def test_r_performance_trial_data_are_summarised():
    trials = pd.DataFrame(
        {
            "trial": [1, 2, 3],
            "scale_id": [
                "s1",
                "s1",
                "s1",
            ],
            "total_rows": [
                1_000_000,
                1_000_000,
                1_000_000,
            ],
            "n_files": [1, 1, 1],
            "rows_per_file": [
                1_000_000,
                1_000_000,
                1_000_000,
            ],
            "operation": [
                "import",
                "import",
                "import",
            ],
            "status": [
                "ok",
                "ok",
                "error",
            ],
            "elapsed_s": [
                100.0,
                120.0,
                999.0,
            ],
            "heap_delta_mb": [
                10.0,
                20.0,
                999.0,
            ],
            "output_size_mb": [
                1.0,
                2.0,
                999.0,
            ],
        }
    )

    out = gp3.check_gazepoint_performance_regression(x=trials)

    row = out["evaluated"].iloc[0]

    assert row["n_trials"] == 3
    assert row["n_success"] == 2
    assert row["median_elapsed_s"] == 110.0
    assert row["median_heap_delta_mb"] == 15.0

    failed = set(out["checks"].query("status == 'fail'")["check"])

    assert "operation_completed" in failed


def test_r_performance_accepts_single_positional_summary():
    out = gp3.check_gazepoint_performance_regression(_summary())

    assert isinstance(out, dict)
    assert out["overall"].iloc[0]["n_checks"] == 6


def test_legacy_python_performance_regression_preserved():
    current = pd.DataFrame(
        {
            "elapsed_seconds": [
                12.0,
                12.0,
            ]
        }
    )

    baseline = pd.DataFrame(
        {
            "elapsed_seconds": [
                10.0,
                10.0,
            ]
        }
    )

    out = gp3.check_gazepoint_performance_regression(
        current,
        baseline,
        tolerance=0.10,
    )

    assert isinstance(out, pd.DataFrame)
    assert np.isclose(
        out.iloc[0]["relative_change"],
        0.2,
    )
    assert bool(out.iloc[0]["regression"])


def test_r_custom_limits():
    limits = pd.DataFrame(
        {
            "operation": ["custom"],
            "max_seconds_per_million_rows": [5.0],
            "max_heap_delta_mb_per_million_rows": [100.0],
            "max_scaling_exponent": [2.0],
        }
    )

    data = _summary(
        operation="custom",
        elapsed=6.0,
    )

    out = gp3.check_gazepoint_performance_regression(
        x=data,
        limits=limits,
    )

    failed = set(out["checks"].query("status == 'fail'")["check"])

    assert "elapsed_absolute_limit" in failed
