"""Headless plotting example suitable for CI and servers."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gp3tools as gp3


def main() -> None:
    master = gp3.load_example_master()
    fig = gp3.plot_sampling_rate(master, time_col="TIME", group_cols=["subject"])
    assert fig is not None
    plt.close("all")
    print("plot smoke: PASS")


if __name__ == "__main__":
    main()
