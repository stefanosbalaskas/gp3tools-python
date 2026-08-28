"""End-to-end gp3tools Python quickstart using bundled synthetic data."""

from __future__ import annotations

import gp3tools as gp3


def main() -> None:
    master = gp3.load_example_master()
    sampling = gp3.check_sampling_rate(master, time_col="TIME", group_cols=["subject"])
    quality = gp3.summarise_tracking_quality(master, group_cols=["subject"])
    print("rows:", len(master))
    print("sampling groups:", len(sampling))
    print("quality groups:", len(quality))
    assert len(master) > 0
    assert len(sampling) > 0
    assert len(quality) > 0


if __name__ == "__main__":
    main()
