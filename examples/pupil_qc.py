"""Pupil-QC and preprocessing example."""

from __future__ import annotations

import gp3tools as gp3


def main() -> None:
    master = gp3.load_example_master()
    pupil = gp3.mean_gazepoint_pupil(master)
    flagged = gp3.flag_gazepoint_pupil(pupil)
    print("pupil rows:", len(flagged))
    assert len(flagged) == len(master)


if __name__ == "__main__":
    main()
