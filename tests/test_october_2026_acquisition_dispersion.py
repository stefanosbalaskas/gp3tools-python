import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from research.october_2026.acquisition_and_dispersion import (  # noqa: E402
    acquisition_integrity_certificate,
    dispersion_interpretation_audit,
)


def test_quantization_catchup_and_rate_provenance():
    t = [0., .01, .02, .06, .061, .071]
    d = acquisition_integrity_certificate(
        t, valid=[True, True, False, True, True, True],
        native_rate_hz=60, analysis_grid_rate_hz=500,
    )
    assert d["catchup_pair_candidates"] == 1
    assert d["maximum_gap_ms"] == pytest.approx(40)
    assert d["invalid_state_fraction"] == pytest.approx(1 / 6)
    assert d["analysis_grid_exceeds_observed_rate"] is True


def test_duplicate_timestamp_retained_and_backward_rejected():
    d = acquisition_integrity_certificate([0, .01, .01, .02])
    assert d["duplicate_timestamp_intervals"] == 1
    with pytest.raises(ValueError, match="reverse"):
        acquisition_integrity_certificate([0, .02, .01])


def test_dispersion_requires_context_and_stays_descriptive():
    f = pd.DataFrame({
        "participant_id": ["p1", "p1"],
        "measurement_context": ["calibration_target", "task_fixation"],
        "target_visual_angle_deg": [.5, 3.0],
        "eccentricity_deg": [0., 4.],
        "task": ["calibration", "reading"],
        "bcea68": [0.1, 0.2],
    })
    a = dispersion_interpretation_audit(f)
    assert "not_tracker_precision" in a.iloc[1].interpretation
    with pytest.raises(ValueError, match="missing"):
        dispersion_interpretation_audit(f.drop(columns=["target_visual_angle_deg"]))
