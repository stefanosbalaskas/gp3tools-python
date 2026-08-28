# Complete Python workflow

This article demonstrates the intended analysis order for `gp3tools` in Python. The exact columns available depend on the Gazepoint export type, so production analyses should inspect and validate the source before modelling.

## 1. Import and inspect

```python
import gp3tools as gp3

raw = gp3.read_gazepoint("participant.csv")
columns = gp3.inspect_gazepoint_columns(raw)
```

## 2. Build and audit a master table

```python
master = gp3.create_gazepoint_master(gaze_data=raw)
audit = gp3.audit_gazepoint_master(master)
```

Use the audit output to document sampling, gaze validity, pupil missingness, coordinate coverage and AOI state coverage before inferential analysis.

## 3. Pupil and event processing

The package includes blink detection, interpolation, artifact flagging, smoothing, binocular combination/reconstruction and trial/window summaries. Preserve preprocessing decisions in the analysis record rather than silently overwriting raw columns.

## 4. AOI, transitions and scanpaths

Static rectangular AOIs, polygon AOIs and time-varying AOIs are supported. Sequence, transition, entropy, recurrence, scanpath geometry and clustering functions can then be applied to validated assignments.

## 5. Time-course and modelling

Time-course preparation, cluster-permutation workflows and model helpers are available. Python-native statistical backends are reported as backend-equivalent/adapted where exact R-backend identity is neither possible nor scientifically meaningful.

## 6. Reporting

Use the reporting/checklist/export helpers to preserve decisions, diagnostics, exclusions and generated outputs with the analysis.

See `docs/PARITY_STATUS.md` for the evidence terminology used by the Python port.
