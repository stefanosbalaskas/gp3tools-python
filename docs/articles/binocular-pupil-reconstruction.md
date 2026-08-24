# Binocular pupil reconstruction and artificial-monocular-loss validation

This Python migration article documents the corresponding gp3tools capability with explicit auditability and no assumption that a different computational backend is numerically identical to the R implementation.

```python
import gp3tools as gp3

data = gp3.load_example_master()
print(gp3.__version__)
```

Use `gp3.api_status()` to inspect implementation status for every public function. Core deterministic transformations are intended for direct Python use; backend-adapted statistical functions should be parity-tested against the R reference for confirmatory analyses.
