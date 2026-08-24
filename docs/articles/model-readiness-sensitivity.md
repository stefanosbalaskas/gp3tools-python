# Model-readiness and sensitivity analysis

This article is the Python counterpart of the corresponding gp3tools workflow. It uses bundled synthetic data so it can be run without private participant exports.

```python
import gp3tools as gp3

data = gp3.load_example_master()
# prepare_gazepoint_pupil_window_model_data is available as gp3.prepare_gazepoint_pupil_window_model_data(...)
# fit_gazepoint_pupil_window_lmm is available as gp3.fit_gazepoint_pupil_window_lmm(...)
# tidy_gazepoint_model_summary is available as gp3.tidy_gazepoint_model_summary(...)
# compare_gazepoint_nested_models is available as gp3.compare_gazepoint_nested_models(...)
```

## Recommended workflow

1. Inspect column availability and create/validate a master table.
2. Run the relevant quality gates before transformation or modelling.
3. Keep preprocessing and exclusion decisions explicit.
4. Save derived tables/plots with package/version metadata.
5. For backend-adapted statistical functions, report the Python backend and validate the inferential target against the R reference when confirmatory equivalence matters.

## Backend note

The Python migration preserves the workflow and estimand intent but does not assert numerical identity with R-specific engines. Validate confirmatory results against the frozen R implementation until a dedicated parity fixture exists.

## Reproducibility checkpoint

```python
import gp3tools as gp3

print(gp3.__version__)
print(gp3.api_status().query("r_export in @funcs"))
```
