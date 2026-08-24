# Experimental Bayesian bridge helpers

This article is the Python counterpart of the corresponding gp3tools workflow. It uses bundled synthetic data so it can be run without private participant exports.

```python
import gp3tools as gp3

data = gp3.load_example_master()
# create_gazepoint_bayesian_sap is available as gp3.create_gazepoint_bayesian_sap(...)
# create_gazepoint_brms_template is available as gp3.create_gazepoint_brms_template(...)
# fit_gazepoint_brms_model is available as gp3.fit_gazepoint_brms_model(...)
# check_gazepoint_bayesian_readiness is available as gp3.check_gazepoint_bayesian_readiness(...)
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
