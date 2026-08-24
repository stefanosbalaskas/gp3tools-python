# Cluster-permutation time-course workflow

This article is the Python counterpart of the corresponding gp3tools workflow. It uses bundled synthetic data so it can be run without private participant exports.

```python
import gp3tools as gp3

data = gp3.load_example_master()
# prepare_gazepoint_cluster_data is available as gp3.prepare_gazepoint_cluster_data(...)
# run_gazepoint_cluster_permutation is available as gp3.run_gazepoint_cluster_permutation(...)
# estimate_gazepoint_divergence_point is available as gp3.estimate_gazepoint_divergence_point(...)
# plot_gazepoint_cluster_results is available as gp3.plot_gazepoint_cluster_results(...)
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
