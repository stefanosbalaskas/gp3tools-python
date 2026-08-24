# Scanpath and quality-control quick wins

This article is the Python counterpart of the corresponding gp3tools workflow. It uses bundled synthetic data so it can be run without private participant exports.

```python
import gp3tools as gp3

data = gp3.load_example_master()
# prepare_gazepoint_aoi_sequences is available as gp3.prepare_gazepoint_aoi_sequences(...)
# compute_gazepoint_sequence_complexity is available as gp3.compute_gazepoint_sequence_complexity(...)
# cluster_gazepoint_scanpaths is available as gp3.cluster_gazepoint_scanpaths(...)
# plot_gazepoint_scanpath is available as gp3.plot_gazepoint_scanpath(...)
```

## Recommended workflow

1. Inspect column availability and create/validate a master table.
2. Run the relevant quality gates before transformation or modelling.
3. Keep preprocessing and exclusion decisions explicit.
4. Save derived tables/plots with package/version metadata.
5. For backend-adapted statistical functions, report the Python backend and validate the inferential target against the R reference when confirmatory equivalence matters.

## Reproducibility checkpoint

```python
import gp3tools as gp3

print(gp3.__version__)
print(gp3.api_status().query("r_export in @funcs"))
```
