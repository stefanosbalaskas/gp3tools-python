# Working safely with private Gazepoint exports

This article is the Python counterpart of the corresponding gp3tools workflow. It uses bundled synthetic data so it can be run without private participant exports.

```python
import gp3tools as gp3

data = gp3.load_example_master()
# load_example_master is available as gp3.load_example_master(...)
# create_gazepoint_master is available as gp3.create_gazepoint_master(...)
# audit_gazepoint_master is available as gp3.audit_gazepoint_master(...)
# run_gazepoint_workflow is available as gp3.run_gazepoint_workflow(...)
```

## Recommended workflow

1. Inspect column availability and create/validate a master table.
2. Run the relevant quality gates before transformation or modelling.
3. Keep preprocessing and exclusion decisions explicit.
4. Save derived tables/plots with package/version metadata.
5. For backend-adapted statistical functions, report the Python backend and validate the inferential target against the R reference when confirmatory equivalence matters.

## Data safety

Keep raw participant exports outside the repository, use synthetic examples for public issues, inspect derived reports before sharing, and check version-control status before every commit.

## Reproducibility checkpoint

```python
import gp3tools as gp3

print(gp3.__version__)
print(gp3.api_status().query("r_export in @funcs"))
```
