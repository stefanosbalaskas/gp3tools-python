# External face-data reporting

This article is the Python counterpart of the corresponding gp3tools workflow. It uses bundled synthetic data so it can be run without private participant exports.

```python
import gp3tools as gp3

data = gp3.load_example_master()
# read_gazepoint_face_export is available as gp3.read_gazepoint_face_export(...)
# audit_gazepoint_face_quality is available as gp3.audit_gazepoint_face_quality(...)
# sync_gazepoint_face_data is available as gp3.sync_gazepoint_face_data(...)
# prepare_gazepoint_multimodal_data is available as gp3.prepare_gazepoint_multimodal_data(...)
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
