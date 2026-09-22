# gp3tools Python release validation

Generated: 2026-09-22
Release candidate: **0.1.1**

## Frozen reference

- Canonical R package: `gp3tools 2.3.0`
- Canonical R exports: **278**
- Python supported versions: **3.11, 3.12, 3.13**

## Release-candidate validation

Validated on exact development head `f144c76074303a310599c8b2fdbf7eda61ca9cae` before the version-only release metadata commit.

- Full pytest suite: **1,424 passed**
- Executable statements: **16,546**
- Missing executable statements: **0**
- Package-wide line coverage: **100.00%**
- Ruff: **PASS**
- Public 278-export API contract: **PASS**
- R4 compatibility coverage contract: **100% PASS**
- Python 3.11 CI: **PASS**
- Python 3.12 CI: **PASS**
- Python 3.13 CI: **PASS**
- Runnable examples smoke: **PASS**
- Wheel build: **PASS**
- Source distribution build: **PASS**
- Fresh-wheel installation and `pip check`: **PASS**
- Fresh-wheel 278-export smoke: **PASS**

The release metadata commit raises the CI package-wide coverage threshold from 90% to **100%**. The release is not considered qualified until the same CI matrix passes again on the exact `0.1.1` metadata head.

## Scientific and API contract

The Python implementation preserves the frozen R 2.3.0 export surface while retaining documented Python compatibility interfaces.

Coverage closure did **not** disable tests, lower scientific validation, add coverage exclusions, or silently change estimator, missing-data, AOI, censoring, or provenance semantics. Residual defensive branches proven unreachable under existing validated invariants were simplified directly; executable edge paths received deterministic regression coverage.

## Distribution contract

The stable release must use the exact validated `0.1.1` source state to produce:

- `gp3tools-0.1.1-py3-none-any.whl`
- `gp3tools-0.1.1.tar.gz`

GitHub Release assets and PyPI publication must refer to those same validated distributions. PyPI publication is performed through the repository's Trusted Publishing workflow.
