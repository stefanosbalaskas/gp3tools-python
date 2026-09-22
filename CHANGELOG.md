# Changelog

## 0.1.1 — 2026-09-22

- enforce **100% package-wide executable statement coverage** across the complete `gp3tools` source tree;
- close residual AOI, pupil, and QC validation/edge paths with deterministic regression tests;
- remove branches proven unreachable under validated API invariants instead of excluding them from coverage;
- retain the frozen **278-export** R gp3tools v2.3.0 public API contract and R/Python compatibility gates;
- validate **1,424 tests** on Python 3.11, 3.12, and 3.13, including the 100% R4 compatibility coverage contract;
- validate runnable examples, wheel and source-distribution builds, and installation from the freshly built wheel;
- promote the post-0.1.0 hardening line from `0.1.1.dev0` to stable `0.1.1`.

## 0.1.0a1 — comprehensive migration alpha

- froze the 278-name public API from R gp3tools v2.3.0;
- implemented native Python import, QC, pupil, binocular, AOI, sequence, fixation/saccade, plotting, reporting, simulation, face, and interoperability layers;
- added Python-native statistical adaptations for mixed-model, spline/GAM-like, Bayesian, cluster-permutation, bootstrap, and sensitivity workflows;
- bundled five synthetic example datasets;
- added article set, API/parity manifests, examples, tests, build metadata, and CI configuration;
- added wheel and source-distribution validation workflow.

This is an alpha migration build; it does not claim exact inferential parity for R-backend-specific models.
