# gp3tools for Python

[![CI](https://github.com/stefanosbalaskas/gp3tools-python/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/stefanosbalaskas/gp3tools-python/actions/workflows/ci.yml)
[![Documentation](https://github.com/stefanosbalaskas/gp3tools-python/actions/workflows/docs.yml/badge.svg?branch=main)](https://github.com/stefanosbalaskas/gp3tools-python/actions/workflows/docs.yml)
[![Release](https://img.shields.io/github/v/release/stefanosbalaskas/gp3tools-python?include_prereleases&label=release)](https://github.com/stefanosbalaskas/gp3tools-python/releases/tag/v0.1.0a1)
![Python](https://img.shields.io/badge/Python-3.11--3.13-3776AB)
![License](https://img.shields.io/badge/license-MIT-success)

**Documentation:** https://stefanosbalaskas.github.io/gp3tools-python/


A Python implementation of the public workflow of **gp3tools**, the R package for importing, inspecting, analysing, visualising, modelling, and reporting Gazepoint GP3 / Gazepoint Analysis exports.

> **Status: 0.1.0a1 comprehensive migration build.** The complete 278-function public name surface frozen from the R `v2.3.0` `NAMESPACE` is present. Core data, QC, pupil, AOI, sequence, event, plotting, reporting, face, simulation, and export workflows are native Python. Statistical functions tied to R-specific backends are **Python-native adaptations**, not claims of coefficient-for-coefficient equivalence to `lme4`, `glmmTMB`, `mgcv`, or `brms`.

## Install a local wheel

```powershell
uv pip install .\gp3tools-0.1.0a1-py3-none-any.whl
```

or

```powershell
pip install .\gp3tools-0.1.0a1-py3-none-any.whl
```

## Quick start

```python
import gp3tools as gp3

master = gp3.load_example_master()

sampling = gp3.check_sampling_rate(
    master,
    time_col="TIME",
    group_cols=["subject", "trial_global"],
)

pupil = gp3.preprocess_gazepoint_signals(
    master,
    pupil_col="pupil",
    time_col="TIME",
)

transitions = gp3.compute_gazepoint_aoi_transition_matrix(
    master,
    aoi_col="aoi_current",
)

fig = gp3.plot_gazepoint_heatmap(master)
```

## Public API contract

```python
import gp3tools as gp3

len(gp3.R_EXPORTS)
# 278

gp3.api_status()["status"].value_counts()
```

The build distinguishes:

- `native`: direct Python implementation;
- `native-adapted`: implemented in Python but the R package used a backend or algorithm whose exact numerical identity is not asserted;
- `native-adapter`: interoperability adapter expressed natively in Python.

See `docs/API_PARITY.md` and the bundled `api_manifest.csv` for every exported function.

## Main modules

- `io`: Gazepoint CSV, folder, summary, and face-export import
- `qc`: master tables, sampling, tracking, missingness, screen bounds, exclusions, readiness
- `pupil`: artifact flags, blinks, interpolation, baseline correction, smoothing, binocular reconstruction
- `aoi`: static/dynamic/polygon AOIs, entries, windows, transitions, entropy, scanpaths, clustering
- `events`: fixation/saccade detection, agreement and benchmark workflows
- `stats`: model preparation, LMM/GLM/spline adaptations, Bayesian adapters, cluster permutation, bootstrap and sensitivity
- `face`: external facial-analysis QC and time synchronisation
- `interop`: eyetrackingR/pupillometryR/gazer/eyetools/HDDM/BIDS/gpbiometrics adapters
- `plotting`: Matplotlib visualisations
- `reporting`: CSV/HTML outputs, reporting checklists and end-to-end workflow
- `simulation`: synthetic Gazepoint data for examples and tests

## Scientific compatibility note

A Python port should preserve scientific intent without pretending different statistical engines are identical. Accordingly, the modelling layer documents backend adaptation explicitly. The package is suitable for software migration, workflow prototyping, reproducible preprocessing, data/QC/plotting tasks, and validation development; advanced inferential parity should be checked against the R reference implementation before confirmatory use.

## R reference

The public API was frozen against `gp3tools` R `v2.3.0`. `R_NAMESPACE_REFERENCE.txt` is included in the source distribution for auditability.

## Citation

Balaskas, S. (2026). *gp3tools: An R Package for Reproducible Analysis and Reporting of Gazepoint GP3 Eye-Tracking Exports*. Journal of Eye Movement Research, 19(4), 76. DOI: 10.3390/jemr19040076.

## License

MIT.

<!-- GP3TOOLS_PARITY_STATUS -->
## R 2.3.0 parity and validation

The Python package freezes the **278-export** R 2.3.0 public API while exposing **285** Python public names in the validated parity branch. Behavioral equality is evidence graded: exact and numerical-tolerance claims are made only where frozen R oracles exist; backend-adapted statistical families are labelled accordingly.

The release gate includes R1β€“R3 frozen behavioral regressions, R4 canonical structure/compatibility tests plus frozen-oracle integrity checks, whole-surface smoke, plot catalog smoke, runnable examples, Ruff, compilation, strict documentation build, **β‰¥90% test coverage**, wheel/sdist construction and isolated-wheel validation. See `docs/PARITY_STATUS.md`.
