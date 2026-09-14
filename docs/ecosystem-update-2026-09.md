# Ecosystem update — September 2026

## Related methodological addition

Within the wider Gazepoint research-software ecosystem, the related **gpbiometricspy** package now includes a fully exact-main-certified crossed participant–item Gaussian hierarchical location–scale model with **one location random slope for each crossed factor**.

The model targets continuous repeated outcomes where both participants and items/stimuli contribute heterogeneity. Each crossed factor has a location intercept, a location slope and a log-scale intercept, with its own unrestricted positive-definite 3 × 3 covariance matrix. The method keeps seen-level conditional prediction distinct from unseen-level population prediction and binds converged fits to deterministic reproducibility certificates.

This is a related downstream modelling option, not a change to `gp3tools` for Python. The package's frozen R-parity target, release version, tests, API and scientific claims remain unchanged.

### Certification record

Certified gpbiometricspy PR #129 is pinned to merge SHA `d078e0366ace49c3ebeb2f6800bad6394d70631e`:

- 14/14 exact-main push workflow families green;
- 12/12 Ubuntu/macOS/Windows × Python 3.11–3.14 test lanes green;
- 782/782 tests;
- 14,015/14,015 statements;
- 6,757/6,776 raw branches = 99.7196%;
- 19 unchanged audited structural arcs, with 0 unexpected, 0 stale and 0 unaudited debt;
- frozen `gpbiometrics 2.0.0` parity surface unchanged at 406/406.

[Read the crossed participant–item random-slope guide](https://stefanosbalaskas.github.io/gpbiometricspy/methods/crossed-random-slopes-location-scale/)

[Open gpbiometricspy PR #129](https://github.com/stefanosbalaskas/gpbiometricspy/pull/129)

For `gp3tools` users, the method belongs downstream of defensible import, quality control, pupil/gaze preparation, AOI construction and measurement workflows; it does not replace those steps or establish measurement validity by itself.
