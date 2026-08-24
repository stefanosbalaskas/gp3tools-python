# Contributing

Changes should preserve the frozen public API, add tests for behavioural changes, avoid silent imputation, expose failure states, and document statistical backend differences. For deterministic functions, cross-language fixtures from the R implementation are preferred. For model functions, compare estimands and diagnostics rather than assuming numerical identity across engines.
