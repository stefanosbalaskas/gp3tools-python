# R 2.3.0 β†’ Python parity status

This repository freezes the public API against **gp3tools R 2.3.0**.

## Contracts

- Canonical R exports: **278**
- Python public namespace: **285 unique names**
- Frozen R4 oracle: **3,786 assertions**
- Frozen R4 oracle SHA256: `7749F214B3F16A4A03CEBD030C597999B3BDDA38C1BEB9A4CAEDA5380735F684`
- Coverage gate: **β‰¥ 90%**

## Evidence levels

The project deliberately separates API availability from scientific behavioral evidence:

| Status | Meaning |
|---|---|
| `EXACT` | Frozen deterministic R/Python evidence agrees exactly. |
| `NUMERIC_TOLERANCE` | Values agree within a declared numerical tolerance. |
| `STRUCTURAL_EQUIVALENT` | Structure and scientific meaning agree while representation differs. |
| `BACKEND_EQUIVALENT` | Python uses an appropriate Python backend for the same analysis family. |
| `INTENTIONAL_PYTHON_ADAPTATION` | Python-native adaptation is documented and tested. |
| `BLOCKED_OPTIONAL_DEP` | Requires an optional backend not installed in the core environment. |
| `FAIL` | Measured parity defect. |

A whole-surface smoke pass is **not** presented as universal exact behavioral parity. Exact claims are limited to frozen R oracle evidence.

## Behavioral freeze

R1, R2 and R3 are retained as permanent canonical behavioral-parity suites. R4 adds the pupil/QC/workflow/blink/fixation/AOI/geometry family and preserves legacy Python call contracts through semantic dual dispatch. The 3,786-row R4 oracle is frozen and hash-verified; this completion runner does not over-claim a fresh all-row R4 re-score unless a dedicated regenerated scorer is present.

## Release gates

The completion runner enforces:

1. 285/278 namespace/export integrity.
2. R1β€“R3 frozen behavioral regressions plus R4 canonical structure/compatibility gates.
3. Legacy R-alias compatibility.
4. All 278 exports resolving to introspectable callables.
5. Plot export catalog plus headless core plot smoke.
6. Full pytest with β‰¥90% coverage.
7. Static call-surface audit, Ruff and compileall.
8. Runnable examples.
9. Strict MkDocs build.
10. Wheel + sdist build and isolated wheel import smoke.
