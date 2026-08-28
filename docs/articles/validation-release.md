# Validation and release engineering

The Python port uses layered validation rather than treating API-name parity as proof of scientific parity.

## Layer 1 β€” public surface

The frozen R 2.3.0 namespace contains 278 exports. Every export must resolve to an introspectable Python callable, while the Python package may expose a small number of Python-specific utility names.

## Layer 2 β€” semantic call contracts

R spellings, required arguments, aliases and legacy Python convenience interfaces are tested together. Where a canonical R return shape conflicts with an earlier Python convenience return shape, semantic dual dispatch preserves both rather than silently breaking one interface.

## Layer 3 β€” frozen behavioral oracles

Deterministic R fixtures are flattened into typed assertions and compared against Python. Numerical tolerances are explicit. Frozen oracle files are hash checked during the completion run. R1β€“R3 are executed as their permanent oracle test suites. For R4, the completion runner executes the permanent canonical target/structure tests and preserves the immutable 3,786-row oracle; a full regenerated 3,786-row scorer is a distinct evidence layer and is not implied by hash verification.

## Layer 4 β€” release integrity

The repository must pass linting, compilation, full tests, β‰¥90% coverage, documentation build, examples, wheel/sdist construction and isolated wheel installation.

## Scientific interpretation

`BACKEND_EQUIVALENT` does not mean byte-for-byte numerical equality. For model families implemented with different statistical libraries, the contract is the analysis family, inputs, outputs, diagnostics and documented interpretation.
