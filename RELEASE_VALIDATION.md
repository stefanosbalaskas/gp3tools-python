# gp3tools Python release validation

Generated: 2026-08-29T00:30:22+03:00

## Frozen reference

- Canonical R package: `gp3tools 2.3.0`
- Canonical R exports: **278**
- Python public namespace: **285**
- Frozen R4 oracle assertions: **3,786**
- Frozen R4 oracle SHA256: `7749F214B3F16A4A03CEBD030C597999B3BDDA38C1BEB9A4CAEDA5380735F684`

## Final validation

- Static R/Python call-surface hard failures: **0**
- Full pytest suite: **684 passed**
- Measured line coverage: **90.06%**
- Required line coverage: **>= 90%**
- Ruff: **PASS**
- `compileall`: **PASS**
- `git diff --check`: **PASS**
- R1-R4 behavioral/semantic regression suites: **PASS**
- Runnable Python examples: **6/6 PASS**
- Headless plot smoke: **PASS**
- MkDocs strict build: **PASS**
- Wheel build: **PASS**
- Source distribution build: **PASS**
- Offline extracted-wheel artifact smoke: **PASS**
- All 278 R exports callable: **PASS**

## Distribution artifacts

- Wheel: `gp3tools-0.1.0a1-py3-none-any.whl`
- Wheel SHA256: `6B5DE8C402BA0C585014765C71704CF11B5AA6EAC6048F2F31BD9D1D5323222B`
- Source distribution: `gp3tools-0.1.0a1.tar.gz`
- Source distribution SHA256: `B4E6C56FA0318958881A48736C82DD6695A6773D8500754BF39736B63921F592`

## Runnable examples

- `binocular_pupil.py`
- `full_workflow.py`
- `plot_gallery.py`
- `pupil_qc.py`
- `quickstart.py`
- `timecourse.py`

## Documentation

The documentation build completed successfully under `mkdocs build --strict --clean`.

The documentation tree contains the complete workflow, AOI, pupil,
binocular, scanpath, transition, time-course, statistical, Bayesian,
face, multimodal, interoperability, QC, plotting, validation and release
articles currently implemented in the repository.

## Compatibility and parity

The Python implementation preserves the frozen R 2.3.0 export surface
while retaining documented Python compatibility interfaces.

Parity validation distinguishes exact structural behavior, numerical
tolerance where appropriate, and backend-equivalent statistical behavior
where the Python ecosystem does not use the identical R implementation.

The frozen R4 oracle was not modified during completion.