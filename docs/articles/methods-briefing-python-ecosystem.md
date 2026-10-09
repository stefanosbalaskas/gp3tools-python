# Methods briefing: Python measurement integrity and package boundaries

This page maps the September–October 2026 methodological recommendations to existing **Python** workflows. It does not announce a new gp3tools API or release.

## What to use now

| Task | Recommended Python owner | Why |
| --- | --- | --- |
| GP3 sample import, AOI/sequence QC, pupil preprocessing | `gp3tools` | Frozen R-parity namespace and legacy GP3 workflows |
| Trial/participant missingness sensitivity | [eyeprocesspy](https://github.com/stefanosbalaskas/eyeprocesspy/pull/37) | Experimental explicitly observed-versus-missing and pattern-mixture audits |
| Physiological rate lineage and reference agreement | [gpbiometricspy](https://github.com/stefanosbalaskas/gpbiometricspy/pull/173) | Device-versus-delivered-rate and agreement-versus-construct responsiveness |
| Cross-modal pairing, split scope, quality weighting | [GazeForge](https://github.com/stefanosbalaskas/GazeForge/pull/249) | Structurally auditable multimodal evidence |
| Cluster-aware nonlinear mediation robustness | [gp3bayespy](https://github.com/stefanosbalaskas/gp3bayespy/pull/18) | Experimental diagnostic, not automatic causal identification |
| Functional trajectories and sparse Bayesian development | [eyetrajectoriespy](https://stefanosbalaskas.github.io/eyetrajectoriespy/) | 1.1 stable versus unreleased 1.2 research boundaries |
| Ordered scanpaths/transitions | [gp3sequencespy](https://stefanosbalaskas.github.io/gp3sequencespy/) | Sequence-specific analyses, not independent evidence of experimental generalization |

**Important:** newly proposed functions in linked repositories remain development-only until reviewed, merged, scientifically qualified and documented on published default branches. Cross-package pointers describe intended ownership, not guaranteed current released availability.

## A common reporting contract

Record participant and trial keys, native/SDK/observed/analysis sampling rates where known, source clocks and their alignment evidence, explicit dropout masks, QC flags, target truth if simulated, inferential estimand and grouped validation scope. Never equate sample-level CV with unseen-person validity. Do not silently replace gaps or declare subjective physiological states from a fused signal.

## Deliberately not added

No duplicate root functions, new fixation detector, or automatic scanpath estimator were warranted by this briefing cycle. This handoff preserves the 278-name stable public gp3tools namespace and existing parity evidence.
