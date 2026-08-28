---
title: Getting started
---

# Getting started

Use the bundled synthetic data to run a complete first workflow without
private participant exports.

## Install

=== "pip"

    ```bash
    python -m pip install "gp3tools==0.1.0a1"
    ```

=== "latest alpha"

    ```bash
    python -m pip install --pre gp3tools
    ```

=== "uv"

    ```bash
    uv pip install "gp3tools==0.1.0a1"
    ```

The GitHub CI matrix validates Python **3.11, 3.12 and 3.13**.

The published distributions are available from [PyPI](https://pypi.org/project/gp3tools/), and the archived software release is available at [Zenodo DOI 10.5281/zenodo.22150772](https://doi.org/10.5281/zenodo.22150772).

## Verify

```python
import gp3tools as gp3

print(gp3.__version__)
print(len(gp3.R_EXPORTS))
print(len(gp3.__all__))
```

Validated contracts:

```text
R exports:           278
Python public names: 285
```

## Load example data

```python
master = gp3.load_example_master()
print(master.shape)
```

## Check sampling

```python
sampling = gp3.check_sampling_rate(
    master,
    time_col="TIME",
    group_cols=["subject", "trial_global"],
)
```

## Preprocess pupil data

```python
processed = gp3.preprocess_gazepoint_signals(
    master,
    pupil_col="pupil",
    time_col="TIME",
)
```

## Analyse AOI transitions

```python
transitions = gp3.compute_gazepoint_aoi_transition_matrix(
    master,
    aoi_col="aoi_current",
)
```

## Create a figure

```python
fig = gp3.plot_gazepoint_heatmap(master)
```

## Where next?

<div class="grid cards" markdown>

- **Complete workflow**

    [Open guide →](articles/complete-python-workflow.md)

- **Pupil analysis**

    [Open guide →](articles/pupil-workflow.md)

- **AOI analysis**

    [Open guide →](articles/aoi-workflow.md)

- **Plot gallery**

    [Open gallery →](articles/plot-gallery.md)

- **Validation**

    [Open status →](PARITY_STATUS.md)

- **API reference**

    [Browse functions →](API_REFERENCE.md)

</div>
