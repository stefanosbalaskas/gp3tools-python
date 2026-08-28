---
title: Getting started
---

# Getting started

Use the bundled synthetic data to run a complete first workflow without
private participant exports.

## Install

=== "pip"

    ```bash
    python -m pip install https://github.com/stefanosbalaskas/gp3tools-python/releases/download/v0.1.0a1/gp3tools-0.1.0a1-py3-none-any.whl
    ```

=== "uv"

    ```bash
    uv pip install https://github.com/stefanosbalaskas/gp3tools-python/releases/download/v0.1.0a1/gp3tools-0.1.0a1-py3-none-any.whl
    ```

The GitHub CI matrix validates Python **3.11, 3.12 and 3.13**.

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
