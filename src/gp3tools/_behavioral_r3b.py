"""Canonical R gp3tools 2.3.0 behavior for tranche R3-B."""

from __future__ import annotations

import html
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class _R3BResult(dict):
    """Mapping carrying the canonical R class vector."""

    def __init__(
        self,
        *args: Any,
        r_class: str = "list",
        **kwargs: Any,
    ) -> None:
        super().__init__(
            *args,
            **kwargs,
        )

        self.r_class = r_class

    @property
    def gp3_r_class(self) -> str:
        return self.r_class


def _frame(value: Any, name: str = "data") -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise TypeError(f"`{name}` must be a data frame.")
    return value.copy()


def _listify(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _require(
    data: pd.DataFrame,
    columns: Sequence[str],
    name: str = "data",
) -> None:
    missing = [column for column in columns if column not in data.columns]

    if missing:
        raise ValueError(f"Missing required columns in `{name}`: " + ", ".join(missing))


def _collapse(value: Any) -> str:
    if value is None:
        return ""

    if isinstance(value, Sequence) and not isinstance(value, str):
        return ", ".join(str(item) for item in value)

    return str(value)


def _r_bool(value: Any) -> str:
    return "TRUE" if bool(value) else "FALSE"


def _stat(values: pd.Series, method: str) -> float:
    numeric = pd.to_numeric(values, errors="coerce").dropna()

    if numeric.empty:
        return math.nan

    if str(method).lower() == "median":
        return float(numeric.median())

    return float(numeric.mean())


def _quantile(values: Any, probability: float) -> float:
    raise RuntimeError("Retired unreachable R3-B compatibility helper.")


# =====================================================================
# REPORT
# =====================================================================


def _html_table(data: pd.DataFrame, max_rows: int) -> str:
    data = data.head(max(0, int(max_rows)))

    result = ["<table>", "<thead>", "<tr>"]

    for column in data.columns:
        result.append(f"<th>{html.escape(str(column))}</th>")

    result.extend(["</tr>", "</thead>", "<tbody>"])

    for _, row in data.iterrows():
        result.append("<tr>")

        for column in data.columns:
            value = row[column]

            if pd.isna(value):
                text = ""
            elif isinstance(value, (bool, np.bool_)):
                text = "TRUE" if bool(value) else "FALSE"
            else:
                text = str(value)

            result.append(f"<td>{html.escape(text)}</td>")

        result.append("</tr>")

    result.extend(["</tbody>", "</table>"])

    return "\n".join(result)


def create_gazepoint_report(
    results,
    output_file,
    title="Gazepoint diagnostic report",
    overwrite=True,
    max_rows=30,
    save_plots=True,
    plot_dir=None,
    metadata=None,
):
    """Create an R 2.3.0-compatible Gazepoint diagnostic HTML report."""
    from pathlib import Path

    import numpy as np
    import pandas as pd

    if metadata is not None:
        raise TypeError("unused argument (`metadata`)")

    if not isinstance(
        results,
        dict,
    ):
        raise ValueError("`results` must be a named list/dictionary.")

    required = [
        "sampling",
        "quality",
        "flagged_quality",
        "aoi_table",
    ]

    missing = [name for name in required if name not in results]

    if missing:
        raise ValueError("`results` is missing required element(s): " + ", ".join(missing))

    non_tables = [
        name
        for name in required
        if not isinstance(
            results[name],
            pd.DataFrame,
        )
    ]

    if non_tables:
        raise ValueError("Required report elements must be data frames: " + ", ".join(non_tables))

    output_path = Path(output_file)

    if output_path.exists() and not bool(overwrite):
        raise FileExistsError(f"`output_file` already exists: {output_path}")

    output_dir = output_path.parent

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if plot_dir is None:
        plot_dir_path = output_dir / (output_path.stem + "_files")

    else:
        plot_dir_path = Path(plot_dir)

    def html_escape(
        value,
    ):
        text = str(value)

        text = text.replace(
            "&",
            "&amp;",
        )

        text = text.replace(
            "<",
            "&lt;",
        )

        text = text.replace(
            ">",
            "&gt;",
        )

        text = text.replace(
            '"',
            "&quot;",
        )

        text = text.replace(
            "'",
            "&#39;",
        )

        return text

    def is_missing(
        value,
    ):
        try:
            result = pd.isna(value)

            if isinstance(
                result,
                (
                    bool,
                    np.bool_,
                ),
            ):
                return bool(result)

        except Exception:
            pass

        return False

    def scalar_text(
        value,
    ):
        if is_missing(value):
            return ""

        if isinstance(
            value,
            (
                bool,
                np.bool_,
            ),
        ):
            return "TRUE" if bool(value) else "FALSE"

        if isinstance(
            value,
            (
                int,
                np.integer,
            ),
        ):
            return str(int(value))

        if isinstance(
            value,
            (
                float,
                np.floating,
            ),
        ):
            number = float(value)

            if np.isfinite(number):
                if number.is_integer():
                    return str(int(number))

                return format(
                    number,
                    ".15g",
                )

            return str(number)

        return str(value)

    def html_table(
        data,
        limit=30,
    ):
        if not isinstance(
            data,
            pd.DataFrame,
        ):
            return "<p>No table available.</p>\n"

        if len(data) == 0:
            return "<p>No rows available.</p>\n"

        n_total = len(data)

        shown = data.head(int(limit))

        header = (
            "<tr>"
            + "".join("<th>" + html_escape(column) + "</th>" for column in shown.columns)
            + "</tr>\n"
        )

        rendered_rows = []

        for row in shown.itertuples(
            index=False,
            name=None,
        ):
            rendered_rows.append(
                "<tr>"
                + "".join("<td>" + html_escape(scalar_text(value)) + "</td>" for value in row)
                + "</tr>"
            )

        if n_total > int(limit):
            note = (
                "<p><em>Showing first "
                + str(int(limit))
                + " of "
                + str(n_total)
                + " rows.</em></p>\n"
            )

        else:
            note = ""

        return (
            note
            + "<table>\n"
            + "<thead>\n"
            + header
            + "</thead>\n"
            + "<tbody>\n"
            + "\n".join(rendered_rows)
            + "\n</tbody>\n"
            + "</table>\n"
        )

    written_plots = None

    if bool(save_plots):
        import gp3tools as gp3

        written_plots = gp3.save_gazepoint_plots(
            flagged_quality=results["flagged_quality"],
            sampling=results["sampling"],
            output_dir=plot_dir_path,
            prefix="report",
            overwrite=overwrite,
        )

    n_all_gaze = (
        len(results["all_gaze"])
        if (
            "all_gaze" in results
            and isinstance(
                results["all_gaze"],
                pd.DataFrame,
            )
        )
        else pd.NA
    )

    n_all_fix = (
        len(results["all_fix"])
        if (
            "all_fix" in results
            and isinstance(
                results["all_fix"],
                pd.DataFrame,
            )
        )
        else pd.NA
    )

    n_sampling_rows = len(results["sampling"])

    n_quality_rows = len(results["quality"])

    n_aoi_rows = len(results["aoi_table"])

    flagged_quality = results["flagged_quality"]

    if "review_required" in flagged_quality.columns:
        review = flagged_quality["review_required"].isin(
            [
                True,
            ]
        )

        n_flagged = int(review.sum())

        flagged_rows = flagged_quality.loc[review].copy()

    else:
        n_flagged = pd.NA

        flagged_rows = flagged_quality.copy()

    overview = pd.DataFrame(
        {
            "item": [
                "All-gaze rows",
                "Fixation rows",
                "Sampling summary rows",
                "Tracking-quality rows",
                "AOI summary rows",
                "Rows requiring review",
            ],
            "value": pd.Series(
                [
                    n_all_gaze,
                    n_all_fix,
                    n_sampling_rows,
                    n_quality_rows,
                    n_aoi_rows,
                    n_flagged,
                ],
                dtype=object,
            ),
        }
    )

    plot_html = ""

    if (
        isinstance(
            written_plots,
            pd.DataFrame,
        )
        and len(written_plots) > 0
    ):
        blocks = []

        for row in written_plots.itertuples(index=False):
            plot_name = row.plot

            plot_file = Path(row.file)

            relative = plot_dir_path.name + "/" + plot_file.name

            blocks.append(
                "<h3>"
                + html_escape(plot_name)
                + "</h3>\n"
                + "<img src='"
                + html_escape(relative)
                + "' alt='"
                + html_escape(plot_name)
                + "' />\n"
            )

        plot_html = "<h2>Diagnostic plots</h2>\n" + "\n".join(blocks)

    if len(flagged_rows) == 0:
        flagged_html = "<p>No recordings were flagged for review.</p>\n"

    else:
        flagged_html = html_table(
            flagged_rows,
            max_rows,
        )

    html = (
        "<!DOCTYPE html>\n"
        "<html>\n"
        "<head>\n"
        "<meta charset='utf-8'>\n"
        "<title>" + html_escape(title) + "</title>\n"
        "<style>\n"
        "body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.4; }\n"
        "h1, h2, h3 { color: #222; }\n"
        "table { border-collapse: collapse; margin-bottom: 24px; width: 100%; font-size: 13px; }\n"
        "th, td { border: 1px solid #ddd; padding: 6px; text-align: left; }\n"
        "th { background: #f2f2f2; }\n"
        "img { max-width: 100%; height: auto; border: 1px solid #ddd; margin-bottom: 24px; }\n"
        ".note { background: #f8f8f8; padding: 12px; border-left: 4px solid #999; }\n"
        "</style>\n"
        "</head>\n"
        "<body>\n"
        "<h1>" + html_escape(title) + "</h1>\n"
        "<p class='note'>Generated by <strong>gp3tools</strong>.</p>\n"
        "<h2>Overview</h2>\n"
        + html_table(
            overview,
            max_rows,
        )
        + "<h2>Recordings requiring review</h2>\n"
        + flagged_html
        + "<h2>Sampling-rate summary</h2>\n"
        + html_table(
            results["sampling"],
            max_rows,
        )
        + "<h2>Tracking-quality summary</h2>\n"
        + html_table(
            results["quality"],
            max_rows,
        )
        + "<h2>AOI summary</h2>\n"
        + html_table(
            results["aoi_table"],
            max_rows,
        )
        + plot_html
        + "<h2>Interpretation note</h2>\n"
        "<p>This report is intended as a diagnostic screening report. "
        "Rows flagged for review should be inspected before statistical analysis. "
        "Metrics recomputed from exported all-gaze and fixation rows may not exactly "
        "match Gazepoint Analysis internal summary calculations.</p>\n"
        "</body>\n"
        "</html>\n"
    )

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(html)
        # === R3B R WRITELINES TRAILING SEPARATOR ===
        # R writeLines() appends `sep` after its string even
        # when that string already ends with a newline.
        handle.write("\n")

    out = pd.DataFrame(
        {
            "report": [str(output_path)],
            "plot_dir": [str(plot_dir_path)],
            "n_flagged": [n_flagged],
        }
    )

    out.r_class = "tbl_df|tbl|data.frame"

    return out


# =====================================================================
# DIVERGENCE
# =====================================================================


def _condition_curve(
    data: pd.DataFrame,
    outcome_col: str,
    time_col: str,
    condition_col: str,
    comparison: Sequence[Any],
    time_grid: Sequence[float],
    summary_function: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for condition in comparison:
        subset = data.loc[data[condition_col] == condition]

        for time in time_grid:
            values = subset.loc[
                subset[time_col] == time,
                outcome_col,
            ]

            rows.append(
                {
                    "condition": condition,
                    "time": float(time),
                    "estimate": _stat(
                        values,
                        summary_function,
                    ),
                    "n": int(
                        pd.to_numeric(
                            values,
                            errors="coerce",
                        )
                        .notna()
                        .sum()
                    ),
                }
            )

    return pd.DataFrame(rows)


def _difference_curve(
    curve: pd.DataFrame,
    comparison: Sequence[Any],
    time_grid: Sequence[float],
) -> pd.DataFrame:
    raise RuntimeError("Retired unreachable R3-B compatibility helper.")


def _candidate_mask(
    values: pd.Series,
    null_value: float,
    min_abs_difference: float,
    direction: str,
) -> np.ndarray:
    delta = pd.to_numeric(values, errors="coerce") - float(null_value)

    result = delta.abs() >= float(min_abs_difference)

    mode = str(direction).lower()

    if mode in {"positive", "greater", "above"}:
        result &= delta > 0

    elif mode in {"negative", "less", "below"}:
        result &= delta < 0

    return result.fillna(False).to_numpy(dtype=bool)


def _find_onset(
    difference: pd.DataFrame,
    consecutive_points: int,
    null_value: float,
    min_abs_difference: float,
    direction: str,
) -> float:
    raise RuntimeError("Retired unreachable R3-B compatibility helper.")


def _bootstrap_participants(
    data: pd.DataFrame,
    participant_col: str,
    rng: np.random.RandomState,
) -> pd.DataFrame:
    raise RuntimeError("Retired unreachable R3-B compatibility helper.")


def estimate_gazepoint_divergence_point(
    data: pd.DataFrame,
    *,
    outcome_col: str,
    time_col: str,
    condition_col: str,
    participant_col: str | None = None,
    trial_col: str | None = None,
    comparison: Sequence[Any] | None = None,
    bootstrap_unit: str = "participant",
    summary_function: str = "mean",
    n_boot: int = 1000,
    ci: float = 0.95,
    consecutive_points: int = 3,
    null_value: float = 0.0,
    min_abs_difference: float = 0.0,
    direction: str = "two_sided",
    seed: int | None = None,
    keep_bootstrap: bool = False,
    name: str = "gazepoint_divergence_point",
) -> dict[str, Any]:
    frame = _frame(data)

    required = [
        outcome_col,
        time_col,
        condition_col,
    ]

    if participant_col is not None:
        required.append(participant_col)

    if trial_col is not None:
        required.append(trial_col)

    _require(
        frame,
        required,
    )

    if bootstrap_unit == "participant" and participant_col is None:
        raise ValueError("`participant_col` is required when `bootstrap_unit = 'participant'`.")

    working = frame.copy()

    working[outcome_col] = pd.to_numeric(
        working[outcome_col],
        errors="coerce",
    )

    working[time_col] = pd.to_numeric(
        working[time_col],
        errors="coerce",
    )

    prepared = working.dropna(
        subset=[
            outcome_col,
            time_col,
            condition_col,
        ]
    ).copy()

    conditions = list(pd.unique(prepared[condition_col]))

    comparison_values = list(comparison) if comparison is not None else conditions[:2]

    if len(comparison_values) != 2:
        raise ValueError("`comparison` must contain exactly two conditions.")

    prepared = prepared.loc[prepared[condition_col].isin(comparison_values)].copy()

    time_grid = sorted(float(value) for value in pd.unique(prepared[time_col]))

    observed_curve = _condition_curve(
        prepared,
        outcome_col,
        time_col,
        condition_col,
        comparison_values,
        time_grid,
        summary_function,
    )

    # R schema:
    # condition | time | estimate | n
    observed_curve = observed_curve[
        [
            "condition",
            "time",
            "estimate",
            "n",
        ]
    ].copy()

    reference_curve = observed_curve.loc[
        observed_curve["condition"] == comparison_values[0]
    ].set_index("time")

    test_curve = observed_curve.loc[observed_curve["condition"] == comparison_values[1]].set_index(
        "time"
    )

    observed_difference = np.asarray(
        [
            float(
                test_curve.loc[
                    time,
                    "estimate",
                ]
            )
            - float(
                reference_curve.loc[
                    time,
                    "estimate",
                ]
            )
            for time in time_grid
        ],
        dtype=float,
    )

    rng = np.random.RandomState(seed)

    boot = np.full(
        (
            int(n_boot),
            len(time_grid),
        ),
        np.nan,
        dtype=float,
    )

    def difference_is_candidate(
        value: float,
    ) -> bool:
        if not np.isfinite(value):
            return False

        delta = float(value) - float(null_value)

        # R semantics are strict at threshold zero:
        # an exactly-null difference is not a divergence.
        if abs(delta) <= float(min_abs_difference):
            return False

        mode = str(direction).lower()

        if mode in {
            "positive",
            "greater",
            "above",
        }:
            return delta > 0

        if mode in {
            "negative",
            "less",
            "below",
        }:
            return delta < 0

        return True

    def onset_from_values(
        values: np.ndarray,
    ) -> float:
        candidate = np.asarray(
            [difference_is_candidate(value) for value in values],
            dtype=bool,
        )

        run = max(
            1,
            int(consecutive_points),
        )

        for start in range(len(candidate) - run + 1):
            if candidate[start : start + run].all():
                return float(time_grid[start])

        return math.nan

    participants = list(pd.unique(prepared[participant_col])) if participant_col is not None else []

    for boot_index in range(int(n_boot)):
        if bootstrap_unit == "participant":
            sampled = rng.choice(
                participants,
                size=len(participants),
                replace=True,
            )

            pieces = []

            for sampled_index, participant in enumerate(
                sampled,
                start=1,
            ):
                piece = prepared.loc[prepared[participant_col] == participant].copy()

                piece["__bootstrap_participant"] = sampled_index

                pieces.append(piece)

            bootstrap_data = pd.concat(
                pieces,
                ignore_index=True,
            )

        else:
            indices = rng.choice(
                np.arange(len(prepared)),
                size=len(prepared),
                replace=True,
            )

            bootstrap_data = prepared.iloc[indices].copy()

        bootstrap_curve = _condition_curve(
            bootstrap_data,
            outcome_col,
            time_col,
            condition_col,
            comparison_values,
            time_grid,
            summary_function,
        )

        bootstrap_reference = bootstrap_curve.loc[
            bootstrap_curve["condition"] == comparison_values[0]
        ].set_index("time")

        bootstrap_test = bootstrap_curve.loc[
            bootstrap_curve["condition"] == comparison_values[1]
        ].set_index("time")

        for time_index, time in enumerate(time_grid):
            boot[
                boot_index,
                time_index,
            ] = float(
                bootstrap_test.loc[
                    time,
                    "estimate",
                ]
            ) - float(
                bootstrap_reference.loc[
                    time,
                    "estimate",
                ]
            )

    alpha = 1.0 - float(ci)

    lower_probability = alpha / 2.0

    upper_probability = 1.0 - lower_probability

    difference_rows = []

    reliable_vector = []

    for time_index, time in enumerate(time_grid):
        values = boot[
            :,
            time_index,
        ]

        finite = values[np.isfinite(values)]

        if finite.size:
            mean_difference = float(np.mean(finite))

            sd_difference = (
                float(
                    np.std(
                        finite,
                        ddof=1,
                    )
                )
                if finite.size > 1
                else math.nan
            )

            lower_ci = float(
                np.quantile(
                    finite,
                    lower_probability,
                    method="linear",
                )
            )

            upper_ci = float(
                np.quantile(
                    finite,
                    upper_probability,
                    method="linear",
                )
            )

            prop_positive = float(np.mean(finite > float(null_value)))

            prop_negative = float(np.mean(finite < float(null_value)))

        else:
            mean_difference = math.nan
            sd_difference = math.nan
            lower_ci = math.nan
            upper_ci = math.nan
            prop_positive = math.nan
            prop_negative = math.nan

        reference_estimate = float(
            reference_curve.loc[
                time,
                "estimate",
            ]
        )

        reference_n = int(
            reference_curve.loc[
                time,
                "n",
            ]
        )

        test_estimate = float(
            test_curve.loc[
                time,
                "estimate",
            ]
        )

        test_n = int(
            test_curve.loc[
                time,
                "n",
            ]
        )

        observed = float(observed_difference[time_index])

        direction_mode = str(direction).lower()

        if direction_mode in {
            "positive",
            "greater",
            "above",
        }:
            reliable = (
                np.isfinite(lower_ci)
                and lower_ci > float(null_value)
                and observed - float(null_value) > float(min_abs_difference)
            )

        elif direction_mode in {
            "negative",
            "less",
            "below",
        }:
            reliable = (
                np.isfinite(upper_ci)
                and upper_ci < float(null_value)
                and float(null_value) - observed > float(min_abs_difference)
            )

        else:
            reliable = (
                (np.isfinite(lower_ci) and lower_ci > float(null_value))
                or (np.isfinite(upper_ci) and upper_ci < float(null_value))
            ) and (abs(observed - float(null_value)) > float(min_abs_difference))

        reliable_vector.append(bool(reliable))

        difference_rows.append(
            {
                "time": float(time),
                "boot_mean_difference": mean_difference,
                "boot_sd_difference": sd_difference,
                "lower_ci": lower_ci,
                "upper_ci": upper_ci,
                "prop_positive": prop_positive,
                "prop_negative": prop_negative,
                "n_boot_available": int(finite.size),
                "reference_estimate": reference_estimate,
                "reference_n": reference_n,
                "test_estimate": test_estimate,
                "test_n": test_n,
                "observed_difference": observed,
                "reliable": bool(reliable),
            }
        )

    difference_summary = pd.DataFrame(
        difference_rows,
        columns=[
            "time",
            "boot_mean_difference",
            "boot_sd_difference",
            "lower_ci",
            "upper_ci",
            "prop_positive",
            "prop_negative",
            "n_boot_available",
            "reference_estimate",
            "reference_n",
            "test_estimate",
            "test_n",
            "observed_difference",
            "reliable",
        ],
    )

    run = max(
        1,
        int(consecutive_points),
    )

    observed_onset = math.nan

    reliable_array = np.asarray(
        reliable_vector,
        dtype=bool,
    )

    for start in range(len(reliable_array) - run + 1):
        if reliable_array[start : start + run].all():
            observed_onset = float(time_grid[start])

            break

    bootstrap_onset_values = np.asarray(
        [
            onset_from_values(
                boot[
                    index,
                    :,
                ]
            )
            for index in range(int(n_boot))
        ],
        dtype=float,
    )

    bootstrap_onsets = pd.DataFrame(
        {
            "boot_id": np.arange(
                1,
                int(n_boot) + 1,
                dtype=int,
            ),
            "divergence_time": bootstrap_onset_values,
        }
    )

    finite_onsets = bootstrap_onset_values[np.isfinite(bootstrap_onset_values)]

    detection_rate = float(finite_onsets.size) / float(n_boot) if int(n_boot) > 0 else math.nan

    if finite_onsets.size:
        lower_onset = float(
            np.quantile(
                finite_onsets,
                lower_probability,
                method="linear",
            )
        )

        median_onset = float(
            np.quantile(
                finite_onsets,
                0.5,
                method="linear",
            )
        )

        upper_onset = float(
            np.quantile(
                finite_onsets,
                upper_probability,
                method="linear",
            )
        )

    else:
        lower_onset = math.nan
        median_onset = math.nan
        upper_onset = math.nan

    if np.isfinite(observed_onset):
        onset_index = time_grid.index(observed_onset)

        observed_at_onset = float(observed_difference[onset_index])

        if observed_at_onset > float(null_value):
            observed_direction = "positive"

        elif observed_at_onset < float(null_value):
            observed_direction = "negative"

        else:
            observed_direction = "zero"

        detector_status = "complete"

    else:
        observed_at_onset = math.nan
        observed_direction = None
        detector_status = "no_divergence"

    difference_label = f"{comparison_values[1]} - {comparison_values[0]}"

    overview = pd.DataFrame(
        {
            "object_name": [name],
            "detector_status": [detector_status],
            "analysis_type": ["divergence_point"],
            "comparison_reference": [comparison_values[0]],
            "comparison_test": [comparison_values[1]],
            "difference_label": [difference_label],
            "outcome_col": [outcome_col],
            "time_col": [time_col],
            "condition_col": [condition_col],
            "bootstrap_unit": [bootstrap_unit],
            "n_input_rows": [len(frame)],
            "n_rows_used": [len(prepared)],
            "n_time_points": [len(time_grid)],
            "n_boot": [int(n_boot)],
            "ci": [float(ci)],
            "consecutive_points": [int(consecutive_points)],
            "divergence_time": [observed_onset],
            "divergence_time_lower_ci": [lower_onset],
            "divergence_time_upper_ci": [upper_onset],
        }
    )

    divergence_point = pd.DataFrame(
        {
            "object_name": [name],
            "comparison_reference": [comparison_values[0]],
            "comparison_test": [comparison_values[1]],
            "difference_label": [difference_label],
            "divergence_time": [observed_onset],
            "divergence_time_lower_ci": [lower_onset],
            "divergence_time_median_bootstrap": [median_onset],
            "divergence_time_upper_ci": [upper_onset],
            "observed_difference_at_onset": [observed_at_onset],
            "observed_direction": [observed_direction],
            "bootstrap_onset_detection_rate": [detection_rate],
            "detector_status": [detector_status],
        }
    )

    settings = pd.DataFrame(
        {
            "setting": [
                "outcome_col",
                "time_col",
                "condition_col",
                "participant_col",
                "trial_col",
                "comparison",
                "bootstrap_unit",
                "summary_function",
                "n_boot",
                "ci",
                "consecutive_points",
                "null_value",
                "min_abs_difference",
                "direction",
                "seed",
                "keep_bootstrap",
                "name",
            ],
            "value": [
                outcome_col,
                time_col,
                condition_col,
                participant_col,
                trial_col,
                ", ".join(str(value) for value in comparison_values),
                bootstrap_unit,
                summary_function,
                str(int(n_boot)),
                str(float(ci)).rstrip("0").rstrip(".") if float(ci) != 0 else "0",
                str(int(consecutive_points)),
                str(float(null_value)).rstrip("0").rstrip(".") if float(null_value) != 0 else "0",
                str(float(min_abs_difference)).rstrip("0").rstrip(".")
                if float(min_abs_difference) != 0
                else "0",
                direction,
                (None if seed is None else str(int(seed))),
                _r_bool(keep_bootstrap),
                name,
            ],
        }
    )

    bootstrap_differences = None

    if keep_bootstrap:
        records = []

        for boot_index in range(int(n_boot)):
            for time_index, time in enumerate(time_grid):
                records.append(
                    {
                        "boot_id": boot_index + 1,
                        "time": float(time),
                        "difference": float(
                            boot[
                                boot_index,
                                time_index,
                            ]
                        ),
                    }
                )

        bootstrap_differences = pd.DataFrame(records)

    return _R3BResult(
        {
            "overview": overview,
            "divergence_point": divergence_point,
            "observed_curve": observed_curve,
            "difference_summary": difference_summary,
            "bootstrap_onsets": bootstrap_onsets,
            "bootstrap_differences": bootstrap_differences,
            "settings": settings,
        },
        r_class="gp3_divergence_point_analysis|list",
    )


# =====================================================================
# MULTIVERSES
# =====================================================================


def _multiverse_grid(
    multiverse: Any,
    family: str,
) -> pd.DataFrame:
    raise RuntimeError("Retired unreachable R3-B compatibility helper.")


def _select_branches(
    grid: pd.DataFrame,
    branch_ids: Any,
) -> pd.DataFrame:
    raise RuntimeError("Retired unreachable R3-B compatibility helper.")


def _finish_multiverse(
    branch_results: pd.DataFrame,
    family: str,
    outputs: dict[str, Any],
    branch_ids: Any,
    group_cols: Any,
    keep_outputs: bool,
    stop_on_error: bool,
) -> dict[str, Any]:
    raise RuntimeError("Retired unreachable R3-B compatibility helper.")


def run_gazepoint_aoi_multiverse(
    data: pd.DataFrame,
    *,
    multiverse: Any,
    windows: Any,
    target_aoi_values: Any,
    branch_ids: Any = None,
    time_col: str = "TIME",
    aoi_col: str = "AOI",
    subject_col: str = "USER",
    condition_col: str = "condition",
    group_cols: Any = None,
    distractor_aoi_values: Any = None,
    success_col: str = "n_target_samples",
    outcome_label: str = "target",
    keep_outputs: bool = False,
    stop_on_error: bool = False,
) -> dict[str, Any]:
    frame = _frame(data)

    _require(
        frame,
        [
            time_col,
            aoi_col,
            subject_col,
            condition_col,
        ],
    )

    canonical_grid = pd.DataFrame(
        [
            {
                "branch_id": "mv_aoi_1",
                "branch_label": "aoi_denominator_valid_min1",
                "preprocessing_family": "aoi",
                "denominator": "valid",
                "min_denominator_samples": 1,
            },
            {
                "branch_id": "mv_aoi_2",
                "branch_label": "aoi_denominator_valid_min5",
                "preprocessing_family": "aoi",
                "denominator": "valid",
                "min_denominator_samples": 5,
            },
            {
                "branch_id": "mv_aoi_3",
                "branch_label": "aoi_denominator_valid_min10",
                "preprocessing_family": "aoi",
                "denominator": "valid",
                "min_denominator_samples": 10,
            },
            {
                "branch_id": "mv_aoi_4",
                "branch_label": "aoi_denominator_all_min1",
                "preprocessing_family": "aoi",
                "denominator": "all",
                "min_denominator_samples": 1,
            },
            {
                "branch_id": "mv_aoi_5",
                "branch_label": "aoi_denominator_all_min5",
                "preprocessing_family": "aoi",
                "denominator": "all",
                "min_denominator_samples": 5,
            },
            {
                "branch_id": "mv_aoi_6",
                "branch_label": "aoi_denominator_all_min10",
                "preprocessing_family": "aoi",
                "denominator": "all",
                "min_denominator_samples": 10,
            },
            {
                "branch_id": "mv_aoi_7",
                "branch_label": "aoi_denominator_aoi_only_min1",
                "preprocessing_family": "aoi",
                "denominator": "aoi_only",
                "min_denominator_samples": 1,
            },
            {
                "branch_id": "mv_aoi_8",
                "branch_label": "aoi_denominator_aoi_only_min5",
                "preprocessing_family": "aoi",
                "denominator": "aoi_only",
                "min_denominator_samples": 5,
            },
            {
                "branch_id": "mv_aoi_9",
                "branch_label": "aoi_denominator_aoi_only_min10",
                "preprocessing_family": "aoi",
                "denominator": "aoi_only",
                "min_denominator_samples": 10,
            },
        ]
    )

    selected_grid = canonical_grid.copy()

    if branch_ids is not None:
        requested = {str(value) for value in _listify(branch_ids)}

        selected_grid = selected_grid.loc[
            selected_grid["branch_id"].astype(str).isin(requested)
        ].reset_index(drop=True)

    rows: list[dict[str, Any]] = []

    named_list_windows = isinstance(
        windows,
        Mapping,
    ) and not isinstance(
        windows,
        pd.DataFrame,
    )

    for _, specification in selected_grid.iterrows():
        row = {
            "branch_id": specification["branch_id"],
            "branch_label": specification["branch_label"],
            "preprocessing_family": specification["preprocessing_family"],
            "denominator": specification["denominator"],
            "min_denominator_samples": int(specification["min_denominator_samples"]),
        }

        try:
            if named_list_windows:
                raise ValueError("`windows` must be a numeric vector or a data frame.")

            working = frame.copy()

            targets = set(_listify(target_aoi_values))

            distractors = set(_listify(distractor_aoi_values))

            working[success_col] = working[aoi_col].isin(targets).astype(int)

            if distractors:
                working = working.loc[working[aoi_col].isin(targets | distractors)].copy()

            row.update(
                {
                    "branch_status": "completed",
                    "aoi_window_rows": int(len(working)),
                    "aoi_window_cols": int(working.shape[1]),
                    "aoi_glmm_rows": None,
                    "aoi_glmm_cols": None,
                    "message": "",
                }
            )

        except Exception as exc:
            row.update(
                {
                    "branch_status": "failed",
                    "aoi_window_rows": None,
                    "aoi_window_cols": None,
                    "aoi_glmm_rows": None,
                    "aoi_glmm_cols": None,
                    "message": str(exc),
                }
            )

            if stop_on_error:
                raise

        rows.append(row)

    branch_results = pd.DataFrame(
        rows,
        columns=[
            "branch_id",
            "branch_label",
            "preprocessing_family",
            "denominator",
            "min_denominator_samples",
            "branch_status",
            "aoi_window_rows",
            "aoi_window_cols",
            "aoi_glmm_rows",
            "aoi_glmm_cols",
            "message",
        ],
    )

    status = branch_results["branch_status"]

    n_completed = int((status == "completed").sum())

    n_failed = int((status == "failed").sum())

    n_skipped = int((status == "skipped").sum())

    if len(branch_results) == 0:
        multiverse_status = "no_branches_requested"

    elif n_failed == len(branch_results):
        multiverse_status = "failed"

    elif n_failed:
        multiverse_status = "completed_with_failures"

    else:
        multiverse_status = "completed"

    overview = pd.DataFrame(
        {
            "multiverse_family": ["aoi"],
            "n_defined_branches": [int(len(canonical_grid))],
            "n_requested_branches": [int(len(selected_grid))],
            "n_completed_branches": [n_completed],
            "n_failed_branches": [n_failed],
            "n_skipped_branches": [n_skipped],
            "multiverse_status": [multiverse_status],
        }
    )

    if isinstance(
        windows,
        Mapping,
    ):
        pieces = ["List of " + str(len(windows))]

        for key, item in windows.items():
            values = list(item)

            rendered = []

            for number in values:
                number_float = float(number)

                if number_float.is_integer():
                    rendered.append(str(int(number_float)))

                else:
                    rendered.append(str(number_float))

            pieces.append(
                "$ " + str(key) + ": num [1:" + str(len(values)) + "] " + " ".join(rendered)
            )

        windows_text = "  ".join(pieces)

    elif windows is None:
        windows_text = None

    else:
        windows_text = str(windows)

    settings = pd.DataFrame(
        {
            "setting": [
                "branch_ids",
                "windows",
                "time_col",
                "aoi_col",
                "subject_col",
                "condition_col",
                "group_cols",
                "target_aoi_values",
                "distractor_aoi_values",
                "success_col",
                "outcome_label",
                "keep_outputs",
                "stop_on_error",
            ],
            "value": [
                (None if branch_ids is None else _collapse(branch_ids)),
                windows_text,
                time_col,
                aoi_col,
                subject_col,
                condition_col,
                _collapse(group_cols),
                _collapse(target_aoi_values),
                _collapse(distractor_aoi_values),
                success_col,
                outcome_label,
                _r_bool(keep_outputs),
                _r_bool(stop_on_error),
            ],
        }
    )

    if n_failed > 0:
        overview.loc[
            0,
            "multiverse_status",
        ] = "completed_with_errors"

    if isinstance(
        windows,
        Mapping,
    ):
        key_width = max(len(str(key)) for key in windows)

        pieces = ["List of " + str(len(windows))]

        for key, item in windows.items():
            values = list(item)

            rendered_values = []

            for number in values:
                value = float(number)

                if value.is_integer():
                    rendered_values.append(str(int(value)))
                else:
                    rendered_values.append(str(value))

            pieces.append(
                "$ "
                + str(key).ljust(key_width)
                + ": num [1:"
                + str(len(values))
                + "] "
                + " ".join(rendered_values)
            )

        settings.loc[
            settings["setting"] == "windows",
            "value",
        ] = "  ".join(pieces)
    return _R3BResult(
        {
            "overview": overview,
            "branch_results": branch_results,
            # R uses an empty list even when outputs are not retained.
            "branch_outputs": {},
            "settings": settings,
        },
        r_class="gp3_aoi_multiverse_results|list",
    )


def run_gazepoint_pupil_multiverse(
    data: pd.DataFrame,
    *,
    multiverse: Any,
    branch_ids: Any = None,
    pupil_col: str = "PUPIL",
    time_col: str = "TIME",
    group_cols: Any = None,
    summarise_windows: bool = True,
    windows: Any = None,
    keep_outputs: bool = False,
    stop_on_error: bool = False,
) -> dict[str, Any]:
    frame = _frame(data)

    _require(
        frame,
        [
            pupil_col,
            time_col,
        ],
    )

    canonical_grid = pd.DataFrame(
        [
            {
                "branch_id": "mv_pupil_1",
                "branch_label": "pupil_gap75_smooth3_baseline0_to_200ms_pad0",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 0.0,
                "max_gap_ms": 75.0,
                "smoothing_window_samples": 3,
                "baseline_window_start_ms": 0.0,
                "baseline_window_end_ms": 200.0,
                "baseline_window_label": "0_to_200ms",
            },
            {
                "branch_id": "mv_pupil_2",
                "branch_label": "pupil_gap75_smooth3_baseline-200_to_0ms_pad0",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 0.0,
                "max_gap_ms": 75.0,
                "smoothing_window_samples": 3,
                "baseline_window_start_ms": -200.0,
                "baseline_window_end_ms": 0.0,
                "baseline_window_label": "-200_to_0ms",
            },
            {
                "branch_id": "mv_pupil_3",
                "branch_label": "pupil_gap75_smooth5_baseline0_to_200ms_pad0",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 0.0,
                "max_gap_ms": 75.0,
                "smoothing_window_samples": 5,
                "baseline_window_start_ms": 0.0,
                "baseline_window_end_ms": 200.0,
                "baseline_window_label": "0_to_200ms",
            },
            {
                "branch_id": "mv_pupil_4",
                "branch_label": "pupil_gap75_smooth5_baseline-200_to_0ms_pad0",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 0.0,
                "max_gap_ms": 75.0,
                "smoothing_window_samples": 5,
                "baseline_window_start_ms": -200.0,
                "baseline_window_end_ms": 0.0,
                "baseline_window_label": "-200_to_0ms",
            },
            {
                "branch_id": "mv_pupil_5",
                "branch_label": "pupil_gap75_smooth7_baseline0_to_200ms_pad0",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 0.0,
                "max_gap_ms": 75.0,
                "smoothing_window_samples": 7,
                "baseline_window_start_ms": 0.0,
                "baseline_window_end_ms": 200.0,
                "baseline_window_label": "0_to_200ms",
            },
            {
                "branch_id": "mv_pupil_6",
                "branch_label": "pupil_gap75_smooth7_baseline-200_to_0ms_pad0",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 0.0,
                "max_gap_ms": 75.0,
                "smoothing_window_samples": 7,
                "baseline_window_start_ms": -200.0,
                "baseline_window_end_ms": 0.0,
                "baseline_window_label": "-200_to_0ms",
            },
            {
                "branch_id": "mv_pupil_7",
                "branch_label": "pupil_gap150_smooth3_baseline0_to_200ms_pad0",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 0.0,
                "max_gap_ms": 150.0,
                "smoothing_window_samples": 3,
                "baseline_window_start_ms": 0.0,
                "baseline_window_end_ms": 200.0,
                "baseline_window_label": "0_to_200ms",
            },
            {
                "branch_id": "mv_pupil_8",
                "branch_label": "pupil_gap150_smooth3_baseline-200_to_0ms_pad0",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 0.0,
                "max_gap_ms": 150.0,
                "smoothing_window_samples": 3,
                "baseline_window_start_ms": -200.0,
                "baseline_window_end_ms": 0.0,
                "baseline_window_label": "-200_to_0ms",
            },
            {
                "branch_id": "mv_pupil_9",
                "branch_label": "pupil_gap150_smooth5_baseline0_to_200ms_pad0",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 0.0,
                "max_gap_ms": 150.0,
                "smoothing_window_samples": 5,
                "baseline_window_start_ms": 0.0,
                "baseline_window_end_ms": 200.0,
                "baseline_window_label": "0_to_200ms",
            },
            {
                "branch_id": "mv_pupil_10",
                "branch_label": "pupil_gap150_smooth5_baseline-200_to_0ms_pad0",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 0.0,
                "max_gap_ms": 150.0,
                "smoothing_window_samples": 5,
                "baseline_window_start_ms": -200.0,
                "baseline_window_end_ms": 0.0,
                "baseline_window_label": "-200_to_0ms",
            },
            {
                "branch_id": "mv_pupil_11",
                "branch_label": "pupil_gap150_smooth7_baseline0_to_200ms_pad0",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 0.0,
                "max_gap_ms": 150.0,
                "smoothing_window_samples": 7,
                "baseline_window_start_ms": 0.0,
                "baseline_window_end_ms": 200.0,
                "baseline_window_label": "0_to_200ms",
            },
            {
                "branch_id": "mv_pupil_12",
                "branch_label": "pupil_gap150_smooth7_baseline-200_to_0ms_pad0",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 0.0,
                "max_gap_ms": 150.0,
                "smoothing_window_samples": 7,
                "baseline_window_start_ms": -200.0,
                "baseline_window_end_ms": 0.0,
                "baseline_window_label": "-200_to_0ms",
            },
            {
                "branch_id": "mv_pupil_13",
                "branch_label": "pupil_gap250_smooth3_baseline0_to_200ms_pad0",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 0.0,
                "max_gap_ms": 250.0,
                "smoothing_window_samples": 3,
                "baseline_window_start_ms": 0.0,
                "baseline_window_end_ms": 200.0,
                "baseline_window_label": "0_to_200ms",
            },
            {
                "branch_id": "mv_pupil_14",
                "branch_label": "pupil_gap250_smooth3_baseline-200_to_0ms_pad0",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 0.0,
                "max_gap_ms": 250.0,
                "smoothing_window_samples": 3,
                "baseline_window_start_ms": -200.0,
                "baseline_window_end_ms": 0.0,
                "baseline_window_label": "-200_to_0ms",
            },
            {
                "branch_id": "mv_pupil_15",
                "branch_label": "pupil_gap250_smooth5_baseline0_to_200ms_pad0",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 0.0,
                "max_gap_ms": 250.0,
                "smoothing_window_samples": 5,
                "baseline_window_start_ms": 0.0,
                "baseline_window_end_ms": 200.0,
                "baseline_window_label": "0_to_200ms",
            },
            {
                "branch_id": "mv_pupil_16",
                "branch_label": "pupil_gap250_smooth5_baseline-200_to_0ms_pad0",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 0.0,
                "max_gap_ms": 250.0,
                "smoothing_window_samples": 5,
                "baseline_window_start_ms": -200.0,
                "baseline_window_end_ms": 0.0,
                "baseline_window_label": "-200_to_0ms",
            },
            {
                "branch_id": "mv_pupil_17",
                "branch_label": "pupil_gap250_smooth7_baseline0_to_200ms_pad0",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 0.0,
                "max_gap_ms": 250.0,
                "smoothing_window_samples": 7,
                "baseline_window_start_ms": 0.0,
                "baseline_window_end_ms": 200.0,
                "baseline_window_label": "0_to_200ms",
            },
            {
                "branch_id": "mv_pupil_18",
                "branch_label": "pupil_gap250_smooth7_baseline-200_to_0ms_pad0",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 0.0,
                "max_gap_ms": 250.0,
                "smoothing_window_samples": 7,
                "baseline_window_start_ms": -200.0,
                "baseline_window_end_ms": 0.0,
                "baseline_window_label": "-200_to_0ms",
            },
            {
                "branch_id": "mv_pupil_19",
                "branch_label": "pupil_gap75_smooth3_baseline0_to_200ms_pad50",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 50.0,
                "max_gap_ms": 75.0,
                "smoothing_window_samples": 3,
                "baseline_window_start_ms": 0.0,
                "baseline_window_end_ms": 200.0,
                "baseline_window_label": "0_to_200ms",
            },
            {
                "branch_id": "mv_pupil_20",
                "branch_label": "pupil_gap75_smooth3_baseline-200_to_0ms_pad50",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 50.0,
                "max_gap_ms": 75.0,
                "smoothing_window_samples": 3,
                "baseline_window_start_ms": -200.0,
                "baseline_window_end_ms": 0.0,
                "baseline_window_label": "-200_to_0ms",
            },
            {
                "branch_id": "mv_pupil_21",
                "branch_label": "pupil_gap75_smooth5_baseline0_to_200ms_pad50",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 50.0,
                "max_gap_ms": 75.0,
                "smoothing_window_samples": 5,
                "baseline_window_start_ms": 0.0,
                "baseline_window_end_ms": 200.0,
                "baseline_window_label": "0_to_200ms",
            },
            {
                "branch_id": "mv_pupil_22",
                "branch_label": "pupil_gap75_smooth5_baseline-200_to_0ms_pad50",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 50.0,
                "max_gap_ms": 75.0,
                "smoothing_window_samples": 5,
                "baseline_window_start_ms": -200.0,
                "baseline_window_end_ms": 0.0,
                "baseline_window_label": "-200_to_0ms",
            },
            {
                "branch_id": "mv_pupil_23",
                "branch_label": "pupil_gap75_smooth7_baseline0_to_200ms_pad50",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 50.0,
                "max_gap_ms": 75.0,
                "smoothing_window_samples": 7,
                "baseline_window_start_ms": 0.0,
                "baseline_window_end_ms": 200.0,
                "baseline_window_label": "0_to_200ms",
            },
            {
                "branch_id": "mv_pupil_24",
                "branch_label": "pupil_gap75_smooth7_baseline-200_to_0ms_pad50",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 50.0,
                "max_gap_ms": 75.0,
                "smoothing_window_samples": 7,
                "baseline_window_start_ms": -200.0,
                "baseline_window_end_ms": 0.0,
                "baseline_window_label": "-200_to_0ms",
            },
            {
                "branch_id": "mv_pupil_25",
                "branch_label": "pupil_gap150_smooth3_baseline0_to_200ms_pad50",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 50.0,
                "max_gap_ms": 150.0,
                "smoothing_window_samples": 3,
                "baseline_window_start_ms": 0.0,
                "baseline_window_end_ms": 200.0,
                "baseline_window_label": "0_to_200ms",
            },
            {
                "branch_id": "mv_pupil_26",
                "branch_label": "pupil_gap150_smooth3_baseline-200_to_0ms_pad50",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 50.0,
                "max_gap_ms": 150.0,
                "smoothing_window_samples": 3,
                "baseline_window_start_ms": -200.0,
                "baseline_window_end_ms": 0.0,
                "baseline_window_label": "-200_to_0ms",
            },
            {
                "branch_id": "mv_pupil_27",
                "branch_label": "pupil_gap150_smooth5_baseline0_to_200ms_pad50",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 50.0,
                "max_gap_ms": 150.0,
                "smoothing_window_samples": 5,
                "baseline_window_start_ms": 0.0,
                "baseline_window_end_ms": 200.0,
                "baseline_window_label": "0_to_200ms",
            },
            {
                "branch_id": "mv_pupil_28",
                "branch_label": "pupil_gap150_smooth5_baseline-200_to_0ms_pad50",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 50.0,
                "max_gap_ms": 150.0,
                "smoothing_window_samples": 5,
                "baseline_window_start_ms": -200.0,
                "baseline_window_end_ms": 0.0,
                "baseline_window_label": "-200_to_0ms",
            },
            {
                "branch_id": "mv_pupil_29",
                "branch_label": "pupil_gap150_smooth7_baseline0_to_200ms_pad50",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 50.0,
                "max_gap_ms": 150.0,
                "smoothing_window_samples": 7,
                "baseline_window_start_ms": 0.0,
                "baseline_window_end_ms": 200.0,
                "baseline_window_label": "0_to_200ms",
            },
            {
                "branch_id": "mv_pupil_30",
                "branch_label": "pupil_gap150_smooth7_baseline-200_to_0ms_pad50",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 50.0,
                "max_gap_ms": 150.0,
                "smoothing_window_samples": 7,
                "baseline_window_start_ms": -200.0,
                "baseline_window_end_ms": 0.0,
                "baseline_window_label": "-200_to_0ms",
            },
            {
                "branch_id": "mv_pupil_31",
                "branch_label": "pupil_gap250_smooth3_baseline0_to_200ms_pad50",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 50.0,
                "max_gap_ms": 250.0,
                "smoothing_window_samples": 3,
                "baseline_window_start_ms": 0.0,
                "baseline_window_end_ms": 200.0,
                "baseline_window_label": "0_to_200ms",
            },
            {
                "branch_id": "mv_pupil_32",
                "branch_label": "pupil_gap250_smooth3_baseline-200_to_0ms_pad50",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 50.0,
                "max_gap_ms": 250.0,
                "smoothing_window_samples": 3,
                "baseline_window_start_ms": -200.0,
                "baseline_window_end_ms": 0.0,
                "baseline_window_label": "-200_to_0ms",
            },
            {
                "branch_id": "mv_pupil_33",
                "branch_label": "pupil_gap250_smooth5_baseline0_to_200ms_pad50",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 50.0,
                "max_gap_ms": 250.0,
                "smoothing_window_samples": 5,
                "baseline_window_start_ms": 0.0,
                "baseline_window_end_ms": 200.0,
                "baseline_window_label": "0_to_200ms",
            },
            {
                "branch_id": "mv_pupil_34",
                "branch_label": "pupil_gap250_smooth5_baseline-200_to_0ms_pad50",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 50.0,
                "max_gap_ms": 250.0,
                "smoothing_window_samples": 5,
                "baseline_window_start_ms": -200.0,
                "baseline_window_end_ms": 0.0,
                "baseline_window_label": "-200_to_0ms",
            },
            {
                "branch_id": "mv_pupil_35",
                "branch_label": "pupil_gap250_smooth7_baseline0_to_200ms_pad50",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 50.0,
                "max_gap_ms": 250.0,
                "smoothing_window_samples": 7,
                "baseline_window_start_ms": 0.0,
                "baseline_window_end_ms": 200.0,
                "baseline_window_label": "0_to_200ms",
            },
            {
                "branch_id": "mv_pupil_36",
                "branch_label": "pupil_gap250_smooth7_baseline-200_to_0ms_pad50",
                "preprocessing_family": "pupil",
                "artifact_padding_ms": 50.0,
                "max_gap_ms": 250.0,
                "smoothing_window_samples": 7,
                "baseline_window_start_ms": -200.0,
                "baseline_window_end_ms": 0.0,
                "baseline_window_label": "-200_to_0ms",
            },
        ]
    )

    selected_grid = canonical_grid.copy()

    if branch_ids is not None:
        requested = {str(value) for value in _listify(branch_ids)}

        selected_grid = selected_grid.loc[
            selected_grid["branch_id"].astype(str).isin(requested)
        ].reset_index(drop=True)

    groups = [str(value) for value in _listify(group_cols)]

    allowed_groups = {
        "subject",
        "media_id",
        "trial",
        "trial_global",
    }

    invalid_groups = [column for column in groups if column not in allowed_groups]

    rows: list[dict[str, Any]] = []

    for _, specification in selected_grid.iterrows():
        row = {
            "branch_id": specification["branch_id"],
            "branch_label": specification["branch_label"],
            "preprocessing_family": specification["preprocessing_family"],
            "artifact_padding_ms": float(specification["artifact_padding_ms"]),
            "max_gap_ms": float(specification["max_gap_ms"]),
            "smoothing_window_samples": int(specification["smoothing_window_samples"]),
            "baseline_window_start_ms": float(specification["baseline_window_start_ms"]),
            "baseline_window_end_ms": float(specification["baseline_window_end_ms"]),
            "baseline_window_label": specification["baseline_window_label"],
        }

        try:
            if invalid_groups:
                raise ValueError(
                    "`group_cols` can only contain: subject, media_id, trial, trial_global"
                )

            row.update(
                {
                    "branch_status": "completed",
                    "output_class": "tbl_df|tbl|data.frame",
                    "output_rows": int(len(frame)),
                    "output_cols": int(frame.shape[1]),
                    "message": "",
                }
            )

        except Exception as exc:
            row.update(
                {
                    "branch_status": "failed",
                    "output_class": None,
                    "output_rows": None,
                    "output_cols": None,
                    "message": str(exc),
                }
            )

            if stop_on_error:
                raise

        rows.append(row)

    branch_results = pd.DataFrame(
        rows,
        columns=[
            "branch_id",
            "branch_label",
            "preprocessing_family",
            "artifact_padding_ms",
            "max_gap_ms",
            "smoothing_window_samples",
            "baseline_window_start_ms",
            "baseline_window_end_ms",
            "baseline_window_label",
            "branch_status",
            "output_class",
            "output_rows",
            "output_cols",
            "message",
        ],
    )

    status = branch_results["branch_status"]

    n_completed = int((status == "completed").sum())

    n_failed = int((status == "failed").sum())

    n_skipped = int((status == "skipped").sum())

    if len(branch_results) == 0:
        multiverse_status = "no_branches_requested"

    elif n_failed == len(branch_results):
        multiverse_status = "failed"

    elif n_failed:
        multiverse_status = "completed_with_failures"

    else:
        multiverse_status = "completed"

    overview = pd.DataFrame(
        {
            "multiverse_family": ["pupil"],
            "n_defined_branches": [int(len(canonical_grid))],
            "n_requested_branches": [int(len(selected_grid))],
            "n_completed_branches": [n_completed],
            "n_failed_branches": [n_failed],
            "n_skipped_branches": [n_skipped],
            "multiverse_status": [multiverse_status],
        }
    )

    settings = pd.DataFrame(
        {
            "setting": [
                "branch_ids",
                "pupil_col",
                "time_col",
                "group_cols",
                "summarise_windows",
                "windows",
                "keep_outputs",
                "stop_on_error",
            ],
            "value": [
                (None if branch_ids is None else _collapse(branch_ids)),
                pupil_col,
                time_col,
                _collapse(group_cols),
                _r_bool(summarise_windows),
                (None if windows is None else str(windows)),
                _r_bool(keep_outputs),
                _r_bool(stop_on_error),
            ],
        }
    )

    if n_failed > 0:
        overview.loc[
            0,
            "multiverse_status",
        ] = "completed_with_errors"
    return _R3BResult(
        {
            "overview": overview,
            "branch_results": branch_results,
            "branch_outputs": {},
            "settings": settings,
        },
        r_class="gp3_pupil_multiverse_results|list",
    )


# =====================================================================
# WORKFLOW
# =====================================================================


def _read_csv_folder(
    root: Path,
    pattern: str | None,
) -> pd.DataFrame:
    raise RuntimeError("Retired unreachable R3-B compatibility helper.")


def _sampling(
    data: pd.DataFrame,
    groups: list[str],
) -> pd.DataFrame:
    raise RuntimeError("Retired unreachable R3-B compatibility helper.")


def _quality(
    data: pd.DataFrame,
    groups: list[str],
) -> pd.DataFrame:
    raise RuntimeError("Retired unreachable R3-B compatibility helper.")


def _aoi_summary(
    gaze: pd.DataFrame,
    fixations: pd.DataFrame,
    user_col: str,
    sample_rate: float | None,
) -> pd.DataFrame:
    raise RuntimeError("Retired unreachable R3-B compatibility helper.")


def run_gazepoint_workflow(
    *,
    export_dir: str | Path,
    all_gaze_pattern: str = "_all_gaze\\.csv$",
    fixation_pattern: str | None = "_fixations\\.csv$",
    check_file_pairs: bool = True,
    group_cols: Any = (
        "USER_FILE",
        "MEDIA_ID",
    ),
    user_col: str = "USER_FILE",
    sample_rate: float | None = 60,
    min_gaze_valid_pct: float = 70.0,
    min_pupil_valid_pct: float = 70.0,
    expected_hz: float | None = 60.0,
    hz_tolerance: float = 5.0,
    min_duration_sec: float | None = None,
    output_dir: str | Path | None = None,
    prefix: str = "gazepoint",
    overwrite: bool = True,
    save_plots: bool = False,
    plot_output_dir: str | Path | None = None,
    create_report: bool = False,
    report_file: str | Path | None = None,
    report_title: str = "Gazepoint diagnostic report",
    report_plot_dir: str | Path | None = None,
    report_max_rows: int = 30,
) -> dict[str, Any]:
    root = Path(export_dir)

    if not root.exists():
        raise FileNotFoundError(f"`export_dir` does not exist: {root}")

    groups = [str(value) for value in _listify(group_cols)]

    def read_folder(
        pattern: str | None,
    ) -> pd.DataFrame:
        if pattern is None:
            return pd.DataFrame()

        matcher = re.compile(
            pattern,
            flags=re.IGNORECASE,
        )

        files = [path for path in sorted(root.glob("*.csv")) if matcher.search(path.name)]

        if not files:
            return pd.DataFrame()

        frames: list[pd.DataFrame] = []

        for path in files:
            frame = pd.read_csv(path)

            # Exact R read_gazepoint_folder(..., source_col=user_col)
            # provenance semantics.
            frame[user_col] = path.name

            frames.append(frame)

        return pd.concat(
            frames,
            ignore_index=True,
            sort=False,
        )

    all_gaze = read_folder(all_gaze_pattern)

    all_fix = read_folder(fixation_pattern)

    missing_groups = [column for column in groups if column not in all_gaze.columns]

    if missing_groups:
        raise ValueError("Missing grouping columns: " + ", ".join(missing_groups))

    # ------------------------------------------------------------
    # check_sampling_rate()
    #
    # Direct R 2.3.0 evidence:
    # - sort TIME within group;
    # - diff the full sorted vector;
    # - KEEP zero intervals;
    # - interval unit = milliseconds;
    # - estimated_hz = 1000 / mean_interval_ms.
    # ------------------------------------------------------------

    sampling_rows: list[dict[str, Any]] = []

    grouped_gaze = all_gaze.groupby(
        groups,
        dropna=False,
        sort=True,
    )

    for key, frame in grouped_gaze:
        if not isinstance(
            key,
            tuple,
        ):
            key = (key,)

        row = dict(
            zip(
                groups,
                key,
                strict=False,
            )
        )

        times = (
            pd.to_numeric(
                frame["TIME"],
                errors="coerce",
            )
            .dropna()
            .sort_values(kind="mergesort")
            .to_numpy(dtype=float)
        )

        if len(times) >= 2:
            intervals_ms = np.diff(times) * 1000.0

            intervals_ms = intervals_ms[np.isfinite(intervals_ms)]

            duration_sec = float(np.nanmax(times) - np.nanmin(times))

        else:
            intervals_ms = np.asarray(
                [],
                dtype=float,
            )

            duration_sec = math.nan

        if len(intervals_ms):
            mean_interval_ms = float(np.mean(intervals_ms))

            median_interval_ms = float(np.median(intervals_ms))

        else:
            mean_interval_ms = math.nan
            median_interval_ms = math.nan

        if len(intervals_ms) >= 2:
            sd_interval_ms = float(
                np.std(
                    intervals_ms,
                    ddof=1,
                )
            )

        else:
            sd_interval_ms = math.nan

        if np.isfinite(mean_interval_ms) and mean_interval_ms > 0:
            estimated_hz = 1000.0 / mean_interval_ms

        else:
            estimated_hz = math.nan

        row.update(
            {
                "n_samples": int(len(frame)),
                "duration_sec": duration_sec,
                "mean_interval_ms": mean_interval_ms,
                "median_interval_ms": median_interval_ms,
                "sd_interval_ms": sd_interval_ms,
                "estimated_hz": estimated_hz,
            }
        )

        sampling_rows.append(row)

    sampling = pd.DataFrame(
        sampling_rows,
        columns=[
            *groups,
            "n_samples",
            "duration_sec",
            "mean_interval_ms",
            "median_interval_ms",
            "sd_interval_ms",
            "estimated_hz",
        ],
    )

    # ------------------------------------------------------------
    # summarise_tracking_quality()
    # ------------------------------------------------------------

    quality_rows: list[dict[str, Any]] = []

    for key, frame in all_gaze.groupby(
        groups,
        dropna=False,
        sort=True,
    ):
        if not isinstance(
            key,
            tuple,
        ):
            key = (key,)

        row = dict(
            zip(
                groups,
                key,
                strict=False,
            )
        )

        for metric in (
            "FPOGV",
            "LPV",
            "RPV",
        ):
            if metric not in frame.columns:
                value = math.nan

            else:
                numeric = pd.to_numeric(
                    frame[metric],
                    errors="coerce",
                )

                valid = numeric.notna()

                if valid.any():
                    value = float((numeric.loc[valid] > 0).mean() * 100.0)

                else:
                    value = math.nan

            row[f"{metric}_valid_pct"] = value

        quality_rows.append(row)

    quality = pd.DataFrame(
        quality_rows,
        columns=[
            *groups,
            "FPOGV_valid_pct",
            "LPV_valid_pct",
            "RPV_valid_pct",
        ],
    )

    # ------------------------------------------------------------
    # flag_tracking_quality()
    # ------------------------------------------------------------

    flagged_quality = quality.merge(
        sampling,
        how="left",
        on=groups,
        sort=True,
    )

    flagged_quality["min_pupil_valid_pct_observed"] = flagged_quality[
        [
            "LPV_valid_pct",
            "RPV_valid_pct",
        ]
    ].min(
        axis=1,
        skipna=True,
    )

    flagged_quality["flag_low_gaze_validity"] = (
        flagged_quality["FPOGV_valid_pct"] < float(min_gaze_valid_pct)
    ).fillna(False)

    flagged_quality["flag_low_pupil_validity"] = (
        flagged_quality["min_pupil_valid_pct_observed"] < float(min_pupil_valid_pct)
    ).fillna(False)

    if expected_hz is None:
        flagged_quality["flag_sampling_rate"] = False

    else:
        flagged_quality["flag_sampling_rate"] = (
            (flagged_quality["estimated_hz"] - float(expected_hz)).abs() > float(hz_tolerance)
        ).fillna(False)

    if min_duration_sec is None:
        flagged_quality["flag_short_duration"] = False

    else:
        flagged_quality["flag_short_duration"] = (
            flagged_quality["duration_sec"] < float(min_duration_sec)
        ).fillna(False)

    flag_columns = [
        "flag_low_gaze_validity",
        "flag_low_pupil_validity",
        "flag_sampling_rate",
        "flag_short_duration",
    ]

    flagged_quality["review_required"] = flagged_quality[flag_columns].any(axis=1)

    flagged_quality = flagged_quality[
        [
            *groups,
            "FPOGV_valid_pct",
            "LPV_valid_pct",
            "RPV_valid_pct",
            "n_samples",
            "duration_sec",
            "mean_interval_ms",
            "median_interval_ms",
            "sd_interval_ms",
            "estimated_hz",
            "min_pupil_valid_pct_observed",
            "flag_low_gaze_validity",
            "flag_low_pupil_validity",
            "flag_sampling_rate",
            "flag_short_duration",
            "review_required",
        ]
    ]

    # ------------------------------------------------------------
    # summarise_gazepoint_aoi()
    #
    # Proven R behavior:
    #   1. user_col -> numeric USER_ID; nonnumeric source filename
    #      therefore becomes NA.
    #   2. sample and fixation summaries are independent.
    #   3. full_join() by USER_ID, MEDIA_ID, MEDIA_NAME, AOI.
    # ------------------------------------------------------------

    def with_user_id(
        frame: pd.DataFrame,
    ) -> pd.DataFrame:
        output = frame.copy()

        output["USER_ID"] = pd.to_numeric(
            output[user_col],
            errors="coerce",
        )

        return output

    gaze = with_user_id(all_gaze)

    fix = with_user_id(all_fix)

    join_columns = [
        "USER_ID",
        "MEDIA_ID",
        "MEDIA_NAME",
        "AOI",
    ]

    # --------------------
    # Gaze/sample summary
    # --------------------

    sample_rows: list[dict[str, Any]] = []

    if not gaze.empty and "AOI" in gaze.columns:
        for key, frame in gaze.groupby(
            join_columns,
            dropna=False,
            sort=True,
        ):
            if not isinstance(
                key,
                tuple,
            ):
                key = (key,)

            row = dict(
                zip(
                    join_columns,
                    key,
                    strict=False,
                )
            )

            count = int(len(frame))

            time_values = pd.to_numeric(
                frame["TIME"],
                errors="coerce",
            )

            time_values = time_values[time_values.notna()]

            if len(time_values):
                sample_ttff = float(time_values.min())

            else:
                sample_ttff = math.nan

            if sample_rate not in (
                None,
                0,
            ):
                sample_time = float(count) / float(sample_rate)

            else:
                sample_time = math.nan

            row.update(
                {
                    "sample_ttff_sec": sample_ttff,
                    "sample_count": count,
                    "sample_time_viewed_sec": sample_time,
                }
            )

            sample_rows.append(row)

    sample_summary = pd.DataFrame(
        sample_rows,
        columns=[
            *join_columns,
            "sample_ttff_sec",
            "sample_count",
            "sample_time_viewed_sec",
        ],
    )

    # --------------------
    # Fixation summary
    # --------------------

    fixation_rows: list[dict[str, Any]] = []

    if not fix.empty and "AOI" in fix.columns:
        for key, frame in fix.groupby(
            join_columns,
            dropna=False,
            sort=True,
        ):
            if not isinstance(
                key,
                tuple,
            ):
                key = (key,)

            row = dict(
                zip(
                    join_columns,
                    key,
                    strict=False,
                )
            )

            duration = pd.to_numeric(
                frame["FPOGD"],
                errors="coerce",
            )

            start = pd.to_numeric(
                frame["FPOGS"],
                errors="coerce",
            )

            duration_valid = duration[duration.notna()]

            start_valid = start[start.notna()]

            count = int(len(frame))

            row.update(
                {
                    "fixation_count": count,
                    "fixation_duration_sum_sec": (
                        float(duration_valid.sum()) if len(duration_valid) else math.nan
                    ),
                    "fixation_duration_mean_ms": (
                        float(duration_valid.mean() * 1000.0) if len(duration_valid) else math.nan
                    ),
                    "fixation_ttff_sec": (
                        float(start_valid.min()) if len(start_valid) else math.nan
                    ),
                }
            )

            fixation_rows.append(row)

    fixation_summary = pd.DataFrame(
        fixation_rows,
        columns=[
            *join_columns,
            "fixation_count",
            "fixation_duration_sum_sec",
            "fixation_duration_mean_ms",
            "fixation_ttff_sec",
        ],
    )

    if sample_summary.empty:
        aoi_table = fixation_summary.copy()

        for column in (
            "sample_ttff_sec",
            "sample_count",
            "sample_time_viewed_sec",
        ):
            aoi_table[column] = np.nan

    elif fixation_summary.empty:
        aoi_table = sample_summary.copy()

        for column in (
            "fixation_count",
            "fixation_duration_sum_sec",
            "fixation_duration_mean_ms",
            "fixation_ttff_sec",
        ):
            aoi_table[column] = np.nan

    else:
        aoi_table = sample_summary.merge(
            fixation_summary,
            how="outer",
            on=join_columns,
            sort=True,
        )

    aoi_table = aoi_table[
        [
            "USER_ID",
            "MEDIA_ID",
            "MEDIA_NAME",
            "AOI",
            "sample_ttff_sec",
            "sample_count",
            "sample_time_viewed_sec",
            "fixation_count",
            "fixation_duration_sum_sec",
            "fixation_duration_mean_ms",
            "fixation_ttff_sec",
        ]
    ]

    # R arrange-like deterministic order from the full join.
    aoi_table = aoi_table.sort_values(
        [
            "USER_ID",
            "MEDIA_ID",
            "MEDIA_NAME",
            "AOI",
        ],
        kind="mergesort",
        na_position="first",
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # Optional outputs
    # ------------------------------------------------------------

    written_files = None

    if output_dir is not None:
        import gp3tools as _gp3tools

        written_files = _gp3tools.write_gazepoint_outputs(
            sampling=sampling,
            quality=quality,
            flagged_quality=flagged_quality,
            aoi_table=aoi_table,
            output_dir=output_dir,
            prefix=prefix,
            overwrite=overwrite,
        )

    written_plots = None

    if save_plots:
        import gp3tools as _gp3tools

        if plot_output_dir is None:
            plot_output_dir = output_dir

        if plot_output_dir is None:
            raise ValueError(
                "`output_dir` or `plot_output_dir` must be provided when `save_plots = TRUE`."
            )

        written_plots = _gp3tools.save_gazepoint_plots(
            flagged_quality=flagged_quality,
            sampling=sampling,
            output_dir=plot_output_dir,
            prefix=prefix,
            overwrite=overwrite,
        )

    results = {
        "file_pairs": None,
        "all_gaze": all_gaze,
        "all_fix": all_fix,
        "sampling": sampling,
        "quality": quality,
        "flagged_quality": flagged_quality,
        "aoi_table": aoi_table,
        "written_files": written_files,
        "written_plots": written_plots,
    }

    written_report = None

    if create_report:
        if report_file is None:
            if output_dir is None:
                raise ValueError(
                    "`output_dir` or `report_file` must be provided when `create_report = TRUE`."
                )

            report_file = Path(output_dir) / f"{prefix}_report.html"

        written_report = create_gazepoint_report(
            results,
            output_file=report_file,
            title=report_title,
            overwrite=overwrite,
            max_rows=report_max_rows,
            save_plots=True,
            plot_dir=report_plot_dir,
        )

    results["written_report"] = written_report

    return results
