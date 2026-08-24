"""Master-table, sampling, tracking, missingness, and QC helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ._utils import (
    as_bool,
    attach_attrs,
    ensure_dataframe,
    finite_numeric,
    infer_column,
    normalize_group_cols,
    result_table,
    time_to_seconds,
)


def as_gazepoint_master(data, copy: bool = True) -> pd.DataFrame:
    """Coerce sample-level data to a standard Gazepoint master table."""
    df = ensure_dataframe(data, copy=copy)
    mapping = {}
    roles = {
        "subject": "subject",
        "trial": "trial_global",
        "time": "time",
        "x": "x",
        "y": "y",
        "pupil": "pupil",
        "aoi": "aoi_current",
        "condition": "condition",
        "media": "MEDIA_ID",
    }
    for role, canonical in roles.items():
        col = infer_column(df, role)
        if col is not None and canonical not in df.columns:
            mapping[col] = canonical
    df = df.rename(columns=mapping)
    return attach_attrs(df, gp3_class="gazepoint_master")


def create_gazepoint_master(data, **kwargs) -> pd.DataFrame:
    """Create a standardized sample-level master table."""
    df = as_gazepoint_master(data)
    if "sample_index" not in df.columns:
        df["sample_index"] = np.arange(len(df), dtype=int)
    if "subject" not in df.columns:
        source = df.get("USER_FILE")
        if source is not None:
            df["subject"] = (
                source.astype(str).str.extract(r"(\d+)", expand=False).fillna(source.astype(str))
            )
    if "trial_global" not in df.columns:
        media = df.get("MEDIA_ID")
        subject = df.get("subject")
        if media is not None and subject is not None:
            df["trial_global"] = subject.astype(str) + "::" + media.astype(str)
        else:
            df["trial_global"] = 1
    return attach_attrs(df, gp3_class="gazepoint_master")


def validate_gazepoint_master(
    data, required: tuple[str, ...] = ("subject", "time")
) -> dict[str, Any]:
    """Validate a master table and return an explicit pass/fail gate."""
    df = ensure_dataframe(data, copy=False)
    checks = []
    for col in required:
        checks.append(
            {
                "check": f"required:{col}",
                "passed": col in df.columns,
                "detail": "present" if col in df.columns else "missing",
            }
        )
    checks.extend(
        [
            {"check": "nonempty", "passed": len(df) > 0, "detail": f"n={len(df)}"},
            {
                "check": "unique_columns",
                "passed": not df.columns.duplicated().any(),
                "detail": "no duplicate names"
                if not df.columns.duplicated().any()
                else "duplicates present",
            },
        ]
    )
    table = pd.DataFrame(checks)
    return {
        "valid": bool(table["passed"].all()),
        "summary": result_table(
            n_rows=len(df),
            n_columns=df.shape[1],
            n_checks=len(table),
            n_failed=int((~table["passed"]).sum()),
        ),
        "checks": table,
    }


def audit_gazepoint_master(data) -> dict[str, pd.DataFrame]:
    """Create structural and signal-availability summaries for a master table."""
    df = ensure_dataframe(data, copy=False)
    overview = result_table(
        n_rows=len(df),
        n_columns=df.shape[1],
        n_subjects=df["subject"].nunique() if "subject" in df else np.nan,
        n_trials=df["trial_global"].nunique() if "trial_global" in df else np.nan,
    )
    missing = pd.DataFrame(
        {
            "column": df.columns,
            "n_missing": [int(df[c].isna().sum()) for c in df],
            "missing_prop": [float(df[c].isna().mean()) for c in df],
        }
    )
    out: dict[str, pd.DataFrame] = {"overview": overview, "missingness": missing}
    for label, col in (
        ("subject_summary", "subject"),
        ("media_summary", "MEDIA_ID"),
        ("aoi_summary", "aoi_current"),
    ):
        if col in df:
            out[label] = df.groupby(col, dropna=False).size().rename("n_samples").reset_index()
    pupil = infer_column(df, "pupil")
    if pupil:
        x = finite_numeric(df[pupil])
        out["pupil_summary"] = result_table(
            n=int(x.notna().sum()),
            mean=float(x.mean()),
            sd=float(x.std()),
            minimum=float(x.min()),
            maximum=float(x.max()),
        )
    xcol, ycol = infer_column(df, "x"), infer_column(df, "y")
    if xcol and ycol:
        out["coordinate_summary"] = result_table(
            x_min=float(pd.to_numeric(df[xcol], errors="coerce").min()),
            x_max=float(pd.to_numeric(df[xcol], errors="coerce").max()),
            y_min=float(pd.to_numeric(df[ycol], errors="coerce").min()),
            y_max=float(pd.to_numeric(df[ycol], errors="coerce").max()),
        )
    return out


def check_sampling_rate(
    data,
    time_col: str | None = None,
    group_cols=None,
    expected_hz: float = 60.0,
    tolerance_hz: float = 5.0,
) -> pd.DataFrame:
    """Estimate effective sampling rate from timestamp differences."""
    df = ensure_dataframe(data, copy=False)
    time_col = infer_column(df, "time", time_col, required=True)
    groups = normalize_group_cols(df, group_cols)
    rows = []
    iterator = [((), df)] if not groups else df.groupby(groups, dropna=False, sort=False)
    for key, frame in iterator:
        if groups and not isinstance(key, tuple):
            key = (key,)
        t = time_to_seconds(frame[time_col]).dropna().sort_values().to_numpy(float)
        diffs = np.diff(t)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        hz = float(1.0 / np.median(diffs)) if diffs.size else np.nan
        row = {c: v for c, v in zip(groups, key, strict=False)} if groups else {}
        row.update(
            n_samples=len(frame),
            sampling_hz=hz,
            expected_hz=float(expected_hz),
            deviation_hz=hz - expected_hz if np.isfinite(hz) else np.nan,
            within_tolerance=bool(np.isfinite(hz) and abs(hz - expected_hz) <= tolerance_hz),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def summarise_tracking_quality(
    data,
    validity_col: str | None = None,
    group_cols=None,
    x_col: str | None = None,
    y_col: str | None = None,
) -> pd.DataFrame:
    """Summarise usable gaze samples and tracking loss."""
    df = ensure_dataframe(data, copy=False)
    groups = normalize_group_cols(df, group_cols)
    validity_col = infer_column(df, "validity", validity_col)
    x_col = infer_column(df, "x", x_col)
    y_col = infer_column(df, "y", y_col)
    usable = pd.Series(True, index=df.index)
    if validity_col:
        invert = validity_col.lower() == "trackloss"
        usable &= as_bool(df[validity_col], invert_trackloss=invert)
    if x_col:
        usable &= pd.to_numeric(df[x_col], errors="coerce").notna()
    if y_col:
        usable &= pd.to_numeric(df[y_col], errors="coerce").notna()
    work = df.assign(_gp3_usable=usable)
    if groups:
        out = (
            work.groupby(groups, dropna=False)
            .agg(n_samples=("_gp3_usable", "size"), n_usable=("_gp3_usable", "sum"))
            .reset_index()
        )
    else:
        out = pd.DataFrame({"n_samples": [len(work)], "n_usable": [int(usable.sum())]})
    out["usable_prop"] = out["n_usable"] / out["n_samples"].replace(0, np.nan)
    out["trackloss_prop"] = 1 - out["usable_prop"]
    return out


def flag_tracking_quality(data, min_usable_prop: float = 0.8, **kwargs) -> pd.DataFrame:
    out = summarise_tracking_quality(data, **kwargs)
    out["quality_flag"] = np.where(out["usable_prop"] >= min_usable_prop, "pass", "flag")
    return out


def clean_gazepoint_by_trackloss(
    data, validity_col: str | None = None, drop: bool = True
) -> pd.DataFrame:
    df = ensure_dataframe(data)
    validity_col = infer_column(df, "validity", validity_col)
    if validity_col is None:
        return df
    invert = validity_col.lower() == "trackloss"
    valid = as_bool(df[validity_col], invert_trackloss=invert)
    return df.loc[valid].copy() if drop else df.assign(gp3_track_valid=valid)


def summarise_gazepoint_missingness(data, group_cols=None, columns=None) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    groups = normalize_group_cols(df, group_cols)
    columns = list(columns) if columns is not None else list(df.columns)
    rows = []
    iterator = [((), df)] if not groups else df.groupby(groups, dropna=False, sort=False)
    for key, frame in iterator:
        if groups and not isinstance(key, tuple):
            key = (key,)
        base = {c: v for c, v in zip(groups, key, strict=False)} if groups else {}
        for col in columns:
            if col not in frame:
                continue
            rows.append(
                {
                    **base,
                    "column": col,
                    "n": len(frame),
                    "n_missing": int(frame[col].isna().sum()),
                    "missing_prop": float(frame[col].isna().mean()),
                }
            )
    return pd.DataFrame(rows)


summarize_gazepoint_missingness = summarise_gazepoint_missingness


def audit_gazepoint_gaze_signal_quality(data, **kwargs) -> dict[str, pd.DataFrame]:
    return {
        "tracking": summarise_tracking_quality(data, **kwargs),
        "missingness": summarise_gazepoint_missingness(data),
    }


def audit_gazepoint_screen_bounds(
    data, x_col=None, y_col=None, width: float = 1.0, height: float = 1.0, normalized: bool = True
) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    x_col, y_col = (
        infer_column(df, "x", x_col, required=True),
        infer_column(df, "y", y_col, required=True),
    )
    x, y = finite_numeric(df[x_col]), finite_numeric(df[y_col])
    xmax, ymax = (1.0, 1.0) if normalized else (float(width), float(height))
    inside = x.between(0, xmax) & y.between(0, ymax)
    return result_table(
        n=len(df),
        n_finite=int((x.notna() & y.notna()).sum()),
        n_inside=int(inside.sum()),
        n_outside=int((x.notna() & y.notna() & ~inside).sum()),
        inside_prop=float(inside.mean()),
    )


def harmonize_gazepoint_screen_coordinates(
    data,
    x_col=None,
    y_col=None,
    width: float | None = None,
    height: float | None = None,
    output_x: str = "x_norm",
    output_y: str = "y_norm",
) -> pd.DataFrame:
    df = ensure_dataframe(data)
    x_col, y_col = (
        infer_column(df, "x", x_col, required=True),
        infer_column(df, "y", y_col, required=True),
    )
    x, y = finite_numeric(df[x_col]), finite_numeric(df[y_col])
    if width is None:
        width = 1.0 if x.max(skipna=True) <= 1.5 else float(x.max(skipna=True))
    if height is None:
        height = 1.0 if y.max(skipna=True) <= 1.5 else float(y.max(skipna=True))
    df[output_x] = x / width
    df[output_y] = y / height
    return df


def summarise_gazepoint_coordinate_coverage(
    data, x_col=None, y_col=None, group_cols=None
) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    x_col, y_col = (
        infer_column(df, "x", x_col, required=True),
        infer_column(df, "y", y_col, required=True),
    )
    groups = normalize_group_cols(df, group_cols)
    work = df.assign(_x=finite_numeric(df[x_col]), _y=finite_numeric(df[y_col]))
    if groups:
        return (
            work.groupby(groups, dropna=False)
            .agg(
                n_samples=("_x", "size"),
                n_xy=("_x", lambda s: int((s.notna() & work.loc[s.index, "_y"].notna()).sum())),
                x_min=("_x", "min"),
                x_max=("_x", "max"),
                y_min=("_y", "min"),
                y_max=("_y", "max"),
            )
            .reset_index()
        )
    return result_table(
        n_samples=len(work),
        n_xy=int((work._x.notna() & work._y.notna()).sum()),
        x_min=float(work._x.min()),
        x_max=float(work._x.max()),
        y_min=float(work._y.min()),
        y_max=float(work._y.max()),
    )


summarize_gazepoint_coordinate_coverage = summarise_gazepoint_coordinate_coverage


def audit_gazepoint_design_balance(data, group_cols=("subject", "condition")) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    cols = [c for c in group_cols if c in df]
    if not cols:
        return result_table(n_rows=len(df))
    return df.groupby(cols, dropna=False).size().rename("n").reset_index()


def audit_gazepoint_condition_quality_imbalance(data, condition_col=None, **kwargs) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    condition_col = infer_column(df, "condition", condition_col, required=True)
    return summarise_tracking_quality(df, group_cols=[condition_col], **kwargs)


def audit_gazepoint_post_exclusion_balance(
    data, excluded_col: str = "excluded", group_cols=("condition",)
) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    cols = [c for c in group_cols if c in df]
    if excluded_col in df:
        retained = df.loc[~as_bool(df[excluded_col])]
    else:
        retained = df
    return (
        retained.groupby(cols, dropna=False).size().rename("n_retained").reset_index()
        if cols
        else result_table(n_retained=len(retained))
    )


def audit_gazepoint_exclusion_flow(data, stages: list[str] | None = None) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    if stages is None:
        stages = [c for c in df.columns if c.lower().startswith(("exclude", "flag_", "excluded"))]
    rows = [{"stage": "input", "n": len(df)}]
    current = pd.Series(True, index=df.index)
    for stage in stages:
        current &= ~as_bool(df[stage])
        rows.append({"stage": stage, "n": int(current.sum())})
    return pd.DataFrame(rows)


def check_gazepoint_file_pairs(folder: str | Path) -> pd.DataFrame:
    root = Path(folder)
    files = [p.name for p in root.glob("*.csv")]
    users: dict[str, dict[str, bool]] = {}
    for name in files:
        m = pd.Series([name]).str.extract(r"User\s*([^_]+)", expand=False).iloc[0]
        if pd.isna(m):
            continue
        row = users.setdefault(str(m), {"all_gaze": False, "fixations": False})
        if "all_gaze" in name.lower():
            row["all_gaze"] = True
        if "fix" in name.lower():
            row["fixations"] = True
    return pd.DataFrame(
        [
            {"user": u, **v, "paired": bool(v["all_gaze"] and v["fixations"])}
            for u, v in sorted(users.items())
        ]
    )


def segment_gazepoint_task_phases(
    data, time_col=None, boundaries=None, labels=None, output_col: str = "phase"
) -> pd.DataFrame:
    df = ensure_dataframe(data)
    time_col = infer_column(df, "time", time_col, required=True)
    t = finite_numeric(df[time_col])
    if boundaries is None:
        q = t.quantile([0, 1 / 3, 2 / 3, 1]).to_numpy()
        boundaries = np.unique(q)
    boundaries = np.asarray(boundaries, dtype=float)
    if labels is None:
        labels = [f"phase_{i + 1}" for i in range(len(boundaries) - 1)]
    df[output_col] = pd.cut(t, boundaries, labels=labels, include_lowest=True)
    return df


def summarise_gazepoint_phase_coverage(data, phase_col="phase", group_cols=None) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    groups = normalize_group_cols(df, group_cols) + [phase_col]
    return df.groupby(groups, dropna=False).size().rename("n_samples").reset_index()


summarize_gazepoint_phase_coverage = summarise_gazepoint_phase_coverage


def collect_gazepoint_qc_summaries(data) -> dict[str, Any]:
    return {
        "master": audit_gazepoint_master(data),
        "tracking": summarise_tracking_quality(data),
        "missingness": summarise_gazepoint_missingness(data),
        "screen": audit_gazepoint_screen_bounds(data)
        if infer_column(ensure_dataframe(data, copy=False), "x")
        and infer_column(ensure_dataframe(data, copy=False), "y")
        else None,
    }


def summarise_gazepoint_qc_status(qc) -> pd.DataFrame:
    rows = []
    if isinstance(qc, dict):
        for name, value in qc.items():
            status = "available" if value is not None else "not_available"
            rows.append({"component": name, "status": status})
    return pd.DataFrame(rows)


summarize_gazepoint_qc_status = summarise_gazepoint_qc_status


def check_gazepoint_real_data_readiness(data) -> dict[str, Any]:
    validation = validate_gazepoint_master(as_gazepoint_master(data), required=("subject", "time"))
    df = ensure_dataframe(data, copy=False)
    has_gaze = infer_column(df, "x") is not None and infer_column(df, "y") is not None
    has_pupil = infer_column(df, "pupil") is not None or (
        infer_column(df, "left_pupil") and infer_column(df, "right_pupil")
    )
    checks = validation["checks"].copy()
    checks = pd.concat(
        [
            checks,
            pd.DataFrame(
                [
                    {
                        "check": "gaze_coordinates",
                        "passed": has_gaze,
                        "detail": "available" if has_gaze else "missing",
                    },
                    {
                        "check": "pupil_signal",
                        "passed": bool(has_pupil),
                        "detail": "available" if has_pupil else "missing",
                    },
                ]
            ),
        ],
        ignore_index=True,
    )
    return {"ready": bool(checks["passed"].all()), "checks": checks}


def recommend_gazepoint_exclusions(
    data,
    participant_col=None,
    trial_col=None,
    validity_col=None,
    x_col=None,
    y_col=None,
    pupil_col=None,
    artifact_col=None,
    min_trial_samples: int = 20,
    max_trial_missing_prop: float = 0.5,
    max_trial_artifact_prop: float = 0.5,
    min_participant_trials: int = 1,
    min_participant_valid_trials: int = 1,
    max_participant_missing_prop: float = 0.5,
    max_participant_artifact_prop: float = 0.5,
    require_both_gaze_coordinates: bool = True,
    name: str = "gazepoint_exclusions",
    **kwargs,
) -> dict[str, Any]:
    """Recommend exclusions without removing rows."""
    df = ensure_dataframe(data, copy=False)
    participant_col = infer_column(df, "subject", participant_col, required=True)
    trial_col = infer_column(df, "trial", trial_col) or "__gp3_trial"
    work = df.copy()
    if trial_col == "__gp3_trial":
        work[trial_col] = 1
    validity_col = infer_column(work, "validity", validity_col)
    x_col, y_col = infer_column(work, "x", x_col), infer_column(work, "y", y_col)
    pupil_col = infer_column(work, "pupil", pupil_col)
    usable = pd.Series(True, index=work.index)
    if validity_col:
        usable &= as_bool(work[validity_col], invert_trackloss=validity_col.lower() == "trackloss")
    coords = pd.Series(True, index=work.index)
    if x_col:
        coords &= finite_numeric(work[x_col]).notna()
    if y_col:
        coords &= finite_numeric(work[y_col]).notna()
    if require_both_gaze_coordinates:
        usable &= coords
    if pupil_col:
        usable &= finite_numeric(work[pupil_col]).notna()
    artifact = (
        as_bool(work[artifact_col])
        if artifact_col and artifact_col in work
        else pd.Series(False, index=work.index)
    )
    work = work.assign(_usable=usable, _artifact=artifact)
    trial = (
        work.groupby([participant_col, trial_col], dropna=False)
        .agg(
            n_samples=("_usable", "size"),
            usable_prop=("_usable", "mean"),
            artifact_prop=("_artifact", "mean"),
        )
        .reset_index()
    )
    trial["exclude"] = (
        (trial.n_samples < min_trial_samples)
        | ((1 - trial.usable_prop) > max_trial_missing_prop)
        | (trial.artifact_prop > max_trial_artifact_prop)
    )
    part = (
        trial.groupby(participant_col, dropna=False)
        .agg(n_trials=(trial_col, "size"), n_valid_trials=("exclude", lambda s: int((~s).sum())))
        .reset_index()
    )
    sample = (
        work.groupby(participant_col, dropna=False)
        .agg(
            missing_prop=("_usable", lambda s: float(1 - s.mean())),
            artifact_prop=("_artifact", "mean"),
        )
        .reset_index()
    )
    part = part.merge(sample, on=participant_col, how="left")
    part["exclude"] = (
        (part.n_trials < min_participant_trials)
        | (part.n_valid_trials < min_participant_valid_trials)
        | (part.missing_prop > max_participant_missing_prop)
        | (part.artifact_prop > max_participant_artifact_prop)
    )
    exclusion_table = pd.concat(
        [
            trial.loc[trial.exclude, [participant_col, trial_col]].assign(level="trial"),
            part.loc[part.exclude, [participant_col]].assign(
                **{trial_col: pd.NA}, level="participant"
            ),
        ],
        ignore_index=True,
    )
    return {
        "name": name,
        "overview": result_table(
            n_trial_exclusions=int(trial.exclude.sum()),
            n_participant_exclusions=int(part.exclude.sum()),
        ),
        "trial_recommendations": trial,
        "participant_recommendations": part,
        "exclusions": exclusion_table,
        "settings": kwargs
        | {
            "min_trial_samples": min_trial_samples,
            "max_trial_missing_prop": max_trial_missing_prop,
        },
    }


def audit_gazepoint_naming_consistency(names=None) -> dict[str, Any]:
    names = list(names or [])
    american = [x for x in names if x.startswith("summarize_")]
    british = [x for x in names if x.startswith("summarise_")]
    return {
        "summary": result_table(
            n_names=len(names), n_british=len(british), n_american_aliases=len(american)
        ),
        "british": british,
        "american_aliases": american,
    }


def gp3tools_naming_policy() -> dict[str, Any]:
    return {
        "canonical": "British English summarise_*",
        "compatibility": "Existing summarize_* aliases are retained",
        "rules": pd.DataFrame(
            [
                {"rule": "summary helpers", "canonical": "summarise_*"},
                {"rule": "legacy aliases", "canonical": "retain summarize_*"},
            ]
        ),
    }


def write_gazepoint_naming_audit(path, names=None) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    audit = audit_gazepoint_naming_consistency(names)
    audit["summary"].to_csv(out, index=False)
    return out
