"""AOI assignment, sequence, transition, and scanpath helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, Point, Polygon
from sklearn.cluster import AgglomerativeClustering

from ._compat import r_aliases
from ._utils import (
    collapse_consecutive,
    ensure_dataframe,
    finite_numeric,
    infer_column,
    normalize_group_cols,
    ordered_unique,
)
from .io import standardise_gazepoint_names


def add_gazepoint_aoi(
    data,
    x_col=None,
    y_col=None,
    aoi_geometry=None,
    output_col="aoi_current",
    outside_label="outside",
) -> pd.DataFrame:
    """Assign rectangular AOIs from a geometry table.

    ``aoi_geometry`` should contain name/aoi plus xmin, xmax, ymin, ymax columns.
    """
    df = ensure_dataframe(data)
    x_col = infer_column(df, "x", x_col, required=True)
    y_col = infer_column(df, "y", y_col, required=True)
    geom = (
        ensure_dataframe(aoi_geometry, copy=False) if aoi_geometry is not None else pd.DataFrame()
    )
    name_col = next((c for c in ("aoi", "name", "AOI", "label") if c in geom), None)
    required = {"xmin", "xmax", "ymin", "ymax"}
    if name_col is None or not required.issubset(geom.columns):
        raise ValueError("aoi_geometry must contain name/aoi and xmin,xmax,ymin,ymax columns")
    x = finite_numeric(df[x_col])
    y = finite_numeric(df[y_col])
    labels = pd.Series(outside_label, index=df.index, dtype="object")
    for _, row in geom.iterrows():
        mask = x.between(float(row.xmin), float(row.xmax)) & y.between(
            float(row.ymin), float(row.ymax)
        )
        labels.loc[mask] = row[name_col]
    labels.loc[x.isna() | y.isna()] = pd.NA
    df[output_col] = labels
    return df


def add_gazepoint_polygon_aoi(
    data, polygons, x_col=None, y_col=None, output_col="aoi_current", outside_label="outside"
) -> pd.DataFrame:
    df = ensure_dataframe(data)
    x_col = infer_column(df, "x", x_col, required=True)
    y_col = infer_column(df, "y", y_col, required=True)
    items = []
    if isinstance(polygons, dict):
        items = list(polygons.items())
    else:
        g = ensure_dataframe(polygons, copy=False)
        name_col = next((c for c in ("aoi", "name", "label") if c in g), None)
        poly_col = next((c for c in ("polygon", "vertices", "geometry") if c in g), None)
        if not name_col or not poly_col:
            raise ValueError("Polygon table must contain aoi/name and polygon/vertices columns")
        items = [(row[name_col], row[poly_col]) for _, row in g.iterrows()]
    shp = []
    for name, vertices in items:
        shp.append((name, vertices if isinstance(vertices, Polygon) else Polygon(vertices)))
    labels = []
    for xv, yv in zip(finite_numeric(df[x_col]), finite_numeric(df[y_col]), strict=False):
        if not np.isfinite(xv) or not np.isfinite(yv):
            labels.append(pd.NA)
            continue
        p = Point(float(xv), float(yv))
        hits = [name for name, poly in shp if poly.contains(p) or poly.touches(p)]
        labels.append(hits[-1] if hits else outside_label)
    df[output_col] = labels
    return df


def add_gazepoint_dynamic_aoi(
    data,
    aoi_data,
    time_col=None,
    x_col=None,
    y_col=None,
    aoi_time_col=None,
    output_col="aoi_current",
    tolerance=None,
) -> pd.DataFrame:
    """Assign time-varying rectangular AOIs using nearest-time geometry rows."""
    df = ensure_dataframe(data)
    geom = ensure_dataframe(aoi_data, copy=False)
    time_col = infer_column(df, "time", time_col, required=True)
    aoi_time_col = infer_column(geom, "time", aoi_time_col, required=True)
    x_col = infer_column(df, "x", x_col, required=True)
    y_col = infer_column(df, "y", y_col, required=True)
    name_col = next((c for c in ("aoi", "name", "label") if c in geom), None)
    if name_col is None or not {"xmin", "xmax", "ymin", "ymax"}.issubset(geom):
        raise ValueError("Dynamic AOI geometry requires aoi/name,xmin,xmax,ymin,ymax")
    left = df.copy().sort_values(time_col)
    right = geom.copy().sort_values(aoi_time_col)
    merged = pd.merge_asof(
        left,
        right,
        left_on=time_col,
        right_on=aoi_time_col,
        direction="nearest",
        tolerance=tolerance,
        suffixes=("", "_aoi"),
    )
    x = finite_numeric(merged[x_col])
    y = finite_numeric(merged[y_col])
    inside = x.between(merged.xmin, merged.xmax) & y.between(merged.ymin, merged.ymax)
    merged[output_col] = merged[name_col].where(inside, "outside")
    return merged.sort_index()


def audit_gazepoint_aoi_geometry(aoi_geometry) -> dict[str, Any]:
    g = ensure_dataframe(aoi_geometry, copy=False)
    issues = []
    required = ["xmin", "xmax", "ymin", "ymax"]
    missing = [c for c in required if c not in g]
    if missing:
        issues.append(f"missing columns: {missing}")
    if not missing:
        bad = ((g.xmin >= g.xmax) | (g.ymin >= g.ymax)).sum()
        if bad:
            issues.append(f"{int(bad)} invalid rectangles")
    return {
        "valid": not issues,
        "issues": issues,
        "summary": pd.DataFrame([{"n_aois": len(g), "n_issues": len(issues)}]),
    }


def audit_gazepoint_aoi_overlap(aoi_geometry) -> pd.DataFrame:
    g = ensure_dataframe(aoi_geometry, copy=False)
    name = next((c for c in ("aoi", "name", "label") if c in g), None)
    rows = []
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            a, b = g.iloc[i], g.iloc[j]
            x = max(0, min(a.xmax, b.xmax) - max(a.xmin, b.xmin))
            y = max(0, min(a.ymax, b.ymax) - max(a.ymin, b.ymin))
            area = x * y
            if area > 0:
                rows.append(
                    {
                        "aoi1": a[name] if name else i,
                        "aoi2": b[name] if name else j,
                        "overlap_area": area,
                    }
                )
    return pd.DataFrame(rows, columns=["aoi1", "aoi2", "overlap_area"])


def audit_gazepoint_aoi_screen_coverage(aoi_geometry, width=1.0, height=1.0) -> pd.DataFrame:
    g = ensure_dataframe(aoi_geometry, copy=False)
    total = float(width * height)
    areas = (g.xmax - g.xmin).clip(lower=0) * (g.ymax - g.ymin).clip(lower=0)
    return pd.DataFrame(
        [
            {
                "n_aois": len(g),
                "sum_aoi_area": float(areas.sum()),
                "screen_area": total,
                "nominal_coverage_prop": float(areas.sum() / total) if total else np.nan,
            }
        ]
    )


def audit_gazepoint_dynamic_aoi_coverage(data, aoi_col="aoi_current") -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    s = df[aoi_col] if aoi_col in df else pd.Series(pd.NA, index=df.index)
    return pd.DataFrame(
        [
            {
                "n_samples": len(df),
                "n_assigned": int(s.notna().sum()),
                "assigned_prop": float(s.notna().mean()),
                "n_outside": int(s.astype("string").eq("outside").sum()),
            }
        ]
    )


def audit_gazepoint_aoi_coding_matrix(data, aoi_col=None, group_cols=None) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    aoi_col = infer_column(df, "aoi", aoi_col, required=True)
    groups = normalize_group_cols(df, group_cols)
    return df.groupby(groups + [aoi_col], dropna=False).size().rename("n").reset_index()


def audit_gazepoint_aoi_margin_sensitivity(
    data, aoi_geometry, margins=(-0.02, 0, 0.02), x_col=None, y_col=None
) -> pd.DataFrame:
    g = ensure_dataframe(aoi_geometry)
    rows = []
    for m in margins:
        gm = g.copy()
        gm["xmin"] -= m
        gm["xmax"] += m
        gm["ymin"] -= m
        gm["ymax"] += m
        assigned = add_gazepoint_aoi(data, x_col=x_col, y_col=y_col, aoi_geometry=gm)
        rows.append(
            {
                "margin": m,
                "assigned_prop": float(assigned.aoi_current.notna().mean()),
                "outside_prop": float(assigned.aoi_current.astype("string").eq("outside").mean()),
            }
        )
    return pd.DataFrame(rows)


def summarise_aoi_samples(data, aoi_col=None, group_cols=None) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    aoi_col = infer_column(df, "aoi", aoi_col, required=True)
    groups = normalize_group_cols(df, group_cols)
    out = df.groupby(groups + [aoi_col], dropna=False).size().rename("n_samples").reset_index()
    totals = (
        out.groupby(groups, dropna=False).n_samples.transform("sum")
        if groups
        else out.n_samples.sum()
    )
    out["proportion"] = out.n_samples / totals
    return out


def summarise_gazepoint_aoi(
    data=None,
    aoi_col=None,
    group_cols=None,
    *,
    gaze_data=None,
    fixation_data=None,
    user_col="USER_FILE",
    sample_rate=60,
) -> pd.DataFrame:
    """Summarise Gazepoint AOI data.

    The historical Python sample-count interface remains available.
    Supplying ``gaze_data``/``fixation_data`` or passing two DataFrames
    positionally activates the R gp3tools v2.3.0 interface.
    """
    positional_r_mode = (
        isinstance(
            data,
            pd.DataFrame,
        )
        and isinstance(
            aoi_col,
            pd.DataFrame,
        )
        and gaze_data is None
        and fixation_data is None
    )

    r_mode = positional_r_mode or gaze_data is not None or fixation_data is not None

    if not r_mode:
        if data is None:
            raise TypeError("data is required for the Python interface")

        return summarise_aoi_samples(
            data,
            aoi_col=aoi_col,
            group_cols=group_cols,
        )

    if positional_r_mode:
        gaze_data = data
        fixation_data = aoi_col

    elif gaze_data is None:
        if data is None:
            raise TypeError("gaze_data is required")

        gaze_data = data

    if fixation_data is None:
        raise TypeError("fixation_data is required")

    gaze = ensure_dataframe(
        gaze_data,
        copy=False,
    )

    fix = ensure_dataframe(
        fixation_data,
        copy=False,
    )

    if not isinstance(user_col, str) or not user_col:
        raise ValueError("user_col must be a non-empty string")

    required_gaze = [
        user_col,
        "MEDIA_ID",
        "MEDIA_NAME",
        "AOI",
        "TIME",
    ]

    required_fix = [
        user_col,
        "MEDIA_ID",
        "MEDIA_NAME",
        "AOI",
        "FPOGD",
        "FPOGS",
    ]

    missing_gaze = [col for col in required_gaze if col not in gaze.columns]

    if missing_gaze:
        raise ValueError("Missing required columns in `gaze_data`: " + ", ".join(missing_gaze))

    missing_fix = [col for col in required_fix if col not in fix.columns]

    if missing_fix:
        raise ValueError("Missing required columns in `fixation_data`: " + ", ".join(missing_fix))

    def user_ids(frame):
        extracted = (
            frame[user_col]
            .astype("string")
            .str.extract(
                r"(\d+)",
                expand=False,
            )
        )

        return pd.to_numeric(
            extracted,
            errors="coerce",
        ).astype("Int64")

    gaze_work = gaze.loc[gaze["AOI"].notna() & gaze["AOI"].ne("")].copy()

    gaze_work["USER_ID"] = user_ids(gaze_work)

    gaze_work["_TIME"] = pd.to_numeric(
        gaze_work["TIME"],
        errors="coerce",
    )

    keys = [
        "USER_ID",
        "MEDIA_ID",
        "MEDIA_NAME",
        "AOI",
    ]

    sample_rows = []

    for key, part in gaze_work.groupby(
        keys,
        dropna=False,
        sort=False,
    ):
        values = part["_TIME"].dropna().to_numpy(float)

        sample_ttff = float(np.min(values)) if len(values) else np.inf

        row = dict(
            zip(
                keys,
                (
                    key
                    if isinstance(
                        key,
                        tuple,
                    )
                    else (key,)
                ),
                strict=True,
            )
        )

        row.update(
            {
                "sample_ttff_sec": sample_ttff,
                "sample_count": int(len(part)),
            }
        )

        sample_rows.append(row)

    sample_summary = pd.DataFrame(
        sample_rows,
        columns=[
            *keys,
            "sample_ttff_sec",
            "sample_count",
        ],
    )

    if len(sample_summary):
        sample_summary["sample_time_viewed_sec"] = sample_summary["sample_count"] / sample_rate
    else:
        sample_summary["sample_time_viewed_sec"] = pd.Series(dtype=float)

    fix_work = fix.loc[fix["AOI"].notna() & fix["AOI"].ne("")].copy()

    fix_work["USER_ID"] = user_ids(fix_work)

    fix_work["_FPOGD"] = pd.to_numeric(
        fix_work["FPOGD"],
        errors="coerce",
    )

    fix_work["_FPOGS"] = pd.to_numeric(
        fix_work["FPOGS"],
        errors="coerce",
    )

    fixation_rows = []

    for key, part in fix_work.groupby(
        keys,
        dropna=False,
        sort=False,
    ):
        duration = part["_FPOGD"]

        starts = part["_FPOGS"].dropna().to_numpy(float)

        fixation_ttff = float(np.min(starts)) if len(starts) else np.inf

        row = dict(
            zip(
                keys,
                (
                    key
                    if isinstance(
                        key,
                        tuple,
                    )
                    else (key,)
                ),
                strict=True,
            )
        )

        row.update(
            {
                "fixation_count": int(len(part)),
                "fixation_duration_sum_sec": float(duration.sum(skipna=True)),
                "fixation_duration_mean_ms": float(duration.mean(skipna=True) * 1000),
                "fixation_ttff_sec": fixation_ttff,
            }
        )

        fixation_rows.append(row)

    fixation_summary = pd.DataFrame(
        fixation_rows,
        columns=[
            *keys,
            "fixation_count",
            "fixation_duration_sum_sec",
            "fixation_duration_mean_ms",
            "fixation_ttff_sec",
        ],
    )

    out = sample_summary.merge(
        fixation_summary,
        on=keys,
        how="outer",
        sort=False,
    )

    if len(out):
        out = out.sort_values(
            [
                "USER_ID",
                "MEDIA_ID",
                "AOI",
            ],
            kind="stable",
            na_position="last",
        ).reset_index(drop=True)

    return out


def _sequence_frame(
    data,
    aoi_col=None,
    group_cols=None,
    time_col=None,
    include_missing=False,
    missing_label="missing",
    collapse_repeats=False,
):
    df = ensure_dataframe(data, copy=False)
    aoi_col = infer_column(df, "aoi", aoi_col, required=True)
    groups = normalize_group_cols(df, group_cols)
    work = df.copy()
    if time_col and time_col in work:
        work = work.sort_values(groups + [time_col] if groups else [time_col])
    if include_missing:
        work[aoi_col] = work[aoi_col].astype("string").fillna(missing_label)
    else:
        work = work.loc[work[aoi_col].notna()]
    rows = []
    iterator = [((), work)] if not groups else work.groupby(groups, dropna=False, sort=False)
    for key, frame in iterator:
        if groups and not isinstance(key, tuple):
            key = (key,)
        seq = frame[aoi_col].astype(str).tolist()
        seq = collapse_consecutive(seq) if collapse_repeats else seq
        row = {c: v for c, v in zip(groups, key, strict=False)} if groups else {}
        row["sequence"] = seq
        rows.append(row)
    return pd.DataFrame(rows), groups


def prepare_gazepoint_aoi_sequences(
    data, aoi_col=None, group_cols=None, time_col=None, collapse_repeats=True, **kwargs
) -> pd.DataFrame:
    out, _ = _sequence_frame(
        data, aoi_col, group_cols, time_col, collapse_repeats=collapse_repeats, **kwargs
    )
    out["sequence_string"] = out.sequence.map(lambda x: " > ".join(map(str, x)))
    out["sequence_length"] = out.sequence.map(len)
    return out


def summarise_gazepoint_aoi_entries(
    data, aoi_col=None, group_cols=None, time_col=None
) -> pd.DataFrame:
    seq, groups = _sequence_frame(data, aoi_col, group_cols, time_col, collapse_repeats=True)
    rows = []
    for _, r in seq.iterrows():
        base = {c: r[c] for c in groups}
        counts = Counter(r.sequence)
        for a, n in counts.items():
            rows.append({**base, "aoi": a, "n_entries": n})
    return pd.DataFrame(rows)


def summarise_gazepoint_aoi_transitions(
    data, aoi_col=None, group_cols=None, time_col=None, include_self=False
) -> pd.DataFrame:
    seq, groups = _sequence_frame(
        data, aoi_col, group_cols, time_col, collapse_repeats=not include_self
    )
    rows = []
    for _, r in seq.iterrows():
        base = {c: r[c] for c in groups}
        s = r.sequence
        for a, b in zip(s[:-1], s[1:], strict=False):
            if include_self or a != b:
                rows.append({**base, "from_aoi": a, "to_aoi": b})
    if not rows:
        return pd.DataFrame(columns=groups + ["from_aoi", "to_aoi", "n_transitions"])
    return (
        pd.DataFrame(rows)
        .groupby(groups + ["from_aoi", "to_aoi"], dropna=False)
        .size()
        .rename("n_transitions")
        .reset_index()
    )


def compute_transition_matrix(
    sequence=None,
    normalize=False,
    *,
    data=None,
    group_cols="MEDIA_ID",
    aoi_col="AOI",
    time_col="TIME",
    collapse_repeats=True,
) -> pd.DataFrame:
    """Compute AOI transitions.

    A sequence input retains the historical Python square-matrix
    interface. A DataFrame, either positionally or via ``data=``,
    activates the R gp3tools v2.3.0 long-table interface.
    """
    if data is not None and sequence is not None:
        raise TypeError("supply either sequence or data, not both")

    r_data = (
        data
        if data is not None
        else (
            sequence
            if isinstance(
                sequence,
                pd.DataFrame,
            )
            else None
        )
    )

    if r_data is None:
        values = list([] if sequence is None else sequence)

        states = ordered_unique(values)

        matrix = pd.DataFrame(
            0.0,
            index=states,
            columns=states,
        )

        for from_aoi, to_aoi in zip(
            values[:-1],
            values[1:],
            strict=False,
        ):
            matrix.loc[
                from_aoi,
                to_aoi,
            ] += 1

        if normalize:
            matrix = matrix.div(
                matrix.sum(axis=1).replace(
                    0,
                    np.nan,
                ),
                axis=0,
            ).fillna(0)

        matrix.index.name = "from_aoi"
        matrix.columns.name = "to_aoi"

        return matrix

    frame = standardise_gazepoint_names(r_data)

    if isinstance(
        group_cols,
        str,
    ):
        groups = [group_cols]

    else:
        try:
            groups = list(group_cols)
        except TypeError as exc:
            raise ValueError("group_cols must be a string or sequence of strings") from exc

        if not all(
            isinstance(
                col,
                str,
            )
            for col in groups
        ):
            raise ValueError("group_cols must contain strings")

    needed = [
        *groups,
        aoi_col,
        time_col,
    ]

    missing = [col for col in needed if col not in frame.columns]

    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    visits = frame.sort_values(
        [
            *groups,
            time_col,
        ]
        if groups
        else [time_col],
        kind="stable",
        na_position="last",
    )

    visits = visits.loc[visits[aoi_col].notna() & visits[aoi_col].ne("")].copy()

    if collapse_repeats:
        if groups:
            previous = visits.groupby(
                groups,
                dropna=False,
                sort=False,
            )[aoi_col].shift(1)
        else:
            previous = visits[aoi_col].shift(1)

        visits = visits.loc[previous.isna() | visits[aoi_col].ne(previous)].copy()

    visits["from"] = visits[aoi_col]

    if groups:
        visits["to"] = visits.groupby(
            groups,
            dropna=False,
            sort=False,
        )[aoi_col].shift(-1)
    else:
        visits["to"] = visits[aoi_col].shift(-1)

    visits = visits.loc[visits["to"].notna()]

    output_columns = [
        *groups,
        "from",
        "to",
        "n",
        "prob",
    ]

    if not len(visits):
        return pd.DataFrame(columns=output_columns)

    count_groups = [
        *groups,
        "from",
        "to",
    ]

    result = (
        visits.groupby(
            count_groups,
            dropna=False,
            sort=True,
        )
        .size()
        .rename("n")
        .reset_index()
    )

    probability_groups = [
        *groups,
        "from",
    ]

    result["prob"] = result["n"] / result.groupby(
        probability_groups,
        dropna=False,
        sort=False,
    )["n"].transform("sum")

    return result[output_columns].reset_index(drop=True)


def compute_gazepoint_aoi_transition_matrix(
    data=None,
    sequence=None,
    aoi_col=None,
    group_cols=None,
    time_col=None,
    normalize=False,
    include_self=True,
):
    if data is None:
        return compute_transition_matrix(sequence or [], normalize)
    seq, groups = _sequence_frame(
        data, aoi_col, group_cols, time_col, collapse_repeats=not include_self
    )
    if not groups:
        return compute_transition_matrix(seq.iloc[0].sequence if len(seq) else [], normalize)
    rows = []
    for _, r in seq.iterrows():
        m = compute_transition_matrix(r.sequence, normalize).stack().rename("value").reset_index()
        for c in groups:
            m[c] = r[c]
        rows.append(m)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compute_gazepoint_time_varying_transition_matrix(
    data, aoi_col=None, time_col=None, bin_width=500, group_cols=None, normalize=False
) -> pd.DataFrame:
    df = ensure_dataframe(data)
    time_col = infer_column(df, "time", time_col, required=True)
    t = finite_numeric(df[time_col])
    df["_time_bin"] = (np.floor(t / bin_width) * bin_width).astype("Int64")
    groups = normalize_group_cols(df, group_cols) + ["_time_bin"]
    out = compute_gazepoint_aoi_transition_matrix(
        df, aoi_col=aoi_col, group_cols=groups, time_col=time_col, normalize=normalize
    )
    return out.rename(columns={"_time_bin": "time_bin"})


def compute_gazepoint_aoi_entropy(
    data=None, sequence=None, aoi_col=None, group_cols=None, time_col=None, normalize=True
) -> pd.DataFrame:
    if data is None:
        seqdf = pd.DataFrame([{"sequence": list(sequence or [])}])
        groups = []
    else:
        seqdf, groups = _sequence_frame(data, aoi_col, group_cols, time_col)
    rows = []
    for _, r in seqdf.iterrows():
        seq = r.sequence
        counts = np.array(list(Counter(seq).values()), dtype=float)
        p = counts / counts.sum() if counts.sum() else np.array([])
        h = float(-(p * np.log2(p)).sum()) if len(p) else 0.0
        k = len(counts)
        hn = h / np.log2(k) if normalize and k > 1 else h
        base = {c: r[c] for c in groups}
        rows.append({**base, "n": len(seq), "n_states": k, "entropy": h, "normalized_entropy": hn})
    return pd.DataFrame(rows)


def compute_gazepoint_aoi_sequence_metrics(data=None, sequence=None, **kwargs) -> pd.DataFrame:
    comp = compute_gazepoint_sequence_complexity(data=data, sequence=sequence, **kwargs)
    return comp


def compute_gazepoint_sequence_complexity(
    data=None,
    sequence=None,
    aoi_col=None,
    group_cols=None,
    time_col=None,
    include_missing=False,
    missing_label="missing",
    collapse_repeats=False,
) -> pd.DataFrame:
    if data is None:
        seqdf = pd.DataFrame([{"sequence": list(sequence or [])}])
        groups = []
    else:
        seqdf, groups = _sequence_frame(
            data, aoi_col, group_cols, time_col, include_missing, missing_label, collapse_repeats
        )
    rows = []
    for _, r in seqdf.iterrows():
        s = r.sequence
        n = len(s)
        k = len(set(s))
        trans = sum(a != b for a, b in zip(s[:-1], s[1:], strict=False))
        ent = compute_gazepoint_aoi_entropy(sequence=s).iloc[0].normalized_entropy
        ttr = k / n if n else 0
        dens = trans / max(n - 1, 1) if n > 1 else 0
        ci = float(np.mean([ttr, dens, ent])) if n else 0
        base = {c: r[c] for c in groups}
        rows.append(
            {
                **base,
                "sequence_length": n,
                "unique_states": k,
                "entropy": float(ent),
                "transition_density": dens,
                "type_token_ratio": ttr,
                "complexity_index": ci,
            }
        )
    return pd.DataFrame(rows)


def _levenshtein(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def compute_gazepoint_sequence_distance(
    sequence_a, sequence_b, method="levenshtein", normalize=True
) -> float:
    a = list(sequence_a)
    b = list(sequence_b)
    if method in {"levenshtein", "edit"}:
        d = float(_levenshtein(a, b))
    elif method == "jaccard":
        d = 1 - len(set(a) & set(b)) / max(len(set(a) | set(b)), 1)
    else:
        raise ValueError("method must be levenshtein/edit or jaccard")
    return d / max(len(a), len(b), 1) if normalize and method in {"levenshtein", "edit"} else d


def compute_gazepoint_sequence_recurrence(sequence, lag=1) -> dict[str, float]:
    s = list(sequence)
    matches = sum(s[i] == s[i - lag] for i in range(lag, len(s))) if len(s) > lag else 0
    denom = max(len(s) - lag, 0)
    return {"lag": lag, "n_pairs": denom, "recurrence": matches / denom if denom else np.nan}


def compute_gazepoint_scanpath_geometry(
    data,
    x_col=None,
    y_col=None,
    time_col=None,
    group_cols=None,
    *,
    x=None,
    y=None,
    subject=None,
    trial=None,
    time=None,
    condition=None,
) -> pd.DataFrame:
    """Compute scanpath geometry.

    Supplying the R argument names ``x``, ``y``, ``subject`` and
    ``trial`` activates the gp3tools v2.3.0 interface. Otherwise the
    historical Python group-based summary is retained.
    """
    df = ensure_dataframe(
        data,
        copy=False,
    )

    r_mode = any(
        value is not None
        for value in (
            x,
            y,
            subject,
            trial,
            time,
            condition,
        )
    )

    if not r_mode:
        x_col = infer_column(
            df,
            "x",
            x_col,
            required=True,
        )
        y_col = infer_column(
            df,
            "y",
            y_col,
            required=True,
        )

        groups = normalize_group_cols(
            df,
            group_cols,
        )

        rows = []

        iterator = (
            [((), df)]
            if not groups
            else df.groupby(
                groups,
                dropna=False,
                sort=False,
            )
        )

        for key, part in iterator:
            if time_col and time_col in part:
                part = part.sort_values(
                    time_col,
                    kind="stable",
                )

            px = finite_numeric(part[x_col]).to_numpy(float)

            py = finite_numeric(part[y_col]).to_numpy(float)

            ok = np.isfinite(px) & np.isfinite(py)

            px = px[ok]
            py = py[ok]

            dist = np.sqrt(np.diff(px) ** 2 + np.diff(py) ** 2)

            base = (
                {
                    col: value
                    for col, value in zip(
                        groups,
                        (
                            key
                            if isinstance(
                                key,
                                tuple,
                            )
                            else (key,)
                        ),
                        strict=False,
                    )
                }
                if groups
                else {}
            )

            rows.append(
                {
                    **base,
                    "n_points": len(px),
                    "path_length": float(dist.sum()),
                    "mean_step": (float(dist.mean()) if len(dist) else 0),
                    "dispersion_x": (float(np.std(px)) if len(px) else np.nan),
                    "dispersion_y": (float(np.std(py)) if len(py) else np.nan),
                }
            )

        return pd.DataFrame(rows)

    required_arguments = {
        "x": x,
        "y": y,
        "subject": subject,
        "trial": trial,
    }

    missing_arguments = [
        name
        for name, value in required_arguments.items()
        if (not isinstance(value, str) or not value)
    ]

    if missing_arguments:
        raise ValueError("R-compatible scanpath geometry requires: " + ", ".join(missing_arguments))

    for optional_name, value in {
        "time": time,
        "condition": condition,
    }.items():
        if value is not None and (
            not isinstance(
                value,
                str,
            )
            or not value
        ):
            raise ValueError(f"{optional_name} must be None or a non-empty string")

    required_columns = [
        x,
        y,
        subject,
        trial,
    ]

    if time is not None:
        required_columns.append(time)

    if condition is not None:
        required_columns.append(condition)

    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError("Missing columns: " + ", ".join(missing_columns))

    rows = []

    for _, part in df.groupby(
        [subject, trial],
        dropna=True,
        sort=True,
    ):
        if time is not None:
            part = part.sort_values(
                time,
                kind="stable",
                na_position="last",
            )

        px = pd.to_numeric(
            part[x],
            errors="coerce",
        ).to_numpy(float)

        py = pd.to_numeric(
            part[y],
            errors="coerce",
        ).to_numpy(float)

        ok = np.isfinite(px) & np.isfinite(py)

        px = px[ok]
        py = py[ok]

        n_points = len(px)

        if n_points < 2:
            scanpath_length = np.nan
            straight_line_distance = np.nan
            efficiency = np.nan

        else:
            step_distance = np.sqrt(np.diff(px) ** 2 + np.diff(py) ** 2)

            scanpath_length = float(np.sum(step_distance))

            straight_line_distance = float(np.sqrt((px[-1] - px[0]) ** 2 + (py[-1] - py[0]) ** 2))

            efficiency = straight_line_distance / scanpath_length if scanpath_length > 0 else np.nan

        if n_points:
            centroid_x = float(np.mean(px))
            centroid_y = float(np.mean(py))

            spatial_dispersion = float(
                np.mean(np.sqrt((px - centroid_x) ** 2 + (py - centroid_y) ** 2))
            )

            if not np.isfinite(spatial_dispersion):
                spatial_dispersion = np.nan
        else:
            spatial_dispersion = np.nan

        if n_points < 3:
            convex_hull_area = np.nan
        else:
            convex_hull_area = float(
                MultiPoint(
                    np.column_stack(
                        [
                            px,
                            py,
                        ]
                    )
                ).convex_hull.area
            )

        row = {
            "subject": part.iloc[0][subject],
            "trial": part.iloc[0][trial],
            "n_points": int(n_points),
            "scanpath_length": scanpath_length,
            "straight_line_distance": straight_line_distance,
            "scanpath_efficiency": efficiency,
            "convex_hull_area": convex_hull_area,
            "spatial_dispersion": spatial_dispersion,
        }

        if condition is not None:
            row["condition"] = part.iloc[0][condition]

        rows.append(row)

    return pd.DataFrame(rows)


def _gp3_scanpath_r_group_id(key):
    if not isinstance(key, tuple):
        key = (key,)
    return "|".join("<NA>" if pd.isna(v) else str(v) for v in key)


def _gp3_scanpath_r_sequences(
    data,
    *,
    aoi_col,
    group_cols,
    time_col,
    include_missing,
    missing_label,
    collapse_repeats,
    max_sequences,
):
    frame = ensure_dataframe(data, copy=False)
    if not isinstance(aoi_col, str) or not aoi_col:
        raise ValueError("aoi_col must be a non-empty string")

    groups = [group_cols] if isinstance(group_cols, str) else list(group_cols or [])
    if not groups or not all(isinstance(c, str) and c for c in groups):
        raise ValueError("group_cols must be a non-empty character vector")

    if time_col is not None and (not isinstance(time_col, str) or not time_col):
        raise ValueError("time_col must be None or a non-empty string")

    needed = [aoi_col, *groups]
    if time_col is not None:
        needed.append(time_col)

    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    try:
        max_sequences = int(max_sequences)
    except (TypeError, ValueError) as exc:
        raise ValueError("max_sequences must be a number of at least 2") from exc

    if max_sequences < 2:
        raise ValueError("max_sequences must be a number of at least 2")

    ids, seqs = [], []

    for key, part in frame.groupby(groups, dropna=False, sort=False):
        if time_col is not None:
            part = part.sort_values(time_col, kind="stable", na_position="last")

        values = []
        for value in part[aoi_col]:
            missing_value = pd.isna(value) or (isinstance(value, str) and value == "")
            if missing_value:
                if include_missing:
                    values.append(str(missing_label))
                continue
            values.append(str(value))

        if collapse_repeats:
            values = list(collapse_consecutive(values))

        ids.append(_gp3_scanpath_r_group_id(key))
        seqs.append(values)

    if len(seqs) > max_sequences:
        raise ValueError(
            "Too many grouped sequences. Increase max_sequences if this is intentional."
        )

    return ids, seqs


def _gp3_scanpath_r_validate_distance_matrix(matrix, labels=None):
    if isinstance(matrix, pd.DataFrame):
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Distance matrix must be square")
        values = matrix.to_numpy(dtype=float)
        if labels is None:
            labels = [str(x) for x in matrix.index]
    else:
        values = np.asarray(matrix, dtype=float)
        if values.ndim != 2 or values.shape[0] != values.shape[1]:
            raise ValueError("Distance matrix must be square")
        if labels is None:
            labels = [f"sequence_{i}" for i in range(1, values.shape[0] + 1)]

    labels = [str(x) for x in labels]

    if len(labels) != values.shape[0] or len(set(labels)) != len(labels):
        raise ValueError("Distance labels must be unique and match matrix dimensions")

    if not np.isfinite(values).all():
        raise ValueError("Distance matrix must contain finite values")

    if (values < 0).any():
        raise ValueError("Distance matrix must be non-negative")

    if not np.allclose(values, values.T, atol=1e-12, rtol=1e-12):
        raise ValueError("Distance matrix must be symmetric")

    if not np.allclose(np.diag(values), 0.0, atol=1e-12, rtol=1e-12):
        raise ValueError("Distance matrix diagonal must be zero")

    return pd.DataFrame(values, index=labels, columns=labels)


def _gp3_scanpath_r_pairs_to_matrix(pairs, distance_col):
    frame = ensure_dataframe(pairs, copy=False)
    needed = ["sequence_a", "sequence_b", distance_col]
    missing = [c for c in needed if c not in frame.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    labels = []
    for col in ("sequence_a", "sequence_b"):
        for value in frame[col]:
            value = str(value)
            if value not in labels:
                labels.append(value)

    matrix = pd.DataFrame(
        np.nan,
        index=labels,
        columns=labels,
        dtype=float,
    )

    for label in labels:
        matrix.loc[label, label] = 0.0

    for _, row in frame.iterrows():
        a = str(row["sequence_a"])
        b = str(row["sequence_b"])
        value = float(row[distance_col])
        matrix.loc[a, b] = value
        matrix.loc[b, a] = value

    if matrix.isna().any().any():
        raise ValueError("Pairwise distance table does not contain every sequence pair")

    return _gp3_scanpath_r_validate_distance_matrix(matrix)


def _gp3_scanpath_r_cluster_matrix(distance, *, k, method, linkage):
    from scipy.cluster.hierarchy import cut_tree
    from scipy.cluster.hierarchy import linkage as scipy_linkage
    from scipy.spatial.distance import squareform

    n = len(distance)
    if n < 3:
        raise ValueError("At least three scanpaths are required for clustering")

    try:
        k = int(k)
    except (TypeError, ValueError) as exc:
        raise ValueError("k must be one finite integer") from exc

    if k < 2 or k >= n:
        raise ValueError("k must be at least 2 and smaller than the number of scanpaths")

    if method == "hierarchical":
        linkage_map = {
            "average": "average",
            "complete": "complete",
            "single": "single",
            "ward.D2": "ward",
            "ward.D": "ward",
            "mcquitty": "weighted",
            "median": "median",
            "centroid": "centroid",
        }

        if linkage not in linkage_map:
            raise ValueError("Unsupported hierarchical linkage")

        model = scipy_linkage(
            squareform(distance.to_numpy(float), checks=False),
            method=linkage_map[linkage],
        )
        labels = cut_tree(model, n_clusters=[k]).reshape(-1).astype(int) + 1
        return labels, model, None

    if method != "pam":
        raise ValueError("method must be 'hierarchical' or 'pam'")

    values = distance.to_numpy(float)
    medoids = [int(np.argmin(values.sum(axis=1)))]

    while len(medoids) < k:
        nearest = np.min(values[:, medoids], axis=1)
        nearest[medoids] = -np.inf
        medoids.append(int(np.argmax(nearest)))

    for _ in range(100):
        assigned = np.argmin(values[:, medoids], axis=1)
        updated = medoids.copy()

        for cluster_index in range(k):
            members = np.flatnonzero(assigned == cluster_index)
            if not len(members):
                continue

            within = values[np.ix_(members, members)]
            updated[cluster_index] = int(members[np.argmin(within.sum(axis=1))])

        if updated == medoids:
            break
        medoids = updated

    labels = np.argmin(values[:, medoids], axis=1).astype(int) + 1
    return labels, {"method": "pam", "medoid_indices": medoids}, medoids


def _gp3_scanpath_r_representatives(fit):
    distance = fit["distance"]
    assignments = fit["assignments"].set_index("sequence_id")["cluster"].astype(int)

    rows = []

    for cluster_id in sorted(assignments.unique()):
        members = [
            sequence_id
            for sequence_id in distance.index
            if assignments.loc[sequence_id] == cluster_id
        ]

        if len(members) == 1:
            means = pd.Series([0.0], index=members)
        else:
            means = distance.loc[members, members].sum(axis=1) / (len(members) - 1)

        sequence_id = sorted(
            members,
            key=lambda value: (float(means.loc[value]), str(value)),
        )[0]

        rows.append(
            {
                "cluster": int(cluster_id),
                "sequence_id": sequence_id,
                "mean_within_cluster_distance": float(means.loc[sequence_id]),
                "cluster_size": len(members),
            }
        )

    return pd.DataFrame(rows)


def _gp3_scanpath_r_map_clusters(reference, resampled):
    reference = np.asarray(reference, dtype=int)
    resampled = np.asarray(resampled, dtype=int)
    mapping = {}

    for cluster_id in sorted(np.unique(resampled)):
        values, counts = np.unique(
            reference[resampled == cluster_id],
            return_counts=True,
        )
        order = np.lexsort((values, -counts))
        mapping[int(cluster_id)] = int(values[order[0]])

    return mapping


def compute_gazepoint_scanpath_similarity(
    path_a=None,
    path_b=None,
    method="sequence",
    *,
    data=None,
    aoi_col=None,
    group_cols=None,
    time_col=None,
    include_missing=False,
    missing_label="missing",
    collapse_repeats=False,
    max_sequences=200,
):
    """Compute scalar Python similarity or an R-compatible pairwise table."""
    if data is not None and path_a is not None:
        raise TypeError("supply either path_a or data, not both")

    r_data = data
    if r_data is None and isinstance(path_a, pd.DataFrame) and path_b is None:
        r_data = path_a

    if r_data is None:
        if path_b is None:
            raise TypeError("path_b is required for the legacy Python interface")

        if method == "sequence":
            return 1 - compute_gazepoint_sequence_distance(
                path_a,
                path_b,
                normalize=True,
            )

        a = np.asarray(path_a, float)
        b = np.asarray(path_b, float)
        n = min(len(a), len(b))

        if n == 0:
            return np.nan

        return float(
            np.exp(
                -np.mean(
                    np.linalg.norm(
                        a[:n] - b[:n],
                        axis=1,
                    )
                )
            )
        )

    ids, seqs = _gp3_scanpath_r_sequences(
        r_data,
        aoi_col=aoi_col,
        group_cols=group_cols,
        time_col=time_col,
        include_missing=include_missing,
        missing_label=missing_label,
        collapse_repeats=collapse_repeats,
        max_sequences=max_sequences,
    )

    rows = []

    for i in range(len(seqs)):
        for j in range(i, len(seqs)):
            edit = int(
                compute_gazepoint_sequence_distance(
                    seqs[i],
                    seqs[j],
                    normalize=False,
                )
            )

            denominator = max(len(seqs[i]), len(seqs[j]))
            normalized = 0.0 if denominator == 0 else edit / denominator

            rows.append(
                {
                    "sequence_a": ids[i],
                    "sequence_b": ids[j],
                    "edit_distance": edit,
                    "normalized_distance": float(normalized),
                    "similarity": float(1 - normalized),
                    "sequence_a_length": len(seqs[i]),
                    "sequence_b_length": len(seqs[j]),
                    "n_sequences": len(seqs),
                    "similarity_status": "ok",
                }
            )

    return pd.DataFrame(rows)


def compute_gazepoint_transition_network_metrics(matrix) -> pd.DataFrame:
    if isinstance(matrix, pd.DataFrame):
        G = nx.from_pandas_adjacency(matrix, create_using=nx.DiGraph)
    else:
        G = nx.from_numpy_array(np.asarray(matrix), create_using=nx.DiGraph)
    rows = []
    for node in G.nodes:
        rows.append(
            {
                "node": node,
                "in_degree": G.in_degree(node, weight="weight"),
                "out_degree": G.out_degree(node, weight="weight"),
                "pagerank": nx.pagerank(G, weight="weight").get(node, np.nan),
            }
        )
    return pd.DataFrame(rows)


def summarise_gazepoint_markovchain(data=None, sequence=None, **kwargs) -> dict[str, Any]:
    m = compute_gazepoint_aoi_transition_matrix(
        data=data, sequence=sequence, normalize=True, **kwargs
    )
    return {
        "transition_matrix": m,
        "stationary": _stationary(m)
        if isinstance(m, pd.DataFrame) and m.shape[0] == m.shape[1]
        else None,
    }


def _stationary(m: pd.DataFrame) -> pd.DataFrame:
    arr = m.to_numpy(float)
    vals, vecs = np.linalg.eig(arr.T)
    i = int(np.argmin(abs(vals - 1)))
    v = np.real(vecs[:, i])
    v = np.abs(v)
    v = v / v.sum() if v.sum() else v
    return pd.DataFrame({"state": m.index, "stationary_probability": v})


def create_gazepoint_markovchain_object(data=None, sequence=None, **kwargs):
    return summarise_gazepoint_markovchain(data, sequence, **kwargs)


def prepare_gazepoint_semimarkov_data(data, **kwargs):
    return prepare_gazepoint_aoi_sequences(data, **kwargs)


def summarise_gazepoint_semimarkov(data=None, sequence=None, **kwargs):
    return summarise_gazepoint_markovchain(data, sequence, **kwargs)


def prepare_gazepoint_traminer_data(data, **kwargs):
    return prepare_gazepoint_aoi_sequences(data, **kwargs)


def flag_gazepoint_sequence_anomalies(
    data=None, sequence=None, z_threshold=3.0, **kwargs
) -> pd.DataFrame:
    comp = compute_gazepoint_sequence_complexity(data=data, sequence=sequence, **kwargs)
    x = comp.complexity_index
    z = (
        (x - x.mean()) / x.std(ddof=0)
        if len(x) > 1 and x.std(ddof=0) > 0
        else pd.Series(0, index=x.index)
    )
    comp["anomaly_score"] = z.abs()
    comp["anomaly"] = comp.anomaly_score > z_threshold
    return comp


def summarise_gazepoint_aoi_trial_features(
    data, aoi_col=None, trial_col=None, group_cols=None
) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    trial_col = infer_column(df, "trial", trial_col, required=True)
    groups = normalize_group_cols(df, group_cols) + [trial_col]
    s = summarise_aoi_samples(df, aoi_col=aoi_col, group_cols=groups)
    return s


def summarise_gazepoint_aoi_windows(
    data, aoi_col=None, time_col=None, windows=None, group_cols=None
) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    time_col = infer_column(df, "time", time_col, required=True)
    t = finite_numeric(df[time_col])
    windows = windows or {"window": (float(t.min()), float(t.max()))}
    rows = []
    for name, (lo, hi) in windows.items() if isinstance(windows, dict) else windows:
        tmp = summarise_aoi_samples(
            df.loc[t.between(lo, hi)], aoi_col=aoi_col, group_cols=group_cols
        )
        tmp["window"] = name
        tmp["window_start"] = lo
        tmp["window_end"] = hi
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def transform_gazepoint_aoi_empirical_logit(
    data, success_col="success", total_col="total", adjustment=0.5, output_col="empirical_logit"
) -> pd.DataFrame:
    df = ensure_dataframe(data)
    s = finite_numeric(df[success_col])
    n = finite_numeric(df[total_col])
    df[output_col] = np.log((s + adjustment) / (n - s + adjustment))
    return df


def audit_gazepoint_aoi_window_denominators(
    data, success_col="success", total_col="total"
) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    s = finite_numeric(df[success_col])
    n = finite_numeric(df[total_col])
    return pd.DataFrame(
        [
            {
                "n_rows": len(df),
                "n_invalid": int(((s < 0) | (n < 0) | (s > n)).sum()),
                "min_total": float(n.min()),
                "max_total": float(n.max()),
            }
        ]
    )


def cluster_gazepoint_scanpaths(
    data=None,
    aoi_col=None,
    group_cols=None,
    time_col=None,
    n_clusters=3,
    *,
    x=None,
    k=None,
    method=None,
    linkage="average",
    distance_col="normalized_distance",
    include_missing=False,
    missing_label="missing",
    collapse_repeats=False,
    max_sequences=200,
):
    """Cluster scanpaths with legacy Python or R-compatible structured output."""
    r_mode = x is not None or k is not None or method is not None

    if not r_mode:
        seq = prepare_gazepoint_aoi_sequences(
            data,
            aoi_col=aoi_col,
            group_cols=group_cols,
            time_col=time_col,
        )

        n = len(seq)
        if n == 0:
            return seq.assign(cluster=pd.Series(dtype=int))

        distance_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                distance_matrix[i, j] = distance_matrix[j, i] = compute_gazepoint_sequence_distance(
                    seq.iloc[i].sequence,
                    seq.iloc[j].sequence,
                )

        if n == 1:
            labels = np.array([0])
        else:
            labels = AgglomerativeClustering(
                n_clusters=min(n_clusters, n),
                metric="precomputed",
                linkage="average",
            ).fit_predict(distance_matrix)

        seq["cluster"] = labels
        seq.attrs["distance_matrix"] = distance_matrix
        return seq

    source = x if x is not None else data
    if source is None:
        raise TypeError("x is required for the R-compatible interface")

    k = 3 if k is None else k
    method = "hierarchical" if method is None else str(method)
    pairwise = None

    if isinstance(source, np.ndarray):
        distance = _gp3_scanpath_r_validate_distance_matrix(source)
        distance_source = "distance_matrix"

    elif (
        isinstance(source, pd.DataFrame)
        and source.shape[0] == source.shape[1]
        and [str(value) for value in source.index] == [str(value) for value in source.columns]
    ):
        distance = _gp3_scanpath_r_validate_distance_matrix(
            source,
            labels=[str(value) for value in source.index],
        )
        distance_source = "distance_matrix"

    elif isinstance(source, pd.DataFrame) and {
        "sequence_a",
        "sequence_b",
        distance_col,
    }.issubset(source.columns):
        pairwise = source.copy()
        distance = _gp3_scanpath_r_pairs_to_matrix(source, distance_col)
        distance_source = "pairwise_distance_table"

    elif isinstance(source, pd.DataFrame):
        if aoi_col is None or group_cols is None:
            raise ValueError("Supply aoi_col and group_cols when x is long-format AOI data")

        pairwise = compute_gazepoint_scanpath_similarity(
            data=source,
            aoi_col=aoi_col,
            group_cols=group_cols,
            time_col=time_col,
            include_missing=include_missing,
            missing_label=missing_label,
            collapse_repeats=collapse_repeats,
            max_sequences=max_sequences,
        )

        distance = _gp3_scanpath_r_pairs_to_matrix(
            pairwise,
            distance_col,
        )
        distance_source = "long_aoi_data"

    else:
        distance = _gp3_scanpath_r_validate_distance_matrix(source)
        distance_source = "distance_matrix"

    labels, model_object, medoid_indices = _gp3_scanpath_r_cluster_matrix(
        distance,
        k=k,
        method=method,
        linkage=linkage,
    )

    sequence_ids = list(distance.index)
    assignments = pd.DataFrame(
        {
            "sequence_id": sequence_ids,
            "cluster": labels.astype(int),
        }
    )

    medoids = (
        [sequence_ids[index] for index in medoid_indices] if medoid_indices is not None else None
    )

    silhouette = None
    mean_silhouette_width = np.nan

    unique_clusters = np.unique(labels)

    if 2 <= len(unique_clusters) < len(labels):
        from sklearn.metrics import silhouette_samples

        widths = silhouette_samples(
            distance.to_numpy(float),
            labels,
            metric="precomputed",
        )

        silhouette = pd.DataFrame(
            {
                "sequence_id": sequence_ids,
                "cluster": labels.astype(int),
                "neighbor_cluster": np.nan,
                "silhouette_width": widths.astype(float),
            }
        )

        mean_silhouette_width = float(np.mean(widths))

    return {
        "assignments": assignments,
        "distance": distance,
        "model": model_object,
        "medoids": medoids,
        "silhouette": silhouette,
        "mean_silhouette_width": mean_silhouette_width,
        "pairwise_distances": pairwise,
        "k": int(k),
        "method": method,
        "linkage": linkage if method == "hierarchical" else None,
        "distance_source": distance_source,
        "clustering_status": "ok",
        "_gp3_class": "gp3_scanpath_clusters",
    }


def select_gazepoint_scanpath_clusters(data, max_clusters=6, **kwargs) -> pd.DataFrame:
    rows = []
    for k in range(2, max_clusters + 1):
        try:
            cl = cluster_gazepoint_scanpaths(data, n_clusters=k, **kwargs)
            counts = cl.cluster.value_counts()
            balance = float(counts.min() / counts.max()) if len(counts) else np.nan
            rows.append(
                {
                    "n_clusters": k,
                    "min_cluster": int(counts.min()),
                    "max_cluster": int(counts.max()),
                    "balance": balance,
                }
            )
        except Exception:
            continue
    return pd.DataFrame(rows)


def extract_gazepoint_representative_scanpaths(clustered) -> pd.DataFrame:
    df = ensure_dataframe(clustered, copy=False)
    rows = []
    for cluster, g in df.groupby("cluster", dropna=False):
        seqs = g.sequence.tolist()
        n = len(seqs)
        dist = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist[i, j] = dist[j, i] = compute_gazepoint_sequence_distance(seqs[i], seqs[j])
        med = int(np.argmin(dist.mean(axis=1))) if n else 0
        rows.append(
            {"cluster": cluster, "representative_sequence": seqs[med] if n else [], "n_members": n}
        )
    return pd.DataFrame(rows)


def bootstrap_gazepoint_scanpath_clusters(
    data=None,
    n_boot=100,
    random_state=123,
    *,
    x=None,
    k=None,
    sample_fraction=None,
    method=None,
    linkages=None,
    seed=None,
    aoi_col=None,
    group_cols=None,
    time_col=None,
    distance_col="normalized_distance",
    include_missing=False,
    missing_label="missing",
    collapse_repeats=False,
    max_sequences=200,
    **kwargs,
):
    """Bootstrap scanpath clusters with legacy Python or R-compatible output."""
    r_mode = any(value is not None for value in (x, k, sample_fraction, method, linkages, seed))

    if not r_mode:
        rng = np.random.default_rng(random_state)
        cluster_gazepoint_scanpaths(data, **kwargs)

        rows = []

        for bootstrap_index in range(n_boot):
            indices = rng.integers(
                0,
                len(data),
                len(data),
            )

            sample = ensure_dataframe(data, copy=False).iloc[indices].reset_index(drop=True)

            try:
                clustered = cluster_gazepoint_scanpaths(
                    sample,
                    **kwargs,
                )

                rows.append(
                    {
                        "bootstrap": bootstrap_index,
                        "n_clusters": int(clustered.cluster.nunique()),
                        "largest_cluster_prop": float(
                            clustered.cluster.value_counts(normalize=True).max()
                        ),
                    }
                )

            except Exception:
                rows.append(
                    {
                        "bootstrap": bootstrap_index,
                        "n_clusters": np.nan,
                        "largest_cluster_prop": np.nan,
                    }
                )

        return pd.DataFrame(rows)

    if kwargs:
        raise TypeError("legacy clustering kwargs are not accepted in R-compatible mode")

    source = x if x is not None else data

    if source is None:
        raise TypeError("x is required for R-compatible bootstrap mode")

    k = 3 if k is None else int(k)
    sample_fraction = 0.8 if sample_fraction is None else float(sample_fraction)
    method = "hierarchical" if method is None else str(method)

    if linkages is None:
        linkages = ["average"]
    elif isinstance(linkages, str):
        linkages = [linkages]
    else:
        linkages = list(linkages)

    if not np.isfinite(sample_fraction) or not (0 < sample_fraction <= 1):
        raise ValueError("sample_fraction must be greater than 0 and at most 1")

    n_boot = int(n_boot)
    if n_boot < 1:
        raise ValueError("n_boot must be one positive integer")

    if method == "pam":
        specifications = pd.DataFrame(
            [
                {
                    "specification": "pam",
                    "method": "pam",
                    "linkage": "average",
                }
            ]
        )
    else:
        specifications = pd.DataFrame(
            [
                {
                    "specification": f"hierarchical_{value}",
                    "method": "hierarchical",
                    "linkage": str(value),
                }
                for value in linkages
            ]
        )

    preparation = cluster_gazepoint_scanpaths(
        x=source,
        k=2,
        method="hierarchical",
        linkage="average",
        aoi_col=aoi_col,
        group_cols=group_cols,
        time_col=time_col,
        distance_col=distance_col,
        include_missing=include_missing,
        missing_label=missing_label,
        collapse_repeats=collapse_repeats,
        max_sequences=max_sequences,
    )

    distance = preparation["distance"]
    sequence_ids = list(distance.index)
    n_sequences = len(sequence_ids)

    if k < 2 or k >= n_sequences:
        raise ValueError("k must be at least 2 and smaller than the number of scanpaths")

    sample_size = min(
        max(
            k + 1,
            int(np.ceil(n_sequences * sample_fraction)),
        ),
        n_sequences,
    )

    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(seed)

    reference_fits = {}
    co_clustering = {}
    pair_coverage = {}
    same_counts = {}
    seen_counts = {}
    inclusion_counts = {}
    iteration_rows = []
    representative_rows = []

    index_lookup = {sequence_id: index for index, sequence_id in enumerate(sequence_ids)}

    for specification in specifications.itertuples(index=False):
        reference_fit = cluster_gazepoint_scanpaths(
            x=distance,
            k=k,
            method=specification.method,
            linkage=specification.linkage,
        )

        reference_fits[specification.specification] = reference_fit

        reference_map = (
            reference_fit["assignments"].set_index("sequence_id")["cluster"].astype(int).to_dict()
        )

        same_matrix = np.zeros(
            (n_sequences, n_sequences),
            dtype=int,
        )

        seen_matrix = np.zeros_like(same_matrix)

        included = {sequence_id: 0 for sequence_id in sequence_ids}

        for iteration in range(1, n_boot + 1):
            sampled_ids = list(
                rng.choice(
                    sequence_ids,
                    size=sample_size,
                    replace=False,
                )
            )

            for sequence_id in sampled_ids:
                included[sequence_id] += 1

            fit = cluster_gazepoint_scanpaths(
                x=distance.loc[sampled_ids, sampled_ids],
                k=k,
                method=specification.method,
                linkage=specification.linkage,
            )

            sampled_map = (
                fit["assignments"].set_index("sequence_id")["cluster"].astype(int).to_dict()
            )

            sampled_cluster = np.array(
                [sampled_map[sequence_id] for sequence_id in sampled_ids],
                dtype=int,
            )

            sampled_reference = np.array(
                [reference_map[sequence_id] for sequence_id in sampled_ids],
                dtype=int,
            )

            sampled_indices = np.array(
                [index_lookup[sequence_id] for sequence_id in sampled_ids],
                dtype=int,
            )

            seen_matrix[np.ix_(sampled_indices, sampled_indices)] += 1

            same_matrix[np.ix_(sampled_indices, sampled_indices)] += (
                sampled_cluster[:, None] == sampled_cluster[None, :]
            ).astype(int)

            iteration_rows.append(
                {
                    "specification": specification.specification,
                    "method": specification.method,
                    "linkage": specification.linkage,
                    "iteration": iteration,
                    "n_sampled": sample_size,
                    "adjusted_rand_index": float(
                        adjusted_rand_score(
                            sampled_reference,
                            sampled_cluster,
                        )
                    ),
                    "mean_silhouette_width": fit["mean_silhouette_width"],
                }
            )

            mapping = _gp3_scanpath_r_map_clusters(
                sampled_reference,
                sampled_cluster,
            )

            representatives = _gp3_scanpath_r_representatives(fit)

            for row in representatives.itertuples(index=False):
                representative_rows.append(
                    {
                        "specification": specification.specification,
                        "iteration": iteration,
                        "sequence_id": row.sequence_id,
                        "resampled_cluster": int(row.cluster),
                        "reference_cluster": mapping[int(row.cluster)],
                    }
                )

        with np.errstate(divide="ignore", invalid="ignore"):
            co_matrix = same_matrix / seen_matrix

        co_matrix = co_matrix.astype(float)
        co_matrix[seen_matrix == 0] = np.nan
        np.fill_diagonal(co_matrix, 1.0)

        coverage_matrix = seen_matrix / n_boot

        co_clustering[specification.specification] = pd.DataFrame(
            co_matrix,
            index=sequence_ids,
            columns=sequence_ids,
        )

        pair_coverage[specification.specification] = pd.DataFrame(
            coverage_matrix,
            index=sequence_ids,
            columns=sequence_ids,
        )

        same_counts[specification.specification] = pd.DataFrame(
            same_matrix,
            index=sequence_ids,
            columns=sequence_ids,
        )

        seen_counts[specification.specification] = pd.DataFrame(
            seen_matrix,
            index=sequence_ids,
            columns=sequence_ids,
        )

        inclusion_counts[specification.specification] = included

    iteration_summary = pd.DataFrame(iteration_rows)

    representative_events = pd.DataFrame(
        representative_rows,
        columns=[
            "specification",
            "iteration",
            "sequence_id",
            "resampled_cluster",
            "reference_cluster",
        ],
    )

    if representative_events.empty:
        representative_stability = pd.DataFrame(
            columns=[
                "specification",
                "reference_cluster",
                "sequence_id",
                "n_representative",
                "n_included",
                "representative_rate",
            ]
        )
    else:
        representative_stability = (
            representative_events.groupby(
                [
                    "specification",
                    "reference_cluster",
                    "sequence_id",
                ],
                dropna=False,
                sort=False,
            )
            .size()
            .rename("n_representative")
            .reset_index()
        )

        representative_stability["n_included"] = [
            inclusion_counts[row.specification].get(
                str(row.sequence_id),
                0,
            )
            for row in representative_stability.itertuples(index=False)
        ]

        representative_stability["representative_rate"] = np.where(
            representative_stability["n_included"] > 0,
            representative_stability["n_representative"] / representative_stability["n_included"],
            np.nan,
        )

    return {
        "reference_fits": reference_fits,
        "co_clustering": co_clustering,
        "pair_coverage": pair_coverage,
        "same_counts": same_counts,
        "seen_counts": seen_counts,
        "inclusion_counts": inclusion_counts,
        "iteration_summary": iteration_summary,
        "representative_events": representative_events,
        "representative_stability": representative_stability,
        "distance": distance,
        "specifications": specifications,
        "settings": {
            "k": k,
            "n_boot": n_boot,
            "sample_fraction": sample_fraction,
            "sample_size": sample_size,
            "method": method,
            "linkages": specifications["linkage"].tolist(),
            "seed": seed,
            "distance_source": preparation["distance_source"],
        },
        "bootstrap_status": "ok",
        "_gp3_class": "gp3_scanpath_cluster_bootstrap",
    }


def summarise_gazepoint_scanpath_cluster_stability(
    data=None,
    *,
    x=None,
    min_pair_coverage=0.5,
    stable_threshold=0.75,
):
    """Summarise legacy bootstrap tables or R-compatible bootstrap objects."""
    source = x if x is not None else data

    if isinstance(source, pd.DataFrame):
        df = ensure_dataframe(source, copy=False)

        return pd.DataFrame(
            [
                {
                    "n_boot": len(df),
                    "mean_n_clusters": (
                        float(
                            pd.to_numeric(
                                df.n_clusters,
                                errors="coerce",
                            ).mean()
                        )
                        if "n_clusters" in df
                        else np.nan
                    ),
                    "mean_largest_cluster_prop": (
                        float(
                            pd.to_numeric(
                                df.largest_cluster_prop,
                                errors="coerce",
                            ).mean()
                        )
                        if "largest_cluster_prop" in df
                        else np.nan
                    ),
                }
            ]
        )

    if not isinstance(source, dict):
        raise ValueError("x must be returned by the R-compatible bootstrap function")

    required = {
        "reference_fits",
        "co_clustering",
        "pair_coverage",
        "iteration_summary",
        "representative_stability",
        "specifications",
        "settings",
    }

    missing = sorted(required - set(source))

    if missing:
        raise ValueError("Bootstrap result is missing: " + ", ".join(missing))

    for name, value in {
        "min_pair_coverage": min_pair_coverage,
        "stable_threshold": stable_threshold,
    }.items():
        value = float(value)

        if not np.isfinite(value) or not (0 <= value <= 1):
            raise ValueError(f"{name} must be between 0 and 1")

    overview_rows = []
    sequence_rows = []
    pair_rows = []

    for specification in source["specifications"].itertuples(index=False):
        fit = source["reference_fits"][specification.specification]
        co_matrix = source["co_clustering"][specification.specification]
        coverage_matrix = source["pair_coverage"][specification.specification]

        reference = fit["assignments"].set_index("sequence_id")["cluster"].astype(int).to_dict()

        sequence_ids = list(reference)
        local_pair_rows = []

        for i in range(len(sequence_ids)):
            for j in range(i + 1, len(sequence_ids)):
                sequence_a = sequence_ids[i]
                sequence_b = sequence_ids[j]

                probability = float(co_matrix.loc[sequence_a, sequence_b])

                coverage = float(coverage_matrix.loc[sequence_a, sequence_b])

                row = {
                    "specification": specification.specification,
                    "sequence_a": sequence_a,
                    "sequence_b": sequence_b,
                    "co_clustering_probability": probability,
                    "pair_coverage": coverage,
                    "same_reference_cluster": (reference[sequence_a] == reference[sequence_b]),
                    "included_in_summary": bool(
                        np.isfinite(probability) and coverage >= min_pair_coverage
                    ),
                }

                local_pair_rows.append(row)
                pair_rows.append(row)

        for sequence_id in sequence_ids:
            other_ids = [value for value in sequence_ids if value != sequence_id]

            same_ids = [value for value in other_ids if reference[value] == reference[sequence_id]]

            different_ids = [
                value for value in other_ids if reference[value] != reference[sequence_id]
            ]

            def eligible(
                other_ids,
                co_matrix=co_matrix,
                coverage_matrix=coverage_matrix,
                sequence_id=sequence_id,
            ):
                values = []

                for other_id in other_ids:
                    value = float(co_matrix.loc[sequence_id, other_id])

                    coverage = float(coverage_matrix.loc[sequence_id, other_id])

                    if np.isfinite(value) and coverage >= min_pair_coverage:
                        values.append(value)

                return values

            within_values = eligible(same_ids)
            between_values = eligible(different_ids)

            within_mean = float(np.mean(within_values)) if within_values else np.nan

            between_mean = float(np.mean(between_values)) if between_values else np.nan

            coverages = [
                float(coverage_matrix.loc[sequence_id, other_id])
                for other_id in other_ids
                if np.isfinite(float(coverage_matrix.loc[sequence_id, other_id]))
            ]

            sequence_rows.append(
                {
                    "specification": specification.specification,
                    "sequence_id": sequence_id,
                    "reference_cluster": int(reference[sequence_id]),
                    "within_cluster_stability": within_mean,
                    "between_cluster_coclustering": between_mean,
                    "stability_separation": (
                        within_mean - between_mean
                        if np.isfinite(within_mean) and np.isfinite(between_mean)
                        else np.nan
                    ),
                    "n_within_pairs": len(within_values),
                    "n_between_pairs": len(between_values),
                    "mean_pair_coverage": (float(np.mean(coverages)) if coverages else np.nan),
                    "stable": bool(np.isfinite(within_mean) and within_mean >= stable_threshold),
                }
            )

        sequence_table = pd.DataFrame(
            [row for row in sequence_rows if row["specification"] == specification.specification]
        )

        pair_table = pd.DataFrame(local_pair_rows)

        valid_pairs = pair_table.loc[pair_table["included_in_summary"]]

        within_pairs = valid_pairs.loc[
            valid_pairs["same_reference_cluster"],
            "co_clustering_probability",
        ].to_numpy(float)

        between_pairs = valid_pairs.loc[
            ~valid_pairs["same_reference_cluster"],
            "co_clustering_probability",
        ].to_numpy(float)

        iteration_summary = source["iteration_summary"]
        iteration_table = iteration_summary.loc[
            iteration_summary["specification"].eq(specification.specification)
        ]

        ari = pd.to_numeric(
            iteration_table["adjusted_rand_index"],
            errors="coerce",
        ).dropna()

        stability_values = pd.to_numeric(
            sequence_table["within_cluster_stability"],
            errors="coerce",
        )

        finite_stability = stability_values.loc[np.isfinite(stability_values)]

        acceptable = sequence_table["stable"] | ~np.isfinite(stability_values)

        overview_rows.append(
            {
                "specification": specification.specification,
                "method": specification.method,
                "linkage": specification.linkage,
                "k": source["settings"]["k"],
                "n_boot": source["settings"]["n_boot"],
                "sample_size": source["settings"]["sample_size"],
                "mean_adjusted_rand_index": (float(ari.mean()) if len(ari) else np.nan),
                "sd_adjusted_rand_index": (float(ari.std(ddof=1)) if len(ari) >= 2 else np.nan),
                "min_adjusted_rand_index": (float(ari.min()) if len(ari) else np.nan),
                "mean_within_cluster_coclustering": (
                    float(np.mean(within_pairs)) if len(within_pairs) else np.nan
                ),
                "mean_between_cluster_coclustering": (
                    float(np.mean(between_pairs)) if len(between_pairs) else np.nan
                ),
                "mean_sequence_stability": (
                    float(finite_stability.mean()) if len(finite_stability) else np.nan
                ),
                "min_sequence_stability": (
                    float(finite_stability.min()) if len(finite_stability) else np.nan
                ),
                "pct_sequences_stable": (100 * float(sequence_table["stable"].mean())),
                "stability_status": ("stable" if bool(acceptable.all()) else "review"),
            }
        )

    return {
        "overview": pd.DataFrame(overview_rows),
        "sequence_summary": pd.DataFrame(sequence_rows),
        "pairwise_summary": pd.DataFrame(pair_rows),
        "representative_stability": source["representative_stability"],
        "settings": {
            "min_pair_coverage": float(min_pair_coverage),
            "stable_threshold": float(stable_threshold),
        },
        "_gp3_class": "gp3_scanpath_cluster_stability_summary",
    }


# BEGIN R V2.3.0 CALL-SURFACE ALIASES
add_gazepoint_aoi = r_aliases(
    add_gazepoint_aoi, master_df="data", aoi_defs="aoi_geometry", label_col="output_col"
)
add_gazepoint_dynamic_aoi = r_aliases(
    add_gazepoint_dynamic_aoi, master_df="data", aoi_defs="aoi_data", label_col="output_col"
)
audit_gazepoint_aoi_geometry = r_aliases(audit_gazepoint_aoi_geometry, data="aoi_geometry")
audit_gazepoint_aoi_margin_sensitivity = r_aliases(
    audit_gazepoint_aoi_margin_sensitivity, gaze_data="data"
)
audit_gazepoint_aoi_overlap = r_aliases(audit_gazepoint_aoi_overlap, data="aoi_geometry")
audit_gazepoint_aoi_screen_coverage = r_aliases(
    audit_gazepoint_aoi_screen_coverage,
    data="aoi_geometry",
    screen_width="width",
    screen_height="height",
)
extract_gazepoint_representative_scanpaths = r_aliases(
    extract_gazepoint_representative_scanpaths, x="clustered"
)
# END R V2.3.0 CALL-SURFACE ALIASES
