"""AOI assignment, sequence, transition, and scanpath helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon
from sklearn.cluster import AgglomerativeClustering

from ._utils import (
    collapse_consecutive,
    ensure_dataframe,
    finite_numeric,
    infer_column,
    normalize_group_cols,
    ordered_unique,
)


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


summarise_gazepoint_aoi = summarise_aoi_samples


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


def compute_transition_matrix(sequence, normalize=False) -> pd.DataFrame:
    s = list(sequence)
    states = ordered_unique(s)
    mat = pd.DataFrame(0.0, index=states, columns=states)
    for a, b in zip(s[:-1], s[1:], strict=False):
        mat.loc[a, b] += 1
    if normalize:
        mat = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    mat.index.name = "from_aoi"
    mat.columns.name = "to_aoi"
    return mat


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
    data, x_col=None, y_col=None, time_col=None, group_cols=None
) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    x_col = infer_column(df, "x", x_col, required=True)
    y_col = infer_column(df, "y", y_col, required=True)
    groups = normalize_group_cols(df, group_cols)
    rows = []
    iterator = [((), df)] if not groups else df.groupby(groups, dropna=False, sort=False)
    for key, f in iterator:
        if time_col and time_col in f:
            f = f.sort_values(time_col)
        x = finite_numeric(f[x_col]).to_numpy(float)
        y = finite_numeric(f[y_col]).to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(y)
        x = x[ok]
        y = y[ok]
        dist = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        base = (
            {c: v for c, v in zip(groups, key if isinstance(key, tuple) else (key,), strict=False)}
            if groups
            else {}
        )
        rows.append(
            {
                **base,
                "n_points": len(x),
                "path_length": float(dist.sum()),
                "mean_step": float(dist.mean()) if len(dist) else 0,
                "dispersion_x": float(np.std(x)) if len(x) else np.nan,
                "dispersion_y": float(np.std(y)) if len(y) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def compute_gazepoint_scanpath_similarity(path_a, path_b, method="sequence") -> float:
    if method == "sequence":
        return 1 - compute_gazepoint_sequence_distance(path_a, path_b, normalize=True)
    a = np.asarray(path_a, float)
    b = np.asarray(path_b, float)
    n = min(len(a), len(b))
    if n == 0:
        return np.nan
    return float(np.exp(-np.mean(np.linalg.norm(a[:n] - b[:n], axis=1))))


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
    data, aoi_col=None, group_cols=None, time_col=None, n_clusters=3
) -> pd.DataFrame:
    seq = prepare_gazepoint_aoi_sequences(
        data, aoi_col=aoi_col, group_cols=group_cols, time_col=time_col
    )
    n = len(seq)
    if n == 0:
        return seq.assign(cluster=pd.Series(dtype=int))
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            dist[i, j] = dist[j, i] = compute_gazepoint_sequence_distance(
                seq.iloc[i].sequence, seq.iloc[j].sequence
            )
    if n == 1:
        labels = np.array([0])
    else:
        labels = AgglomerativeClustering(
            n_clusters=min(n_clusters, n), metric="precomputed", linkage="average"
        ).fit_predict(dist)
    seq["cluster"] = labels
    seq.attrs["distance_matrix"] = dist
    return seq


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
    data, n_boot=100, random_state=123, **kwargs
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    cluster_gazepoint_scanpaths(data, **kwargs)
    rows = []
    for b in range(n_boot):
        idx = rng.integers(0, len(data), len(data))
        sample = ensure_dataframe(data, copy=False).iloc[idx].reset_index(drop=True)
        try:
            cl = cluster_gazepoint_scanpaths(sample, **kwargs)
            rows.append(
                {
                    "bootstrap": b,
                    "n_clusters": int(cl.cluster.nunique()),
                    "largest_cluster_prop": float(cl.cluster.value_counts(normalize=True).max()),
                }
            )
        except Exception:
            rows.append({"bootstrap": b, "n_clusters": np.nan, "largest_cluster_prop": np.nan})
    return pd.DataFrame(rows)


def summarise_gazepoint_scanpath_cluster_stability(data) -> pd.DataFrame:
    df = ensure_dataframe(data, copy=False)
    return pd.DataFrame(
        [
            {
                "n_boot": len(df),
                "mean_n_clusters": float(pd.to_numeric(df.n_clusters, errors="coerce").mean())
                if "n_clusters" in df
                else np.nan,
                "mean_largest_cluster_prop": float(
                    pd.to_numeric(df.largest_cluster_prop, errors="coerce").mean()
                )
                if "largest_cluster_prop" in df
                else np.nan,
            }
        ]
    )
