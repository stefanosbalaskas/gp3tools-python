"""AOI assignment, sequence, transition, and scanpath helpers."""

from __future__ import annotations

from collections import Counter
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, Point, Polygon
from sklearn.cluster import AgglomerativeClustering

from ._behavioral_r4 import geometry_validation_bridge as _r4_geometry_validation_bridge
from ._behavioral_r4 import wrap_r4 as _r4_wrap
from ._compat import r_aliases
from ._r4_dual_contract import r4_dual_contract
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
    data=None,
    x_col=None,
    y_col=None,
    aoi_geometry=None,
    output_col="aoi_current",
    outside_label="outside",
    *,
    aoi_name=None,
    output=None,
    prefix="aoi_",
    overlap=None,
    include_overlap_count=None,
) -> pd.DataFrame:
    """Assign rectangular AOIs with legacy and R v2.3.0-compatible semantics."""
    df = ensure_dataframe(data)
    geom = ensure_dataframe(aoi_geometry, copy=False)
    if geom.empty:
        raise ValueError("aoi_geometry/aoi_defs must contain at least one AOI")
    x_col = infer_column(df, "x", x_col, required=True)
    y_col = infer_column(df, "y", y_col, required=True)

    aliases = {
        "name": ("name", "aoi_name", "AOI", "aoi", "label"),
        "left": ("L", "left", "xmin", "x_min"),
        "right": ("R", "right", "xmax", "x_max"),
        "top": ("T", "top", "ymin", "y_min"),
        "bottom": ("B", "bottom", "ymax", "y_max"),
    }
    resolved = {
        role: next((candidate for candidate in candidates if candidate in geom.columns), None)
        for role, candidates in aliases.items()
    }
    missing = [role for role, column in resolved.items() if column is None]
    if missing:
        raise ValueError("Could not resolve AOI definition fields: " + ", ".join(missing))

    defs = pd.DataFrame(
        {
            "name": geom[resolved["name"]].astype("string"),
            "left": pd.to_numeric(geom[resolved["left"]], errors="coerce"),
            "right": pd.to_numeric(geom[resolved["right"]], errors="coerce"),
            "top": pd.to_numeric(geom[resolved["top"]], errors="coerce"),
            "bottom": pd.to_numeric(geom[resolved["bottom"]], errors="coerce"),
        }
    )
    if aoi_name is not None:
        wanted = (
            {str(aoi_name)} if isinstance(aoi_name, str) else {str(value) for value in aoi_name}
        )
        defs = defs.loc[defs["name"].astype(str).isin(wanted)].copy()
        if defs.empty:
            raise ValueError("aoi_name did not match any AOI definition")
    if defs["name"].duplicated().any():
        raise ValueError("AOI names must be unique")
    if not np.isfinite(defs[["left", "right", "top", "bottom"]].to_numpy(float)).all():
        raise ValueError("AOI boundaries must be finite numeric values")

    left = np.minimum(defs["left"].to_numpy(float), defs["right"].to_numpy(float))
    right = np.maximum(defs["left"].to_numpy(float), defs["right"].to_numpy(float))
    top = np.minimum(defs["top"].to_numpy(float), defs["bottom"].to_numpy(float))
    bottom = np.maximum(defs["top"].to_numpy(float), defs["bottom"].to_numpy(float))
    x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(float)
    y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(float)
    valid = np.isfinite(x) & np.isfinite(y)
    membership = np.column_stack(
        [
            valid & (x >= left[i]) & (x <= right[i]) & (y >= top[i]) & (y <= bottom[i])
            for i in range(len(defs))
        ]
    )
    names = defs["name"].astype(str).tolist()
    overlap_count = membership.sum(axis=1)

    r_mode = any(
        (
            aoi_name is not None,
            output is not None,
            prefix != "aoi_",
            overlap is not None,
            include_overlap_count is not None,
        )
    )
    if r_mode:
        output = "logical" if output is None else str(output)
        overlap = "first" if overlap is None else str(overlap)
        include_overlap_count = (
            True if include_overlap_count is None else bool(include_overlap_count)
        )
    else:
        output = "label"
        overlap = "last"
        include_overlap_count = False

    if output not in {"logical", "label", "both"}:
        raise ValueError("output must be one of: logical, label, both")
    if overlap not in {"first", "last", "error"}:
        raise ValueError("overlap must be one of: first, last, error")
    if overlap == "error" and np.any(overlap_count > 1):
        raise ValueError(f"{int(np.sum(overlap_count > 1))} sample(s) fall inside overlapping AOIs")

    out = df.copy()
    if output in {"logical", "both"}:
        logical_names = [prefix + value for value in _gp3_polygon_r_make_names(names)]
        for index, column in enumerate(logical_names):
            out[column] = membership[:, index]
    if output in {"label", "both"}:
        labels = np.full(len(out), outside_label, dtype=object)
        labels[~valid] = pd.NA
        if overlap == "last":
            for index, name in enumerate(names):
                labels[membership[:, index]] = name
        else:
            unassigned = valid.copy()
            for index, name in enumerate(names):
                hit = membership[:, index] & unassigned
                labels[hit] = name
                unassigned[hit] = False
        out[output_col] = labels
    if include_overlap_count:
        out["aoi_overlap_count"] = overlap_count.astype(int)

    out.attrs.update(df.attrs)
    out.attrs["gazepoint_aoi_definitions"] = defs.reset_index(drop=True)
    return out


def _gp3_polygon_r_prepare(
    vertices,
    *,
    aoi_col,
    vertex_x_col,
    vertex_y_col,
    vertex_order_col,
):
    required = [
        aoi_col,
        vertex_x_col,
        vertex_y_col,
    ]

    if vertex_order_col is not None:
        required.append(vertex_order_col)

    missing = [column for column in required if column not in vertices.columns]

    if missing:
        raise ValueError("vertices is missing required column(s): " + ", ".join(missing))

    names = vertices[aoi_col]

    if names.isna().any() or names.astype(str).eq("").any():
        raise ValueError("Polygon AOI names must be non-missing and non-empty")

    definitions = []

    for name in sorted(names.astype(str).unique()):
        block = vertices.loc[names.astype(str).eq(name)].copy()

        if vertex_order_col is not None:
            block = block.sort_values(
                vertex_order_col,
                kind="stable",
                na_position="last",
            )

        x = pd.to_numeric(
            block[vertex_x_col],
            errors="coerce",
        ).to_numpy(dtype=float)

        y = pd.to_numeric(
            block[vertex_y_col],
            errors="coerce",
        ).to_numpy(dtype=float)

        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError(f"Polygon `{name}` contains non-finite vertices")

        points = (
            pd.DataFrame(
                {
                    "x": x,
                    "y": y,
                }
            )
            .drop_duplicates()
            .reset_index(drop=True)
        )

        if len(points) < 3:
            raise ValueError(f"Polygon `{name}` must contain at least three unique vertices")

        definitions.append(
            {
                "name": name,
                "x": points["x"].to_numpy(dtype=float),
                "y": points["y"].to_numpy(dtype=float),
            }
        )

    return definitions


def _gp3_polygon_r_point_on_segment(
    point_x,
    point_y,
    x1,
    y1,
    x2,
    y2,
    tolerance,
):
    cross = (point_y - y1) * (x2 - x1) - (point_x - x1) * (y2 - y1)

    if abs(cross) > tolerance:
        return False

    within_x = point_x >= min(x1, x2) - tolerance and point_x <= max(x1, x2) + tolerance

    within_y = point_y >= min(y1, y2) - tolerance and point_y <= max(y1, y2) + tolerance

    return within_x and within_y


def _gp3_polygon_r_points_in_polygon(
    x,
    y,
    polygon_x,
    polygon_y,
    *,
    boundary,
):
    output = np.zeros(
        len(x),
        dtype=bool,
    )

    valid = np.isfinite(x) & np.isfinite(y)

    if not valid.any():
        return output

    scale_values = np.concatenate(
        [
            np.abs(x[valid]),
            np.abs(y[valid]),
            np.abs(polygon_x),
            np.abs(polygon_y),
        ]
    )

    tolerance = np.sqrt(np.finfo(float).eps) * max(
        1.0,
        float(np.max(scale_values)),
    )

    n_vertices = len(polygon_x)

    for point_index in np.flatnonzero(valid):
        point_x = float(x[point_index])

        point_y = float(y[point_index])

        on_boundary = False

        for vertex_index in range(n_vertices):
            next_index = 0 if vertex_index == n_vertices - 1 else vertex_index + 1

            if _gp3_polygon_r_point_on_segment(
                point_x,
                point_y,
                polygon_x[vertex_index],
                polygon_y[vertex_index],
                polygon_x[next_index],
                polygon_y[next_index],
                tolerance,
            ):
                on_boundary = True
                break

        if on_boundary:
            output[point_index] = boundary == "inside"
            continue

        inside = False
        previous = n_vertices - 1

        for current in range(n_vertices):
            yi = polygon_y[current]
            yj = polygon_y[previous]
            xi = polygon_x[current]
            xj = polygon_x[previous]

            crosses = (yi > point_y) != (yj > point_y)

            if crosses:
                intersection_x = ((xj - xi) * (point_y - yi) / (yj - yi)) + xi

                if point_x < intersection_x:
                    inside = not inside

            previous = current

        output[point_index] = inside

    return output


def _gp3_polygon_r_membership(
    x,
    y,
    definitions,
    *,
    boundary,
):
    return pd.DataFrame(
        {
            definition["name"]: _gp3_polygon_r_points_in_polygon(
                x,
                y,
                definition["x"],
                definition["y"],
                boundary=boundary,
            )
            for definition in definitions
        }
    )


def _gp3_polygon_r_make_names(names):
    import keyword
    import re

    output = []
    counts = {}

    for value in names:
        name = re.sub(
            r"[^A-Za-z0-9_.]",
            ".",
            str(value),
        )

        if (
            not name
            or name[0].isdigit()
            or (name.startswith(".") and len(name) > 1 and name[1].isdigit())
        ):
            name = "X" + name

        if keyword.iskeyword(name):
            name += "."

        base = name
        occurrence = counts.get(
            base,
            0,
        )

        if occurrence:
            name = f"{base}.{occurrence}"

        counts[base] = occurrence + 1
        output.append(name)

    return output


def add_gazepoint_polygon_aoi(
    data=None,
    polygons=None,
    x_col=None,
    y_col=None,
    output_col="aoi_current",
    outside_label="outside",
    *,
    master_df=None,
    vertices=None,
    aoi_col="aoi_name",
    vertex_x_col="vertex_x",
    vertex_y_col="vertex_y",
    vertex_order_col=None,
    output=None,
    prefix="aoi_",
    label_col=None,
    overlap="first",
    boundary="inside",
    include_overlap_count=True,
) -> pd.DataFrame:
    """Assign polygon AOIs using legacy Python or R v2.3.0 semantics."""
    looks_like_r_vertices = isinstance(
        polygons,
        pd.DataFrame,
    ) and {
        aoi_col,
        vertex_x_col,
        vertex_y_col,
    }.issubset(polygons.columns)

    r_mode = (
        master_df is not None
        or vertices is not None
        or looks_like_r_vertices
        or output is not None
        or aoi_col != "aoi_name"
        or vertex_x_col != "vertex_x"
        or vertex_y_col != "vertex_y"
        or vertex_order_col is not None
        or prefix != "aoi_"
        or label_col is not None
        or overlap != "first"
        or boundary != "inside"
        or include_overlap_count is not True
    )

    if not r_mode:
        df = ensure_dataframe(data)

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

        if isinstance(
            polygons,
            dict,
        ):
            items = list(polygons.items())
        else:
            geometry = ensure_dataframe(
                polygons,
                copy=False,
            )

            name_col = next(
                (
                    column
                    for column in (
                        "aoi",
                        "name",
                        "label",
                    )
                    if column in geometry
                ),
                None,
            )

            poly_col = next(
                (
                    column
                    for column in (
                        "polygon",
                        "vertices",
                        "geometry",
                    )
                    if column in geometry
                ),
                None,
            )

            if not name_col or not poly_col:
                raise ValueError("Polygon table must contain aoi/name and polygon/vertices columns")

            items = [
                (
                    row[name_col],
                    row[poly_col],
                )
                for _, row in geometry.iterrows()
            ]

        shapes = [
            (
                name,
                vertices_value
                if isinstance(
                    vertices_value,
                    Polygon,
                )
                else Polygon(vertices_value),
            )
            for (
                name,
                vertices_value,
            ) in items
        ]

        labels = []

        for xv, yv in zip(
            finite_numeric(df[x_col]),
            finite_numeric(df[y_col]),
            strict=False,
        ):
            if not np.isfinite(xv) or not np.isfinite(yv):
                labels.append(pd.NA)
                continue

            point = Point(
                float(xv),
                float(yv),
            )

            hits = [
                name
                for name, polygon in shapes
                if (polygon.contains(point) or polygon.touches(point))
            ]

            labels.append(hits[-1] if hits else outside_label)

        df[output_col] = labels
        return df

    if master_df is not None and data is not None:
        raise TypeError("supply either data or master_df, not both")

    if vertices is not None and polygons is not None:
        raise TypeError("supply either polygons or vertices, not both")

    frame = ensure_dataframe(
        master_df if master_df is not None else data,
        copy=False,
    )

    vertex_frame = ensure_dataframe(
        vertices if vertices is not None else polygons,
        copy=False,
    )

    x_col = "FPOGX" if x_col is None else x_col

    y_col = "FPOGY" if y_col is None else y_col

    output = "label" if output is None else output

    label_col = output_col if label_col is None else label_col

    if output not in {
        "label",
        "logical",
        "both",
    }:
        raise ValueError("output must be 'label', 'logical', or 'both'")

    if overlap not in {
        "first",
        "last",
        "error",
    }:
        raise ValueError("overlap must be 'first', 'last', or 'error'")

    if boundary not in {
        "inside",
        "outside",
    }:
        raise ValueError("boundary must be 'inside' or 'outside'")

    missing = [
        column
        for column in (
            x_col,
            y_col,
        )
        if column not in frame.columns
    ]

    if missing:
        raise ValueError("master_df is missing required column(s): " + ", ".join(missing))

    definitions = _gp3_polygon_r_prepare(
        vertex_frame,
        aoi_col=aoi_col,
        vertex_x_col=vertex_x_col,
        vertex_y_col=vertex_y_col,
        vertex_order_col=vertex_order_col,
    )

    x = pd.to_numeric(
        frame[x_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    y = pd.to_numeric(
        frame[y_col],
        errors="coerce",
    ).to_numpy(dtype=float)

    membership = _gp3_polygon_r_membership(
        x,
        y,
        definitions,
        boundary=boundary,
    )

    overlap_count = membership.sum(axis=1).astype(int)

    if overlap == "error" and (overlap_count > 1).any():
        n_overlap = int((overlap_count > 1).sum())

        raise ValueError(f"{n_overlap} sample(s) fall inside overlapping AOIs")

    result = frame.copy()

    if output in {
        "logical",
        "both",
    }:
        logical_names = [prefix + value for value in _gp3_polygon_r_make_names(membership.columns)]

        for source_name, target_name in zip(
            membership.columns,
            logical_names,
            strict=True,
        ):
            result[target_name] = membership[source_name].to_numpy(dtype=bool)

    if output in {
        "label",
        "both",
    }:
        labels = np.full(
            len(frame),
            outside_label,
            dtype=object,
        )

        valid_xy = np.isfinite(x) & np.isfinite(y)

        labels[~valid_xy] = pd.NA

        for name in membership.columns:
            hit = membership[name].to_numpy(dtype=bool)

            if overlap == "last":
                labels[hit] = name
            else:
                assign = hit & pd.Series(labels).eq(outside_label).fillna(False).to_numpy()

                labels[assign] = name

        result[label_col] = labels

    if include_overlap_count is True:
        result["aoi_overlap_count"] = overlap_count.to_numpy()

    result.attrs["gazepoint_polygon_aoi_definitions"] = definitions

    return result


def _gp3_dynamic_match_time(value, definitions, mode):
    if not np.isfinite(value) or len(definitions) == 0:
        return np.nan
    values = np.asarray(definitions, dtype=float)
    if mode == "nearest":
        # R's which.min() resolves exact distance ties to the first sorted time.
        return float(values[int(np.argmin(np.abs(values - value)))])
    if mode == "previous":
        eligible = values[values <= value]
        return float(np.max(eligible)) if len(eligible) else np.nan
    eligible = values[values >= value]
    return float(np.min(eligible)) if len(eligible) else np.nan


def _gp3_dynamic_group_keys(frame, cols):
    if not cols:
        return pd.Series("__all__", index=frame.index, dtype="string")
    return frame[list(cols)].astype("string").fillna("<NA>").agg("\x1f".join, axis=1)


def _gp3_dynamic_apply_membership(
    frame,
    membership,
    names,
    *,
    output,
    prefix,
    label_col,
    outside_label,
    overlap,
    include_overlap_count,
    valid_xy,
):
    counts = membership.sum(axis=1).astype(int)
    if overlap == "error" and np.any(counts > 1):
        raise ValueError(f"{int(np.sum(counts > 1))} sample(s) fall inside overlapping AOIs")
    out = frame.copy()
    if output in {"logical", "both"}:
        logical_names = [prefix + x for x in _gp3_polygon_r_make_names(names)]
        for j, col in enumerate(logical_names):
            out[col] = membership[:, j]
    if output in {"label", "both"}:
        labels = np.full(len(out), outside_label, dtype=object)
        labels[~valid_xy] = pd.NA
        if overlap == "last":
            for j, name in enumerate(names):
                labels[membership[:, j]] = name
        else:
            available = valid_xy.copy()
            for j, name in enumerate(names):
                hit = membership[:, j] & available
                labels[hit] = name
                available[hit] = False
        out[label_col] = labels
    if include_overlap_count:
        out["aoi_overlap_count"] = counts
    return out


def _gp3_margin_resolve_column(frame, explicit, candidates, *, required, arg):
    if explicit is not None:
        if not isinstance(explicit, str) or not explicit:
            raise ValueError(f"{arg} must be a non-empty column name or None")
        if explicit not in frame:
            if required:
                raise KeyError(f"Missing required column: {explicit}")
            return None
        return explicit
    found = next((c for c in candidates if c in frame), None)
    if required and found is None:
        raise KeyError(f"Could not resolve {arg}")
    return found


def _gp3_margin_safe_stat(values, func):
    x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    x = x[np.isfinite(x)]
    return float(func(x)) if len(x) else np.nan


def add_gazepoint_dynamic_aoi(
    data,
    aoi_data,
    time_col=None,
    x_col=None,
    y_col=None,
    aoi_time_col=None,
    output_col="aoi_current",
    tolerance=None,
    *,
    aoi_name_col=None,
    shape=None,
    group_cols=None,
    match=None,
    max_time_gap=None,
    left_col="left",
    right_col="right",
    top_col="top",
    bottom_col="bottom",
    vertex_x_col="vertex_x",
    vertex_y_col="vertex_y",
    vertex_order_col=None,
    output=None,
    prefix="aoi_",
    outside_label="outside",
    overlap=None,
    boundary="inside",
    definition_time_col="aoi_definition_time",
    time_gap_col="aoi_time_gap",
    include_overlap_count=None,
) -> pd.DataFrame:
    """Assign time-varying rectangular or polygon AOIs with R v2.3.0 semantics."""
    r_mode = any(
        (
            aoi_name_col is not None,
            shape is not None,
            group_cols is not None,
            match is not None,
            max_time_gap is not None,
            left_col != "left",
            right_col != "right",
            top_col != "top",
            bottom_col != "bottom",
            vertex_x_col != "vertex_x",
            vertex_y_col != "vertex_y",
            vertex_order_col is not None,
            output is not None,
            prefix != "aoi_",
            outside_label != "outside",
            overlap is not None,
            boundary != "inside",
            definition_time_col != "aoi_definition_time",
            time_gap_col != "aoi_time_gap",
            include_overlap_count is not None,
        )
    )
    if not r_mode:
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

    frame = ensure_dataframe(data, copy=False)
    defs = ensure_dataframe(aoi_data, copy=False)
    time_col = "TIME" if time_col is None else time_col
    x_col = "FPOGX" if x_col is None else x_col
    y_col = "FPOGY" if y_col is None else y_col
    aoi_time_col = "aoi_time" if aoi_time_col is None else aoi_time_col
    aoi_name_col = "aoi_name" if aoi_name_col is None else aoi_name_col
    shape = "auto" if shape is None else str(shape)
    match = "nearest" if match is None else str(match)
    max_time_gap = np.inf if max_time_gap is None else float(max_time_gap)
    output = "label" if output is None else str(output)
    overlap = "first" if overlap is None else str(overlap)
    include_overlap_count = True if include_overlap_count is None else bool(include_overlap_count)
    groups = (
        []
        if group_cols is None
        else ([group_cols] if isinstance(group_cols, str) else list(dict.fromkeys(group_cols)))
    )
    if shape not in {"auto", "rectangle", "polygon"}:
        raise ValueError("shape must be auto, rectangle, or polygon")
    if match not in {"nearest", "previous", "next"}:
        raise ValueError("match must be nearest, previous, or next")
    if output not in {"label", "logical", "both"}:
        raise ValueError("output must be label, logical, or both")
    if overlap not in {"first", "last", "error"}:
        raise ValueError("overlap must be first, last, or error")
    if boundary not in {"inside", "outside"}:
        raise ValueError("boundary must be inside or outside")
    if not np.isfinite(max_time_gap) and not np.isinf(max_time_gap):
        raise ValueError("max_time_gap must be non-negative or Inf")
    if max_time_gap < 0:
        raise ValueError("max_time_gap must be non-negative or Inf")
    for col in [x_col, y_col, time_col, *groups]:
        if col not in frame:
            raise KeyError(f"Missing required column: {col}")
    for col in [aoi_time_col, aoi_name_col, *groups]:
        if col not in defs:
            raise KeyError(f"Missing required AOI column: {col}")
    rect_ok = {left_col, right_col, top_col, bottom_col}.issubset(defs.columns)
    poly_ok = {vertex_x_col, vertex_y_col}.issubset(defs.columns)
    resolved_shape = shape
    if shape == "auto":
        if rect_ok:
            resolved_shape = "rectangle"
        elif poly_ok:
            resolved_shape = "polygon"
        else:
            raise ValueError("Could not infer dynamic AOI shape")
    required_shape = (
        [left_col, right_col, top_col, bottom_col]
        if resolved_shape == "rectangle"
        else [vertex_x_col, vertex_y_col]
    )
    if resolved_shape == "polygon" and vertex_order_col is not None:
        required_shape.append(vertex_order_col)
    missing = [c for c in required_shape if c not in defs]
    if missing:
        raise KeyError("Missing dynamic AOI geometry columns: " + ", ".join(missing))
    names = defs[aoi_name_col].astype("string")
    if names.isna().any() or names.str.strip().eq("").any():
        raise ValueError("Dynamic AOI names must be non-missing and non-empty")
    all_names = list(dict.fromkeys(names.astype(str).tolist()))
    membership = np.zeros((len(frame), len(all_names)), dtype=bool)
    matched_times = np.full(len(frame), np.nan)
    time_gaps = np.full(len(frame), np.nan)
    x_all = pd.to_numeric(frame[x_col], errors="coerce").to_numpy(float)
    y_all = pd.to_numeric(frame[y_col], errors="coerce").to_numpy(float)
    sample_times = pd.to_numeric(frame[time_col], errors="coerce").to_numpy(float)
    definition_times = pd.to_numeric(defs[aoi_time_col], errors="coerce").to_numpy(float)
    sample_keys = _gp3_dynamic_group_keys(frame, groups).to_numpy(dtype=str)
    definition_keys = _gp3_dynamic_group_keys(defs, groups).to_numpy(dtype=str)
    name_to_col = {name: i for i, name in enumerate(all_names)}
    for key in pd.unique(sample_keys):
        sample_idx = np.flatnonzero(sample_keys == key)
        def_idx = np.flatnonzero(definition_keys == key)
        available = np.unique(definition_times[def_idx][np.isfinite(definition_times[def_idx])])
        if len(def_idx) == 0 or len(available) == 0:
            continue
        selected = np.asarray(
            [_gp3_dynamic_match_time(v, available, match) for v in sample_times[sample_idx]],
            dtype=float,
        )
        gaps = np.abs(sample_times[sample_idx] - selected)
        valid = (
            np.isfinite(sample_times[sample_idx])
            & np.isfinite(selected)
            & np.isfinite(gaps)
            & (gaps <= max_time_gap)
        )
        selected[~valid] = np.nan
        gaps[~valid] = np.nan
        matched_times[sample_idx] = selected
        time_gaps[sample_idx] = gaps
        for definition_time in np.unique(selected[np.isfinite(selected)]):
            local_sample = sample_idx[np.isfinite(selected) & (selected == definition_time)]
            local_defs = defs.iloc[
                def_idx[
                    np.isfinite(definition_times[def_idx])
                    & (definition_times[def_idx] == definition_time)
                ]
            ].copy()
            if resolved_shape == "rectangle":
                lx = pd.to_numeric(local_defs[left_col], errors="coerce").to_numpy(float)
                rx = pd.to_numeric(local_defs[right_col], errors="coerce").to_numpy(float)
                ty = pd.to_numeric(local_defs[top_col], errors="coerce").to_numpy(float)
                by = pd.to_numeric(local_defs[bottom_col], errors="coerce").to_numpy(float)
                if not np.isfinite(np.column_stack([lx, rx, ty, by])).all():
                    raise ValueError("Dynamic AOI rectangle bounds must be finite")
                for j, (_, row) in enumerate(local_defs.iterrows()):
                    hit = (
                        np.isfinite(x_all[local_sample])
                        & np.isfinite(y_all[local_sample])
                        & (x_all[local_sample] >= min(lx[j], rx[j]))
                        & (x_all[local_sample] <= max(lx[j], rx[j]))
                        & (y_all[local_sample] >= min(ty[j], by[j]))
                        & (y_all[local_sample] <= max(ty[j], by[j]))
                    )
                    membership[local_sample, name_to_col[str(row[aoi_name_col])]] = hit
            else:
                definitions = _gp3_polygon_r_prepare(
                    local_defs,
                    aoi_col=aoi_name_col,
                    vertex_x_col=vertex_x_col,
                    vertex_y_col=vertex_y_col,
                    vertex_order_col=vertex_order_col,
                )
                local_membership = _gp3_polygon_r_membership(
                    x_all[local_sample], y_all[local_sample], definitions, boundary=boundary
                )
                for name in local_membership.columns:
                    membership[local_sample, name_to_col[str(name)]] = local_membership[
                        name
                    ].to_numpy(bool)
    out = _gp3_dynamic_apply_membership(
        frame,
        membership,
        all_names,
        output=output,
        prefix=prefix,
        label_col=output_col,
        outside_label=outside_label,
        overlap=overlap,
        include_overlap_count=include_overlap_count,
        valid_xy=np.isfinite(x_all) & np.isfinite(y_all),
    )
    out[definition_time_col] = matched_times
    out[time_gap_col] = time_gaps
    if output in {"label", "both"}:
        out.loc[~np.isfinite(matched_times), output_col] = pd.NA
    out.attrs.update(frame.attrs)
    out.attrs["gazepoint_dynamic_aoi_settings"] = {
        "shape": resolved_shape,
        "group_cols": groups,
        "match": match,
        "max_time_gap": max_time_gap,
        "aoi_time_col": aoi_time_col,
        "aoi_name_col": aoi_name_col,
        "definition_time_col": definition_time_col,
        "time_gap_col": time_gap_col,
    }
    return out


def _gp3_aoi_geometry_r_audit(
    data,
    *,
    aoi_col=None,
    stimulus_col=None,
    x_min_col=None,
    y_min_col=None,
    x_max_col=None,
    y_max_col=None,
    x_col=None,
    y_col=None,
    width_col=None,
    height_col=None,
    screen_x_range=(0, 1),
    screen_y_range=(0, 1),
    min_width=0,
    min_height=0,
    min_area=0,
    max_area_prop=1,
    require_within_screen=True,
):
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a data frame")
    if data.empty:
        raise ValueError("data must contain at least one row")

    frame = _gp3_aoi_coding_r_aliases(data)
    columns = frame.columns

    resolved_aoi = _gp3_aoi_coding_r_resolve(
        aoi_col,
        columns,
        "aoi_col",
        ("aoi", "aoi_name", "aoi_id", "AOI", "AOI_NAME", "AOI_ID"),
        True,
    )
    resolved_stimulus = _gp3_aoi_coding_r_resolve(
        stimulus_col,
        columns,
        "stimulus_col",
    )
    resolved_x_min = _gp3_aoi_coding_r_resolve(
        x_min_col,
        columns,
        "x_min_col",
        ("x_min", "xmin", "left", "Left", "AOI_X_MIN", "AOI_LEFT"),
    )
    resolved_y_min = _gp3_aoi_coding_r_resolve(
        y_min_col,
        columns,
        "y_min_col",
        ("y_min", "ymin", "top", "Top", "AOI_Y_MIN", "AOI_TOP"),
    )
    resolved_x_max = _gp3_aoi_coding_r_resolve(
        x_max_col,
        columns,
        "x_max_col",
        ("x_max", "xmax", "right", "Right", "AOI_X_MAX", "AOI_RIGHT"),
    )
    resolved_y_max = _gp3_aoi_coding_r_resolve(
        y_max_col,
        columns,
        "y_max_col",
        ("y_max", "ymax", "bottom", "Bottom", "AOI_Y_MAX", "AOI_BOTTOM"),
    )
    resolved_x = _gp3_aoi_coding_r_resolve(
        x_col,
        columns,
        "x_col",
        ("x", "X", "aoi_x", "AOI_X"),
    )
    resolved_y = _gp3_aoi_coding_r_resolve(
        y_col,
        columns,
        "y_col",
        ("y", "Y", "aoi_y", "AOI_Y"),
    )
    resolved_width = _gp3_aoi_coding_r_resolve(
        width_col,
        columns,
        "width_col",
        ("width", "Width", "aoi_width", "AOI_WIDTH"),
    )
    resolved_height = _gp3_aoi_coding_r_resolve(
        height_col,
        columns,
        "height_col",
        ("height", "Height", "aoi_height", "AOI_HEIGHT"),
    )

    has_bounds = all(
        value is not None
        for value in (
            resolved_x_min,
            resolved_y_min,
            resolved_x_max,
            resolved_y_max,
        )
    )
    has_origin_size = all(
        value is not None
        for value in (
            resolved_x,
            resolved_y,
            resolved_width,
            resolved_height,
        )
    )
    if not has_bounds and not has_origin_size:
        raise ValueError(
            "AOI geometry requires either x/y min-max columns or x/y plus width/height columns"
        )

    screen_x = _gp3_aoi_coding_r_range(screen_x_range, "screen_x_range")
    screen_y = _gp3_aoi_coding_r_range(screen_y_range, "screen_y_range")

    for value, arg in (
        (min_width, "min_width"),
        (min_height, "min_height"),
        (min_area, "min_area"),
    ):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"{arg} must be a non-negative numeric scalar")

    if (
        isinstance(max_area_prop, (bool, np.bool_))
        or not isinstance(max_area_prop, (int, float, np.integer, np.floating))
        or not np.isfinite(max_area_prop)
        or max_area_prop < 0
        or max_area_prop > 1
    ):
        raise ValueError("max_area_prop must be a numeric scalar between 0 and 1")

    if not isinstance(require_within_screen, (bool, np.bool_)):
        raise ValueError("require_within_screen must be TRUE or FALSE")

    if has_bounds:
        xmin = _gp3_aoi_coding_r_numeric(frame[resolved_x_min])
        ymin = _gp3_aoi_coding_r_numeric(frame[resolved_y_min])
        xmax = _gp3_aoi_coding_r_numeric(frame[resolved_x_max])
        ymax = _gp3_aoi_coding_r_numeric(frame[resolved_y_max])
        coordinate_format = "bounds"
    else:
        xmin = _gp3_aoi_coding_r_numeric(frame[resolved_x])
        ymin = _gp3_aoi_coding_r_numeric(frame[resolved_y])
        width_input = _gp3_aoi_coding_r_numeric(frame[resolved_width])
        height_input = _gp3_aoi_coding_r_numeric(frame[resolved_height])
        xmax = xmin + width_input
        ymax = ymin + height_input
        coordinate_format = "origin_size"

    width = xmax - xmin
    height = ymax - ymin
    area = width * height
    screen_width = float(screen_x[1] - screen_x[0])
    screen_height = float(screen_y[1] - screen_y[0])
    screen_area = screen_width * screen_height
    area_prop = area / screen_area
    center_x = xmin + width / 2
    center_y = ymin + height / 2

    invalid_coordinate = ~(
        np.isfinite(xmin) & np.isfinite(ymin) & np.isfinite(xmax) & np.isfinite(ymax)
    )
    invalid_dimension = ~invalid_coordinate & ((width <= 0) | (height <= 0))
    too_small = (
        ~invalid_coordinate
        & ~invalid_dimension
        & ((width < float(min_width)) | (height < float(min_height)) | (area < float(min_area)))
    )
    too_large = ~invalid_coordinate & ~invalid_dimension & (area_prop > float(max_area_prop))
    outside_screen = (
        ~invalid_coordinate
        & ~invalid_dimension
        & (
            (xmin < screen_x[0])
            | (xmax > screen_x[1])
            | (ymin < screen_y[0])
            | (ymax > screen_y[1])
        )
    )

    status = np.full(len(frame), "ok", dtype=object)
    status[too_large] = "too_large"
    status[too_small] = "too_small"
    if bool(require_within_screen):
        status[outside_screen] = "outside_screen"
    status[invalid_dimension] = "invalid_dimension"
    status[invalid_coordinate] = "invalid_coordinate"

    id_cols = [resolved_aoi]
    if resolved_stimulus is not None:
        id_cols.append(resolved_stimulus)

    geometry_summary = frame[id_cols].copy().reset_index(drop=True)
    geometry_summary["x_min"] = xmin
    geometry_summary["y_min"] = ymin
    geometry_summary["x_max"] = xmax
    geometry_summary["y_max"] = ymax
    geometry_summary["width"] = width
    geometry_summary["height"] = height
    geometry_summary["area"] = area
    geometry_summary["area_prop"] = area_prop
    geometry_summary["center_x"] = center_x
    geometry_summary["center_y"] = center_y
    geometry_summary["outside_screen"] = outside_screen
    geometry_summary["aoi_geometry_status"] = status

    def _safe(series, op):
        values = pd.to_numeric(series, errors="coerce").to_numpy(float)
        values = values[np.isfinite(values)]
        if len(values) == 0:
            return np.nan
        if op == "min":
            return float(np.min(values))
        if op == "median":
            return float(np.median(values))
        return float(np.max(values))

    size_summary = pd.DataFrame(
        [
            {
                "n_aois": len(geometry_summary),
                "min_width": _safe(geometry_summary["width"], "min"),
                "median_width": _safe(geometry_summary["width"], "median"),
                "max_width": _safe(geometry_summary["width"], "max"),
                "min_height": _safe(geometry_summary["height"], "min"),
                "median_height": _safe(geometry_summary["height"], "median"),
                "max_height": _safe(geometry_summary["height"], "max"),
                "min_area": _safe(geometry_summary["area"], "min"),
                "median_area": _safe(geometry_summary["area"], "median"),
                "max_area": _safe(geometry_summary["area"], "max"),
                "min_area_prop": _safe(geometry_summary["area_prop"], "min"),
                "median_area_prop": _safe(geometry_summary["area_prop"], "median"),
                "max_area_prop": _safe(geometry_summary["area_prop"], "max"),
            }
        ]
    )

    duplicate_group_cols = []
    if resolved_stimulus is not None:
        duplicate_group_cols.append(resolved_stimulus)
    duplicate_group_cols.extend(["x_min", "y_min", "x_max", "y_max"])
    duplicate_rows = []
    for key, block in geometry_summary.groupby(
        duplicate_group_cols,
        dropna=True,
        sort=True,
    ):
        if len(block) <= 1:
            continue
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(duplicate_group_cols, key, strict=True))
        row["n_aois"] = len(block)
        row["aoi_values"] = ", ".join(block[resolved_aoi].astype(str))
        row["duplicate_geometry_status"] = "duplicate_geometry"
        duplicate_rows.append(row)

    duplicate_columns = duplicate_group_cols + [
        "n_aois",
        "aoi_values",
        "duplicate_geometry_status",
    ]
    duplicate_geometry = pd.DataFrame(
        duplicate_rows,
        columns=duplicate_columns,
    )
    if not duplicate_rows:
        duplicate_geometry = pd.DataFrame(
            columns=[
                "n_aois",
                "aoi_values",
                "duplicate_geometry_status",
            ]
        )

    flagged_aois = geometry_summary.loc[
        geometry_summary["aoi_geometry_status"].ne("ok")
    ].reset_index(drop=True)

    n_duplicate_groups = int(len(duplicate_geometry))
    overview = pd.DataFrame(
        [
            {
                "n_rows": len(frame),
                "n_aois": len(geometry_summary),
                "n_stimuli": (
                    int(geometry_summary[resolved_stimulus].nunique(dropna=False))
                    if resolved_stimulus is not None
                    else pd.NA
                ),
                "n_flagged_aois": len(flagged_aois),
                "n_duplicate_geometry_groups": n_duplicate_groups,
                "coordinate_format": coordinate_format,
                "screen_width": screen_width,
                "screen_height": screen_height,
                "screen_area": screen_area,
                "aoi_geometry_status": (
                    "review" if len(flagged_aois) > 0 or n_duplicate_groups > 0 else "ok"
                ),
            }
        ]
    )

    def _setting(value):
        if value is None:
            return pd.NA
        if isinstance(value, (bool, np.bool_)):
            return "TRUE" if bool(value) else "FALSE"
        return str(value)

    settings = pd.DataFrame(
        {
            "setting": [
                "aoi_col",
                "stimulus_col",
                "x_min_col",
                "y_min_col",
                "x_max_col",
                "y_max_col",
                "x_col",
                "y_col",
                "width_col",
                "height_col",
                "screen_x_range",
                "screen_y_range",
                "min_width",
                "min_height",
                "min_area",
                "max_area_prop",
                "require_within_screen",
            ],
            "value": [
                resolved_aoi,
                _setting(resolved_stimulus),
                _setting(resolved_x_min),
                _setting(resolved_y_min),
                _setting(resolved_x_max),
                _setting(resolved_y_max),
                _setting(resolved_x),
                _setting(resolved_y),
                _setting(resolved_width),
                _setting(resolved_height),
                f"{screen_x[0]:g}, {screen_x[1]:g}",
                f"{screen_y[0]:g}, {screen_y[1]:g}",
                _setting(min_width),
                _setting(min_height),
                _setting(min_area),
                _setting(max_area_prop),
                _setting(require_within_screen),
            ],
        }
    )

    return {
        "overview": overview,
        "geometry_summary": geometry_summary,
        "size_summary": size_summary,
        "duplicate_geometry": duplicate_geometry,
        "flagged_aois": flagged_aois,
        "settings": settings,
        "_gp3_class": "gp3_aoi_geometry_audit",
    }


def audit_gazepoint_aoi_geometry(
    aoi_geometry=None,
    *,
    data=None,
    aoi_col=None,
    stimulus_col=None,
    x_min_col=None,
    y_min_col=None,
    x_max_col=None,
    y_max_col=None,
    x_col=None,
    y_col=None,
    width_col=None,
    height_col=None,
    screen_x_range=(0, 1),
    screen_y_range=(0, 1),
    min_width=0,
    min_height=0,
    min_area=0,
    max_area_prop=1,
    require_within_screen=True,
):
    """Audit AOI geometry with legacy and R v2.3.0-compatible contracts."""
    r_mode = any(
        [
            aoi_col is not None,
            stimulus_col is not None,
            x_min_col is not None,
            y_min_col is not None,
            x_max_col is not None,
            y_max_col is not None,
            x_col is not None,
            y_col is not None,
            width_col is not None,
            height_col is not None,
            tuple(screen_x_range) != (0, 1),
            tuple(screen_y_range) != (0, 1),
            min_width != 0,
            min_height != 0,
            min_area != 0,
            max_area_prop != 1,
            require_within_screen is not True,
        ]
    )
    if not r_mode:
        if data is not None:
            if aoi_geometry is not None:
                raise TypeError("supply either aoi_geometry or data, not both")
            aoi_geometry = data
        g = ensure_dataframe(aoi_geometry, copy=False)
        issues = []
        required = ["xmin", "xmax", "ymin", "ymax"]
        missing = [column for column in required if column not in g]
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

    if data is not None:
        if aoi_geometry is not None:
            raise TypeError("supply either aoi_geometry or data, not both")
        aoi_geometry = data

    return _gp3_aoi_geometry_r_audit(
        aoi_geometry,
        aoi_col=aoi_col,
        stimulus_col=stimulus_col,
        x_min_col=x_min_col,
        y_min_col=y_min_col,
        x_max_col=x_max_col,
        y_max_col=y_max_col,
        x_col=x_col,
        y_col=y_col,
        width_col=width_col,
        height_col=height_col,
        screen_x_range=screen_x_range,
        screen_y_range=screen_y_range,
        min_width=min_width,
        min_height=min_height,
        min_area=min_area,
        max_area_prop=max_area_prop,
        require_within_screen=require_within_screen,
    )


def audit_gazepoint_aoi_overlap(
    aoi_geometry=None,
    *,
    data=None,
    aoi_col=None,
    stimulus_col=None,
    x_min_col=None,
    y_min_col=None,
    x_max_col=None,
    y_max_col=None,
    x_col=None,
    y_col=None,
    width_col=None,
    height_col=None,
    screen_x_range=(0, 1),
    screen_y_range=(0, 1),
    min_overlap_area=0,
    min_overlap_prop=0,
    ignore_invalid_geometry=True,
):
    """Audit pairwise AOI overlap with legacy and R v2.3.0-compatible contracts."""
    r_mode = any(
        [
            aoi_col is not None,
            stimulus_col is not None,
            x_min_col is not None,
            y_min_col is not None,
            x_max_col is not None,
            y_max_col is not None,
            x_col is not None,
            y_col is not None,
            width_col is not None,
            height_col is not None,
            tuple(screen_x_range) != (0, 1),
            tuple(screen_y_range) != (0, 1),
            min_overlap_area != 0,
            min_overlap_prop != 0,
            ignore_invalid_geometry is not True,
        ]
    )
    if not r_mode:
        if data is not None:
            if aoi_geometry is not None:
                raise TypeError("supply either aoi_geometry or data, not both")
            aoi_geometry = data
        g = ensure_dataframe(aoi_geometry, copy=False)
        name = next((column for column in ("aoi", "name", "label") if column in g), None)
        rows = []
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g.iloc[i], g.iloc[j]
                x_overlap = max(0, min(a.xmax, b.xmax) - max(a.xmin, b.xmin))
                y_overlap = max(0, min(a.ymax, b.ymax) - max(a.ymin, b.ymin))
                area = x_overlap * y_overlap
                if area > 0:
                    rows.append(
                        {
                            "aoi1": a[name] if name else i,
                            "aoi2": b[name] if name else j,
                            "overlap_area": area,
                        }
                    )
        return pd.DataFrame(
            rows,
            columns=["aoi1", "aoi2", "overlap_area"],
        )

    if data is not None:
        if aoi_geometry is not None:
            raise TypeError("supply either aoi_geometry or data, not both")
        aoi_geometry = data

    if (
        isinstance(min_overlap_area, (bool, np.bool_))
        or not isinstance(
            min_overlap_area,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(min_overlap_area)
        or min_overlap_area < 0
    ):
        raise ValueError("min_overlap_area must be a non-negative numeric scalar")
    if (
        isinstance(min_overlap_prop, (bool, np.bool_))
        or not isinstance(
            min_overlap_prop,
            (int, float, np.integer, np.floating),
        )
        or not np.isfinite(min_overlap_prop)
        or min_overlap_prop < 0
        or min_overlap_prop > 1
    ):
        raise ValueError("min_overlap_prop must be a numeric scalar between 0 and 1")
    if not isinstance(ignore_invalid_geometry, (bool, np.bool_)):
        raise ValueError("ignore_invalid_geometry must be TRUE or FALSE")

    geometry_audit = _gp3_aoi_geometry_r_audit(
        aoi_geometry,
        aoi_col=aoi_col,
        stimulus_col=stimulus_col,
        x_min_col=x_min_col,
        y_min_col=y_min_col,
        x_max_col=x_max_col,
        y_max_col=y_max_col,
        x_col=x_col,
        y_col=y_col,
        width_col=width_col,
        height_col=height_col,
        screen_x_range=screen_x_range,
        screen_y_range=screen_y_range,
        require_within_screen=False,
    )
    geometry_summary = geometry_audit["geometry_summary"]
    settings_lookup = dict(
        zip(
            geometry_audit["settings"]["setting"],
            geometry_audit["settings"]["value"],
            strict=True,
        )
    )
    resolved_aoi = settings_lookup["aoi_col"]
    resolved_stimulus = settings_lookup["stimulus_col"]
    if pd.isna(resolved_stimulus) or str(resolved_stimulus) == "":
        resolved_stimulus = None
    else:
        resolved_stimulus = str(resolved_stimulus)

    geometry_for_overlap = geometry_summary
    if bool(ignore_invalid_geometry):
        geometry_for_overlap = geometry_for_overlap.loc[
            ~geometry_for_overlap["aoi_geometry_status"].isin(
                ["invalid_coordinate", "invalid_dimension"]
            )
        ].copy()

    pair_columns = [
        "aoi_1",
        "aoi_2",
        "x_min_1",
        "y_min_1",
        "x_max_1",
        "y_max_1",
        "x_min_2",
        "y_min_2",
        "x_max_2",
        "y_max_2",
        "overlap_x_min",
        "overlap_y_min",
        "overlap_x_max",
        "overlap_y_max",
        "overlap_width",
        "overlap_height",
        "overlap_area",
        "overlap_prop_aoi_1",
        "overlap_prop_aoi_2",
        "overlap_prop_smaller",
        "aoi_overlap_status",
    ]
    if resolved_stimulus is not None:
        pair_columns = [resolved_stimulus] + pair_columns

    pair_rows = []
    if resolved_stimulus is not None:
        grouped = geometry_for_overlap.groupby(
            resolved_stimulus,
            dropna=True,
            sort=True,
        )
    else:
        grouped = [("all_stimuli", geometry_for_overlap)]

    for stimulus_value, block in grouped:
        block = block.reset_index(drop=True)
        for i in range(len(block)):
            for j in range(i + 1, len(block)):
                a = block.iloc[i]
                b = block.iloc[j]
                overlap_x_min = max(a["x_min"], b["x_min"])
                overlap_y_min = max(a["y_min"], b["y_min"])
                overlap_x_max = min(a["x_max"], b["x_max"])
                overlap_y_max = min(a["y_max"], b["y_max"])
                overlap_width = max(0.0, overlap_x_max - overlap_x_min)
                overlap_height = max(0.0, overlap_y_max - overlap_y_min)
                overlap_area = overlap_width * overlap_height

                area_a = float(a["area"]) if np.isfinite(a["area"]) else np.nan
                area_b = float(b["area"]) if np.isfinite(b["area"]) else np.nan
                prop_a = overlap_area / area_a if np.isfinite(area_a) and area_a > 0 else np.nan
                prop_b = overlap_area / area_b if np.isfinite(area_b) and area_b > 0 else np.nan
                smaller = (
                    min(area_a, area_b) if np.isfinite(area_a) and np.isfinite(area_b) else np.nan
                )
                prop_smaller = (
                    overlap_area / smaller if np.isfinite(smaller) and smaller > 0 else np.nan
                )
                flagged = overlap_area > float(min_overlap_area) or (
                    np.isfinite(prop_smaller) and prop_smaller > float(min_overlap_prop)
                )
                row = {
                    "aoi_1": str(a[resolved_aoi]),
                    "aoi_2": str(b[resolved_aoi]),
                    "x_min_1": a["x_min"],
                    "y_min_1": a["y_min"],
                    "x_max_1": a["x_max"],
                    "y_max_1": a["y_max"],
                    "x_min_2": b["x_min"],
                    "y_min_2": b["y_min"],
                    "x_max_2": b["x_max"],
                    "y_max_2": b["y_max"],
                    "overlap_x_min": overlap_x_min,
                    "overlap_y_min": overlap_y_min,
                    "overlap_x_max": overlap_x_max,
                    "overlap_y_max": overlap_y_max,
                    "overlap_width": overlap_width,
                    "overlap_height": overlap_height,
                    "overlap_area": overlap_area,
                    "overlap_prop_aoi_1": prop_a,
                    "overlap_prop_aoi_2": prop_b,
                    "overlap_prop_smaller": prop_smaller,
                    "aoi_overlap_status": "overlap" if flagged else "ok",
                }
                if resolved_stimulus is not None:
                    row = {resolved_stimulus: str(stimulus_value), **row}
                pair_rows.append(row)

    pairwise_overlap = pd.DataFrame(pair_rows, columns=pair_columns)
    flagged_overlaps = pairwise_overlap.loc[
        pairwise_overlap["aoi_overlap_status"].ne("ok")
    ].reset_index(drop=True)

    def _safe_max(values):
        numeric = pd.to_numeric(values, errors="coerce").to_numpy(float)
        numeric = numeric[np.isfinite(numeric)]
        return float(np.max(numeric)) if len(numeric) else np.nan

    summary_columns = [
        "n_aoi_pairs",
        "n_overlapping_pairs",
        "n_flagged_overlaps",
        "max_overlap_area",
        "max_overlap_prop_smaller",
        "aoi_overlap_summary_status",
    ]
    summary_rows = []
    if len(pairwise_overlap):
        if resolved_stimulus is not None:
            summary_groups = pairwise_overlap.groupby(
                resolved_stimulus,
                dropna=True,
                sort=True,
            )
        else:
            summary_groups = [(None, pairwise_overlap)]
        for stimulus_value, block in summary_groups:
            row = {
                "n_aoi_pairs": len(block),
                "n_overlapping_pairs": int((block["overlap_area"] > 0).sum()),
                "n_flagged_overlaps": int(block["aoi_overlap_status"].ne("ok").sum()),
                "max_overlap_area": _safe_max(block["overlap_area"]),
                "max_overlap_prop_smaller": _safe_max(block["overlap_prop_smaller"]),
                "aoi_overlap_summary_status": (
                    "review" if block["aoi_overlap_status"].ne("ok").any() else "ok"
                ),
            }
            if resolved_stimulus is not None:
                row = {resolved_stimulus: str(stimulus_value), **row}
            summary_rows.append(row)

    overlap_summary_columns = (
        [resolved_stimulus] + summary_columns if resolved_stimulus is not None else summary_columns
    )
    overlap_summary = pd.DataFrame(
        summary_rows,
        columns=overlap_summary_columns,
    )

    overview = pd.DataFrame(
        [
            {
                "n_rows": len(aoi_geometry),
                "n_aois": len(geometry_summary),
                "n_aois_used": len(geometry_for_overlap),
                "n_stimuli": (
                    int(geometry_summary[resolved_stimulus].nunique(dropna=False))
                    if resolved_stimulus is not None
                    else pd.NA
                ),
                "n_aoi_pairs": len(pairwise_overlap),
                "n_overlapping_pairs": int((pairwise_overlap["overlap_area"] > 0).sum()),
                "n_flagged_overlaps": len(flagged_overlaps),
                "max_overlap_area": _safe_max(pairwise_overlap["overlap_area"]),
                "max_overlap_prop_smaller": _safe_max(pairwise_overlap["overlap_prop_smaller"]),
                "aoi_overlap_status": ("review" if len(flagged_overlaps) > 0 else "ok"),
            }
        ]
    )

    def _text(value):
        if value is None or pd.isna(value):
            return pd.NA
        return str(value)

    settings = pd.DataFrame(
        {
            "setting": [
                "aoi_col",
                "stimulus_col",
                "x_min_col",
                "y_min_col",
                "x_max_col",
                "y_max_col",
                "x_col",
                "y_col",
                "width_col",
                "height_col",
                "screen_x_range",
                "screen_y_range",
                "min_overlap_area",
                "min_overlap_prop",
                "ignore_invalid_geometry",
            ],
            "value": [
                resolved_aoi,
                _text(resolved_stimulus),
                settings_lookup["x_min_col"],
                settings_lookup["y_min_col"],
                settings_lookup["x_max_col"],
                settings_lookup["y_max_col"],
                settings_lookup["x_col"],
                settings_lookup["y_col"],
                settings_lookup["width_col"],
                settings_lookup["height_col"],
                settings_lookup["screen_x_range"],
                settings_lookup["screen_y_range"],
                str(min_overlap_area),
                str(min_overlap_prop),
                "TRUE" if bool(ignore_invalid_geometry) else "FALSE",
            ],
        }
    )

    return {
        "overview": overview,
        "geometry_summary": geometry_summary,
        "pairwise_overlap": pairwise_overlap,
        "overlap_summary": overlap_summary,
        "flagged_overlaps": flagged_overlaps,
        "settings": settings,
        "_gp3_class": "gp3_aoi_overlap_audit",
    }


def audit_gazepoint_aoi_screen_coverage(
    aoi_geometry,
    width=1.0,
    height=1.0,
    aoi_col=None,
    x_min_col="x_min",
    x_max_col="x_max",
    y_min_col="y_min",
    y_max_col="y_max",
    margin=0,
):
    """Audit AOI screen coverage; legacy summary is retained for default calls."""
    g = ensure_dataframe(aoi_geometry, copy=False)
    r_mode = (
        aoi_col is not None
        or x_min_col != "x_min"
        or x_max_col != "x_max"
        or y_min_col != "y_min"
        or y_max_col != "y_max"
        or margin != 0
    )
    if not r_mode:
        total = float(width * height)
        x0 = "xmin" if "xmin" in g.columns else x_min_col
        x1 = "xmax" if "xmax" in g.columns else x_max_col
        y0 = "ymin" if "ymin" in g.columns else y_min_col
        y1 = "ymax" if "ymax" in g.columns else y_max_col
        areas = (g[x1] - g[x0]).clip(lower=0) * (g[y1] - g[y0]).clip(lower=0)
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

    screen_width = float(width)
    screen_height = float(height)
    if not np.isfinite(screen_width) or screen_width <= 0:
        raise ValueError("screen_width must be positive")
    if not np.isfinite(screen_height) or screen_height <= 0:
        raise ValueError("screen_height must be positive")
    if not isinstance(margin, (int, float, np.integer, np.floating)) or margin < 0:
        raise ValueError("margin must be a single non-negative numeric value")

    required = [x_min_col, x_max_col, y_min_col, y_max_col]
    if aoi_col is not None:
        required.insert(0, aoi_col)
    missing = [column for column in required if column not in g.columns]
    if missing:
        raise ValueError("data is missing required column(s): " + ", ".join(missing))

    x_min = pd.to_numeric(g[x_min_col], errors="coerce").to_numpy(float)
    x_max = pd.to_numeric(g[x_max_col], errors="coerce").to_numpy(float)
    y_min = pd.to_numeric(g[y_min_col], errors="coerce").to_numpy(float)
    y_max = pd.to_numeric(g[y_max_col], errors="coerce").to_numpy(float)
    aoi_id = (
        g[aoi_col].astype(str).to_numpy()
        if aoi_col is not None
        else np.asarray([f"AOI_{i}" for i in range(1, len(g) + 1)], dtype=object)
    )

    missing_geometry = ~(
        np.isfinite(x_min) & np.isfinite(x_max) & np.isfinite(y_min) & np.isfinite(y_max)
    )
    invalid_rectangle = ~missing_geometry & ((x_max <= x_min) | (y_max <= y_min))
    offscreen_left = ~missing_geometry & (x_min < -margin)
    offscreen_right = ~missing_geometry & (x_max > screen_width + margin)
    offscreen_top = ~missing_geometry & (y_min < -margin)
    offscreen_bottom = ~missing_geometry & (y_max > screen_height + margin)
    outside_screen = offscreen_left | offscreen_right | offscreen_top | offscreen_bottom

    raw_width = np.where(missing_geometry | invalid_rectangle, np.nan, x_max - x_min)
    raw_height = np.where(missing_geometry | invalid_rectangle, np.nan, y_max - y_min)
    raw_area = raw_width * raw_height
    clipped_x_min = np.maximum(0, np.minimum(screen_width, x_min))
    clipped_x_max = np.maximum(0, np.minimum(screen_width, x_max))
    clipped_y_min = np.maximum(0, np.minimum(screen_height, y_min))
    clipped_y_max = np.maximum(0, np.minimum(screen_height, y_max))
    clipped_area = np.maximum(0, clipped_x_max - clipped_x_min) * np.maximum(
        0, clipped_y_max - clipped_y_min
    )
    clipped_area[missing_geometry | invalid_rectangle] = np.nan
    screen_area = screen_width * screen_height

    aoi_summary = pd.DataFrame(
        {
            "aoi_id": aoi_id,
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "width": raw_width,
            "height": raw_height,
            "raw_area": raw_area,
            "clipped_area": clipped_area,
            "raw_screen_coverage": raw_area / screen_area,
            "clipped_screen_coverage": clipped_area / screen_area,
            "missing_geometry": missing_geometry,
            "invalid_rectangle": invalid_rectangle,
            "outside_screen": outside_screen,
            "offscreen_left": offscreen_left,
            "offscreen_right": offscreen_right,
            "offscreen_top": offscreen_top,
            "offscreen_bottom": offscreen_bottom,
        }
    )
    overall_summary = pd.DataFrame(
        [
            {
                "n_aois": len(aoi_summary),
                "n_missing_geometry": int(missing_geometry.sum()),
                "n_invalid_rectangles": int(invalid_rectangle.sum()),
                "n_outside_screen": int(outside_screen.sum()),
                "total_raw_area": float(np.nansum(raw_area)),
                "total_clipped_area": float(np.nansum(clipped_area)),
                "total_raw_screen_coverage": float(np.nansum(raw_area / screen_area)),
                "total_clipped_screen_coverage": float(np.nansum(clipped_area / screen_area)),
                "coverage_note": "Coverage sums are descriptive and do not correct for AOI overlap.",
            }
        ]
    )
    return {
        "aoi_summary": aoi_summary,
        "overall_summary": overall_summary,
        "settings": {
            "screen_width": screen_width,
            "screen_height": screen_height,
            "aoi_col": aoi_col,
            "x_min_col": x_min_col,
            "x_max_col": x_max_col,
            "y_min_col": y_min_col,
            "y_max_col": y_max_col,
            "margin": margin,
        },
    }


def audit_gazepoint_dynamic_aoi_coverage(
    data,
    aoi_col="aoi_current",
    *,
    label_col=None,
    definition_time_col=None,
    time_gap_col=None,
    group_cols=None,
    outside_label="outside",
    max_time_gap=None,
    x_col=None,
    y_col=None,
):
    """Audit dynamic-AOI coverage while preserving the historical summary mode."""
    r_mode = (
        any(
            value is not None
            for value in (
                label_col,
                definition_time_col,
                time_gap_col,
                group_cols,
                max_time_gap,
                x_col,
                y_col,
            )
        )
        or outside_label != "outside"
    )
    df = ensure_dataframe(data, copy=False)
    if not r_mode:
        values = df[aoi_col] if aoi_col in df else pd.Series(pd.NA, index=df.index)
        return pd.DataFrame(
            [
                {
                    "n_samples": len(df),
                    "n_assigned": int(values.notna().sum()),
                    "assigned_prop": float(values.notna().mean()),
                    "n_outside": int(values.astype("string").eq("outside").sum()),
                }
            ]
        )

    label_col = "aoi_current" if label_col is None else label_col
    definition_time_col = (
        "aoi_definition_time" if definition_time_col is None else definition_time_col
    )
    time_gap_col = "aoi_time_gap" if time_gap_col is None else time_gap_col
    max_time_gap = np.inf if max_time_gap is None else float(max_time_gap)
    groups = (
        []
        if group_cols is None
        else ([group_cols] if isinstance(group_cols, str) else list(group_cols))
    )
    required = [label_col, definition_time_col, time_gap_col, *groups]
    required.extend(column for column in (x_col, y_col) if column is not None)
    missing = [column for column in dict.fromkeys(required) if column not in df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    if max_time_gap < 0 or not (np.isfinite(max_time_gap) or np.isinf(max_time_gap)):
        raise ValueError("max_time_gap must be one non-negative number or Inf")

    label = df[label_col].astype("string")
    definition_time = pd.to_numeric(df[definition_time_col], errors="coerce").to_numpy(float)
    time_gap = pd.to_numeric(df[time_gap_col], errors="coerce").to_numpy(float)
    has_definition = np.isfinite(definition_time)
    inside = has_definition & label.notna().to_numpy() & label.ne(outside_label).to_numpy()
    outside = has_definition & label.notna().to_numpy() & label.eq(outside_label).to_numpy()
    missing_gaze = np.zeros(len(df), dtype=bool)
    if x_col is not None and y_col is not None:
        x = pd.to_numeric(df[x_col], errors="coerce").to_numpy(float)
        y = pd.to_numeric(df[y_col], errors="coerce").to_numpy(float)
        missing_gaze = ~np.isfinite(x) | ~np.isfinite(y)
    excessive_gap = has_definition & np.isfinite(time_gap) & (time_gap > max_time_gap)
    issue = np.where(
        missing_gaze,
        "missing_gaze",
        np.where(
            ~has_definition,
            "no_dynamic_definition",
            np.where(
                excessive_gap,
                "definition_gap_exceeds_threshold",
                np.where(outside, "outside_all_aoi", "ok"),
            ),
        ),
    )

    def percent(value, total):
        return 100 * value / total if total else np.nan

    finite_gap = time_gap[np.isfinite(time_gap)]
    overview = pd.DataFrame(
        [
            {
                "n_rows": len(df),
                "n_with_definition": int(has_definition.sum()),
                "pct_with_definition": percent(has_definition.sum(), len(df)),
                "n_inside_aoi": int(inside.sum()),
                "pct_inside_aoi": percent(inside.sum(), len(df)),
                "n_outside_aoi": int(outside.sum()),
                "pct_outside_aoi": percent(outside.sum(), len(df)),
                "n_missing_gaze": int(missing_gaze.sum()),
                "n_excessive_gap": int(excessive_gap.sum()),
                "mean_time_gap": float(np.mean(finite_gap)) if len(finite_gap) else np.nan,
                "max_time_gap_observed": float(np.max(finite_gap)) if len(finite_gap) else np.nan,
                "audit_status": "ok" if np.all(issue == "ok") else "review",
            }
        ]
    )
    group_summary = pd.DataFrame()
    if groups:
        temp = df.assign(
            _has_definition=has_definition,
            _inside=inside,
            _outside=outside,
            _missing_gaze=missing_gaze,
            _excessive_gap=excessive_gap,
            _time_gap=time_gap,
        )
        rows = []
        for key, block in temp.groupby(groups, sort=True, dropna=False):
            key = key if isinstance(key, tuple) else (key,)
            finite = pd.to_numeric(block["_time_gap"], errors="coerce")
            finite = finite[np.isfinite(finite)]
            row = {column: value for column, value in zip(groups, key, strict=True)}
            row.update(
                {
                    "n_rows": len(block),
                    "n_with_definition": int(block["_has_definition"].sum()),
                    "pct_with_definition": percent(block["_has_definition"].sum(), len(block)),
                    "n_inside_aoi": int(block["_inside"].sum()),
                    "pct_inside_aoi": percent(block["_inside"].sum(), len(block)),
                    "n_outside_aoi": int(block["_outside"].sum()),
                    "pct_outside_aoi": percent(block["_outside"].sum(), len(block)),
                    "n_missing_gaze": int(block["_missing_gaze"].sum()),
                    "n_excessive_gap": int(block["_excessive_gap"].sum()),
                    "mean_time_gap": float(finite.mean()) if len(finite) else np.nan,
                    "max_time_gap_observed": float(finite.max()) if len(finite) else np.nan,
                }
            )
            rows.append(row)
        group_summary = pd.DataFrame(rows)

    levels = sorted({str(value) for value in label.dropna() if str(value) != outside_label})
    aoi_summary = pd.DataFrame(
        [
            {
                "aoi": value,
                "n_samples": int(label.eq(value).sum()),
                "pct_all_samples": percent(label.eq(value).sum(), len(df)),
                "pct_defined_samples": percent(label.eq(value).sum(), has_definition.sum()),
            }
            for value in levels
        ],
        columns=["aoi", "n_samples", "pct_all_samples", "pct_defined_samples"],
    )
    flagged = df.loc[issue != "ok"].copy()
    flagged["dynamic_aoi_issue"] = issue[issue != "ok"]
    return {
        "overview": overview,
        "group_summary": group_summary,
        "aoi_summary": aoi_summary,
        "flagged_rows": flagged,
        "settings": {
            "label_col": label_col,
            "definition_time_col": definition_time_col,
            "time_gap_col": time_gap_col,
            "group_cols": groups,
            "outside_label": outside_label,
            "max_time_gap": max_time_gap,
            "x_col": x_col,
            "y_col": y_col,
        },
        "_gp3_class": "gp3_dynamic_aoi_coverage_audit",
    }


def _gp3_aoi_coding_r_aliases(data):
    frame = data.copy()
    if "MEDIA_ID" in frame.columns and "media_id" not in frame.columns:
        frame["media_id"] = frame["MEDIA_ID"]
    if "AOI" in frame.columns and "aoi" not in frame.columns:
        frame["aoi"] = frame["AOI"]
    return frame


def _gp3_aoi_coding_r_resolve(col, columns, arg, candidates=(), required=False):
    if col is not None:
        if not isinstance(col, str) or not col:
            raise ValueError(f"{arg} must be a non-missing character scalar")
        if col == "MEDIA_ID" and "media_id" in columns:
            return "media_id"
        if col == "AOI" and "aoi" in columns:
            return "aoi"
        if col not in columns:
            raise ValueError(f"{arg} must be present in data")
        return col
    found = [candidate for candidate in candidates if candidate in columns]
    if found:
        first = found[0]
        if first == "MEDIA_ID" and "media_id" in columns:
            return "media_id"
        if first == "AOI" and "aoi" in columns:
            return "aoi"
        return first
    if required:
        raise ValueError(f"{arg} could not be detected and must be supplied")
    return None


def _gp3_aoi_coding_r_numeric(series):
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def _gp3_aoi_coding_r_range(value, arg):
    try:
        values = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{arg} must be a numeric length-2 vector with lower < upper") from exc
    if (
        values.ndim != 1
        or len(values) != 2
        or not np.isfinite(values).all()
        or values[0] >= values[1]
    ):
        raise ValueError(f"{arg} must be a numeric length-2 vector with lower < upper")
    return values


def _gp3_aoi_coding_r_prop(value, arg):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value < 0
        or value > 1
    ):
        raise ValueError(f"{arg} must be a numeric scalar between 0 and 1")
    return float(value)


def _gp3_aoi_coding_r_label(value, arg):
    if not isinstance(value, str) or not value:
        raise ValueError(f"{arg} must be a non-missing character scalar")
    return value


def _gp3_aoi_coding_r_character_vector(value, arg):
    if isinstance(value, str):
        values = [value]
    else:
        try:
            values = list(value)
        except TypeError as exc:
            raise ValueError(f"{arg} must be a non-empty character vector") from exc
    if not values or any(not isinstance(item, str) or not item for item in values):
        raise ValueError(f"{arg} must be a non-empty character vector")
    return values


def _gp3_aoi_coding_r_geometry(
    data,
    *,
    aoi_col,
    stimulus_col,
    x_min_col,
    y_min_col,
    x_max_col,
    y_max_col,
    x_col,
    y_col,
    width_col,
    height_col,
    screen_x_range,
    screen_y_range,
):
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("aoi_geometry must contain at least one row")
    frame = _gp3_aoi_coding_r_aliases(data)
    columns = frame.columns
    aoi_col = _gp3_aoi_coding_r_resolve(
        aoi_col,
        columns,
        "geometry_aoi_col",
        ("aoi", "aoi_name", "aoi_id", "AOI", "AOI_NAME", "AOI_ID"),
        True,
    )
    stimulus_col = _gp3_aoi_coding_r_resolve(stimulus_col, columns, "geometry_stimulus_col")
    x_min_col = _gp3_aoi_coding_r_resolve(
        x_min_col,
        columns,
        "x_min_col",
        ("x_min", "xmin", "left", "Left", "AOI_X_MIN", "AOI_LEFT"),
    )
    y_min_col = _gp3_aoi_coding_r_resolve(
        y_min_col,
        columns,
        "y_min_col",
        ("y_min", "ymin", "top", "Top", "AOI_Y_MIN", "AOI_TOP"),
    )
    x_max_col = _gp3_aoi_coding_r_resolve(
        x_max_col,
        columns,
        "x_max_col",
        ("x_max", "xmax", "right", "Right", "AOI_X_MAX", "AOI_RIGHT"),
    )
    y_max_col = _gp3_aoi_coding_r_resolve(
        y_max_col,
        columns,
        "y_max_col",
        ("y_max", "ymax", "bottom", "Bottom", "AOI_Y_MAX", "AOI_BOTTOM"),
    )
    x_col = _gp3_aoi_coding_r_resolve(x_col, columns, "x_col", ("x", "X", "aoi_x", "AOI_X"))
    y_col = _gp3_aoi_coding_r_resolve(y_col, columns, "y_col", ("y", "Y", "aoi_y", "AOI_Y"))
    width_col = _gp3_aoi_coding_r_resolve(
        width_col, columns, "width_col", ("width", "Width", "aoi_width", "AOI_WIDTH")
    )
    height_col = _gp3_aoi_coding_r_resolve(
        height_col,
        columns,
        "height_col",
        ("height", "Height", "aoi_height", "AOI_HEIGHT"),
    )
    has_bounds = all(value is not None for value in (x_min_col, y_min_col, x_max_col, y_max_col))
    has_origin_size = all(value is not None for value in (x_col, y_col, width_col, height_col))
    if not has_bounds and not has_origin_size:
        raise ValueError(
            "AOI geometry requires either x/y min-max columns or x/y plus width/height columns"
        )
    screen_x = _gp3_aoi_coding_r_range(screen_x_range, "screen_x_range")
    screen_y = _gp3_aoi_coding_r_range(screen_y_range, "screen_y_range")
    if has_bounds:
        xmin = _gp3_aoi_coding_r_numeric(frame[x_min_col])
        ymin = _gp3_aoi_coding_r_numeric(frame[y_min_col])
        xmax = _gp3_aoi_coding_r_numeric(frame[x_max_col])
        ymax = _gp3_aoi_coding_r_numeric(frame[y_max_col])
    else:
        xmin = _gp3_aoi_coding_r_numeric(frame[x_col])
        ymin = _gp3_aoi_coding_r_numeric(frame[y_col])
        width = _gp3_aoi_coding_r_numeric(frame[width_col])
        height = _gp3_aoi_coding_r_numeric(frame[height_col])
        xmax = xmin + width
        ymax = ymin + height
    width = xmax - xmin
    height = ymax - ymin
    area = width * height
    screen_area = (screen_x[1] - screen_x[0]) * (screen_y[1] - screen_y[0])
    area_prop = area / screen_area
    invalid_coordinate = ~(
        np.isfinite(xmin) & np.isfinite(ymin) & np.isfinite(xmax) & np.isfinite(ymax)
    )
    invalid_dimension = ~invalid_coordinate & ((width <= 0) | (height <= 0))
    too_large = ~invalid_coordinate & ~invalid_dimension & (area_prop > 1)
    status = np.full(len(frame), "ok", dtype=object)
    status[too_large] = "too_large"
    status[invalid_dimension] = "invalid_dimension"
    status[invalid_coordinate] = "invalid_coordinate"
    id_cols = [aoi_col] + ([stimulus_col] if stimulus_col is not None else [])
    summary = frame[id_cols].copy().reset_index(drop=True)
    summary["x_min"] = xmin
    summary["y_min"] = ymin
    summary["x_max"] = xmax
    summary["y_max"] = ymax
    summary["width"] = width
    summary["height"] = height
    summary["area"] = area
    summary["area_prop"] = area_prop
    summary["center_x"] = xmin + width / 2
    summary["center_y"] = ymin + height / 2
    summary["outside_screen"] = (
        ~invalid_coordinate
        & ~invalid_dimension
        & (
            (xmin < screen_x[0])
            | (xmax > screen_x[1])
            | (ymin < screen_y[0])
            | (ymax > screen_y[1])
        )
    )
    summary["aoi_geometry_status"] = status
    return summary, aoi_col, stimulus_col


def _gp3_aoi_coding_r_standardise_observed(values, outside_label, outside_values):
    raw = values.astype("string")
    lower = raw.str.strip().str.lower()
    result = raw.astype(object)
    outside_lookup = {value.lower() for value in outside_values}
    result.loc[lower.isin(outside_lookup)] = outside_label
    result.loc[lower.eq("") | raw.isna()] = pd.NA
    return result


def _gp3_aoi_coding_r_assign(
    gaze,
    geometry,
    *,
    gaze_x_col,
    gaze_y_col,
    gaze_stimulus_col,
    geometry_aoi_col,
    geometry_stimulus_col,
    tie_method,
    outside_label,
    ambiguous_label,
    missing_label,
):
    x = _gp3_aoi_coding_r_numeric(gaze[gaze_x_col])
    y = _gp3_aoi_coding_r_numeric(gaze[gaze_y_col])
    derived = np.full(len(gaze), outside_label, dtype=object)
    n_matching = np.zeros(len(gaze), dtype=int)
    assignment = np.full(len(gaze), "no_aoi", dtype=object)
    missing = ~(np.isfinite(x) & np.isfinite(y))
    derived[missing] = missing_label
    assignment[missing] = "missing_coordinate"
    for i in range(len(gaze)):
        if missing[i]:
            continue
        candidate = geometry
        if gaze_stimulus_col is not None and geometry_stimulus_col is not None:
            gaze_value = gaze.iloc[i][gaze_stimulus_col]
            candidate = candidate.loc[
                candidate[geometry_stimulus_col].astype("string").eq(str(gaze_value)).fillna(False)
            ]
        if candidate.empty:
            continue
        inside = (
            (x[i] >= candidate["x_min"].to_numpy(float))
            & (x[i] <= candidate["x_max"].to_numpy(float))
            & (y[i] >= candidate["y_min"].to_numpy(float))
            & (y[i] <= candidate["y_max"].to_numpy(float))
        )
        hits = np.flatnonzero(inside)
        n_matching[i] = len(hits)
        if len(hits) == 0:
            continue
        if len(hits) == 1:
            derived[i] = str(candidate.iloc[hits[0]][geometry_aoi_col])
            assignment[i] = "single_aoi"
        elif tie_method == "first":
            derived[i] = str(candidate.iloc[hits[0]][geometry_aoi_col])
            assignment[i] = "multiple_aoi_resolved"
        else:
            derived[i] = ambiguous_label
            assignment[i] = "ambiguous_aoi"
    return pd.DataFrame(
        {
            "derived_aoi": derived,
            "n_matching_aois": n_matching,
            "derived_assignment_status": assignment,
        }
    )


def _gp3_aoi_coding_r_summary(sample_coding, key):
    rows = []
    grouped = sample_coding.loc[sample_coding[key].notna()].groupby(key, sort=True, dropna=True)
    for value, block in grouped:
        comparable = int(block["comparable_sample"].fillna(False).sum())
        mismatches = int(block["aoi_coding_status"].eq("mismatch").sum())
        rows.append(
            {
                key: str(value),
                "n_samples": len(block),
                "n_comparable_samples": comparable,
                "n_matches": int(block["coding_match"].eq(True).sum()),
                "n_mismatches": mismatches,
            }
        )
    out = pd.DataFrame(
        rows,
        columns=[
            key,
            "n_samples",
            "n_comparable_samples",
            "n_matches",
            "n_mismatches",
        ],
    )
    total = int(out["n_samples"].sum()) if len(out) else 0
    out["sample_prop"] = out["n_samples"] / total if total else np.nan
    out["mismatch_prop"] = np.where(
        out["n_comparable_samples"] > 0,
        out["n_mismatches"] / out["n_comparable_samples"],
        np.nan,
    )
    return out


def _gp3_aoi_coding_r_text(value):
    if value is None:
        return pd.NA
    if isinstance(value, (list, tuple)):
        if not value:
            return pd.NA
        return ", ".join(str(item) for item in value)
    return str(value)


def audit_gazepoint_aoi_coding_matrix(
    data=None,
    aoi_col=None,
    group_cols=None,
    *,
    gaze_data=None,
    aoi_geometry=None,
    observed_aoi_col=None,
    gaze_x_col=None,
    gaze_y_col=None,
    gaze_stimulus_col=None,
    sample_id_cols=None,
    geometry_aoi_col=None,
    geometry_stimulus_col=None,
    x_min_col=None,
    y_min_col=None,
    x_max_col=None,
    y_max_col=None,
    x_col=None,
    y_col=None,
    width_col=None,
    height_col=None,
    screen_x_range=(0, 1),
    screen_y_range=(0, 1),
    tie_method="ambiguous",
    outside_label="outside",
    ambiguous_label="ambiguous",
    missing_label="missing_coordinate",
    observed_outside_values=(
        "outside",
        "none",
        "no_aoi",
        "non_aoi",
        "background",
        "off_aoi",
    ),
    max_mismatch_prop=0.05,
    max_ambiguous_prop=0.05,
    max_missing_coordinate_prop=0.2,
    ignore_invalid_geometry=True,
):
    """Audit observed versus geometry-derived AOI coding.

    The historical Python frequency-table interface is preserved. Supplying
    ``aoi_geometry``/``gaze_data`` or two DataFrames positionally activates
    the R gp3tools v2.3.0 audit interface.
    """
    positional_r_mode = (
        isinstance(data, pd.DataFrame)
        and isinstance(aoi_col, pd.DataFrame)
        and gaze_data is None
        and aoi_geometry is None
    )
    r_mode = positional_r_mode or gaze_data is not None or aoi_geometry is not None
    if not r_mode:
        df = ensure_dataframe(data, copy=False)
        resolved_aoi = infer_column(df, "aoi", aoi_col, required=True)
        groups = normalize_group_cols(df, group_cols)
        return df.groupby(groups + [resolved_aoi], dropna=False).size().rename("n").reset_index()
    if positional_r_mode:
        gaze_data, aoi_geometry = data, aoi_col
    else:
        if gaze_data is not None and data is not None:
            raise TypeError("supply either data or gaze_data, not both")
        if gaze_data is None:
            gaze_data = data
    if not isinstance(gaze_data, pd.DataFrame) or gaze_data.empty:
        raise ValueError("gaze_data must contain at least one row")
    if not isinstance(aoi_geometry, pd.DataFrame) or aoi_geometry.empty:
        raise ValueError("aoi_geometry must contain at least one row")
    if tie_method not in {"ambiguous", "first"}:
        raise ValueError("tie_method must be 'ambiguous' or 'first'")
    outside_label = _gp3_aoi_coding_r_label(outside_label, "outside_label")
    ambiguous_label = _gp3_aoi_coding_r_label(ambiguous_label, "ambiguous_label")
    missing_label = _gp3_aoi_coding_r_label(missing_label, "missing_label")
    outside_values = _gp3_aoi_coding_r_character_vector(
        observed_outside_values, "observed_outside_values"
    )
    max_mismatch_prop = _gp3_aoi_coding_r_prop(max_mismatch_prop, "max_mismatch_prop")
    max_ambiguous_prop = _gp3_aoi_coding_r_prop(max_ambiguous_prop, "max_ambiguous_prop")
    max_missing_coordinate_prop = _gp3_aoi_coding_r_prop(
        max_missing_coordinate_prop, "max_missing_coordinate_prop"
    )
    if not isinstance(ignore_invalid_geometry, (bool, np.bool_)):
        raise ValueError("ignore_invalid_geometry must be TRUE or FALSE")
    gaze = _gp3_aoi_coding_r_aliases(gaze_data).reset_index(drop=True)
    observed_aoi_col = _gp3_aoi_coding_r_resolve(
        observed_aoi_col,
        gaze.columns,
        "observed_aoi_col",
        (
            "observed_aoi",
            "observed_aoi_label",
            "coded_aoi",
            "aoi",
            "AOI",
            "aoi_current",
            "aoi_label",
            "AOI_LABEL",
        ),
        True,
    )
    gaze_x_col = _gp3_aoi_coding_r_resolve(
        gaze_x_col,
        gaze.columns,
        "gaze_x_col",
        ("x", "X", "gaze_x", "gaze_x_norm", "FPOGX", "BPOGX"),
        True,
    )
    gaze_y_col = _gp3_aoi_coding_r_resolve(
        gaze_y_col,
        gaze.columns,
        "gaze_y_col",
        ("y", "Y", "gaze_y", "gaze_y_norm", "FPOGY", "BPOGY"),
        True,
    )
    gaze_stimulus_col = _gp3_aoi_coding_r_resolve(
        gaze_stimulus_col,
        gaze.columns,
        "gaze_stimulus_col",
        ("media_id", "MEDIA_ID", "stimulus", "stimulus_id"),
    )
    if sample_id_cols is None:
        sample_id_cols = []
    elif isinstance(sample_id_cols, str):
        sample_id_cols = [sample_id_cols]
    else:
        sample_id_cols = [str(value) for value in sample_id_cols]
    sample_id_cols = [
        "media_id" if value == "MEDIA_ID" else "aoi" if value == "AOI" else value
        for value in sample_id_cols
    ]
    sample_id_cols = [value for value in sample_id_cols if value in gaze.columns]
    geometry_summary, resolved_geometry_aoi_col, resolved_geometry_stimulus_col = (
        _gp3_aoi_coding_r_geometry(
            aoi_geometry,
            aoi_col=geometry_aoi_col,
            stimulus_col=geometry_stimulus_col,
            x_min_col=x_min_col,
            y_min_col=y_min_col,
            x_max_col=x_max_col,
            y_max_col=y_max_col,
            x_col=x_col,
            y_col=y_col,
            width_col=width_col,
            height_col=height_col,
            screen_x_range=screen_x_range,
            screen_y_range=screen_y_range,
        )
    )
    if (
        resolved_geometry_stimulus_col is not None
        and gaze_stimulus_col is None
        and geometry_summary[resolved_geometry_stimulus_col].nunique(dropna=False) > 1
    ):
        raise ValueError(
            "gaze_stimulus_col is required when aoi_geometry contains multiple stimuli"
        )
    geometry_for_coding = geometry_summary
    if bool(ignore_invalid_geometry):
        geometry_for_coding = geometry_summary.loc[
            ~geometry_summary["aoi_geometry_status"].isin(
                ["invalid_coordinate", "invalid_dimension"]
            )
        ].copy()
    base = pd.DataFrame({".gp3_sample_index": np.arange(1, len(gaze) + 1)})
    for column in sample_id_cols:
        base[column] = gaze[column].to_numpy()
    if gaze_stimulus_col is not None and gaze_stimulus_col not in base.columns:
        base[gaze_stimulus_col] = gaze[gaze_stimulus_col].to_numpy()
    observed_raw = gaze[observed_aoi_col].astype("string")
    observed = _gp3_aoi_coding_r_standardise_observed(observed_raw, outside_label, outside_values)
    derived = _gp3_aoi_coding_r_assign(
        gaze,
        geometry_for_coding,
        gaze_x_col=gaze_x_col,
        gaze_y_col=gaze_y_col,
        gaze_stimulus_col=gaze_stimulus_col,
        geometry_aoi_col=resolved_geometry_aoi_col,
        geometry_stimulus_col=resolved_geometry_stimulus_col,
        tie_method=tie_method,
        outside_label=outside_label,
        ambiguous_label=ambiguous_label,
        missing_label=missing_label,
    )
    comparable = observed.notna() & ~derived["derived_aoi"].isin([ambiguous_label, missing_label])
    coding_match = pd.Series(pd.NA, index=gaze.index, dtype="boolean")
    coding_match.loc[comparable] = (
        observed.loc[comparable].astype(str).to_numpy()
        == derived.loc[comparable, "derived_aoi"].astype(str).to_numpy()
    )
    status = pd.Series("ok", index=gaze.index, dtype="object")
    status.loc[observed.isna()] = "observed_missing"
    status.loc[derived["derived_assignment_status"].eq("missing_coordinate")] = "missing_coordinate"
    status.loc[derived["derived_aoi"].eq(ambiguous_label)] = "ambiguous_derived"
    status.loc[comparable & coding_match.eq(False)] = "mismatch"
    sample_coding = pd.concat(
        [
            base.reset_index(drop=True),
            pd.DataFrame(
                {
                    "observed_aoi_raw": observed_raw.reset_index(drop=True),
                    "observed_aoi": observed.reset_index(drop=True),
                    "derived_aoi": derived["derived_aoi"].reset_index(drop=True),
                    "n_matching_aois": derived["n_matching_aois"].reset_index(drop=True),
                    "derived_assignment_status": derived["derived_assignment_status"].reset_index(
                        drop=True
                    ),
                    "comparable_sample": comparable.reset_index(drop=True),
                    "coding_match": coding_match.reset_index(drop=True),
                    "aoi_coding_status": status.reset_index(drop=True),
                }
            ),
        ],
        axis=1,
    )
    coding_matrix = (
        sample_coding.groupby(["observed_aoi", "derived_aoi"], dropna=False, sort=True)
        .size()
        .rename("n_samples")
        .reset_index()
    )
    total_matrix = int(coding_matrix["n_samples"].sum())
    coding_matrix["sample_prop"] = (
        coding_matrix["n_samples"] / total_matrix if total_matrix else np.nan
    )
    observed_summary = _gp3_aoi_coding_r_summary(sample_coding, "observed_aoi")
    derived_summary = _gp3_aoi_coding_r_summary(sample_coding, "derived_aoi")
    flagged_samples = sample_coding.loc[~sample_coding["aoi_coding_status"].eq("ok")].copy()
    n_comparable = int(sample_coding["comparable_sample"].fillna(False).sum())
    n_mismatched = int(sample_coding["aoi_coding_status"].eq("mismatch").sum())
    n_ambiguous = int(sample_coding["derived_assignment_status"].eq("ambiguous_aoi").sum())
    n_missing = int(sample_coding["derived_assignment_status"].eq("missing_coordinate").sum())
    mismatch_prop = n_mismatched / n_comparable if n_comparable else np.nan
    ambiguous_prop = n_ambiguous / len(sample_coding)
    missing_prop = n_missing / len(sample_coding)
    overview_status = (
        "review"
        if (
            (np.isfinite(mismatch_prop) and mismatch_prop > max_mismatch_prop)
            or ambiguous_prop > max_ambiguous_prop
            or missing_prop > max_missing_coordinate_prop
        )
        else "ok"
    )
    overview = pd.DataFrame(
        [
            {
                "n_gaze_rows": len(gaze),
                "n_geometry_rows": len(aoi_geometry),
                "n_aois": len(geometry_summary),
                "n_aois_used": len(geometry_for_coding),
                "n_coded_samples": len(sample_coding),
                "n_comparable_samples": n_comparable,
                "n_mismatched_samples": n_mismatched,
                "mismatch_prop": mismatch_prop,
                "n_ambiguous_samples": n_ambiguous,
                "ambiguous_prop": ambiguous_prop,
                "n_missing_coordinate_samples": n_missing,
                "missing_coordinate_prop": missing_prop,
                "n_flagged_samples": len(flagged_samples),
                "aoi_coding_matrix_status": overview_status,
            }
        ]
    )
    settings = pd.DataFrame(
        {
            "setting": [
                "observed_aoi_col",
                "gaze_x_col",
                "gaze_y_col",
                "gaze_stimulus_col",
                "sample_id_cols",
                "geometry_aoi_col",
                "geometry_stimulus_col",
                "screen_x_range",
                "screen_y_range",
                "tie_method",
                "outside_label",
                "ambiguous_label",
                "missing_label",
                "observed_outside_values",
                "max_mismatch_prop",
                "max_ambiguous_prop",
                "max_missing_coordinate_prop",
                "ignore_invalid_geometry",
            ],
            "value": [
                observed_aoi_col,
                gaze_x_col,
                gaze_y_col,
                _gp3_aoi_coding_r_text(gaze_stimulus_col),
                _gp3_aoi_coding_r_text(sample_id_cols),
                resolved_geometry_aoi_col,
                _gp3_aoi_coding_r_text(resolved_geometry_stimulus_col),
                ", ".join(
                    f"{value:g}"
                    for value in _gp3_aoi_coding_r_range(screen_x_range, "screen_x_range")
                ),
                ", ".join(
                    f"{value:g}"
                    for value in _gp3_aoi_coding_r_range(screen_y_range, "screen_y_range")
                ),
                tie_method,
                outside_label,
                ambiguous_label,
                missing_label,
                ", ".join(outside_values),
                f"{max_mismatch_prop:g}",
                f"{max_ambiguous_prop:g}",
                f"{max_missing_coordinate_prop:g}",
                "TRUE" if bool(ignore_invalid_geometry) else "FALSE",
            ],
        }
    )
    return {
        "overview": overview,
        "geometry_summary": geometry_summary,
        "sample_coding": sample_coding,
        "coding_matrix": coding_matrix,
        "observed_summary": observed_summary,
        "derived_summary": derived_summary,
        "flagged_samples": flagged_samples,
        "settings": settings,
        "_gp3_class": "gp3_aoi_coding_matrix_audit",
    }


def audit_gazepoint_aoi_margin_sensitivity(
    data,
    aoi_geometry,
    margins=(-0.02, 0, 0.02),
    x_col=None,
    y_col=None,
    *,
    gaze_x_col=None,
    gaze_y_col=None,
    gaze_stimulus_col=None,
    sample_id_cols=None,
    geometry_aoi_col=None,
    geometry_stimulus_col=None,
    x_min_col=None,
    y_min_col=None,
    x_max_col=None,
    y_max_col=None,
    width_col=None,
    height_col=None,
    screen_x_range=(0, 1),
    screen_y_range=(0, 1),
    tie_method="ambiguous",
    outside_label="outside",
    ambiguous_label="ambiguous",
    missing_label="missing_coordinate",
    max_margin_change_prop=0.10,
    max_ambiguous_prop=0.05,
    ignore_invalid_geometry=True,
):
    """Audit AOI coding sensitivity to geometry expansion and contraction."""
    r_mode = any(
        (
            gaze_x_col is not None,
            gaze_y_col is not None,
            gaze_stimulus_col is not None,
            sample_id_cols is not None,
            geometry_aoi_col is not None,
            geometry_stimulus_col is not None,
            x_min_col is not None,
            y_min_col is not None,
            x_max_col is not None,
            y_max_col is not None,
            width_col is not None,
            height_col is not None,
            tuple(screen_x_range) != (0, 1),
            tuple(screen_y_range) != (0, 1),
            tie_method != "ambiguous",
            outside_label != "outside",
            ambiguous_label != "ambiguous",
            missing_label != "missing_coordinate",
            max_margin_change_prop != 0.10,
            max_ambiguous_prop != 0.05,
            ignore_invalid_geometry is not True,
        )
    )
    if not r_mode:
        g = ensure_dataframe(aoi_geometry)
        rows = []
        for margin in margins:
            gm = g.copy()
            gm["xmin"] -= margin
            gm["xmax"] += margin
            gm["ymin"] -= margin
            gm["ymax"] += margin
            assigned = add_gazepoint_aoi(data, x_col=x_col, y_col=y_col, aoi_geometry=gm)
            rows.append(
                {
                    "margin": margin,
                    "assigned_prop": float(assigned.aoi_current.notna().mean()),
                    "outside_prop": float(
                        assigned.aoi_current.astype("string").eq("outside").mean()
                    ),
                }
            )
        return pd.DataFrame(rows)

    gaze = ensure_dataframe(data, copy=False)
    geometry = ensure_dataframe(aoi_geometry, copy=False)
    if gaze.empty or geometry.empty:
        raise ValueError("gaze_data and aoi_geometry must contain at least one row")
    gaze_x_col = _gp3_margin_resolve_column(
        gaze,
        gaze_x_col,
        ("x", "X", "gaze_x", "gaze_x_norm", "FPOGX", "BPOGX"),
        required=True,
        arg="gaze_x_col",
    )
    gaze_y_col = _gp3_margin_resolve_column(
        gaze,
        gaze_y_col,
        ("y", "Y", "gaze_y", "gaze_y_norm", "FPOGY", "BPOGY"),
        required=True,
        arg="gaze_y_col",
    )
    gaze_stimulus_col = _gp3_margin_resolve_column(
        gaze,
        gaze_stimulus_col,
        ("media_id", "MEDIA_ID", "stimulus", "stimulus_id"),
        required=False,
        arg="gaze_stimulus_col",
    )
    ids = (
        []
        if sample_id_cols is None
        else ([sample_id_cols] if isinstance(sample_id_cols, str) else list(sample_id_cols))
    )
    ids = [c for c in ids if c in gaze]
    margins_array = np.asarray(margins, dtype=float)
    if margins_array.ndim != 1 or not np.isfinite(margins_array).all():
        raise ValueError("margins must be a finite numeric vector")
    if not np.isfinite(max_margin_change_prop) or max_margin_change_prop < 0:
        raise ValueError("max_margin_change_prop must be non-negative")
    if not np.isfinite(max_ambiguous_prop) or not 0 <= max_ambiguous_prop <= 1:
        raise ValueError("max_ambiguous_prop must be in [0, 1]")
    if tie_method not in {"ambiguous", "first"}:
        raise ValueError("tie_method must be ambiguous or first")
    if not isinstance(ignore_invalid_geometry, (bool, np.bool_)):
        raise ValueError("ignore_invalid_geometry must be boolean")
    geometry_audit = _gp3_aoi_geometry_r_audit(
        geometry,
        aoi_col=geometry_aoi_col,
        stimulus_col=geometry_stimulus_col,
        x_min_col=x_min_col,
        y_min_col=y_min_col,
        x_max_col=x_max_col,
        y_max_col=y_max_col,
        x_col=x_col,
        y_col=y_col,
        width_col=width_col,
        height_col=height_col,
        screen_x_range=screen_x_range,
        screen_y_range=screen_y_range,
        require_within_screen=False,
    )
    geometry_summary = geometry_audit["geometry_summary"].copy()
    settings_lookup = dict(
        zip(geometry_audit["settings"]["setting"], geometry_audit["settings"]["value"], strict=True)
    )
    resolved_aoi = str(settings_lookup["aoi_col"])
    resolved_stimulus = settings_lookup["stimulus_col"]
    if pd.isna(resolved_stimulus) or str(resolved_stimulus) == "":
        resolved_stimulus = None
    else:
        resolved_stimulus = str(resolved_stimulus)
    if (
        resolved_stimulus is not None
        and gaze_stimulus_col is None
        and geometry_summary[resolved_stimulus].nunique(dropna=False) > 1
    ):
        raise ValueError("gaze_stimulus_col is required when geometry contains multiple stimuli")
    geometry_for_coding = geometry_summary
    if ignore_invalid_geometry:
        geometry_for_coding = geometry_for_coding.loc[
            ~geometry_for_coding["aoi_geometry_status"].isin(
                ["invalid_coordinate", "invalid_dimension"]
            )
        ].copy()
    margins_used = sorted(set([0.0, *margins_array.tolist()]))
    x = pd.to_numeric(gaze[gaze_x_col], errors="coerce").to_numpy(float)
    y = pd.to_numeric(gaze[gaze_y_col], errors="coerce").to_numpy(float)
    sensitivity_rows = []
    for i in range(len(gaze)):
        base = {".gp3_sample_index": i + 1}
        for col in ids:
            base[col] = gaze.iloc[i][col]
        if gaze_stimulus_col is not None and gaze_stimulus_col not in base:
            base[gaze_stimulus_col] = gaze.iloc[i][gaze_stimulus_col]
        for margin in margins_used:
            assigned = outside_label
            status = "no_aoi"
            n_hits = 0
            if not np.isfinite(x[i]) or not np.isfinite(y[i]):
                assigned = missing_label
                status = "missing_coordinate"
            else:
                candidates = geometry_for_coding
                if gaze_stimulus_col is not None and resolved_stimulus is not None:
                    candidates = candidates.loc[
                        candidates[resolved_stimulus]
                        .astype("string")
                        .eq(str(gaze.iloc[i][gaze_stimulus_col]))
                    ]
                xmin = pd.to_numeric(candidates["x_min"], errors="coerce").to_numpy(float) - margin
                ymin = pd.to_numeric(candidates["y_min"], errors="coerce").to_numpy(float) - margin
                xmax = pd.to_numeric(candidates["x_max"], errors="coerce").to_numpy(float) + margin
                ymax = pd.to_numeric(candidates["y_max"], errors="coerce").to_numpy(float) + margin
                hit = (
                    (xmin < xmax)
                    & (ymin < ymax)
                    & (x[i] >= xmin)
                    & (x[i] <= xmax)
                    & (y[i] >= ymin)
                    & (y[i] <= ymax)
                )
                indices = np.flatnonzero(hit)
                n_hits = len(indices)
                if n_hits == 1:
                    assigned = str(candidates.iloc[indices[0]][resolved_aoi])
                    status = "single_aoi"
                elif n_hits > 1:
                    if tie_method == "first":
                        assigned = str(candidates.iloc[indices[0]][resolved_aoi])
                        status = "multiple_aoi_resolved"
                    else:
                        assigned = ambiguous_label
                        status = "ambiguous_aoi"
            sensitivity_rows.append(
                {
                    **base,
                    "margin": margin,
                    "assigned_aoi": assigned,
                    "n_matching_aois": n_hits,
                    "margin_assignment_status": status,
                }
            )
    sample = pd.DataFrame(sensitivity_rows)
    base_lookup = sample.loc[
        sample["margin"].eq(0), [".gp3_sample_index", "assigned_aoi"]
    ].set_index(".gp3_sample_index")["assigned_aoi"]
    sample["base_assigned_aoi"] = sample[".gp3_sample_index"].map(base_lookup)
    sample["changed_from_base"] = sample["assigned_aoi"].ne(sample["base_assigned_aoi"])
    sample.loc[sample["margin"].eq(0), "changed_from_base"] = False
    sample = sample.sort_values([".gp3_sample_index", "margin"], kind="stable").reset_index(
        drop=True
    )
    margin_rows = []
    for margin, block in sample.groupby("margin", sort=True):
        n = len(block)
        n_changed = int(block["changed_from_base"].fillna(False).sum())
        n_ambiguous = int(block["assigned_aoi"].eq(ambiguous_label).sum())
        change_prop = n_changed / n if n else np.nan
        ambiguous_prop = n_ambiguous / n if n else np.nan
        if margin == 0:
            status = "base_ambiguous" if ambiguous_prop > max_ambiguous_prop else "base"
        elif change_prop > max_margin_change_prop:
            status = "margin_sensitive"
        elif ambiguous_prop > max_ambiguous_prop:
            status = "ambiguous_margin"
        else:
            status = "ok"
        margin_rows.append(
            {
                "margin": margin,
                "n_samples": n,
                "n_changed_from_base": n_changed,
                "margin_change_prop": change_prop,
                "n_ambiguous": n_ambiguous,
                "ambiguous_prop": ambiguous_prop,
                "n_outside": int(block["assigned_aoi"].eq(outside_label).sum()),
                "outside_prop": float(block["assigned_aoi"].eq(outside_label).mean()),
                "n_missing_coordinate": int(block["assigned_aoi"].eq(missing_label).sum()),
                "missing_coordinate_prop": float(block["assigned_aoi"].eq(missing_label).mean()),
                "margin_sensitivity_status": status,
            }
        )
    margin_summary = pd.DataFrame(margin_rows)
    aoi_summary = (
        sample.groupby(["margin", "assigned_aoi"], dropna=False, sort=True)
        .agg(n_samples=("assigned_aoi", "size"), n_changed_from_base=("changed_from_base", "sum"))
        .reset_index()
    )
    totals = aoi_summary.groupby("margin")["n_samples"].transform("sum")
    aoi_summary["margin_total_samples"] = totals
    aoi_summary["sample_prop"] = aoi_summary["n_samples"] / totals
    flagged = sample.loc[
        sample["margin"].ne(0)
        & (
            sample["changed_from_base"].fillna(False)
            | sample["margin_assignment_status"].eq("ambiguous_aoi")
        )
    ].reset_index(drop=True)
    nonbase = margin_summary.loc[margin_summary["margin"].ne(0), "margin_change_prop"]
    n_flagged = int((~margin_summary["margin_sensitivity_status"].isin(["ok", "base"])).sum())
    overview = pd.DataFrame(
        [
            {
                "n_gaze_rows": len(gaze),
                "n_geometry_rows": len(geometry),
                "n_aois": len(geometry_summary),
                "n_aois_used": len(geometry_for_coding),
                "n_margins": len(margins_used),
                "n_sample_margin_rows": len(sample),
                "n_flagged_margins": n_flagged,
                "max_margin_change_prop_observed": _gp3_margin_safe_stat(nonbase, np.max),
                "max_ambiguous_prop_observed": _gp3_margin_safe_stat(
                    margin_summary["ambiguous_prop"], np.max
                ),
                "aoi_margin_sensitivity_status": "review" if n_flagged > 0 else "ok",
            }
        ]
    )
    settings = pd.DataFrame(
        {
            "setting": [
                "gaze_x_col",
                "gaze_y_col",
                "gaze_stimulus_col",
                "sample_id_cols",
                "geometry_aoi_col",
                "geometry_stimulus_col",
                "margins",
                "margins_used",
                "screen_x_range",
                "screen_y_range",
                "tie_method",
                "outside_label",
                "ambiguous_label",
                "missing_label",
                "max_margin_change_prop",
                "max_ambiguous_prop",
                "ignore_invalid_geometry",
            ],
            "value": [
                gaze_x_col,
                gaze_y_col,
                gaze_stimulus_col if gaze_stimulus_col is not None else pd.NA,
                ", ".join(ids) if ids else pd.NA,
                resolved_aoi,
                resolved_stimulus if resolved_stimulus is not None else pd.NA,
                ", ".join(map(str, margins_array.tolist())),
                ", ".join(map(str, margins_used)),
                ", ".join(map(str, screen_x_range)),
                ", ".join(map(str, screen_y_range)),
                tie_method,
                outside_label,
                ambiguous_label,
                missing_label,
                str(max_margin_change_prop),
                str(max_ambiguous_prop),
                str(bool(ignore_invalid_geometry)),
            ],
        }
    )
    return {
        "overview": overview,
        "geometry_summary": geometry_summary,
        "sample_sensitivity": sample,
        "margin_summary": margin_summary,
        "aoi_margin_summary": aoi_summary,
        "flagged_samples": flagged,
        "settings": settings,
        "_gp3_class": "gp3_aoi_margin_sensitivity_audit",
    }


def summarise_aoi_samples(
    data,
    aoi_col=None,
    group_cols=None,
    time_col=None,
) -> pd.DataFrame:
    """Summarise AOI samples with legacy counts or R v2.3.0 timing semantics."""
    df = ensure_dataframe(data, copy=False)

    # Historical Python mode: sample counts/proportions.
    if time_col is None:
        resolved_aoi = infer_column(df, "aoi", aoi_col, required=True)
        groups = normalize_group_cols(df, group_cols)
        out = (
            df.groupby(groups + [resolved_aoi], dropna=False)
            .size()
            .rename("n_samples")
            .reset_index()
        )
        totals = (
            out.groupby(groups, dropna=False).n_samples.transform("sum")
            if groups
            else out.n_samples.sum()
        )
        out["proportion"] = out.n_samples / totals
        return out

    # R v2.3.0 mode.
    frame = standardise_gazepoint_names(df)
    resolved_aoi = "AOI" if aoi_col is None else aoi_col
    groups = (
        ["MEDIA_ID"]
        if group_cols is None
        else ([group_cols] if isinstance(group_cols, str) else list(group_cols))
    )
    needed = [*groups, resolved_aoi, time_col]
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    work = frame.sort_values(
        [*groups, time_col] if groups else [time_col],
        kind="stable",
        na_position="last",
    ).copy()
    work["_gp3_time_numeric"] = pd.to_numeric(work[time_col], errors="coerce")

    if groups:
        work["_gp3_dt_next"] = (
            work.groupby(groups, dropna=False, sort=False)["_gp3_time_numeric"].shift(-1)
            - work["_gp3_time_numeric"]
        )
        medians = work.groupby(groups, dropna=False, sort=False)["_gp3_dt_next"].transform("median")
    else:
        work["_gp3_dt_next"] = work["_gp3_time_numeric"].shift(-1) - work["_gp3_time_numeric"]
        medians = pd.Series(work["_gp3_dt_next"].median(), index=work.index)

    work["_gp3_dt_next"] = work["_gp3_dt_next"].where(
        work["_gp3_dt_next"].notna(),
        medians,
    )
    work = work.loc[work[resolved_aoi].notna() & work[resolved_aoi].astype(str).ne("")]

    rows = []
    keys = [*groups, resolved_aoi]
    for key, block in work.groupby(keys, dropna=False, sort=True):
        values = key if isinstance(key, tuple) else (key,)
        row = dict(zip(keys, values, strict=True))
        row.update(
            {
                "time_to_first_view_sec": float(block["_gp3_time_numeric"].min()),
                "aoi_sample_count": int(len(block)),
                "approx_time_viewed_sec": float(block["_gp3_dt_next"].sum(skipna=True)),
            }
        )
        rows.append(row)

    return pd.DataFrame(
        rows,
        columns=[*keys, "time_to_first_view_sec", "aoi_sample_count", "approx_time_viewed_sec"],
    )


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


_GP3_R_UNSET = object()
_GP3_AOI_NON_VALUES = (
    "non_aoi",
    "none",
    "background",
    "outside",
    "outside_aoi",
    "missing",
    "missing_aoi",
)


def _gp3_aoi_r_list(value, *, allow_none=True, unique=False, name="value"):
    if value is None and allow_none:
        return []
    if isinstance(value, str):
        out = [value]
    else:
        try:
            out = list(value)
        except TypeError as exc:
            raise ValueError(f"{name} must be a character vector") from exc
    if any(v is None or pd.isna(v) or not isinstance(v, str) or not v for v in out):
        raise ValueError(f"{name} must contain non-missing non-empty strings")
    if unique and len(set(out)) != len(out):
        raise ValueError(f"{name} must contain unique column names")
    return out


def _gp3_aoi_r_scalar_label(value, name, *, allow_none=False):
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-missing character scalar")
    return value


def _gp3_aoi_r_resolve_column(df, value, candidates, *, name, allow_none=False):
    value = _gp3_aoi_r_scalar_label(value, name, allow_none=allow_none)
    if value is None:
        value = next((candidate for candidate in candidates if candidate in df.columns), None)
        if value is None:
            raise ValueError(f"Could not automatically detect {name}")
    if value not in df.columns:
        raise ValueError(f"Missing required columns: {value}")
    return value


def _gp3_aoi_r_groupby(df, cols):
    if cols:
        return df.groupby(cols, dropna=False, sort=False)
    return [(None, df)]


def _gp3_aoi_r_entries(
    data,
    *,
    aoi_col=None,
    time_col="time",
    group_cols=("subject", "MEDIA_ID", "trial_global"),
    include_non_aoi=True,
    non_aoi_values=_GP3_AOI_NON_VALUES,
    missing_aoi_label="missing_aoi",
):
    df = ensure_dataframe(data, copy=False)
    time_col = _gp3_aoi_r_scalar_label(time_col, "time_col")
    groups = _gp3_aoi_r_list(group_cols, allow_none=False, unique=True, name="group_cols")
    if not isinstance(include_non_aoi, (bool, np.bool_)):
        raise ValueError("include_non_aoi must be TRUE or FALSE")
    non_values = _gp3_aoi_r_list(non_aoi_values, allow_none=False, name="non_aoi_values")
    missing_aoi_label = _gp3_aoi_r_scalar_label(missing_aoi_label, "missing_aoi_label")
    aoi_col = _gp3_aoi_r_resolve_column(
        df,
        aoi_col,
        ("aoi_current", "AOI", "aoi_state"),
        name="aoi_col",
        allow_none=True,
    )
    missing = [col for col in [*groups, time_col, aoi_col] if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(dict.fromkeys(missing)))

    work = df.copy()
    work[".gp3_aoi_time"] = pd.to_numeric(work[time_col], errors="coerce")
    state = work[aoi_col].astype("string").str.strip()
    state = state.mask(state.isna() | state.eq(""), missing_aoi_label)
    work[".gp3_aoi_state"] = state.astype(object)
    work = work.loc[work[".gp3_aoi_time"].notna()].copy()
    if work.empty:
        raise ValueError("No non-missing time values remain after filtering")
    sort_cols = [*groups, ".gp3_aoi_time"] if groups else [".gp3_aoi_time"]
    work = work.sort_values(sort_cols, kind="stable", na_position="last").reset_index(drop=True)

    pieces = []
    for _, block in _gp3_aoi_r_groupby(work, groups):
        block = block.copy()
        times = block[".gp3_aoi_time"].to_numpy(dtype=float)
        next_times = np.r_[times[1:], np.nan]
        positive = np.where(
            np.isfinite(next_times) & (next_times > times), next_times - times, np.nan
        )
        finite_pos = positive[np.isfinite(positive) & (positive > 0)]
        default_dt = float(np.median(finite_pos)) if finite_pos.size else np.nan
        sample_duration = np.where(np.isfinite(positive), positive, default_dt)
        sample_duration = np.where(np.isfinite(sample_duration), sample_duration, 0.0)
        block[".gp3_sample_duration_ms"] = sample_duration
        changed = block[".gp3_aoi_state"].ne(block[".gp3_aoi_state"].shift(1)).to_numpy(copy=True)
        if len(changed):
            changed[0] = True
        block[".gp3_entry_id"] = np.cumsum(changed).astype(int)
        pieces.append(block)
    work = pd.concat(pieces, ignore_index=True)

    rows = []
    entry_group_cols = [*groups, ".gp3_entry_id", ".gp3_aoi_state"]
    for key, block in _gp3_aoi_r_groupby(work, entry_group_cols):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(entry_group_cols, key, strict=True))
        t = block[".gp3_aoi_time"].to_numpy(dtype=float)
        dur = block[".gp3_sample_duration_ms"].to_numpy(dtype=float)
        rows.append(
            {
                **base,
                "entry_start_time": float(np.min(t)),
                "entry_end_time": float(np.max(t + dur)),
                "entry_duration_ms": float(np.nansum(dur)),
                "n_samples": int(len(block)),
            }
        )
    entries = pd.DataFrame(rows)
    entries["entry_id"] = entries[".gp3_entry_id"].astype(int)
    entries["aoi_state"] = entries[".gp3_aoi_state"].astype(object)
    entries = entries.drop(columns=[".gp3_entry_id", ".gp3_aoi_state"])
    sort_cols = [*groups, "entry_start_time"] if groups else ["entry_start_time"]
    entries = entries.sort_values(sort_cols, kind="stable").reset_index(drop=True)

    out_parts = []
    bg = {str(v).strip().lower() for v in non_values}
    for _, block in _gp3_aoi_r_groupby(entries, groups):
        block = block.copy()
        block["entry_order"] = np.arange(1, len(block) + 1, dtype=int)
        block["previous_aoi_state"] = block["aoi_state"].shift(1)
        block["next_aoi_state"] = block["aoi_state"].shift(-1)
        block["is_non_aoi"] = block["aoi_state"].astype(str).str.strip().str.lower().isin(bg)
        out_parts.append(block)
    entries = pd.concat(out_parts, ignore_index=True)
    if not include_non_aoi:
        entries = entries.loc[~entries["is_non_aoi"]].reset_index(drop=True)
    columns = [
        *groups,
        "entry_id",
        "entry_order",
        "aoi_state",
        "previous_aoi_state",
        "next_aoi_state",
        "entry_start_time",
        "entry_end_time",
        "entry_duration_ms",
        "n_samples",
        "is_non_aoi",
    ]
    return entries.loc[:, columns]


def _gp3_aoi_r_sequences(
    data,
    *,
    aoi_col=None,
    time_col="time",
    group_cols=("subject", "MEDIA_ID", "trial_global"),
    include_non_aoi=True,
    non_aoi_values=_GP3_AOI_NON_VALUES,
    missing_aoi_label="missing_aoi",
    include_terminal=True,
):
    df = ensure_dataframe(data, copy=False)
    groups = _gp3_aoi_r_list(group_cols, allow_none=False, unique=True, name="group_cols")
    if not isinstance(include_non_aoi, (bool, np.bool_)):
        raise ValueError("include_non_aoi must be TRUE or FALSE")
    if not isinstance(include_terminal, (bool, np.bool_)):
        raise ValueError("include_terminal must be TRUE or FALSE")
    non_values = _gp3_aoi_r_list(non_aoi_values, allow_none=False, name="non_aoi_values")
    missing_aoi_label = _gp3_aoi_r_scalar_label(missing_aoi_label, "missing_aoi_label")

    has_entries = {
        "aoi_state",
        "entry_order",
        "entry_start_time",
        "entry_end_time",
        "entry_duration_ms",
        "n_samples",
    }.issubset(df.columns)
    if has_entries:
        entries = df.copy()
        required = [
            *groups,
            "aoi_state",
            "entry_order",
            "entry_start_time",
            "entry_end_time",
            "entry_duration_ms",
            "n_samples",
        ]
        missing = [col for col in required if col not in entries.columns]
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))
    else:
        entries = _gp3_aoi_r_entries(
            df,
            aoi_col=aoi_col,
            time_col=time_col,
            group_cols=groups,
            include_non_aoi=True,
            non_aoi_values=non_values,
            missing_aoi_label=missing_aoi_label,
        )
    if entries.empty:
        raise ValueError("No AOI entries are available")

    state = entries["aoi_state"].astype("string").str.strip()
    entries["aoi_state"] = state.mask(state.isna() | state.eq(""), missing_aoi_label).astype(object)
    for col in ["entry_start_time", "entry_end_time", "entry_duration_ms"]:
        entries[col] = pd.to_numeric(entries[col], errors="coerce")
    entries["n_samples"] = pd.to_numeric(entries["n_samples"], errors="coerce").astype("Int64")
    entries = entries.loc[entries["entry_start_time"].notna()].copy()
    if "entry_id" not in entries.columns:
        sort_cols = [*groups, "entry_start_time"] if groups else ["entry_start_time"]
        entries = entries.sort_values(sort_cols, kind="stable")
        if groups:
            entries["entry_id"] = entries.groupby(groups, dropna=False, sort=False).cumcount() + 1
        else:
            entries["entry_id"] = np.arange(1, len(entries) + 1)
    bg = {str(v).strip().lower() for v in non_values}
    fallback_non = entries["aoi_state"].astype(str).str.strip().str.lower().isin(bg)
    if "is_non_aoi" not in entries.columns:
        entries["is_non_aoi"] = fallback_non
    else:
        existing = entries["is_non_aoi"].astype("boolean")
        entries["is_non_aoi"] = existing.fillna(
            pd.Series(fallback_non, index=entries.index)
        ).astype(bool)
    if not include_non_aoi:
        entries = entries.loc[~entries["is_non_aoi"]].copy()
    if entries.empty:
        raise ValueError("No AOI states remain after applying include_non_aoi")

    sort_cols = (
        [*groups, "entry_start_time", "entry_order"]
        if groups
        else ["entry_start_time", "entry_order"]
    )
    entries = entries.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    parts = []
    for _, block in _gp3_aoi_r_groupby(entries, groups):
        block = block.copy()
        block["state_order"] = np.arange(1, len(block) + 1, dtype=int)
        block["previous_state"] = block["aoi_state"].shift(1)
        block["next_state"] = block["aoi_state"].shift(-1)
        block["transition_from"] = block["aoi_state"]
        block["transition_to"] = block["next_state"]
        block["dwell_before_transition_ms"] = block["entry_duration_ms"]
        block["is_terminal_state"] = block["next_state"].isna()
        block["is_self_transition"] = (~block["transition_to"].isna()) & block[
            "transition_from"
        ].eq(block["transition_to"])
        trans = (~block["is_terminal_state"]).cumsum().astype("Int64")
        trans.loc[block["is_terminal_state"]] = pd.NA
        block["transition_order"] = trans
        parts.append(block)
    out = pd.concat(parts, ignore_index=True)
    if not include_terminal:
        out = out.loc[~out["is_terminal_state"]].reset_index(drop=True)
    columns = [
        *groups,
        "entry_id",
        "state_order",
        "transition_order",
        "aoi_state",
        "previous_state",
        "next_state",
        "transition_from",
        "transition_to",
        "entry_start_time",
        "entry_end_time",
        "entry_duration_ms",
        "dwell_before_transition_ms",
        "n_samples",
        "is_non_aoi",
        "is_self_transition",
        "is_terminal_state",
    ]
    return out.loc[:, columns]


def _gp3_aoi_r_summary_group_rows(frame, groups):
    return _gp3_aoi_r_groupby(frame, groups)


def prepare_gazepoint_aoi_sequences(
    data,
    aoi_col=None,
    group_cols=None,
    time_col=None,
    collapse_repeats=True,
    *,
    include_non_aoi=_GP3_R_UNSET,
    non_aoi_values=_GP3_R_UNSET,
    missing_aoi_label=_GP3_R_UNSET,
    include_terminal=_GP3_R_UNSET,
    **kwargs,
) -> pd.DataFrame:
    r_mode = any(
        value is not _GP3_R_UNSET
        for value in (include_non_aoi, non_aoi_values, missing_aoi_label, include_terminal)
    )
    if not r_mode:
        out, _ = _sequence_frame(
            data,
            aoi_col,
            group_cols,
            time_col,
            collapse_repeats=collapse_repeats,
            **kwargs,
        )
        out["sequence_string"] = out.sequence.map(lambda x: " > ".join(map(str, x)))
        out["sequence_length"] = out.sequence.map(len)
        return out

    return _gp3_aoi_r_sequences(
        data,
        aoi_col=aoi_col,
        time_col="time" if time_col is None else time_col,
        group_cols=("subject", "MEDIA_ID", "trial_global") if group_cols is None else group_cols,
        include_non_aoi=True if include_non_aoi is _GP3_R_UNSET else include_non_aoi,
        non_aoi_values=_GP3_AOI_NON_VALUES if non_aoi_values is _GP3_R_UNSET else non_aoi_values,
        missing_aoi_label="missing_aoi" if missing_aoi_label is _GP3_R_UNSET else missing_aoi_label,
        include_terminal=True if include_terminal is _GP3_R_UNSET else include_terminal,
    )


def summarise_gazepoint_aoi_entries(
    data,
    aoi_col=None,
    group_cols=None,
    time_col=None,
    *,
    include_non_aoi=_GP3_R_UNSET,
    non_aoi_values=_GP3_R_UNSET,
    missing_aoi_label=_GP3_R_UNSET,
) -> pd.DataFrame:
    r_mode = any(
        value is not _GP3_R_UNSET for value in (include_non_aoi, non_aoi_values, missing_aoi_label)
    )
    if not r_mode:
        seq, groups = _sequence_frame(data, aoi_col, group_cols, time_col, collapse_repeats=True)
        rows = []
        for _, row in seq.iterrows():
            base = {column: row[column] for column in groups}
            counts = Counter(row.sequence)
            for aoi, count in counts.items():
                rows.append({**base, "aoi": aoi, "n_entries": count})
        return pd.DataFrame(rows)

    return _gp3_aoi_r_entries(
        data,
        aoi_col=aoi_col,
        time_col="time" if time_col is None else time_col,
        group_cols=("subject", "MEDIA_ID", "trial_global") if group_cols is None else group_cols,
        include_non_aoi=True if include_non_aoi is _GP3_R_UNSET else include_non_aoi,
        non_aoi_values=_GP3_AOI_NON_VALUES if non_aoi_values is _GP3_R_UNSET else non_aoi_values,
        missing_aoi_label="missing_aoi" if missing_aoi_label is _GP3_R_UNSET else missing_aoi_label,
    )


def summarise_gazepoint_aoi_transitions(
    data,
    aoi_col=None,
    group_cols=None,
    time_col=None,
    include_self=False,
    *,
    include_non_aoi=_GP3_R_UNSET,
    target_aoi_values=_GP3_R_UNSET,
    distractor_aoi_values=_GP3_R_UNSET,
    non_aoi_values=_GP3_R_UNSET,
    missing_aoi_label=_GP3_R_UNSET,
) -> pd.DataFrame:
    r_mode = any(
        value is not _GP3_R_UNSET
        for value in (
            include_non_aoi,
            target_aoi_values,
            distractor_aoi_values,
            non_aoi_values,
            missing_aoi_label,
        )
    )
    if not r_mode:
        seq, groups = _sequence_frame(
            data,
            aoi_col,
            group_cols,
            time_col,
            collapse_repeats=not include_self,
        )
        rows = []
        for _, row in seq.iterrows():
            base = {column: row[column] for column in groups}
            values = row.sequence
            for from_aoi, to_aoi in zip(values[:-1], values[1:], strict=False):
                if include_self or from_aoi != to_aoi:
                    rows.append({**base, "from_aoi": from_aoi, "to_aoi": to_aoi})
        if not rows:
            return pd.DataFrame(columns=groups + ["from_aoi", "to_aoi", "n_transitions"])
        return (
            pd.DataFrame(rows)
            .groupby(groups + ["from_aoi", "to_aoi"], dropna=False)
            .size()
            .rename("n_transitions")
            .reset_index()
        )

    groups = ("subject", "MEDIA_ID", "trial_global") if group_cols is None else group_cols
    groups = _gp3_aoi_r_list(groups, allow_none=False, unique=True, name="group_cols")
    include_non_aoi = True if include_non_aoi is _GP3_R_UNSET else include_non_aoi
    target_aoi_values = None if target_aoi_values is _GP3_R_UNSET else target_aoi_values
    distractor_aoi_values = None if distractor_aoi_values is _GP3_R_UNSET else distractor_aoi_values
    non_aoi_values = _GP3_AOI_NON_VALUES if non_aoi_values is _GP3_R_UNSET else non_aoi_values
    missing_aoi_label = "missing_aoi" if missing_aoi_label is _GP3_R_UNSET else missing_aoi_label
    target_values = {
        str(v).strip().lower() for v in _gp3_aoi_r_list(target_aoi_values, name="target_aoi_values")
    }
    distractor_values = {
        str(v).strip().lower()
        for v in _gp3_aoi_r_list(distractor_aoi_values, name="distractor_aoi_values")
    }
    background_values = {
        str(v).strip().lower()
        for v in _gp3_aoi_r_list(non_aoi_values, allow_none=False, name="non_aoi_values")
    }
    target_defined = bool(target_values)
    distractor_defined = bool(distractor_values)

    frame = ensure_dataframe(data, copy=False)
    sequence_columns = {
        "aoi_state",
        "transition_from",
        "transition_to",
        "dwell_before_transition_ms",
        "is_terminal_state",
    }
    if sequence_columns.issubset(frame.columns):
        sequences = frame.copy()
    else:
        sequences = _gp3_aoi_r_sequences(
            frame,
            aoi_col=aoi_col,
            time_col="time" if time_col is None else time_col,
            group_cols=groups,
            include_non_aoi=include_non_aoi,
            non_aoi_values=non_aoi_values,
            missing_aoi_label=missing_aoi_label,
            include_terminal=True,
        )
    required = [
        *groups,
        "aoi_state",
        "transition_from",
        "transition_to",
        "entry_duration_ms",
        "dwell_before_transition_ms",
        "is_non_aoi",
        "is_terminal_state",
    ]
    missing = [col for col in required if col not in sequences.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if sequences.empty:
        raise ValueError("No AOI sequence rows are available")

    sequences = sequences.copy()
    sequences["entry_duration_ms"] = pd.to_numeric(sequences["entry_duration_ms"], errors="coerce")
    sequences["dwell_before_transition_ms"] = pd.to_numeric(
        sequences["dwell_before_transition_ms"], errors="coerce"
    )
    sequences["is_non_aoi"] = sequences["is_non_aoi"].astype("boolean").fillna(False).astype(bool)
    sequences["is_terminal_state"] = (
        sequences["is_terminal_state"].astype("boolean").fillna(False).astype(bool)
    )

    def classify(value):
        if pd.isna(value):
            return pd.NA
        norm = str(value).strip().lower()
        if target_defined and norm in target_values:
            return "target"
        if distractor_defined and norm in distractor_values:
            return "distractor"
        if norm in background_values:
            return "background"
        return "other"

    state_rows = []
    for key, block in _gp3_aoi_r_summary_group_rows(sequences, groups):
        if not groups:
            base = {}
        else:
            key = key if isinstance(key, tuple) else (key,)
            base = dict(zip(groups, key, strict=True))
        dwell = pd.to_numeric(block["entry_duration_ms"], errors="coerce")
        state_rows.append(
            {
                **base,
                "n_states": int(len(block)),
                "n_aoi_states": int((~block["is_non_aoi"]).sum()),
                "n_non_aoi_states": int(block["is_non_aoi"].sum()),
                "total_state_dwell_ms": float(dwell.sum(skipna=True)),
                "mean_state_dwell_ms": float(dwell.mean()) if dwell.notna().any() else np.nan,
            }
        )
    state_summary = pd.DataFrame(state_rows)

    transitions = sequences.loc[
        (~sequences["is_terminal_state"]) & sequences["transition_to"].notna()
    ].copy()
    transition_rows = []
    for key, block in _gp3_aoi_r_summary_group_rows(transitions, groups):
        if not groups:
            base = {}
        else:
            key = key if isinstance(key, tuple) else (key,)
            base = dict(zip(groups, key, strict=True))
        from_class = block["transition_from"].map(classify)
        to_class = block["transition_to"].map(classify)
        self_reentry = block["transition_from"].eq(block["transition_to"])
        dwell = pd.to_numeric(block["dwell_before_transition_ms"], errors="coerce")
        transition_rows.append(
            {
                **base,
                "total_transitions": int(len(block)),
                "self_reentries": int(self_reentry.fillna(False).sum()),
                "target_to_distractor": int(
                    ((from_class == "target") & (to_class == "distractor")).sum()
                ),
                "distractor_to_target": int(
                    ((from_class == "distractor") & (to_class == "target")).sum()
                ),
                "background_to_target": int(
                    ((from_class == "background") & (to_class == "target")).sum()
                ),
                "target_to_background": int(
                    ((from_class == "target") & (to_class == "background")).sum()
                ),
                "background_to_distractor": int(
                    ((from_class == "background") & (to_class == "distractor")).sum()
                ),
                "distractor_to_background": int(
                    ((from_class == "distractor") & (to_class == "background")).sum()
                ),
                "target_to_target": int(((from_class == "target") & (to_class == "target")).sum()),
                "distractor_to_distractor": int(
                    ((from_class == "distractor") & (to_class == "distractor")).sum()
                ),
                "other_transitions": int(((from_class == "other") | (to_class == "other")).sum()),
                "total_pre_transition_dwell_ms": float(dwell.sum(skipna=True)),
                "mean_pre_transition_dwell_ms": float(dwell.mean())
                if dwell.notna().any()
                else np.nan,
                "median_pre_transition_dwell_ms": float(dwell.median())
                if dwell.notna().any()
                else np.nan,
                "max_pre_transition_dwell_ms": float(dwell.max())
                if dwell.notna().any()
                else np.nan,
            }
        )
    transition_summary = pd.DataFrame(transition_rows)
    if transition_summary.empty:
        transition_summary = state_summary.loc[:, groups].copy() if groups else pd.DataFrame([{}])
        zero_counts = [
            "total_transitions",
            "self_reentries",
            "target_to_distractor",
            "distractor_to_target",
            "background_to_target",
            "target_to_background",
            "background_to_distractor",
            "distractor_to_background",
            "target_to_target",
            "distractor_to_distractor",
            "other_transitions",
        ]
        for col in zero_counts:
            transition_summary[col] = 0
        transition_summary["total_pre_transition_dwell_ms"] = 0.0
        transition_summary["mean_pre_transition_dwell_ms"] = np.nan
        transition_summary["median_pre_transition_dwell_ms"] = np.nan
        transition_summary["max_pre_transition_dwell_ms"] = np.nan
    out = (
        state_summary.merge(transition_summary, on=groups, how="left")
        if groups
        else pd.concat(
            [state_summary.reset_index(drop=True), transition_summary.reset_index(drop=True)],
            axis=1,
        )
    )
    count_cols = [
        "total_transitions",
        "self_reentries",
        "target_to_distractor",
        "distractor_to_target",
        "background_to_target",
        "target_to_background",
        "background_to_distractor",
        "distractor_to_background",
        "target_to_target",
        "distractor_to_distractor",
        "other_transitions",
    ]
    for col in count_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0).astype(int)
    out["total_pre_transition_dwell_ms"] = pd.to_numeric(
        out["total_pre_transition_dwell_ms"], errors="coerce"
    ).fillna(0.0)
    out["target_aoi_defined"] = target_defined
    out["distractor_aoi_defined"] = distractor_defined
    statuses = []
    for _, row in out.iterrows():
        if row["total_transitions"] == 0:
            statuses.append("no_transitions")
        elif not target_defined and not distractor_defined:
            statuses.append("no_target_or_distractor_defined")
        elif not target_defined:
            statuses.append("no_target_defined")
        elif not distractor_defined:
            statuses.append("no_distractor_defined")
        else:
            statuses.append("ok")
    out["transition_feature_status"] = statuses
    return out


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
    *,
    by_cols=_GP3_R_UNSET,
    include_non_aoi=_GP3_R_UNSET,
    include_self_transitions=_GP3_R_UNSET,
    states=_GP3_R_UNSET,
    time_window=_GP3_R_UNSET,
    non_aoi_values=_GP3_R_UNSET,
    missing_aoi_label=_GP3_R_UNSET,
):
    r_mode = any(
        value is not _GP3_R_UNSET
        for value in (
            by_cols,
            include_non_aoi,
            include_self_transitions,
            states,
            time_window,
            non_aoi_values,
            missing_aoi_label,
        )
    )
    if not r_mode:
        if data is None:
            return compute_transition_matrix(sequence or [], normalize)
        seq, groups = _sequence_frame(
            data,
            aoi_col,
            group_cols,
            time_col,
            collapse_repeats=not include_self,
        )
        if not groups:
            return compute_transition_matrix(seq.iloc[0].sequence if len(seq) else [], normalize)
        rows = []
        for _, row in seq.iterrows():
            matrix = (
                compute_transition_matrix(row.sequence, normalize)
                .stack()
                .rename("value")
                .reset_index()
            )
            for column in groups:
                matrix[column] = row[column]
            rows.append(matrix)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    if data is None:
        raise ValueError("data must be supplied for R-compatible transition matrices")
    groups = ("subject", "MEDIA_ID", "trial_global") if group_cols is None else group_cols
    groups = _gp3_aoi_r_list(groups, allow_none=False, unique=True, name="group_cols")
    by = None if by_cols is _GP3_R_UNSET else by_cols
    by = _gp3_aoi_r_list(by, unique=True, name="by_cols")
    matrix_groups = list(dict.fromkeys([*groups, *by]))
    include_non_aoi = True if include_non_aoi is _GP3_R_UNSET else include_non_aoi
    include_self_transitions = (
        True if include_self_transitions is _GP3_R_UNSET else include_self_transitions
    )
    states_arg = None if states is _GP3_R_UNSET else states
    time_window_arg = None if time_window is _GP3_R_UNSET else time_window
    non_aoi_values = _GP3_AOI_NON_VALUES if non_aoi_values is _GP3_R_UNSET else non_aoi_values
    missing_aoi_label = "missing_aoi" if missing_aoi_label is _GP3_R_UNSET else missing_aoi_label
    if not isinstance(include_self_transitions, (bool, np.bool_)):
        raise ValueError("include_self_transitions must be TRUE or FALSE")
    if states_arg is not None:
        states_arg = _gp3_aoi_r_list(states_arg, allow_none=False, unique=True, name="states")
    if time_window_arg is not None:
        try:
            tw = np.asarray(list(time_window_arg), dtype=float)
        except Exception as exc:
            raise ValueError("time_window must be a finite numeric vector of length 2") from exc
        if tw.shape != (2,) or not np.isfinite(tw).all():
            raise ValueError("time_window must be a finite numeric vector of length 2")
        time_window_arg = tw

    frame = ensure_dataframe(data, copy=False)
    sequence_columns = {
        "aoi_state",
        "transition_from",
        "transition_to",
        "entry_start_time",
        "is_terminal_state",
    }
    if sequence_columns.issubset(frame.columns):
        sequences = frame.copy()
    else:
        sequences = _gp3_aoi_r_sequences(
            frame,
            aoi_col=aoi_col,
            time_col="time" if time_col is None else time_col,
            group_cols=matrix_groups,
            include_non_aoi=include_non_aoi,
            non_aoi_values=non_aoi_values,
            missing_aoi_label=missing_aoi_label,
            include_terminal=True,
        )
    required = [
        *matrix_groups,
        "aoi_state",
        "transition_from",
        "transition_to",
        "entry_start_time",
        "is_terminal_state",
    ]
    missing = [col for col in required if col not in sequences.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if time_window_arg is not None:
        start = pd.to_numeric(sequences["entry_start_time"], errors="coerce")
        lo, hi = sorted(time_window_arg.tolist())
        sequences = sequences.loc[(start >= lo) & (start <= hi)].copy()
        if sequences.empty:
            raise ValueError("No AOI sequence rows remain after applying time_window")
    transitions = sequences.loc[
        (~sequences["is_terminal_state"].astype("boolean").fillna(False))
        & sequences["transition_from"].notna()
        & sequences["transition_to"].notna()
    ].copy()
    if not include_self_transitions:
        transitions = transitions.loc[
            ~transitions["transition_from"].eq(transitions["transition_to"])
        ].copy()
    if states_arg is None:
        values = []
        for col in ["aoi_state", "transition_from", "transition_to"]:
            for value in sequences[col].tolist():
                if pd.notna(value) and str(value):
                    if value not in values:
                        values.append(value)
        state_values = [str(v) for v in values]
    else:
        state_values = list(states_arg)
    if not state_values:
        raise ValueError("No AOI states are available for matrix construction")

    if transitions.empty:
        long_table = pd.DataFrame(columns=[*by, "from", "to", "n", "row_total", "prob"])
    else:
        tmp = transitions.copy()
        tmp["from"] = tmp["transition_from"].astype(str)
        tmp["to"] = tmp["transition_to"].astype(str)
        count_cols = [*by, "from", "to"]
        long_table = (
            tmp.groupby(count_cols, dropna=False, sort=False).size().rename("n").reset_index()
        )
        denom_cols = [*by, "from"]
        if denom_cols:
            long_table["row_total"] = long_table.groupby(denom_cols, dropna=False, sort=False)[
                "n"
            ].transform("sum")
        else:
            long_table["row_total"] = long_table["n"].sum()
        long_table["prob"] = long_table["n"] / long_table["row_total"]

    def make_matrix(table, value_col):
        matrix = pd.DataFrame(0.0, index=state_values, columns=state_values)
        matrix.index.name = "from"
        matrix.columns.name = "to"
        for _, row in table.iterrows():
            from_state = str(row["from"])
            to_state = str(row["to"])
            if from_state in matrix.index and to_state in matrix.columns:
                matrix.loc[from_state, to_state] = row[value_col]
        if value_col == "n":
            matrix = matrix.astype(int)
        return matrix

    if not by:
        count_matrix = make_matrix(long_table, "n")
        probability_matrix = make_matrix(long_table, "prob")
        count_matrices = None
        probability_matrices = None
    else:
        count_matrix = None
        probability_matrix = None
        count_matrices = {}
        probability_matrices = {}
        keys = sequences.loc[:, by].drop_duplicates(ignore_index=True)
        for _, key_row in keys.iterrows():
            mask = pd.Series(True, index=long_table.index)
            labels = []
            for col in by:
                value = key_row[col]
                labels.append(f"{col}={'NA' if pd.isna(value) else value}")
                mask &= long_table[col].isna() if pd.isna(value) else long_table[col].eq(value)
            label = " | ".join(labels)
            this_long = long_table.loc[mask].copy()
            count_matrices[label] = make_matrix(this_long, "n")
            probability_matrices[label] = make_matrix(this_long, "prob")
    return {
        "count_matrix": count_matrix,
        "probability_matrix": probability_matrix,
        "count_matrices": count_matrices,
        "probability_matrices": probability_matrices,
        "long_table": long_table,
        "states": state_values,
        "settings": {
            "group_cols": groups,
            "by_cols": by,
            "include_non_aoi": bool(include_non_aoi),
            "include_self_transitions": bool(include_self_transitions),
            "time_window": None if time_window_arg is None else time_window_arg.tolist(),
        },
        "_gp3_class": "gp3_aoi_transition_matrix",
    }


def compute_gazepoint_time_varying_transition_matrix(
    data,
    aoi_col=None,
    time_col=None,
    bin_width=500,
    group_cols=None,
    normalize=False,
    *,
    from_col=_GP3_R_UNSET,
    to_col=_GP3_R_UNSET,
    window_col=_GP3_R_UNSET,
    window_size_ms=_GP3_R_UNSET,
    by_cols=_GP3_R_UNSET,
    count_col=_GP3_R_UNSET,
    states=_GP3_R_UNSET,
    complete_states=_GP3_R_UNSET,
    drop_self_transitions=_GP3_R_UNSET,
    normalise=_GP3_R_UNSET,
    name=_GP3_R_UNSET,
):
    r_mode = any(
        value is not _GP3_R_UNSET
        for value in (
            from_col,
            to_col,
            window_col,
            window_size_ms,
            by_cols,
            count_col,
            states,
            complete_states,
            drop_self_transitions,
            normalise,
            name,
        )
    )
    if not r_mode:
        df = ensure_dataframe(data)
        time_col = infer_column(df, "time", time_col, required=True)
        time_values = finite_numeric(df[time_col])
        df["_time_bin"] = (np.floor(time_values / bin_width) * bin_width).astype("Int64")
        groups = normalize_group_cols(df, group_cols) + ["_time_bin"]
        out = compute_gazepoint_aoi_transition_matrix(
            df,
            aoi_col=aoi_col,
            group_cols=groups,
            time_col=time_col,
            normalize=normalize,
        )
        return out.rename(columns={"_time_bin": "time_bin"})

    df = ensure_dataframe(data, copy=False)
    if df.empty:
        raise ValueError("data must contain at least one row")

    def resolve(value, candidates, arg, required=True):
        if value is not _GP3_R_UNSET and value is not None:
            value = _gp3_aoi_r_scalar_label(value, arg)
            if value not in df.columns:
                raise ValueError(f"{arg} must be present in data")
            return value
        found = next((candidate for candidate in candidates if candidate in df.columns), None)
        if found is None and required:
            raise ValueError(f"{arg} could not be detected and must be supplied")
        return found

    from_resolved = resolve(
        from_col,
        ("from_aoi", "from_state", "from", "origin", "previous_aoi", "previous_state", "AOI_from"),
        "from_col",
    )
    to_resolved = resolve(
        to_col,
        ("to_aoi", "to_state", "to", "destination", "next_aoi", "next_state", "AOI_to"),
        "to_col",
    )
    window_resolved = None
    if window_col is not _GP3_R_UNSET and window_col is not None:
        window_resolved = resolve(window_col, (), "window_col")
    time_resolved = None
    if time_col is not None:
        time_resolved = resolve(time_col, (), "time_col")
    if window_resolved is None:
        time_resolved = resolve(
            time_resolved if time_resolved is not None else _GP3_R_UNSET,
            (
                "time",
                "time_ms",
                "timestamp",
                "TIMESTAMP",
                "TIME",
                "sample_time",
                "transition_time",
                "transition_start_time",
            ),
            "time_col",
        )
        width = None if window_size_ms is _GP3_R_UNSET else window_size_ms
        if (
            isinstance(width, (bool, np.bool_))
            or not isinstance(width, (int, float, np.integer, np.floating))
            or not np.isfinite(width)
            or width <= 0
        ):
            raise ValueError("window_size_ms must be a finite positive number")
        width = float(width)
    else:
        width = None if window_size_ms is _GP3_R_UNSET else window_size_ms

    by = (
        []
        if by_cols is _GP3_R_UNSET or by_cols is None
        else _gp3_aoi_r_list(by_cols, allow_none=False, name="by_cols")
    )
    missing_by = [col for col in by if col not in df.columns]
    if missing_by:
        raise ValueError("All by_cols must be present in data")
    count_resolved = None
    if count_col is not _GP3_R_UNSET and count_col is not None:
        count_resolved = resolve(count_col, (), "count_col")
    states_arg = None if states is _GP3_R_UNSET else states
    complete = True if complete_states is _GP3_R_UNSET else complete_states
    drop_self = False if drop_self_transitions is _GP3_R_UNSET else drop_self_transitions
    norm = "row" if normalise is _GP3_R_UNSET else normalise
    object_name = "gazepoint_time_varying_transition_matrix" if name is _GP3_R_UNSET else name
    if not isinstance(complete, (bool, np.bool_)):
        raise ValueError("complete_states must be TRUE or FALSE")
    if not isinstance(drop_self, (bool, np.bool_)):
        raise ValueError("drop_self_transitions must be TRUE or FALSE")
    if norm not in {"row", "global", "none"}:
        raise ValueError("normalise must be one of: row, global, none")
    object_name = _gp3_aoi_r_scalar_label(object_name, "name")

    tmp = df.copy()
    tmp[".gp3_from"] = tmp[from_resolved].astype("string")
    tmp[".gp3_to"] = tmp[to_resolved].astype("string")
    tmp[".gp3_from"] = tmp[".gp3_from"].mask(tmp[".gp3_from"].isna() | tmp[".gp3_from"].eq(""))
    tmp[".gp3_to"] = tmp[".gp3_to"].mask(tmp[".gp3_to"].isna() | tmp[".gp3_to"].eq(""))
    tmp = tmp.loc[tmp[".gp3_from"].notna() & tmp[".gp3_to"].notna()].copy()
    if tmp.empty:
        raise ValueError("No valid non-missing transitions were found")
    if states_arg is None:
        state_values = sorted(set(tmp[".gp3_from"].astype(str)) | set(tmp[".gp3_to"].astype(str)))
    else:
        state_values = list(
            dict.fromkeys(_gp3_aoi_r_list(states_arg, allow_none=False, name="states"))
        )
    tmp = tmp.loc[tmp[".gp3_from"].isin(state_values) & tmp[".gp3_to"].isin(state_values)].copy()
    if drop_self:
        tmp = tmp.loc[~tmp[".gp3_from"].eq(tmp[".gp3_to"])].copy()
    if tmp.empty:
        raise ValueError("No transitions remain after applying states and drop_self_transitions")
    if count_resolved is None:
        tmp[".gp3_transition_count"] = 1.0
    else:
        counts = pd.to_numeric(tmp[count_resolved], errors="coerce")
        if counts.isna().any() or (~np.isfinite(counts)).any() or (counts < 0).any():
            raise ValueError("count_col must contain finite non-negative values")
        tmp[".gp3_transition_count"] = counts.to_numpy(float)
    if window_resolved is not None:
        labels = tmp[window_resolved].astype("string")
        tmp[".gp3_time_window"] = labels.mask(
            labels.isna() | labels.eq(""), "missing_window"
        ).astype(object)
        tmp[".gp3_time_window_start"] = np.nan
        tmp[".gp3_time_window_end"] = np.nan
    else:
        time_values = pd.to_numeric(tmp[time_resolved], errors="coerce")
        if time_values.isna().any() or (~np.isfinite(time_values)).any():
            raise ValueError(
                "time_col must contain finite numeric values when constructing time windows"
            )
        min_time = float(time_values.min())
        idx = np.floor((time_values - min_time) / width)
        tmp[".gp3_time_window_start"] = min_time + idx * width
        tmp[".gp3_time_window_end"] = tmp[".gp3_time_window_start"] + width
        tmp[".gp3_time_window"] = (
            tmp[".gp3_time_window_start"].map(lambda v: f"{v:g}")
            + "-"
            + tmp[".gp3_time_window_end"].map(lambda v: f"{v:g}")
        )

    window_cols = [*by, ".gp3_time_window", ".gp3_time_window_start", ".gp3_time_window_end"]
    count_groups = [*window_cols, ".gp3_from", ".gp3_to"]
    matrix_long = (
        tmp.groupby(count_groups, dropna=False, sort=False)
        .agg(
            transition_count=(".gp3_transition_count", "sum"),
            n_transition_rows=(".gp3_transition_count", "size"),
        )
        .reset_index()
    )
    if complete:
        windows = tmp.loc[:, window_cols].drop_duplicates(ignore_index=True)
        pairs = pd.MultiIndex.from_product(
            [state_values, state_values], names=[".gp3_from", ".gp3_to"]
        ).to_frame(index=False)
        if drop_self:
            pairs = pairs.loc[~pairs[".gp3_from"].eq(pairs[".gp3_to"])].reset_index(drop=True)
        windows["_key"] = 1
        pairs["_key"] = 1
        grid = windows.merge(pairs, on="_key", how="outer").drop(columns="_key")
        matrix_long = grid.merge(matrix_long, on=count_groups, how="left")
        matrix_long["transition_count"] = matrix_long["transition_count"].fillna(0.0)
        matrix_long["n_transition_rows"] = matrix_long["n_transition_rows"].fillna(0).astype(int)
    if norm == "row":
        denom_groups = [*window_cols, ".gp3_from"]
        matrix_long["transition_denominator"] = matrix_long.groupby(
            denom_groups, dropna=False, sort=False
        )["transition_count"].transform("sum")
        matrix_long["transition_probability"] = np.where(
            matrix_long["transition_denominator"] > 0,
            matrix_long["transition_count"] / matrix_long["transition_denominator"],
            np.nan,
        )
    elif norm == "global":
        matrix_long["transition_denominator"] = matrix_long.groupby(
            window_cols, dropna=False, sort=False
        )["transition_count"].transform("sum")
        matrix_long["transition_probability"] = np.where(
            matrix_long["transition_denominator"] > 0,
            matrix_long["transition_count"] / matrix_long["transition_denominator"],
            np.nan,
        )
    else:
        matrix_long["transition_denominator"] = np.nan
        matrix_long["transition_probability"] = np.nan

    count_wide = matrix_long.pivot_table(
        index=[*window_cols, ".gp3_from"],
        columns=".gp3_to",
        values="transition_count",
        aggfunc="first",
        fill_value=0,
        dropna=False,
    ).reset_index()
    count_wide.columns.name = None
    probability_wide = matrix_long.pivot_table(
        index=[*window_cols, ".gp3_from"],
        columns=".gp3_to",
        values="transition_probability",
        aggfunc="first",
        dropna=False,
    ).reset_index()
    probability_wide.columns.name = None
    time_windows = matrix_long.loc[:, window_cols].drop_duplicates(ignore_index=True)
    time_windows = time_windows.sort_values(
        [".gp3_time_window_start", ".gp3_time_window"], kind="stable", na_position="last"
    ).reset_index(drop=True)
    overview = pd.DataFrame(
        [
            {
                "object_name": object_name,
                "n_input_rows": int(len(df)),
                "n_rows_used": int(len(tmp)),
                "n_states": int(len(state_values)),
                "n_time_windows": int(matrix_long[".gp3_time_window"].nunique(dropna=False)),
                "n_by_groups": int(tmp.loc[:, by].drop_duplicates().shape[0]) if by else 1,
                "n_matrix_rows": int(len(matrix_long)),
                "total_transition_count": float(matrix_long["transition_count"].sum()),
                "normalise": norm,
                "complete_states": bool(complete),
                "drop_self_transitions": bool(drop_self),
            }
        ]
    )

    def collapse(value):
        if value is None or value is _GP3_R_UNSET:
            return pd.NA
        if isinstance(value, (list, tuple, set)):
            return ", ".join(map(str, value)) if value else pd.NA
        return str(value)

    settings = pd.DataFrame(
        {
            "setting": [
                "from_col",
                "to_col",
                "time_col",
                "window_col",
                "window_size_ms",
                "by_cols",
                "count_col",
                "states",
                "complete_states",
                "drop_self_transitions",
                "normalise",
                "name",
            ],
            "value": [
                from_resolved,
                to_resolved,
                collapse(time_resolved),
                collapse(window_resolved),
                collapse(width),
                collapse(by),
                collapse(count_resolved),
                collapse(state_values),
                str(bool(complete)).upper(),
                str(bool(drop_self)).upper(),
                norm,
                object_name,
            ],
        }
    )
    return {
        "overview": overview,
        "time_windows": time_windows,
        "matrix_long": matrix_long.reset_index(drop=True),
        "count_wide": count_wide,
        "probability_wide": probability_wide,
        "settings": settings,
        "_gp3_class": "gp3_time_varying_transition_matrix",
    }


def compute_gazepoint_aoi_entropy(
    data=None,
    sequence=None,
    aoi_col=None,
    group_cols=None,
    time_col=None,
    normalize=True,
    *,
    include_missing=False,
    missing_label="missing",
    collapse_repeats=False,
    log_base=2,
) -> pd.DataFrame:
    """Compute AOI entropy using legacy scalar-sequence or R v2.3.0 semantics."""
    if data is None:
        seqdf = pd.DataFrame([{"sequence": list(sequence or [])}])
        groups = []
        rows = []
        for _, row in seqdf.iterrows():
            values = row.sequence
            counts = np.array(list(Counter(values).values()), dtype=float)
            probabilities = counts / counts.sum() if counts.sum() else np.array([])
            entropy = (
                float(-(probabilities * np.log2(probabilities)).sum())
                if len(probabilities)
                else 0.0
            )
            n_states = len(counts)
            normalized = entropy / np.log2(n_states) if normalize and n_states > 1 else entropy
            rows.append(
                {
                    "n": len(values),
                    "n_states": n_states,
                    "entropy": entropy,
                    "normalized_entropy": normalized,
                }
            )
        return pd.DataFrame(rows)

    frame = ensure_dataframe(data, copy=False)
    if not isinstance(aoi_col, str) or not aoi_col:
        raise ValueError("aoi_col must be a non-empty string")
    groups = (
        []
        if group_cols is None
        else ([group_cols] if isinstance(group_cols, str) else list(group_cols))
    )
    needed = [aoi_col, *groups]
    if time_col is not None:
        if not isinstance(time_col, str) or not time_col:
            raise ValueError("time_col must be None or a non-empty string")
        needed.append(time_col)
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))
    if not isinstance(include_missing, (bool, np.bool_)):
        raise ValueError("include_missing must be TRUE or FALSE")
    if not isinstance(collapse_repeats, (bool, np.bool_)):
        raise ValueError("collapse_repeats must be TRUE or FALSE")
    if not isinstance(missing_label, str) or not missing_label:
        raise ValueError("missing_label must be a non-empty string")
    if (
        isinstance(log_base, (bool, np.bool_))
        or not isinstance(log_base, (int, float, np.integer, np.floating))
        or not np.isfinite(log_base)
        or log_base <= 0
        or log_base == 1
    ):
        raise ValueError("log_base must be a positive finite numeric scalar other than 1")

    def entropy_value(values):
        counts = pd.Series(values, dtype="object").value_counts(dropna=False).to_numpy(float)
        if counts.sum() <= 0:
            return np.nan
        probabilities = counts / counts.sum()
        return float(-(probabilities * (np.log(probabilities) / np.log(log_base))).sum())

    def normalized_entropy(value, n_levels):
        if not np.isfinite(value):
            return np.nan
        if n_levels <= 1:
            return 0.0
        maximum = np.log(n_levels) / np.log(log_base)
        return float(value / maximum) if maximum > 0 else np.nan

    iterator = [((), frame)] if not groups else frame.groupby(groups, sort=True, dropna=True)
    rows = []
    for key, block in iterator:
        if time_col is not None:
            block = block.sort_values(time_col, kind="stable", na_position="last")
        values = []
        for value in block[aoi_col]:
            missing_value = pd.isna(value) or str(value).strip() == ""
            if missing_value:
                if include_missing:
                    values.append(missing_label)
                continue
            values.append(str(value))
        if collapse_repeats:
            values = list(collapse_consecutive(values))

        base = {}
        if groups:
            key = key if isinstance(key, tuple) else (key,)
            base = {column: value for column, value in zip(groups, key, strict=True)}

        n_observations = len(values)
        n_aoi = len(set(values))
        if n_observations == 0:
            rows.append(
                {
                    **base,
                    "n_observations": 0,
                    "n_aoi": 0,
                    "spatial_entropy": np.nan,
                    "spatial_entropy_norm": np.nan,
                    "n_transitions": 0,
                    "n_transition_types": 0,
                    "transition_entropy": np.nan,
                    "transition_entropy_norm": np.nan,
                    "conditional_transition_entropy": np.nan,
                    "conditional_transition_entropy_norm": np.nan,
                    "entropy_status": "no_valid_aoi",
                }
            )
            continue

        spatial_entropy = entropy_value(values)
        spatial_norm = normalized_entropy(spatial_entropy, n_aoi)
        if n_observations < 2:
            rows.append(
                {
                    **base,
                    "n_observations": n_observations,
                    "n_aoi": n_aoi,
                    "spatial_entropy": spatial_entropy,
                    "spatial_entropy_norm": spatial_norm,
                    "n_transitions": 0,
                    "n_transition_types": 0,
                    "transition_entropy": np.nan,
                    "transition_entropy_norm": np.nan,
                    "conditional_transition_entropy": np.nan,
                    "conditional_transition_entropy_norm": np.nan,
                    "entropy_status": "no_transitions",
                }
            )
            continue

        from_values = values[:-1]
        to_values = values[1:]
        transitions = [
            f"{left} -> {right}" for left, right in zip(from_values, to_values, strict=True)
        ]
        transition_entropy = entropy_value(transitions)
        n_transition_types = len(set(transitions))
        transition_norm = normalized_entropy(transition_entropy, n_transition_types)

        conditional = 0.0
        for from_level in dict.fromkeys(from_values):
            indexes = [i for i, value in enumerate(from_values) if value == from_level]
            weight = len(indexes) / len(from_values)
            conditional += weight * entropy_value([to_values[i] for i in indexes])
        conditional_norm = normalized_entropy(conditional, n_aoi)

        rows.append(
            {
                **base,
                "n_observations": n_observations,
                "n_aoi": n_aoi,
                "spatial_entropy": spatial_entropy,
                "spatial_entropy_norm": spatial_norm,
                "n_transitions": len(transitions),
                "n_transition_types": n_transition_types,
                "transition_entropy": transition_entropy,
                "transition_entropy_norm": transition_norm,
                "conditional_transition_entropy": conditional,
                "conditional_transition_entropy_norm": conditional_norm,
                "entropy_status": "ok",
            }
        )
    return pd.DataFrame(rows)


def compute_gazepoint_aoi_sequence_metrics(
    data=None,
    aoi_col=None,
    group_cols=None,
    time_col=None,
    include_missing=False,
    missing_label="missing",
    collapse_repeats=True,
    *,
    sequence=None,
    **kwargs,
) -> pd.DataFrame:
    """Compute AOI sequence metrics with R gp3tools 2.3.0 semantics."""
    import numpy as np
    import pandas as pd

    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unknown}")

    # Preserve the pre-parity convenience path for standalone sequences.
    if data is None and sequence is not None:
        data = pd.DataFrame({"__aoi": list(sequence)})
        aoi_col = "__aoi"
        group_cols = []
    if not isinstance(data, pd.DataFrame):
        raise ValueError("`data` must be a data frame.")
    if not isinstance(aoi_col, str) or not aoi_col:
        raise ValueError("`aoi_col` must be a non-missing character scalar.")
    if group_cols is None:
        groups = []
    elif isinstance(group_cols, str):
        groups = [group_cols]
    else:
        groups = list(group_cols)
    if any(not isinstance(c, str) or not c for c in groups):
        raise ValueError("`group_cols` must be a character vector without missing or empty values.")
    if time_col is not None and (not isinstance(time_col, str) or not time_col):
        raise ValueError("`time_col` must be a non-missing character scalar.")
    if not isinstance(missing_label, str) or not missing_label:
        raise ValueError("`missing_label` must be a non-missing character scalar.")
    if not isinstance(include_missing, (bool, np.bool_)):
        raise ValueError("`include_missing` must be TRUE or FALSE.")
    if not isinstance(collapse_repeats, (bool, np.bool_)):
        raise ValueError("`collapse_repeats` must be TRUE or FALSE.")

    required = [aoi_col] + groups + ([time_col] if time_col else [])
    missing = [c for c in dict.fromkeys(required) if c not in data.columns]
    if missing:
        raise ValueError(f"`data` is missing required column(s): {', '.join(missing)}")

    if groups:
        frames = [frame for _, frame in data.groupby(groups, dropna=True, sort=True)]
    else:
        frames = [data]

    rows = []
    for frame in frames:
        if time_col is not None:
            frame = frame.sort_values(time_col, kind="stable", na_position="last")
        group_values = {c: frame.iloc[0][c] for c in groups} if len(frame) else {}

        raw = []
        for value in frame[aoi_col].tolist():
            missing_value = pd.isna(value) or str(value).strip() == ""
            if missing_value:
                if include_missing:
                    raw.append(missing_label)
            else:
                raw.append(str(value))

        sequence_length = len(raw)
        if sequence_length == 0:
            rows.append(
                {
                    **group_values,
                    "sequence_length": 0,
                    "n_aoi_visits": 0,
                    "n_unique_aoi": 0,
                    "transition_count": 0,
                    "revisit_count": np.nan,
                    "revisit_prop": np.nan,
                    "dominant_aoi": pd.NA,
                    "first_aoi": pd.NA,
                    "last_aoi": pd.NA,
                    "mean_run_length": np.nan,
                    "max_run_length": np.nan,
                    "sequence_status": "no_valid_aoi",
                }
            )
            continue

        run_values = [raw[0]]
        run_lengths = [1]
        for value in raw[1:]:
            if value == run_values[-1]:
                run_lengths[-1] += 1
            else:
                run_values.append(value)
                run_lengths.append(1)

        analysis = run_values if collapse_repeats else raw
        visits = len(analysis)
        transitions = max(visits - 1, 0)
        seen = set()
        revisits = 0
        for value in analysis:
            if value in seen:
                revisits += 1
            else:
                seen.add(value)

        counts = {value: raw.count(value) for value in sorted(set(raw))}
        dominant = max(counts, key=lambda value: counts[value])

        rows.append(
            {
                **group_values,
                "sequence_length": sequence_length,
                "n_aoi_visits": visits,
                "n_unique_aoi": len(set(analysis)),
                "transition_count": transitions,
                "revisit_count": revisits,
                "revisit_prop": revisits / visits if visits else np.nan,
                "dominant_aoi": dominant,
                "first_aoi": analysis[0],
                "last_aoi": analysis[-1],
                "mean_run_length": float(np.mean(run_lengths)),
                "max_run_length": int(max(run_lengths)),
                "sequence_status": "ok",
            }
        )

    columns = groups + [
        "sequence_length",
        "n_aoi_visits",
        "n_unique_aoi",
        "transition_count",
        "revisit_count",
        "revisit_prop",
        "dominant_aoi",
        "first_aoi",
        "last_aoi",
        "mean_run_length",
        "max_run_length",
        "sequence_status",
    ]
    return pd.DataFrame(rows, columns=columns)


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
    sequence_a,
    sequence_b,
    method="levenshtein",
    normalize=True,
    *,
    ignore_missing=None,
    missing_label="missing",
    collapse_repeats=False,
    substitution_cost=1,
    insertion_cost=1,
    deletion_cost=1,
):
    """Compute legacy scalar distance or an R v2.3.0 distance table."""
    r_mode = (
        ignore_missing is not None
        or collapse_repeats
        or substitution_cost != 1
        or insertion_cost != 1
        or deletion_cost != 1
    )

    if not r_mode:
        a = list(sequence_a)
        b = list(sequence_b)
        if method in {"levenshtein", "edit"}:
            d = float(_levenshtein(a, b))
        elif method == "jaccard":
            d = 1 - len(set(a) & set(b)) / max(len(set(a) | set(b)), 1)
        else:
            raise ValueError("method must be levenshtein/edit or jaccard")
        return d / max(len(a), len(b), 1) if normalize and method in {"levenshtein", "edit"} else d

    if not isinstance(ignore_missing, (bool, np.bool_)):
        raise ValueError("ignore_missing must be TRUE or FALSE")
    if not isinstance(collapse_repeats, (bool, np.bool_)):
        raise ValueError("collapse_repeats must be TRUE or FALSE")
    if not isinstance(missing_label, str) or not missing_label:
        raise ValueError("missing_label must be a non-empty string")

    costs = np.asarray(
        [substitution_cost, insertion_cost, deletion_cost],
        dtype=float,
    )
    if not np.isfinite(costs).all() or (costs < 0).any():
        raise ValueError("Edit costs must be non-negative finite numeric values")

    def prepare(values):
        out = []
        for value in values:
            missing = pd.isna(value) or (isinstance(value, str) and value.strip() == "")
            if missing:
                if not ignore_missing:
                    out.append(missing_label)
            else:
                out.append(str(value))
        if collapse_repeats:
            out = list(collapse_consecutive(out))
        return out

    a = prepare(sequence_a)
    b = prepare(sequence_b)

    n = len(a)
    m = len(b)
    d = np.zeros((n + 1, m + 1), dtype=float)
    d[:, 0] = np.arange(n + 1) * float(deletion_cost)
    d[0, :] = np.arange(m + 1) * float(insertion_cost)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0.0 if a[i - 1] == b[j - 1] else float(substitution_cost)
            d[i, j] = min(
                d[i - 1, j] + float(deletion_cost),
                d[i, j - 1] + float(insertion_cost),
                d[i - 1, j - 1] + cost,
            )

    distance = float(d[n, m])
    max_length = max(n, m)
    normalized_distance = 0.0 if max_length == 0 else distance / max_length
    return pd.DataFrame(
        [
            {
                "edit_distance": distance,
                "normalized_distance": float(normalized_distance),
                "sequence_a_length": n,
                "sequence_b_length": m,
                "distance_status": "ok",
            }
        ]
    )


def compute_gazepoint_sequence_recurrence(
    sequence=None,
    lag=1,
    *,
    data=None,
    aoi_col=None,
    group_cols=None,
    time_col=None,
    min_line=2,
    include_missing=False,
    missing_label="missing",
):
    """Compute legacy lag recurrence or R v2.3.0 recurrence-plot metrics."""
    r_mode = (
        data is not None
        or aoi_col is not None
        or group_cols is not None
        or time_col is not None
        or min_line != 2
        or include_missing
        or missing_label != "missing"
    )
    if not r_mode:
        s = list([] if sequence is None else sequence)
        matches = sum(s[i] == s[i - lag] for i in range(lag, len(s))) if len(s) > lag else 0
        denom = max(len(s) - lag, 0)
        return {"lag": lag, "n_pairs": denom, "recurrence": matches / denom if denom else np.nan}

    if not isinstance(min_line, (int, np.integer)) or min_line < 1:
        raise ValueError("min_line must be a positive integer")
    if not isinstance(include_missing, (bool, np.bool_)):
        raise ValueError("include_missing must be TRUE or FALSE")
    if not isinstance(missing_label, str) or not missing_label:
        raise ValueError("missing_label must be a non-empty string")

    def prepare(values):
        out = []
        for value in values:
            missing = pd.isna(value) or (isinstance(value, str) and value.strip() == "")
            if missing:
                if include_missing:
                    out.append(missing_label)
            else:
                out.append(str(value))
        return out

    def one_seq(values):
        x = prepare(values)
        n = len(x)
        if n < 2:
            return {
                "sequence_length": n,
                "recurrence_points": 0,
                "recurrence_rate": np.nan,
                "determinism": np.nan,
                "mean_diagonal_length": np.nan,
                "recurrence_status": "too_short",
            }
        mat = np.equal.outer(x, x)
        np.fill_diagonal(mat, False)
        recurrence_points = int(np.triu(mat, k=1).sum())
        possible = n * (n - 1) // 2
        recurrence_rate = recurrence_points / possible
        line_lengths = []
        for offset in range(1, n):
            diagonal = np.diag(mat, k=offset)
            if not len(diagonal):
                continue
            start = 0
            while start < len(diagonal):
                if not diagonal[start]:
                    start += 1
                    continue
                end = start + 1
                while end < len(diagonal) and diagonal[end]:
                    end += 1
                length = end - start
                if length >= min_line:
                    line_lengths.append(length)
                start = end
        deterministic_points = int(sum(line_lengths))
        determinism = deterministic_points / recurrence_points if recurrence_points > 0 else np.nan
        return {
            "sequence_length": n,
            "recurrence_points": recurrence_points,
            "recurrence_rate": float(recurrence_rate),
            "determinism": float(determinism) if np.isfinite(determinism) else np.nan,
            "mean_diagonal_length": float(np.mean(line_lengths)) if line_lengths else np.nan,
            "recurrence_status": "ok",
        }

    if data is None:
        if sequence is None:
            raise ValueError("Supply either data with aoi_col or sequence")
        return pd.DataFrame([one_seq(sequence)])

    frame = ensure_dataframe(data, copy=False)
    if not isinstance(aoi_col, str) or not aoi_col:
        raise ValueError("aoi_col must be a single non-empty column name")
    groups = (
        []
        if group_cols is None
        else ([group_cols] if isinstance(group_cols, str) else list(group_cols))
    )
    needed = [aoi_col, *groups] + ([time_col] if time_col is not None else [])
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise ValueError("data is missing required column(s): " + ", ".join(missing))

    blocks = [(None, frame)] if not groups else frame.groupby(groups, dropna=True, sort=True)
    rows = []
    for key, block in blocks:
        if time_col is not None:
            block = block.sort_values(time_col, kind="stable", na_position="last")
        row = {}
        if groups:
            values = key if isinstance(key, tuple) else (key,)
            row.update(dict(zip(groups, values, strict=True)))
        row.update(one_seq(block[aoi_col]))
        rows.append(row)
    return pd.DataFrame(rows)


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


def _gp3_transition_r_scalar_string(
    value,
    name,
):
    if (
        value is None
        or not isinstance(
            value,
            str,
        )
        or value == ""
    ):
        raise ValueError(f"{name} must be a single non-empty column name")

    return value


def _gp3_transition_r_group_cols(
    value,
):
    if value is None:
        return None

    if isinstance(
        value,
        str,
    ):
        value = [value]
    else:
        value = list(value)

    if not value or any(item is None or str(item) == "" for item in value):
        raise ValueError("group_cols must contain one or more non-empty column names")

    return [str(item) for item in value]


def _gp3_transition_r_prepare_aoi(
    values,
):
    output = []

    for value in values:
        if pd.isna(value):
            continue

        value = str(value)

        if value.strip() == "":
            continue

        output.append(value)

    return output


def compute_gazepoint_transition_network_metrics(
    matrix=None,
    *,
    data=None,
    aoi_col=None,
    from_col=None,
    to_col=None,
    group_cols=None,
    time_col=None,
    include_self_loops=True,
):
    """Compute legacy NetworkX or R v2.3.0 transition metrics."""
    r_mode = (
        data is not None
        or aoi_col is not None
        or from_col is not None
        or to_col is not None
        or group_cols is not None
        or time_col is not None
        or include_self_loops is not True
    )

    if not r_mode:
        if isinstance(
            matrix,
            pd.DataFrame,
        ):
            graph = nx.from_pandas_adjacency(
                matrix,
                create_using=nx.DiGraph,
            )
        else:
            graph = nx.from_numpy_array(
                np.asarray(matrix),
                create_using=nx.DiGraph,
            )

        pagerank = nx.pagerank(
            graph,
            weight="weight",
        )

        return pd.DataFrame(
            [
                {
                    "node": node,
                    "in_degree": graph.in_degree(
                        node,
                        weight="weight",
                    ),
                    "out_degree": graph.out_degree(
                        node,
                        weight="weight",
                    ),
                    "pagerank": pagerank.get(
                        node,
                        np.nan,
                    ),
                }
                for node in graph.nodes
            ]
        )

    if data is not None and matrix is not None:
        raise TypeError("supply either matrix or data, not both")

    frame = ensure_dataframe(
        data if data is not None else matrix,
        copy=False,
    )

    groups = _gp3_transition_r_group_cols(group_cols)

    transitions = []

    if from_col is not None and to_col is not None:
        from_col = _gp3_transition_r_scalar_string(
            from_col,
            "from_col",
        )

        to_col = _gp3_transition_r_scalar_string(
            to_col,
            "to_col",
        )

        missing = [
            column
            for column in (
                from_col,
                to_col,
            )
            if column not in frame.columns
        ]

        if missing:
            raise ValueError("data is missing required column(s): " + ", ".join(missing))

        for from_value, to_value in zip(
            frame[from_col],
            frame[to_col],
            strict=False,
        ):
            if pd.isna(from_value) or pd.isna(to_value):
                continue

            transitions.append(
                (
                    str(from_value),
                    str(to_value),
                )
            )

    else:
        aoi_col = _gp3_transition_r_scalar_string(
            aoi_col,
            "aoi_col",
        )

        if time_col is not None:
            time_col = _gp3_transition_r_scalar_string(
                time_col,
                "time_col",
            )

        required = [
            aoi_col,
        ]

        if groups:
            required.extend(groups)

        if time_col is not None:
            required.append(time_col)

        missing = [column for column in required if column not in frame.columns]

        if missing:
            raise ValueError("data is missing required column(s): " + ", ".join(missing))

        if groups:
            blocks = [
                block
                for _, block in frame.groupby(
                    groups,
                    sort=True,
                    dropna=True,
                )
            ]
        else:
            blocks = [frame]

        for block in blocks:
            if time_col is not None:
                block = block.sort_values(
                    time_col,
                    kind="stable",
                    na_position="last",
                )

            states = _gp3_transition_r_prepare_aoi(block[aoi_col])

            transitions.extend(
                zip(
                    states[:-1],
                    states[1:],
                    strict=False,
                )
            )

    if include_self_loops is not True:
        transitions = [transition for transition in transitions if transition[0] != transition[1]]

    if not transitions:
        return {
            "graph_summary": pd.DataFrame(
                [
                    {
                        "n_states": 0,
                        "n_edges": 0,
                        "density": np.nan,
                        "self_loops": 0,
                        "total_transitions": 0,
                    }
                ]
            ),
            "state_summary": pd.DataFrame(),
            "network_status": "empty",
        }

    transition_frame = pd.DataFrame(
        transitions,
        columns=[
            "from_state",
            "to_state",
        ],
    )

    edge_summary = (
        transition_frame.value_counts(
            [
                "from_state",
                "to_state",
            ],
            sort=False,
        )
        .rename("count")
        .reset_index()
        .sort_values(
            [
                "from_state",
                "to_state",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )

    states = sorted(set(edge_summary["from_state"]) | set(edge_summary["to_state"]))

    n_states = len(states)
    n_edges = len(edge_summary)

    possible_edges = n_states**2 if include_self_loops is True else n_states * (n_states - 1)

    rows = []

    for state in states:
        outgoing = edge_summary["from_state"].eq(state)

        incoming = edge_summary["to_state"].eq(state)

        rows.append(
            {
                "state": state,
                "out_degree": int(outgoing.sum()),
                "in_degree": int(incoming.sum()),
                "weighted_out_degree": int(
                    edge_summary.loc[
                        outgoing,
                        "count",
                    ].sum()
                ),
                "weighted_in_degree": int(
                    edge_summary.loc[
                        incoming,
                        "count",
                    ].sum()
                ),
            }
        )

    state_summary = pd.DataFrame(rows)

    graph_summary = pd.DataFrame(
        [
            {
                "n_states": n_states,
                "n_edges": n_edges,
                "density": (n_edges / possible_edges if possible_edges else np.nan),
                "self_loops": int(edge_summary["from_state"].eq(edge_summary["to_state"]).sum()),
                "total_transitions": int(edge_summary["count"].sum()),
                "mean_out_degree": float(state_summary["out_degree"].mean()),
                "max_out_degree": int(state_summary["out_degree"].max()),
                "mean_in_degree": float(state_summary["in_degree"].mean()),
                "max_in_degree": int(state_summary["in_degree"].max()),
            }
        ]
    )

    return {
        "graph_summary": graph_summary,
        "state_summary": state_summary,
        "edge_summary": edge_summary,
        "network_status": "ok",
    }


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
    data=None,
    aoi_col=None,
    group_cols=None,
    time_col=None,
    min_length=2,
    max_length=None,
    max_missing_prop=0.5,
    z_threshold=3.0,
    min_unique_aoi=1,
    *,
    sequence=None,
    **kwargs,
) -> pd.DataFrame:
    """Flag unusual AOI sequences using R gp3tools 2.3.0 semantics."""
    import numpy as np
    import pandas as pd

    # Preserve the pre-parity standalone-sequence behavior exactly on the
    # Python-only compatibility route. The canonical R route below remains
    # data/aoi_col/group_cols based and continues to be oracle tested.
    if data is None and sequence is not None:
        comp = compute_gazepoint_sequence_complexity(
            data=None,
            sequence=sequence,
            **kwargs,
        )
        x = pd.to_numeric(comp["complexity_index"], errors="coerce")
        spread = float(x.std(ddof=0)) if len(x) > 1 else 0.0
        if len(x) > 1 and np.isfinite(spread) and spread > 0:
            z = (x - x.mean()) / spread
        else:
            z = pd.Series(0.0, index=x.index)
        comp = comp.copy()
        comp["anomaly_score"] = z.abs()
        comp["anomaly"] = comp["anomaly_score"] > z_threshold
        return comp

    if kwargs:
        unknown = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword argument(s): {unknown}")
    if not isinstance(data, pd.DataFrame):
        raise ValueError("data must be a data frame.")
    if not isinstance(aoi_col, str) or not aoi_col:
        raise ValueError("aoi_col must be a single non-empty column name.")
    if group_cols is None:
        raise ValueError("group_cols must contain one or more non-empty column names.")
    if isinstance(group_cols, str):
        group_cols = [group_cols]
    else:
        group_cols = list(group_cols)
    if not group_cols or any(not isinstance(c, str) or not c for c in group_cols):
        raise ValueError("group_cols must contain one or more non-empty column names.")
    if time_col is not None and (not isinstance(time_col, str) or not time_col):
        raise ValueError("time_col must be a single non-empty column name.")
    required = [aoi_col] + group_cols + ([time_col] if time_col else [])
    missing_cols = [c for c in required if c not in data.columns]
    if missing_cols:
        raise ValueError("`data` is missing required column(s): " + ", ".join(missing_cols))
    if (
        isinstance(min_length, (bool, np.bool_))
        or not isinstance(min_length, (int, float, np.number))
        or pd.isna(min_length)
        or min_length < 0
    ):
        raise ValueError("min_length must be a non-negative number.")
    if max_length is not None and (
        isinstance(max_length, (bool, np.bool_))
        or not isinstance(max_length, (int, float, np.number))
        or pd.isna(max_length)
    ):
        raise ValueError("max_length must be NULL or a single number.")
    if (
        isinstance(max_missing_prop, (bool, np.bool_))
        or not isinstance(max_missing_prop, (int, float, np.number))
        or pd.isna(max_missing_prop)
        or not 0 <= max_missing_prop <= 1
    ):
        raise ValueError("max_missing_prop must be between 0 and 1.")
    if (
        isinstance(z_threshold, (bool, np.bool_))
        or not isinstance(z_threshold, (int, float, np.number))
        or pd.isna(z_threshold)
        or z_threshold <= 0
    ):
        raise ValueError("z_threshold must be positive.")

    grouped = data.groupby(group_cols, dropna=True, sort=True)
    rows = []
    for _, frame in grouped:
        if time_col is not None:
            frame = frame.sort_values(time_col, kind="stable")
        raw = frame[aoi_col]
        missing = raw.isna() | raw.astype("string").str.strip().fillna("").eq("")
        observed = raw.loc[~missing].astype(str).tolist()
        key = "|".join(f"{c}={frame.iloc[0][c]}" for c in group_cols)
        row = {
            ".gp3_key": key,
            "total_observations": int(len(raw)),
            "sequence_length": int(len(observed)),
            "missing_prop": float(missing.mean()) if len(raw) else np.nan,
            "n_unique_aoi": int(len(set(observed))),
        }
        row.update({c: frame.iloc[0][c] for c in group_cols})
        rows.append(row)
    out = pd.DataFrame(rows)
    if out.empty:
        return out

    lengths = out["sequence_length"].astype(float)
    len_sd = float(lengths.std(ddof=1)) if len(lengths) >= 2 else np.nan
    len_mean = float(lengths.mean())
    if not np.isfinite(len_sd) or len_sd == 0:
        out["length_z"] = 0.0
    else:
        out["length_z"] = (lengths - len_mean) / len_sd
    out["flag_short"] = out["sequence_length"] < min_length
    out["flag_long"] = False if max_length is None else out["sequence_length"] > max_length
    out["flag_high_missing"] = out["missing_prop"] > max_missing_prop
    out["flag_length_outlier"] = out["length_z"].abs() > z_threshold
    out["flag_low_unique"] = out["n_unique_aoi"] < min_unique_aoi
    out["anomaly_flag"] = (
        out["flag_short"]
        | out["flag_long"]
        | out["flag_high_missing"]
        | out["flag_length_outlier"]
        | out["flag_low_unique"]
    )

    reasons = []
    for _, row in out.iterrows():
        current = []
        if bool(row["flag_short"]):
            current.append("short_sequence")
        if bool(row["flag_long"]):
            current.append("long_sequence")
        if bool(row["flag_high_missing"]):
            current.append("high_missing")
        if bool(row["flag_length_outlier"]):
            current.append("length_outlier")
        if bool(row["flag_low_unique"]):
            current.append("low_unique_aoi")
        reasons.append(";".join(current) if current else "none")
    out["anomaly_reason"] = reasons
    out["anomaly_status"] = "ok"
    return out.reset_index(drop=True)


def summarise_gazepoint_aoi_trial_features(
    data,
    aoi_col=None,
    trial_col=None,
    group_cols=None,
    *,
    time_col="time",
    include_non_aoi=True,
    target_aoi_values=None,
    distractor_aoi_values=None,
    non_aoi_values=_GP3_AOI_NON_VALUES,
    missing_aoi_label="missing_aoi",
) -> pd.DataFrame:
    if trial_col is not None:
        df = ensure_dataframe(data, copy=False)
        trial_col = infer_column(df, "trial", trial_col, required=True)
        groups = normalize_group_cols(df, group_cols) + [trial_col]
        return summarise_aoi_samples(df, aoi_col=aoi_col, group_cols=groups)

    groups = ("subject", "MEDIA_ID", "trial_global") if group_cols is None else group_cols
    groups = _gp3_aoi_r_list(groups, allow_none=False, unique=True, name="group_cols")
    frame = ensure_dataframe(data, copy=False)
    entry_columns = {
        "aoi_state",
        "entry_start_time",
        "entry_end_time",
        "entry_duration_ms",
        "n_samples",
    }
    if entry_columns.issubset(frame.columns):
        entries = frame.copy()
        missing = [col for col in [*groups, *entry_columns] if col not in entries.columns]
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))
    else:
        entries = _gp3_aoi_r_entries(
            frame,
            aoi_col=aoi_col,
            time_col=time_col,
            group_cols=groups,
            include_non_aoi=True,
            non_aoi_values=non_aoi_values,
            missing_aoi_label=missing_aoi_label,
        )
    if entries.empty:
        raise ValueError("No AOI entries are available")
    target_values = {
        str(v).strip().lower() for v in _gp3_aoi_r_list(target_aoi_values, name="target_aoi_values")
    }
    distractor_values = {
        str(v).strip().lower()
        for v in _gp3_aoi_r_list(distractor_aoi_values, name="distractor_aoi_values")
    }
    background_values = {
        str(v).strip().lower()
        for v in _gp3_aoi_r_list(non_aoi_values, allow_none=False, name="non_aoi_values")
    }
    target_defined = bool(target_values)
    distractor_defined = bool(distractor_values)
    entries = entries.copy()
    state = entries["aoi_state"].astype("string").str.strip()
    entries["aoi_state"] = state.mask(state.isna() | state.eq(""), missing_aoi_label).astype(object)
    for col in ["entry_start_time", "entry_end_time", "entry_duration_ms"]:
        entries[col] = pd.to_numeric(entries[col], errors="coerce")
    entries["n_samples"] = pd.to_numeric(entries["n_samples"], errors="coerce")
    entries = entries.loc[entries["entry_start_time"].notna()].copy()
    fallback_non = entries["aoi_state"].astype(str).str.strip().str.lower().isin(background_values)
    if "is_non_aoi" not in entries:
        entries["is_non_aoi"] = fallback_non
    else:
        entries["is_non_aoi"] = (
            entries["is_non_aoi"]
            .astype("boolean")
            .fillna(pd.Series(fallback_non, index=entries.index))
            .astype(bool)
        )
    if not include_non_aoi:
        entries = entries.loc[~entries["is_non_aoi"]].copy()
    if entries.empty:
        raise ValueError("No AOI entries remain after applying include_non_aoi")

    def classify(row):
        if row["is_non_aoi"]:
            return "background"
        norm = str(row["aoi_state"]).strip().lower()
        if target_defined and norm in target_values:
            return "target"
        if distractor_defined and norm in distractor_values:
            return "distractor"
        return "other_aoi"

    entries["state_class"] = entries.apply(classify, axis=1)
    sort_cols = [*groups, "entry_start_time"] if groups else ["entry_start_time"]
    entries = entries.sort_values(sort_cols, kind="stable")

    rows = []
    for key, block in _gp3_aoi_r_summary_group_rows(entries, groups):
        base = {}
        if groups:
            key = key if isinstance(key, tuple) else (key,)
            base = dict(zip(groups, key, strict=True))
        duration = pd.to_numeric(block["entry_duration_ms"], errors="coerce")
        starts = pd.to_numeric(block["entry_start_time"], errors="coerce")
        ends = pd.to_numeric(block["entry_end_time"], errors="coerce")
        is_aoi = ~block["is_non_aoi"]
        target = block["state_class"].eq("target")
        distractor = block["state_class"].eq("distractor")
        other = block["state_class"].eq("other_aoi")

        def ssum(mask, duration=duration):
            return float(duration.loc[mask].sum(skipna=True))

        def smean(mask, duration=duration):
            values = duration.loc[mask].dropna()
            return float(values.mean()) if len(values) else np.nan

        def smedian(mask, duration=duration):
            values = duration.loc[mask].dropna()
            return float(values.median()) if len(values) else np.nan

        def smax(mask, duration=duration):
            values = duration.loc[mask].dropna()
            return float(values.max()) if len(values) else np.nan

        def smin_time(mask, starts=starts):
            values = starts.loc[mask].dropna()
            return float(values.min()) if len(values) else np.nan

        aoi_states = block.loc[is_aoi, "aoi_state"].astype(str)
        first_aoi = block.loc[is_aoi].sort_values("entry_start_time", kind="stable")
        last_aoi = block.loc[is_aoi].sort_values("entry_start_time", kind="stable", ascending=False)
        total_dwell = float(duration.sum(skipna=True))
        aoi_dwell = ssum(is_aoi)
        target_entries = int(target.sum())
        distractor_entries = int(distractor.sum())
        row = {
            **base,
            "trial_start_time": float(starts.min()) if starts.notna().any() else np.nan,
            "trial_end_time": float(ends.max()) if ends.notna().any() else np.nan,
            "n_entries": int(len(block)),
            "n_samples_in_entries": float(
                pd.to_numeric(block["n_samples"], errors="coerce").sum(skipna=True)
            ),
            "n_aoi_entries": int(is_aoi.sum()),
            "n_non_aoi_entries": int(block["is_non_aoi"].sum()),
            "n_unique_aoi_states": int(aoi_states[aoi_states.ne("")].nunique()),
            "total_entry_dwell_ms": total_dwell,
            "total_aoi_dwell_ms": aoi_dwell,
            "total_non_aoi_dwell_ms": ssum(block["is_non_aoi"]),
            "mean_entry_duration_ms": float(duration.mean()) if duration.notna().any() else np.nan,
            "median_entry_duration_ms": float(duration.median())
            if duration.notna().any()
            else np.nan,
            "max_entry_duration_ms": float(duration.max()) if duration.notna().any() else np.nan,
            "mean_aoi_entry_duration_ms": smean(is_aoi),
            "median_aoi_entry_duration_ms": smedian(is_aoi),
            "max_aoi_entry_duration_ms": smax(is_aoi),
            "first_aoi_state": first_aoi["aoi_state"].iloc[0] if len(first_aoi) else pd.NA,
            "last_aoi_state": last_aoi["aoi_state"].iloc[0] if len(last_aoi) else pd.NA,
            "first_aoi_time_ms": smin_time(is_aoi),
            "last_aoi_time_ms": float(starts.loc[is_aoi].max())
            if starts.loc[is_aoi].notna().any()
            else np.nan,
            "target_entries": target_entries,
            "target_revisits": max(target_entries - 1, 0),
            "target_dwell_ms": ssum(target),
            "target_ttff_ms": smin_time(target),
            "mean_target_entry_duration_ms": smean(target),
            "distractor_entries": distractor_entries,
            "distractor_revisits": max(distractor_entries - 1, 0),
            "distractor_dwell_ms": ssum(distractor),
            "distractor_ttff_ms": smin_time(distractor),
            "mean_distractor_entry_duration_ms": smean(distractor),
            "other_aoi_entries": int(other.sum()),
            "other_aoi_dwell_ms": ssum(other),
        }
        row["trial_duration_ms"] = row["trial_end_time"] - row["trial_start_time"]
        row["aoi_dwell_prop"] = (
            row["total_aoi_dwell_ms"] / total_dwell if total_dwell > 0 else np.nan
        )
        row["non_aoi_dwell_prop"] = (
            row["total_non_aoi_dwell_ms"] / total_dwell if total_dwell > 0 else np.nan
        )
        row["target_dwell_prop_of_aoi"] = (
            row["target_dwell_ms"] / aoi_dwell if aoi_dwell > 0 else np.nan
        )
        row["distractor_dwell_prop_of_aoi"] = (
            row["distractor_dwell_ms"] / aoi_dwell if aoi_dwell > 0 else np.nan
        )
        row["target_aoi_defined"] = target_defined
        row["distractor_aoi_defined"] = distractor_defined
        if row["n_aoi_entries"] == 0:
            row["aoi_trial_feature_status"] = "no_aoi_entries"
        elif not target_defined and not distractor_defined:
            row["aoi_trial_feature_status"] = "no_target_or_distractor_defined"
        elif target_defined and target_entries == 0:
            row["aoi_trial_feature_status"] = "target_not_observed"
        elif distractor_defined and distractor_entries == 0:
            row["aoi_trial_feature_status"] = "distractor_not_observed"
        else:
            row["aoi_trial_feature_status"] = "ok"
        rows.append(row)
    features = pd.DataFrame(rows)
    transitions = summarise_gazepoint_aoi_transitions(
        entries,
        group_cols=groups,
        include_non_aoi=True,
        target_aoi_values=target_aoi_values,
        distractor_aoi_values=distractor_aoi_values,
        non_aoi_values=non_aoi_values,
        missing_aoi_label=missing_aoi_label,
    )
    keep = [
        *groups,
        "total_transitions",
        "self_reentries",
        "target_to_distractor",
        "distractor_to_target",
        "background_to_target",
        "target_to_background",
        "background_to_distractor",
        "distractor_to_background",
        "target_to_target",
        "distractor_to_distractor",
        "other_transitions",
        "mean_pre_transition_dwell_ms",
        "transition_feature_status",
    ]
    transitions = transitions.loc[:, keep]
    return (
        features.merge(transitions, on=groups, how="left")
        if groups
        else pd.concat(
            [features.reset_index(drop=True), transitions.reset_index(drop=True)], axis=1
        )
    )


def summarise_gazepoint_aoi_windows(
    data,
    aoi_col=None,
    time_col=None,
    windows=None,
    group_cols=None,
    *,
    subject_col="subject",
    condition_col="condition",
    target_aoi_values=None,
    distractor_aoi_values=None,
    non_aoi_values=_GP3_AOI_NON_VALUES,
    window_label_col="window_label",
    window_start_col="window_start_ms",
    window_end_col="window_end_ms",
    include_right_endpoint=False,
    missing_condition_label="all_data",
    missing_aoi_label="missing_aoi",
) -> pd.DataFrame:
    if isinstance(windows, dict):
        df = ensure_dataframe(data, copy=False)
        time_col = infer_column(df, "time", time_col, required=True)
        t = finite_numeric(df[time_col])
        rows = []
        for name, (lo, hi) in windows.items():
            tmp = summarise_aoi_samples(
                df.loc[t.between(lo, hi)],
                aoi_col=aoi_col,
                group_cols=group_cols,
            )
            tmp["window"] = name
            tmp["window_start"] = lo
            tmp["window_end"] = hi
            rows.append(tmp)
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    df = ensure_dataframe(data, copy=False).copy()
    if windows is None:
        raise ValueError("windows must be a numeric vector or a data frame")
    time_col = "time" if time_col is None else _gp3_aoi_r_scalar_label(time_col, "time_col")
    subject_col = _gp3_aoi_r_scalar_label(subject_col, "subject_col")
    if condition_col is not None:
        condition_col = _gp3_aoi_r_scalar_label(condition_col, "condition_col")
    aoi_col = _gp3_aoi_r_resolve_column(
        df, aoi_col, ("aoi_current", "AOI", "aoi_state"), name="aoi_col", allow_none=True
    )
    required = [time_col, aoi_col, subject_col]
    if condition_col is not None and condition_col in df.columns:
        required.append(condition_col)
    if group_cols is not None:
        required.extend(_gp3_aoi_r_list(group_cols, allow_none=False, name="group_cols"))
    missing = [col for col in dict.fromkeys(required) if col not in df.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if group_cols is None:
        defaults = [subject_col, condition_col, "MEDIA_ID", "trial_global", "trial"]
        groups = list(
            dict.fromkeys([col for col in defaults if col is not None and col in df.columns])
        )
        if subject_col not in groups:
            groups.insert(0, subject_col)
    else:
        groups = list(
            dict.fromkeys(_gp3_aoi_r_list(group_cols, allow_none=False, name="group_cols"))
        )
    if condition_col is not None:
        if condition_col not in df.columns:
            df[condition_col] = missing_condition_label
        if condition_col not in groups:
            groups.append(condition_col)
    if not groups:
        raise ValueError("No grouping columns could be detected")
    if not isinstance(include_right_endpoint, (bool, np.bool_)):
        raise ValueError("include_right_endpoint must be TRUE or FALSE")

    if isinstance(windows, pd.DataFrame):
        for col in [window_label_col, window_start_col, window_end_col]:
            if col not in windows.columns:
                raise ValueError("Missing required window columns: " + col)
        window_table = pd.DataFrame(
            {
                "window_label": windows[window_label_col].astype("string"),
                "window_start_ms": pd.to_numeric(windows[window_start_col], errors="coerce"),
                "window_end_ms": pd.to_numeric(windows[window_end_col], errors="coerce"),
            }
        )
    else:
        try:
            breaks = np.asarray(list(windows), dtype=float)
        except Exception as exc:
            raise ValueError("windows must be a numeric vector or a data frame") from exc
        if breaks.size < 2 or not np.isfinite(breaks).all():
            raise ValueError("windows must contain at least two finite numeric breakpoints")
        breaks = np.unique(np.sort(breaks))
        if breaks.size < 2:
            raise ValueError("windows must contain at least two distinct breakpoints")
        starts, ends = breaks[:-1], breaks[1:]
        window_table = pd.DataFrame(
            {
                "window_label": [f"{a:g}_{b:g}ms" for a, b in zip(starts, ends, strict=True)],
                "window_start_ms": starts,
                "window_end_ms": ends,
            }
        )
    if (
        window_table["window_label"].isna().any()
        or window_table["window_label"].astype(str).eq("").any()
    ):
        raise ValueError("Window labels must be non-missing and non-empty")
    if not np.isfinite(window_table[["window_start_ms", "window_end_ms"]].to_numpy(float)).all():
        raise ValueError("Window start and end values must be finite")
    if (window_table["window_end_ms"] <= window_table["window_start_ms"]).any():
        raise ValueError("Each AOI window must have window_end_ms greater than window_start_ms")

    df[".gp3_time"] = pd.to_numeric(df[time_col], errors="coerce")
    state = df[aoi_col].astype("string").str.strip()
    df[".gp3_aoi"] = state.mask(state.isna() | state.eq(""), missing_aoi_label).astype(object)
    subject = df[subject_col].astype("string").str.strip()
    df[subject_col] = subject.mask(subject.isna() | subject.eq(""), "unknown_subject").astype(
        object
    )
    if condition_col is not None:
        condition = df[condition_col].astype("string").str.strip()
        df[condition_col] = condition.mask(
            condition.isna() | condition.eq(""), missing_condition_label
        ).astype(object)
    df = df.loc[np.isfinite(df[".gp3_time"])].copy()
    if df.empty:
        raise ValueError("No rows contain finite time values")
    indices = np.full(len(df), -1, dtype=int)
    times = df[".gp3_time"].to_numpy(float)
    for i, row in window_table.iterrows():
        if include_right_endpoint:
            inside = (times >= row["window_start_ms"]) & (times <= row["window_end_ms"])
        else:
            inside = (times >= row["window_start_ms"]) & (times < row["window_end_ms"])
        indices[(indices < 0) & inside] = i
    df[".gp3_window_index"] = indices
    df = df.loc[df[".gp3_window_index"] >= 0].copy()
    if df.empty:
        raise ValueError("No rows fall inside the supplied AOI windows")
    idx = df[".gp3_window_index"].astype(int).to_numpy()
    df["window_label"] = window_table.iloc[idx]["window_label"].to_numpy()
    df["window_start_ms"] = window_table.iloc[idx]["window_start_ms"].to_numpy()
    df["window_end_ms"] = window_table.iloc[idx]["window_end_ms"].to_numpy()
    target_values = set([] if target_aoi_values is None else map(str, target_aoi_values))
    distractor_values = set(
        [] if distractor_aoi_values is None else map(str, distractor_aoi_values)
    )
    non_values = set(map(str, non_aoi_values))
    df[".gp3_is_target"] = df[".gp3_aoi"].isin(target_values)
    df[".gp3_is_distractor"] = df[".gp3_aoi"].isin(distractor_values)
    df[".gp3_is_non_aoi"] = df[".gp3_aoi"].isin(non_values)
    df[".gp3_is_missing_aoi"] = df[".gp3_aoi"].eq(missing_aoi_label) | df[".gp3_aoi"].isin(
        ["missing", "missing_aoi"]
    )
    df[".gp3_is_other_aoi"] = ~(
        df[".gp3_is_target"]
        | df[".gp3_is_distractor"]
        | df[".gp3_is_non_aoi"]
        | df[".gp3_is_missing_aoi"]
    )
    rows = []
    group_all = [*groups, "window_label", "window_start_ms", "window_end_ms"]
    for key, block in df.groupby(group_all, dropna=False, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        base = dict(zip(group_all, key, strict=True))
        n_window = len(block)
        n_target = int(block[".gp3_is_target"].sum())
        n_distractor = int(block[".gp3_is_distractor"].sum())
        n_non = int(block[".gp3_is_non_aoi"].sum())
        n_missing = int(block[".gp3_is_missing_aoi"].sum())
        n_other = int(block[".gp3_is_other_aoi"].sum())
        n_aoi = n_target + n_distractor + n_other
        valid = n_window - n_missing
        rows.append(
            {
                **base,
                "n_window_samples": n_window,
                "n_target_samples": n_target,
                "n_distractor_samples": n_distractor,
                "n_non_aoi_samples": n_non,
                "n_missing_aoi_samples": n_missing,
                "n_other_aoi_samples": n_other,
                "n_unique_aoi_states": int(block[".gp3_aoi"].nunique(dropna=False)),
                "first_aoi_state": block[".gp3_aoi"].iloc[0],
                "last_aoi_state": block[".gp3_aoi"].iloc[-1],
                "n_aoi_samples": n_aoi,
                "n_valid_denominator_samples": valid,
                "target_sample_prop_all": n_target / n_window if n_window > 0 else np.nan,
                "target_sample_prop_valid": n_target / valid if valid > 0 else np.nan,
                "target_sample_prop_aoi": n_target / n_aoi if n_aoi > 0 else np.nan,
                "distractor_sample_prop_all": n_distractor / n_window if n_window > 0 else np.nan,
                "valid_denominator_prop": valid / n_window if n_window > 0 else np.nan,
                "target_aoi_defined": bool(target_values),
                "distractor_aoi_defined": bool(distractor_values),
                "aoi_window_status": (
                    "no_target_aoi_defined"
                    if not target_values
                    else "zero_valid_denominator"
                    if valid == 0
                    else "target_not_observed"
                    if n_target == 0
                    else "target_only"
                    if n_target == valid
                    else "ok"
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary = summary.sort_values([*groups, "window_start_ms"], kind="stable").reset_index(
        drop=True
    )
    summary.attrs["gp3_class"] = "gp3_aoi_window_summary"
    summary.attrs["settings"] = {
        "time_col": time_col,
        "aoi_col": aoi_col,
        "subject_col": subject_col,
        "condition_col": condition_col,
        "group_cols": groups,
        "target_aoi_values": target_aoi_values,
        "distractor_aoi_values": distractor_aoi_values,
        "non_aoi_values": list(non_aoi_values),
        "include_right_endpoint": bool(include_right_endpoint),
        "missing_condition_label": missing_condition_label,
        "missing_aoi_label": missing_aoi_label,
    }
    return summary


def transform_gazepoint_aoi_empirical_logit(
    data,
    numerator_col=None,
    denominator_col=None,
    proportion_col=None,
    correction=0.5,
    pseudo_denominator=1,
    output_col=None,
    adjusted_proportion_col="aoi_proportion_adjusted",
    raw_proportion_col="aoi_proportion_raw",
    numerator_output_col="aoi_numerator",
    denominator_output_col="aoi_denominator",
    status_col="aoi_empirical_logit_status",
    overwrite=False,
    name="gazepoint_aoi_empirical_logit",
    *,
    success_col=None,
    total_col=None,
    adjustment=None,
) -> pd.DataFrame:
    """Apply empirical-logit transformation with legacy success/total compatibility."""
    frame = ensure_dataframe(data)
    legacy = (
        numerator_col is None
        and denominator_col is None
        and proportion_col is None
        and ("success" in frame.columns or success_col is not None)
        and ("total" in frame.columns or total_col is not None)
    )
    if legacy:
        success_col = "success" if success_col is None else success_col
        total_col = "total" if total_col is None else total_col
        adjustment = correction if adjustment is None else adjustment
        target = "empirical_logit" if output_col is None else output_col
        success = finite_numeric(frame[success_col])
        total = finite_numeric(frame[total_col])
        frame[target] = np.log((success + adjustment) / (total - success + adjustment))
        return frame

    if frame.empty:
        raise ValueError("data must contain at least one row")
    if proportion_col is None and (numerator_col is None or denominator_col is None):
        raise ValueError("Supply either proportion_col or both numerator_col and denominator_col")
    for column, argument in (
        (numerator_col, "numerator_col"),
        (denominator_col, "denominator_col"),
        (proportion_col, "proportion_col"),
    ):
        if column is not None and (
            not isinstance(column, str) or not column or column not in frame.columns
        ):
            raise ValueError(f"{argument} must identify a column in data")
    for value, argument in ((correction, "correction"), (pseudo_denominator, "pseudo_denominator")):
        if (
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, float, np.integer, np.floating))
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{argument} must be a positive finite numeric scalar")
    if not isinstance(overwrite, (bool, np.bool_)):
        raise ValueError("overwrite must be TRUE or FALSE")
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")

    output_col = "aoi_empirical_logit" if output_col is None else output_col
    output_columns = [
        output_col,
        adjusted_proportion_col,
        raw_proportion_col,
        numerator_output_col,
        denominator_output_col,
        status_col,
    ]
    if any(not isinstance(column, str) or not column for column in output_columns):
        raise ValueError("output column names must be non-empty strings")
    if len(set(output_columns)) != len(output_columns):
        raise ValueError("Output column names must be unique")
    existing = [column for column in output_columns if column in frame.columns]
    if existing and not overwrite:
        raise ValueError(
            "Output column(s) already exist in data: "
            + ", ".join(existing)
            + ". Use overwrite=True to replace them"
        )

    if numerator_col is not None and denominator_col is not None:
        numerator = pd.to_numeric(frame[numerator_col], errors="coerce").to_numpy(float)
        denominator = pd.to_numeric(frame[denominator_col], errors="coerce").to_numpy(float)
        raw = numerator / denominator
        denominator_source = "observed_denominator"
    elif proportion_col is not None and denominator_col is not None:
        raw = pd.to_numeric(frame[proportion_col], errors="coerce").to_numpy(float)
        denominator = pd.to_numeric(frame[denominator_col], errors="coerce").to_numpy(float)
        numerator = raw * denominator
        denominator_source = "observed_denominator_from_proportion"
    else:
        raw = pd.to_numeric(frame[proportion_col], errors="coerce").to_numpy(float)
        denominator = np.full(len(raw), float(pseudo_denominator))
        numerator = raw * denominator
        denominator_source = "pseudo_denominator_from_proportion"

    non_aoi = denominator - numerator
    status = np.full(len(frame), "complete", dtype=object)
    status[~np.isfinite(raw)] = "missing_or_nonfinite_proportion"
    status[~np.isfinite(numerator)] = "missing_or_nonfinite_numerator"
    status[~np.isfinite(denominator)] = "missing_or_nonfinite_denominator"
    status[np.isfinite(denominator) & (denominator <= 0)] = "invalid_denominator"
    status[np.isfinite(numerator) & (numerator < 0)] = "invalid_numerator"
    status[np.isfinite(non_aoi) & (non_aoi < 0)] = "numerator_exceeds_denominator"
    status[np.isfinite(raw) & ((raw < 0) | (raw > 1))] = "proportion_out_of_bounds"

    valid = status == "complete"
    adjusted = np.full(len(frame), np.nan)
    empirical = np.full(len(frame), np.nan)
    adjusted[valid] = (numerator[valid] + correction) / (denominator[valid] + 2 * correction)
    empirical[valid] = np.log((numerator[valid] + correction) / (non_aoi[valid] + correction))

    out = frame.copy()
    out[raw_proportion_col] = raw
    out[numerator_output_col] = numerator
    out[denominator_output_col] = denominator
    out[adjusted_proportion_col] = adjusted
    out[output_col] = empirical
    out[status_col] = status

    def safe_min(values):
        finite = np.asarray(values, float)
        finite = finite[np.isfinite(finite)]
        return float(finite.min()) if len(finite) else np.nan

    def safe_max(values):
        finite = np.asarray(values, float)
        finite = finite[np.isfinite(finite)]
        return float(finite.max()) if len(finite) else np.nan

    overview = pd.DataFrame(
        [
            {
                "object_name": name,
                "transformation": "aoi_empirical_logit",
                "numerator_col": numerator_col,
                "denominator_col": denominator_col,
                "proportion_col": proportion_col,
                "denominator_source": denominator_source,
                "correction": correction,
                "pseudo_denominator": pseudo_denominator,
                "n_input_rows": len(frame),
                "n_complete": int(np.sum(status == "complete")),
                "n_problem_rows": int(np.sum(status != "complete")),
                "min_raw_proportion": safe_min(raw),
                "max_raw_proportion": safe_max(raw),
                "min_empirical_logit": safe_min(empirical),
                "max_empirical_logit": safe_max(empirical),
            }
        ]
    )
    status_summary = (
        (pd.Series(status, name="status").value_counts(sort=False).rename("n").reset_index())
        .sort_values("status", kind="stable")
        .reset_index(drop=True)
    )
    settings = pd.DataFrame(
        {
            "setting": [
                "numerator_col",
                "denominator_col",
                "proportion_col",
                "correction",
                "pseudo_denominator",
                "output_col",
                "adjusted_proportion_col",
                "raw_proportion_col",
                "numerator_output_col",
                "denominator_output_col",
                "status_col",
                "overwrite",
                "name",
            ],
            "value": [
                numerator_col,
                denominator_col,
                proportion_col,
                str(correction),
                str(pseudo_denominator),
                output_col,
                adjusted_proportion_col,
                raw_proportion_col,
                numerator_output_col,
                denominator_output_col,
                status_col,
                "TRUE" if overwrite else "FALSE",
                name,
            ],
        }
    )
    out.attrs["gp3_empirical_logit_overview"] = overview
    out.attrs["gp3_empirical_logit_status_summary"] = status_summary
    out.attrs["gp3_empirical_logit_settings"] = settings
    return out


def audit_gazepoint_aoi_window_denominators(
    data,
    success_col="success",
    total_col="total",
    *,
    window_col=None,
    window_start_col=None,
    window_end_col=None,
    denominator_col=None,
    target_col=None,
    condition_col=None,
    group_cols=None,
    min_denominator_samples=5,
    min_valid_denominator_prop=0.70,
    max_denominator_cv=0.25,
    max_condition_ratio=2,
):
    """Audit AOI window denominator validity and condition imbalance."""
    r_mode = any(
        (
            window_col is not None,
            window_start_col is not None,
            window_end_col is not None,
            denominator_col is not None,
            target_col is not None,
            condition_col is not None,
            group_cols is not None,
            min_denominator_samples != 5,
            min_valid_denominator_prop != 0.70,
            max_denominator_cv != 0.25,
            max_condition_ratio != 2,
        )
    )
    if not r_mode:
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

    df = ensure_dataframe(data, copy=False)
    window_col = "window_label" if window_col is None else window_col
    window_start_col = "window_start_ms" if window_start_col is None else window_start_col
    window_end_col = "window_end_ms" if window_end_col is None else window_end_col
    denominator_col = "n_valid_denominator_samples" if denominator_col is None else denominator_col
    target_col = "n_target_samples" if target_col is None else target_col
    condition_col = "condition" if condition_col is None else condition_col
    total_col = "n_window_samples" if total_col == "total" else total_col
    if min_denominator_samples <= 0 or not np.isfinite(min_denominator_samples):
        raise ValueError("min_denominator_samples must be positive")
    if not 0 <= min_valid_denominator_prop <= 1 or not np.isfinite(min_valid_denominator_prop):
        raise ValueError("min_valid_denominator_prop must be in [0, 1]")
    if (
        max_denominator_cv <= 0
        or max_condition_ratio <= 0
        or not np.isfinite(max_denominator_cv)
        or not np.isfinite(max_condition_ratio)
    ):
        raise ValueError("denominator thresholds must be positive finite values")
    groups = (
        None
        if group_cols is None
        else ([group_cols] if isinstance(group_cols, str) else list(group_cols))
    )
    required = [window_col, denominator_col, total_col, target_col]
    if groups:
        required += groups
    missing = [c for c in dict.fromkeys(required) if c not in df]
    if missing:
        raise KeyError("Missing required columns: " + ", ".join(missing))
    if groups is None:
        groups = [
            c for c in ["subject", condition_col, "MEDIA_ID", "trial_global", "trial"] if c in df
        ]
    work = df.copy()
    work[".gp3_window"] = work[window_col].astype("string").str.strip().fillna("unknown_window")
    work.loc[work[".gp3_window"].eq(""), ".gp3_window"] = "unknown_window"
    work[".gp3_denominator"] = pd.to_numeric(work[denominator_col], errors="coerce")
    work[".gp3_total"] = pd.to_numeric(work[total_col], errors="coerce")
    work[".gp3_target"] = pd.to_numeric(work[target_col], errors="coerce")
    if condition_col in work:
        work[".gp3_condition"] = work[condition_col].astype("string").str.strip().fillna("all_data")
        work.loc[work[".gp3_condition"].eq(""), ".gp3_condition"] = "all_data"
    else:
        work[".gp3_condition"] = "all_data"
    work[".gp3_window_start"] = (
        pd.to_numeric(work[window_start_col], errors="coerce")
        if window_start_col in work
        else np.nan
    )
    work[".gp3_window_end"] = (
        pd.to_numeric(work[window_end_col], errors="coerce") if window_end_col in work else np.nan
    )
    den = work[".gp3_denominator"].to_numpy(float)
    total = work[".gp3_total"].to_numpy(float)
    target = work[".gp3_target"].to_numpy(float)
    valid_prop = np.divide(
        den, total, out=np.full(len(work), np.nan), where=np.isfinite(total) & (total > 0)
    )
    work[".gp3_valid_denominator_prop"] = valid_prop
    work[".gp3_failure"] = den - target
    flags = {
        "denominator_missing": ~np.isfinite(den),
        "total_missing": ~np.isfinite(total),
        "target_missing": ~np.isfinite(target),
        "denominator_negative": np.isfinite(den) & (den < 0),
        "total_non_positive": np.isfinite(total) & (total <= 0),
        "target_negative": np.isfinite(target) & (target < 0),
        "target_exceeds_denominator": np.isfinite(target) & np.isfinite(den) & (target > den),
        "denominator_zero": np.isfinite(den) & (den == 0),
        "denominator_low": np.isfinite(den) & (den > 0) & (den < min_denominator_samples),
        "valid_denominator_prop_low": np.isfinite(valid_prop)
        & (valid_prop < min_valid_denominator_prop),
        "target_zero": np.isfinite(target) & (target == 0),
        "target_all": np.isfinite(target) & np.isfinite(den) & (den > 0) & (target == den),
    }
    for name, values in flags.items():
        work[name] = values
    status = np.full(len(work), "ok", dtype=object)
    precedence = [
        ("denominator_missing", "missing_denominator"),
        ("total_missing", "missing_total"),
        ("target_missing", "missing_target"),
        ("denominator_negative", "negative_denominator"),
        ("total_non_positive", "non_positive_total"),
        ("target_negative", "negative_target"),
        ("target_exceeds_denominator", "target_exceeds_denominator"),
        ("denominator_zero", "zero_denominator"),
        ("denominator_low", "low_denominator"),
        ("valid_denominator_prop_low", "low_valid_denominator_prop"),
    ]
    unresolved = np.ones(len(work), dtype=bool)
    for flag, label in precedence:
        hit = work[flag].to_numpy(bool) & unresolved
        status[hit] = label
        unresolved[hit] = False
    work["denominator_audit_status"] = status

    def safe(values, fn):
        arr = pd.to_numeric(values, errors="coerce").to_numpy(float)
        arr = arr[np.isfinite(arr)]
        return float(fn(arr)) if len(arr) else np.nan

    overview = pd.DataFrame(
        [
            {
                "n_rows": len(work),
                "n_windows": work[".gp3_window"].nunique(dropna=False),
                "n_conditions": work[".gp3_condition"].nunique(dropna=False),
                "n_missing_denominator": int(work["denominator_missing"].sum()),
                "n_zero_denominator": int(work["denominator_zero"].sum()),
                "n_low_denominator": int(work["denominator_low"].sum()),
                "n_low_valid_denominator_prop": int(work["valid_denominator_prop_low"].sum()),
                "n_target_exceeds_denominator": int(work["target_exceeds_denominator"].sum()),
                "n_target_zero": int(work["target_zero"].sum()),
                "n_target_all": int(work["target_all"].sum()),
                "denominator_min": safe(work[".gp3_denominator"], np.min),
                "denominator_median": safe(work[".gp3_denominator"], np.median),
                "denominator_max": safe(work[".gp3_denominator"], np.max),
                "valid_denominator_prop_min": safe(work[".gp3_valid_denominator_prop"], np.min),
                "valid_denominator_prop_median": safe(
                    work[".gp3_valid_denominator_prop"], np.median
                ),
                "valid_denominator_prop_max": safe(work[".gp3_valid_denominator_prop"], np.max),
            }
        ]
    )
    ov = overview.iloc[0]
    if ov["n_target_exceeds_denominator"] > 0:
        audit_status = "invalid_counts"
    elif ov["n_missing_denominator"] > 0:
        audit_status = "missing_denominators"
    elif ov["n_zero_denominator"] > 0:
        audit_status = "zero_denominators"
    elif ov["n_low_denominator"] > 0 or ov["n_low_valid_denominator_prop"] > 0:
        audit_status = "review_denominators"
    else:
        audit_status = "ok"
    overview["denominator_audit_status"] = audit_status
    wrows = []
    for keys, block in work.groupby(
        [".gp3_window", ".gp3_window_start", ".gp3_window_end"], dropna=False, sort=True
    ):
        mean = safe(block[".gp3_denominator"], np.mean)
        sd = (
            safe(block[".gp3_denominator"], lambda x: np.std(x, ddof=1))
            if block[".gp3_denominator"].notna().sum() > 1
            else np.nan
        )
        cv = sd / mean if np.isfinite(mean) and mean > 0 and np.isfinite(sd) else np.nan
        n_zero = int(block["denominator_zero"].sum())
        n_low = int(block["denominator_low"].sum())
        n_prop = int(block["valid_denominator_prop_low"].sum())
        if n_zero:
            wstatus = "zero_denominator"
        elif n_low:
            wstatus = "low_denominator"
        elif n_prop:
            wstatus = "low_valid_denominator_prop"
        elif np.isfinite(cv) and cv > max_denominator_cv:
            wstatus = "high_denominator_variability"
        else:
            wstatus = "ok"
        wrows.append(
            {
                "window_label": keys[0],
                "window_start_ms": keys[1],
                "window_end_ms": keys[2],
                "n_rows": len(block),
                "denominator_min": safe(block[".gp3_denominator"], np.min),
                "denominator_mean": mean,
                "denominator_median": safe(block[".gp3_denominator"], np.median),
                "denominator_max": safe(block[".gp3_denominator"], np.max),
                "denominator_sd": sd,
                "n_zero_denominator": n_zero,
                "n_low_denominator": n_low,
                "n_low_valid_denominator_prop": n_prop,
                "n_target_zero": int(block["target_zero"].sum()),
                "n_target_all": int(block["target_all"].sum()),
                "valid_denominator_prop_min": safe(block[".gp3_valid_denominator_prop"], np.min),
                "valid_denominator_prop_mean": safe(block[".gp3_valid_denominator_prop"], np.mean),
                "denominator_cv": cv,
                "window_denominator_status": wstatus,
            }
        )
    window_summary = pd.DataFrame(wrows)
    crows = []
    for keys, block in work.groupby(
        [".gp3_condition", ".gp3_window", ".gp3_window_start", ".gp3_window_end"],
        dropna=False,
        sort=True,
    ):
        crows.append(
            {
                "condition": keys[0],
                "window_label": keys[1],
                "window_start_ms": keys[2],
                "window_end_ms": keys[3],
                "n_rows": len(block),
                "denominator_mean": safe(block[".gp3_denominator"], np.mean),
                "denominator_median": safe(block[".gp3_denominator"], np.median),
                "denominator_min": safe(block[".gp3_denominator"], np.min),
                "denominator_max": safe(block[".gp3_denominator"], np.max),
                "valid_denominator_prop_mean": safe(block[".gp3_valid_denominator_prop"], np.mean),
                "n_zero_denominator": int(block["denominator_zero"].sum()),
                "n_low_denominator": int(block["denominator_low"].sum()),
                "n_target_zero": int(block["target_zero"].sum()),
                "n_target_all": int(block["target_all"].sum()),
            }
        )
    condition_summary = pd.DataFrame(crows)
    irows = []
    if len(condition_summary):
        for keys, block in condition_summary.groupby(
            ["window_label", "window_start_ms", "window_end_ms"], dropna=False, sort=True
        ):
            vals = pd.to_numeric(block["denominator_mean"], errors="coerce").to_numpy(float)
            vals = vals[np.isfinite(vals)]
            ncond = block["condition"].nunique(dropna=False)
            dmin = float(np.min(vals)) if len(vals) else np.nan
            dmax = float(np.max(vals)) if len(vals) else np.nan
            dgrand = float(np.mean(vals)) if len(vals) else np.nan
            dsd = float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan
            ratio = dmax / dmin if np.isfinite(dmin) and dmin > 0 else np.nan
            cv = dsd / dgrand if np.isfinite(dgrand) and dgrand > 0 and np.isfinite(dsd) else np.nan
            if ncond < 2:
                istatus = "single_condition"
            elif np.isfinite(ratio) and ratio > max_condition_ratio:
                istatus = "condition_denominator_ratio_high"
            elif np.isfinite(cv) and cv > max_denominator_cv:
                istatus = "condition_denominator_cv_high"
            else:
                istatus = "ok"
            irows.append(
                {
                    "window_label": keys[0],
                    "window_start_ms": keys[1],
                    "window_end_ms": keys[2],
                    "n_conditions": ncond,
                    "denominator_mean_min": dmin,
                    "denominator_mean_max": dmax,
                    "denominator_mean_sd": dsd,
                    "denominator_mean_grand": dgrand,
                    "denominator_condition_ratio": ratio,
                    "denominator_condition_cv": cv,
                    "denominator_imbalance_status": istatus,
                }
            )
    imbalance = pd.DataFrame(irows)
    flagged_rows = work.loc[work["denominator_audit_status"].ne("ok")].reset_index(drop=True)
    return {
        "overview": overview,
        "row_audit": work.reset_index(drop=True),
        "window_summary": window_summary,
        "condition_window_summary": condition_summary,
        "denominator_imbalance": imbalance,
        "flagged_rows": flagged_rows,
        "settings": {
            "window_col": window_col,
            "window_start_col": window_start_col,
            "window_end_col": window_end_col,
            "denominator_col": denominator_col,
            "total_col": total_col,
            "target_col": target_col,
            "condition_col": condition_col,
            "group_cols": groups,
            "min_denominator_samples": min_denominator_samples,
            "min_valid_denominator_prop": min_valid_denominator_prop,
            "max_denominator_cv": max_denominator_cv,
            "max_condition_ratio": max_condition_ratio,
        },
        "_gp3_class": "gp3_aoi_window_denominator_audit",
    }


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


def extract_gazepoint_representative_scanpaths(clustered, n_per_cluster=1) -> pd.DataFrame:
    """Extract legacy representatives or R v2.3.0 ranked cluster representatives."""
    if not isinstance(clustered, dict):
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
                {
                    "cluster": cluster,
                    "representative_sequence": seqs[med] if n else [],
                    "n_members": n,
                }
            )
        return pd.DataFrame(rows)

    if not isinstance(n_per_cluster, (int, np.integer)) or int(n_per_cluster) < 1:
        raise ValueError("n_per_cluster must be one positive integer")
    n_per_cluster = int(n_per_cluster)
    required = {"distance", "assignments"}
    if not required.issubset(clustered):
        raise ValueError("clustered must be an R-compatible scanpath cluster result")

    distance = clustered["distance"]
    distance = (
        distance.copy()
        if isinstance(distance, pd.DataFrame)
        else pd.DataFrame(np.asarray(distance, dtype=float))
    )
    if distance.shape[0] != distance.shape[1]:
        raise ValueError("distance must be square")
    sequence_ids = [str(value) for value in distance.index]
    assignments = ensure_dataframe(clustered["assignments"], copy=True)
    if not {"sequence_id", "cluster"}.issubset(assignments.columns):
        raise ValueError("assignments must contain sequence_id and cluster")
    assignments["sequence_id"] = assignments["sequence_id"].astype(str)
    assignments["cluster"] = pd.to_numeric(assignments["cluster"], errors="raise").astype(int)
    if assignments["sequence_id"].duplicated().any():
        raise ValueError("Cluster assignments contain duplicated sequence identifiers")
    missing = [value for value in sequence_ids if value not in set(assignments["sequence_id"])]
    if missing:
        raise ValueError(
            "Cluster assignments are missing sequence identifier(s): " + ", ".join(missing)
        )
    assignments = assignments.set_index("sequence_id").loc[sequence_ids].reset_index()
    model_medoids = set(str(value) for value in (clustered.get("medoids") or []))

    rows = []
    for cluster_id in sorted(assignments["cluster"].unique()):
        members = assignments.loc[assignments["cluster"].eq(cluster_id), "sequence_id"].tolist()
        within = distance.loc[members, members]
        if len(members) == 1:
            means = pd.Series([0.0], index=members)
        else:
            means = within.sum(axis=1) / (len(members) - 1)
        ordered = sorted(members, key=lambda value: (float(means.loc[value]), value))
        selected = ordered[: min(n_per_cluster, len(ordered))]
        for rank, sequence_id in enumerate(selected, 1):
            rows.append(
                {
                    "cluster": int(cluster_id),
                    "representative_rank": rank,
                    "sequence_id": sequence_id,
                    "mean_within_cluster_distance": float(means.loc[sequence_id]),
                    "cluster_size": len(members),
                    "is_model_medoid": sequence_id in model_medoids,
                }
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
audit_gazepoint_aoi_margin_sensitivity = r_aliases(
    audit_gazepoint_aoi_margin_sensitivity, gaze_data="data"
)
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

# === R4 CANONICAL WRAPPER: add_gazepoint_aoi ===
add_gazepoint_aoi = _r4_wrap(add_gazepoint_aoi, name="add_gazepoint_aoi")

# === R4 CANONICAL WRAPPER: add_gazepoint_dynamic_aoi ===
add_gazepoint_dynamic_aoi = _r4_wrap(add_gazepoint_dynamic_aoi, name="add_gazepoint_dynamic_aoi")

# === R4 CANONICAL WRAPPER: add_gazepoint_polygon_aoi ===
add_gazepoint_polygon_aoi = _r4_wrap(add_gazepoint_polygon_aoi, name="add_gazepoint_polygon_aoi")

# === R4 CANONICAL WRAPPER: audit_gazepoint_aoi_coding_matrix ===
audit_gazepoint_aoi_coding_matrix = _r4_wrap(
    audit_gazepoint_aoi_coding_matrix, name="audit_gazepoint_aoi_coding_matrix"
)

# === R4 CANONICAL WRAPPER: audit_gazepoint_aoi_geometry ===
audit_gazepoint_aoi_geometry = _r4_wrap(
    audit_gazepoint_aoi_geometry, name="audit_gazepoint_aoi_geometry"
)

# === R4 DUAL CONTRACT: add_gazepoint_aoi ===
add_gazepoint_aoi = r4_dual_contract(add_gazepoint_aoi, name="add_gazepoint_aoi")

# === R4 DUAL CONTRACT: audit_gazepoint_aoi_geometry ===
audit_gazepoint_aoi_geometry = r4_dual_contract(
    audit_gazepoint_aoi_geometry, name="audit_gazepoint_aoi_geometry"
)

# === R4 GEOMETRY VALIDATION INSTALL ===

audit_gazepoint_aoi_geometry = _r4_geometry_validation_bridge(audit_gazepoint_aoi_geometry)
