"""
chart_builder.py
================
Transform a DataFrame + chart config into recharts-ready JSON data.

Supported chart types
---------------------
scatter      – {x, y, group?}[]
bar          – {name, value}[]
grouped_bar  – {name, series1, series2, ...}[] + config.series
line         – {date, value}[] or multi-series {date, s1, s2}[]
histogram    – {bin, from, to, count}[]
box          – {name, min, q1, median, q3, max, lower_fence, upper_fence, outliers}
pie          – {name, value, pct}[]
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _safe_rows(rows: list[dict]) -> list[dict]:
    return [{k: _safe(v) for k, v in row.items()} for row in rows]


def _parse_dates(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
        except Exception:
            return pd.Series(pd.NaT, index=series.index)


def _agg(grouped, col: str, func: str) -> pd.Series:
    f = (func or "mean").lower()
    dispatch = {
        "mean":   grouped[col].mean,
        "sum":    grouped[col].sum,
        "count":  grouped[col].count,
        "median": grouped[col].median,
        "min":    grouped[col].min,
        "max":    grouped[col].max,
    }
    return dispatch.get(f, grouped[col].mean)()


def _apply_filters(df: pd.DataFrame, filters: list[dict], notes: list[str]) -> pd.DataFrame:
    for f in (filters or []):
        col = f.get("column")
        op  = f.get("operator", "==")
        val = f.get("value")
        if not col or col not in df.columns:
            continue
        try:
            if op == "==":
                df = df[df[col].astype(str) == str(val)]
            elif op == "!=":
                df = df[df[col].astype(str) != str(val)]
            elif op == ">":
                df = df[pd.to_numeric(df[col], errors="coerce") > float(val)]
            elif op == "<":
                df = df[pd.to_numeric(df[col], errors="coerce") < float(val)]
            elif op == ">=":
                df = df[pd.to_numeric(df[col], errors="coerce") >= float(val)]
            elif op == "<=":
                df = df[pd.to_numeric(df[col], errors="coerce") <= float(val)]
            elif op == "contains":
                df = df[df[col].astype(str).str.contains(str(val), case=False, na=False)]
        except Exception as exc:
            notes.append(f"Filter on '{col}' skipped: {exc}")
    return df


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def _scatter(df, x, y, group_by, notes, title):
    cols = [c for c in [x, y, group_by] if c and c in df.columns]
    sub  = df[cols].dropna(subset=[x, y])
    if len(sub) > 1000:
        notes.append("Showing a random sample of 1 000 points for performance.")
        sub = sub.sample(1000, random_state=42)

    data = []
    for _, row in sub.iterrows():
        pt = {"x": _safe(row[x]), "y": _safe(row[y])}
        if group_by and group_by in row:
            pt["group"] = str(row[group_by])
        data.append(pt)

    return {
        "data": data,
        "config": {
            "chart_type": "scatter",
            "x_label": x, "y_label": y,
            "group_by": group_by,
            "title": title or f"{x} vs {y}",
            "series": None,
        },
        "notes": notes,
    }


def _bar(df, x, y, aggregation, group_by, sort_by, sort_asc, top_n, notes, title):
    count_only = y is None or y == "count"

    if count_only:
        if group_by and group_by in df.columns:
            sub   = df[[x, group_by]].dropna()
            pivot = sub.groupby([x, group_by]).size().unstack(fill_value=0)
            if top_n:
                pivot = pivot.head(top_n)
            series_names = [str(c) for c in pivot.columns]
            data = []
            for idx in pivot.index:
                row = {"name": str(idx)}
                for col in pivot.columns:
                    row[str(col)] = int(pivot.loc[idx, col])
                data.append(row)
            return {
                "data": data,
                "config": {
                    "chart_type": "grouped_bar",
                    "x_label": x, "y_label": "count",
                    "group_by": group_by,
                    "title": title or f"Count of {x} by {group_by}",
                    "series": series_names,
                },
                "notes": notes,
            }
        else:
            vc = df[x].value_counts()
            if top_n:
                vc = vc.head(top_n)
            data = [{"name": str(k), "value": int(v)} for k, v in vc.items()]
            return {
                "data": data,
                "config": {
                    "chart_type": "bar",
                    "x_label": x, "y_label": "count",
                    "group_by": None,
                    "title": title or f"Frequency of {x}",
                    "series": None,
                },
                "notes": notes,
            }

    if group_by and group_by in df.columns:
        sub   = df[[x, y, group_by]].dropna()
        pivot = sub.groupby([x, group_by])[y].agg(aggregation or "mean").unstack(fill_value=0)
        if top_n:
            pivot = pivot.head(top_n)
        series_names = [str(c) for c in pivot.columns]
        data = []
        for idx in pivot.index:
            row = {"name": str(idx)}
            for col in pivot.columns:
                row[str(col)] = _safe(pivot.loc[idx, col])
            data.append(row)
        y_label = f"{aggregation or 'mean'}({y})"
        return {
            "data": data,
            "config": {
                "chart_type": "grouped_bar",
                "x_label": x, "y_label": y_label,
                "group_by": group_by,
                "title": title or f"{y_label} by {x} (grouped by {group_by})",
                "series": series_names,
            },
            "notes": notes,
        }

    sub     = df[[x, y]].dropna()
    result  = _agg(sub.groupby(x), y, aggregation).reset_index()
    result.columns = [x, "value"]
    asc     = sort_asc if sort_by == "value" else False
    result  = result.sort_values("value", ascending=asc)
    if top_n:
        result = result.head(top_n)
    data    = [{"name": str(r[x]), "value": _safe(r["value"])} for _, r in result.iterrows()]
    y_label = f"{aggregation or 'mean'}({y})"
    return {
        "data": data,
        "config": {
            "chart_type": "bar",
            "x_label": x, "y_label": y_label,
            "group_by": None,
            "title": title or f"{y_label} by {x}",
            "series": None,
        },
        "notes": notes,
    }


def _line(df, x, y, aggregation, group_by, notes, title):
    date_series = _parse_dates(df[x])
    valid       = date_series.notna() & (df[y].notna() if y else True)
    sub         = df[valid].copy()
    sub["_dt"]  = date_series[valid]
    sub         = sub.sort_values("_dt")

    if len(sub) == 0:
        return {"data": [], "config": {}, "notes": notes + ["No valid date/value pairs."]}

    if group_by and group_by in df.columns:
        grp   = sub.groupby(["_dt", group_by])[y].agg(aggregation or "mean").reset_index()
        pivot = grp.pivot(index="_dt", columns=group_by, values=y).reset_index()
        data  = []
        for _, row in pivot.iterrows():
            pt = {"date": str(row["_dt"])[:10]}
            for col in pivot.columns:
                if col != "_dt":
                    pt[str(col)] = _safe(row[col])
            data.append(pt)
        series = [str(c) for c in pivot.columns if c != "_dt"]
    else:
        if len(sub) > 200:
            grp = sub.groupby("_dt")[y].agg(aggregation or "mean").reset_index()
            grp.columns = ["_dt", y]
        else:
            grp = sub[["_dt", y]].copy()
        data   = [{"date": str(r["_dt"])[:10], "value": _safe(r[y])} for _, r in grp.iterrows()]
        series = None

    return {
        "data": data,
        "config": {
            "chart_type": "line",
            "x_label": x, "y_label": f"{aggregation or 'mean'}({y})",
            "group_by": group_by,
            "title": title or f"{y} over {x}",
            "series": series,
        },
        "notes": notes,
    }


def _histogram(df, x, notes, title):
    clean = pd.to_numeric(df[x], errors="coerce").dropna()
    if len(clean) == 0:
        # Categorical fallback
        vc   = df[x].astype(str).value_counts().head(30)
        data = [{"bin": str(k), "from": None, "to": None, "count": int(v)} for k, v in vc.items()]
    else:
        bins = min(30, max(5, len(clean) // 20))
        counts, edges = np.histogram(clean, bins=bins)
        data = [
            {
                "bin":   f"{edges[i]:.2f}–{edges[i + 1]:.2f}",
                "from":  round(float(edges[i]), 4),
                "to":    round(float(edges[i + 1]), 4),
                "count": int(counts[i]),
            }
            for i in range(len(counts))
        ]
    return {
        "data": data,
        "config": {
            "chart_type": "histogram",
            "x_label": x, "y_label": "count",
            "title": title or f"Distribution of {x}",
            "series": None,
        },
        "notes": notes,
    }


def _box(df, x, y, notes, title):
    col   = (y if y and y != "count" else x)
    clean = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(clean) < 5:
        return {"data": [], "config": {}, "notes": notes + ["Not enough data for box plot."]}

    q1  = float(clean.quantile(0.25))
    q2  = float(clean.quantile(0.50))
    q3  = float(clean.quantile(0.75))
    iqr = q3 - q1
    lf  = q1 - 1.5 * iqr
    uf  = q3 + 1.5 * iqr
    out = clean[(clean < lf) | (clean > uf)].tolist()[:100]

    return {
        "data": [{
            "name": col,
            "min": float(clean.min()), "max": float(clean.max()),
            "q1": q1, "median": q2, "q3": q3,
            "lower_fence": lf, "upper_fence": uf,
            "outliers": [round(v, 4) for v in out],
        }],
        "config": {
            "chart_type": "box",
            "x_label": col, "y_label": "value",
            "title": title or f"Box plot — {col}",
            "series": None,
        },
        "notes": notes,
    }


def _pie(df, x, y, aggregation, top_n, notes, title):
    if y and y != "count" and y in df.columns:
        sub     = df[[x, y]].dropna()
        grouped = sub.groupby(x)[y].agg(aggregation or "sum").sort_values(ascending=False)
    else:
        grouped = df[x].value_counts()

    if top_n:
        grouped = grouped.head(top_n)
    total = grouped.sum()
    data  = [
        {"name": str(k), "value": _safe(v), "pct": round(float(v) / total * 100, 1) if total else 0}
        for k, v in grouped.items()
    ]
    return {
        "data": data,
        "config": {
            "chart_type": "pie",
            "x_label": x, "y_label": y or "count",
            "title": title or f"{x} distribution",
            "series": None,
        },
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_chart_data(
    df: pd.DataFrame,
    chart_type: str,
    x: str,
    y: str | None = None,
    aggregation: str | None = None,
    group_by: str | None = None,
    filters: list[dict] | None = None,
    sort_by: str | None = None,
    sort_asc: bool = True,
    top_n: int | None = None,
    title: str | None = None,
) -> dict:
    """Return {data, config, notes} ready for the frontend, or {error}."""
    notes: list[str] = []

    if x not in df.columns:
        return {"error": f"Column '{x}' not found in dataset."}
    if y and y not in ("count",) and y not in df.columns:
        return {"error": f"Column '{y}' not found in dataset."}
    if group_by and group_by not in df.columns:
        notes.append(f"Group-by column '{group_by}' not found — ignored.")
        group_by = None

    df = _apply_filters(df, filters or [], notes)
    if len(df) == 0:
        return {"data": [], "config": {}, "notes": notes + ["No rows remain after filters."]}

    try:
        ct = chart_type.lower()
        if ct == "scatter":
            return _scatter(df, x, y, group_by, notes, title)
        if ct in ("bar", "grouped_bar"):
            return _bar(df, x, y, aggregation, group_by, sort_by, sort_asc, top_n, notes, title)
        if ct == "line":
            return _line(df, x, y, aggregation, group_by, notes, title)
        if ct == "histogram":
            return _histogram(df, x, notes, title)
        if ct == "box":
            return _box(df, x, y, notes, title)
        if ct == "pie":
            return _pie(df, x, y, aggregation, top_n, notes, title)
        return {"error": f"Unknown chart type '{chart_type}'."}
    except Exception as exc:
        return {"error": f"Chart build error: {exc}"}
