"""
insight_detector.py
===================
Statistical insight detection engine.

Detects five categories of relationships:
  A. numerical  vs numerical   — Pearson / Spearman correlation
  B. categorical vs numerical  — group means + ANOVA
  C. date       vs numerical   — linear trend (OLS slope)
  D. categorical vs categorical— chi-squared + Cramér's V
  E. data quality / distribution— skewness, outliers, missing values, IDs
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# Column-level guards
# ---------------------------------------------------------------------------

_MAX_MISSING = 0.50      # skip columns with >50 % missing
_MIN_ROWS    = 10        # minimum paired rows for any test
_MAX_CATS    = 40        # max unique values for a categorical column


def _usable_num(series: pd.Series) -> bool:
    if series.isna().sum() / max(len(series), 1) > _MAX_MISSING:
        return False
    clean = series.dropna()
    return len(clean) >= _MIN_ROWS and clean.nunique() > 1


def _usable_cat(series: pd.Series) -> bool:
    if series.isna().sum() / max(len(series), 1) > _MAX_MISSING:
        return False
    n = len(series.dropna())
    u = series.nunique()
    return n >= _MIN_ROWS and 2 <= u <= _MAX_CATS


def _top_by_variance(df: pd.DataFrame, cols: list[str], n: int = 12) -> list[str]:
    """Return up to n numerical columns ranked by coefficient of variation."""
    scored = []
    for c in cols:
        s = df[c].dropna()
        mean = s.mean()
        std  = s.std()
        cv   = std / abs(mean) if abs(mean) > 1e-9 else std
        scored.append((c, cv))
    scored.sort(key=lambda x: -x[1])
    return [c for c, _ in scored[:n]]


def _parse_date_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return pd.to_datetime(series, infer_datetime_format=True, errors="coerce")
        except Exception:
            return pd.Series(pd.NaT, index=series.index)


# ---------------------------------------------------------------------------
# A. Numerical vs Numerical
# ---------------------------------------------------------------------------

def _detect_num_num(df: pd.DataFrame, num_cols: list[str]) -> list[dict]:
    insights = []
    seen: set[tuple] = set()

    for i, col_a in enumerate(num_cols):
        for col_b in num_cols[i + 1:]:
            key = (min(col_a, col_b), max(col_a, col_b))
            if key in seen:
                continue
            seen.add(key)

            pair = df[[col_a, col_b]].dropna()
            n = len(pair)
            if n < _MIN_ROWS:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    corr, p_val = stats.pearsonr(pair[col_a], pair[col_b])
                except Exception:
                    continue

            if math.isnan(corr):
                continue

            abs_c = abs(corr)
            if abs_c < 0.15:
                continue

            if abs_c >= 0.70:
                strength, conf = "strong",   min(0.99, 0.85 + (abs_c - 0.70) * 0.47)
            elif abs_c >= 0.40:
                strength, conf = "moderate", 0.52 + (abs_c - 0.40) * 1.10
            else:
                strength, conf = "weak",     0.20 + (abs_c - 0.15) * 1.20

            direction = "positive" if corr > 0 else "negative"
            if p_val > 0.05:
                conf *= 0.60
            conf = round(min(conf, 0.99), 3)

            insights.append({
                "type":    "numerical_numerical",
                "columns": [col_a, col_b],
                "title":   f"{col_a} vs {col_b}",
                "summary": (
                    f"{col_a} and {col_b} show a {strength} {direction} "
                    f"relationship (r = {corr:.2f}, p = {p_val:.4f})"
                ),
                "strength":   strength,
                "confidence": conf,
                "recommended_chart": {
                    "chart_type": "scatter",
                    "x": col_a, "y": col_b,
                    "aggregation": None, "group_by": None,
                    "title": f"{col_a} vs {col_b}",
                },
                "metadata": {
                    "correlation":  round(corr, 3),
                    "p_value":      round(p_val, 4),
                    "sample_size":  n,
                    "direction":    direction,
                },
            })
    return insights


# ---------------------------------------------------------------------------
# B. Categorical vs Numerical
# ---------------------------------------------------------------------------

def _detect_cat_num(
    df: pd.DataFrame, cat_cols: list[str], num_cols: list[str]
) -> list[dict]:
    insights = []

    for cat_col in cat_cols:
        for num_col in num_cols:
            sub = df[[cat_col, num_col]].dropna()
            if len(sub) < _MIN_ROWS:
                continue

            grp        = sub.groupby(cat_col)[num_col]
            means      = grp.mean().sort_values(ascending=False)
            overall_m  = sub[num_col].mean()
            overall_s  = sub[num_col].std()

            if overall_s == 0 or math.isnan(overall_s) or len(means) < 2:
                continue

            top_cat = means.index[0]
            top_val = means.iloc[0]
            diff_z  = (top_val - overall_m) / overall_s

            if abs(diff_z) < 0.30:
                continue

            if abs(diff_z) >= 1.0:
                strength, conf = "strong",   0.80
            elif abs(diff_z) >= 0.55:
                strength, conf = "moderate", 0.60
            else:
                strength, conf = "weak",     0.35

            # ANOVA significance check
            groups_data = [g.values for _, g in grp if len(g) >= 3]
            p_val = None
            if len(groups_data) >= 2:
                try:
                    _, p_val = stats.f_oneway(*groups_data)
                    if not math.isnan(p_val) and p_val > 0.10:
                        conf *= 0.55
                except Exception:
                    pass

            direction = "higher" if diff_z > 0 else "lower"
            conf = round(min(conf, 0.99), 3)

            insights.append({
                "type":    "categorical_numerical",
                "columns": [cat_col, num_col],
                "title":   f"{cat_col} vs {num_col}",
                "summary": (
                    f"'{top_cat}' in {cat_col} has {direction} average {num_col} "
                    f"({top_val:.1f} vs overall {overall_m:.1f})"
                ),
                "strength":   strength,
                "confidence": conf,
                "recommended_chart": {
                    "chart_type": "bar",
                    "x": cat_col, "y": num_col,
                    "aggregation": "mean", "group_by": None,
                    "title": f"Average {num_col} by {cat_col}",
                },
                "metadata": {
                    "top_category":    str(top_cat),
                    "top_value":       round(top_val, 2),
                    "overall_mean":    round(overall_m, 2),
                    "diff_z":          round(diff_z, 3),
                    "anova_p_value":   round(p_val, 4) if p_val is not None else None,
                    "group_count":     len(means),
                    "sample_size":     len(sub),
                    "group_means":     {str(k): round(v, 2) for k, v in means.items()},
                },
            })
    return insights


# ---------------------------------------------------------------------------
# C. Date vs Numerical
# ---------------------------------------------------------------------------

def _detect_date_num(
    df: pd.DataFrame, date_cols: list[str], num_cols: list[str]
) -> list[dict]:
    insights = []

    for date_col in date_cols:
        dates      = _parse_date_series(df[date_col])
        valid_mask = dates.notna()
        if valid_mask.sum() < _MIN_ROWS:
            continue

        for num_col in num_cols:
            sub = df[[num_col]].copy()
            sub["_date"] = dates
            sub = sub[valid_mask & sub[num_col].notna()].sort_values("_date")

            if len(sub) < _MIN_ROWS:
                continue

            x = sub["_date"].map(lambda d: d.toordinal()).values.astype(float)
            y = sub[num_col].values.astype(float)
            x_norm = (x - x.mean()) / max(x.std(), 1e-9)

            try:
                slope, _, r_val, p_val, _ = stats.linregress(x_norm, y)
            except Exception:
                continue

            r_sq = r_val ** 2
            if r_sq < 0.06 or abs(slope) < 1e-9:
                continue

            trend = "increasing" if slope > 0 else "decreasing"

            if r_sq >= 0.50:
                strength, conf = "strong",   0.82
            elif r_sq >= 0.20:
                strength, conf = "moderate", 0.60
            else:
                strength, conf = "weak",     0.35

            if p_val > 0.05:
                conf *= 0.55
            conf = round(min(conf, 0.99), 3)

            insights.append({
                "type":    "date_numerical",
                "columns": [date_col, num_col],
                "title":   f"{num_col} trend over {date_col}",
                "summary": (
                    f"{num_col} shows a {strength} {trend} trend over {date_col} "
                    f"(R² = {r_sq:.2f})"
                ),
                "strength":   strength,
                "confidence": conf,
                "recommended_chart": {
                    "chart_type": "line",
                    "x": date_col, "y": num_col,
                    "aggregation": "mean", "group_by": None,
                    "title": f"{num_col} over {date_col}",
                },
                "metadata": {
                    "trend_direction": trend,
                    "r_squared":       round(r_sq, 3),
                    "slope":           round(float(slope), 6),
                    "p_value":         round(p_val, 4),
                    "sample_size":     len(sub),
                },
            })
    return insights


# ---------------------------------------------------------------------------
# D. Categorical vs Categorical
# ---------------------------------------------------------------------------

def _detect_cat_cat(df: pd.DataFrame, cat_cols: list[str]) -> list[dict]:
    insights = []
    seen: set[tuple] = set()

    for i, col_a in enumerate(cat_cols):
        for col_b in cat_cols[i + 1:]:
            key = (min(col_a, col_b), max(col_a, col_b))
            if key in seen:
                continue
            seen.add(key)

            sub = df[[col_a, col_b]].dropna()
            if len(sub) < 20:
                continue

            try:
                ct = pd.crosstab(sub[col_a], sub[col_b])
                if ct.shape[0] < 2 or ct.shape[1] < 2:
                    continue
                chi2, p_val, dof, _ = stats.chi2_contingency(ct)
                n         = int(ct.values.sum())
                min_dim   = min(ct.shape) - 1
                cramer_v  = math.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else 0.0
            except Exception:
                continue

            if cramer_v < 0.10:
                continue

            if cramer_v >= 0.50:
                strength, conf = "strong",   0.85
            elif cramer_v >= 0.30:
                strength, conf = "moderate", 0.65
            else:
                strength, conf = "weak",     0.38

            if p_val > 0.05:
                conf *= 0.55
            conf = round(min(conf, 0.99), 3)

            top_idx = ct.stack().idxmax()
            top_a, top_b = str(top_idx[0]), str(top_idx[1])

            insights.append({
                "type":    "categorical_categorical",
                "columns": [col_a, col_b],
                "title":   f"{col_a} vs {col_b}",
                "summary": (
                    f"{col_a} and {col_b} show a {strength} association "
                    f"(Cramér's V = {cramer_v:.2f}). "
                    f"'{top_a}' most frequently pairs with '{top_b}'."
                ),
                "strength":   strength,
                "confidence": conf,
                "recommended_chart": {
                    "chart_type": "grouped_bar",
                    "x": col_a, "y": "count",
                    "aggregation": "count", "group_by": col_b,
                    "title": f"{col_a} distribution by {col_b}",
                },
                "metadata": {
                    "cramers_v":       round(cramer_v, 3),
                    "chi2":            round(chi2, 3),
                    "p_value":         round(p_val, 4),
                    "dof":             int(dof),
                    "sample_size":     n,
                    "top_combination": {"col_a": top_a, "col_b": top_b},
                },
            })
    return insights


# ---------------------------------------------------------------------------
# E. Data Quality / Distribution
# ---------------------------------------------------------------------------

def _detect_quality(df: pd.DataFrame, col_types: dict[str, str]) -> list[dict]:
    insights = []
    n = len(df)

    for col, ctype in col_types.items():
        s            = df[col]
        missing_pct  = s.isna().sum() / max(n, 1) * 100

        # High missing values
        if missing_pct > 30:
            insights.append({
                "type":    "data_quality",
                "columns": [col],
                "title":   f"High missing values in {col}",
                "summary": (
                    f"{col} has {missing_pct:.1f}% missing values. "
                    "Analyses involving this column may be unreliable."
                ),
                "strength":   "high" if missing_pct > 60 else "moderate",
                "confidence": 0.95,
                "recommended_chart": {
                    "chart_type": "bar",
                    "x": col, "y": "count",
                    "aggregation": "count", "group_by": None,
                    "title": f"Value distribution — {col}",
                },
                "metadata": {"missing_pct": round(missing_pct, 1), "col_type": ctype},
            })

        if ctype == "numerical":
            clean = pd.to_numeric(s, errors="coerce").dropna()
            if len(clean) < 5:
                continue

            # Skewness
            try:
                skew = float(stats.skew(clean))
            except Exception:
                skew = 0.0

            if abs(skew) >= 1.5:
                direction = "right" if skew > 0 else "left"
                insights.append({
                    "type":    "distribution",
                    "columns": [col],
                    "title":   f"{col} is heavily {direction}-skewed",
                    "summary": (
                        f"{col} has skewness = {skew:.2f} ({direction}-skewed). "
                        f"{'A few very high values are pulling the distribution up.' if direction == 'right' else 'A few very low values pull the distribution down.'}"
                    ),
                    "strength":   "high" if abs(skew) >= 2.5 else "moderate",
                    "confidence": 0.90,
                    "recommended_chart": {
                        "chart_type": "histogram",
                        "x": col, "y": "count",
                        "aggregation": "count", "group_by": None,
                        "title": f"Distribution of {col}",
                    },
                    "metadata": {"skewness": round(skew, 3), "col_type": "numerical"},
                })

            # Outliers (IQR method)
            q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
            iqr    = q3 - q1
            if iqr > 0:
                n_out    = int(((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum())
                out_pct  = n_out / len(clean) * 100
                if out_pct >= 5:
                    insights.append({
                        "type":    "distribution",
                        "columns": [col],
                        "title":   f"Outliers detected in {col}",
                        "summary": (
                            f"{col} has {n_out} outliers ({out_pct:.1f}% of values) "
                            "beyond the IQR fences."
                        ),
                        "strength":   "high" if out_pct >= 15 else "moderate",
                        "confidence": 0.85,
                        "recommended_chart": {
                            "chart_type": "box",
                            "x": col, "y": col,
                            "aggregation": None, "group_by": None,
                            "title": f"Box plot — {col}",
                        },
                        "metadata": {
                            "outlier_count": n_out,
                            "outlier_pct":   round(out_pct, 1),
                            "q1": round(float(q1), 2),
                            "q3": round(float(q3), 2),
                            "iqr": round(float(iqr), 2),
                        },
                    })

        if ctype == "id":
            insights.append({
                "type":    "data_quality",
                "columns": [col],
                "title":   f"{col} looks like an identifier",
                "summary": (
                    f"'{col}' appears to be a unique identifier (ID) column "
                    "and should not be used in relationship analysis."
                ),
                "strength":   "info",
                "confidence": 0.85,
                "recommended_chart": None,
                "metadata": {"col_type": "id", "unique_count": int(s.nunique())},
            })

    return insights


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_all(df: pd.DataFrame, col_types: dict[str, str]) -> list[dict]:
    """Run all five detectors and return a flat list of raw (unranked) insights."""
    num_cols  = [c for c, t in col_types.items() if t == "numerical"  and _usable_num(df[c])]
    cat_cols  = [c for c, t in col_types.items() if t == "categorical" and _usable_cat(df[c])]
    date_cols = [c for c, t in col_types.items() if t == "date"]

    # Cap to avoid O(n²) blowup on wide tables
    num_cols  = _top_by_variance(df, num_cols, 12)
    cat_cols  = cat_cols[:8]
    date_cols = date_cols[:4]

    insights: list[dict] = []
    insights += _detect_num_num(df, num_cols)
    insights += _detect_cat_num(df, cat_cols, num_cols)
    insights += _detect_date_num(df, date_cols, num_cols)
    insights += _detect_cat_cat(df, cat_cols)
    insights += _detect_quality(df, col_types)
    return insights
