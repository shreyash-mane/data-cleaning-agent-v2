"""
insight_detector.py
===================
Statistical insight detection engine.

Always produces usable insights:
  A. Numerical vs Numerical   — Pearson correlation (threshold 0.10)
  B. Categorical vs Numerical — group means, always generates insight
  C. Date vs Numerical        — linear trend
  D. Categorical vs Categorical — chi-squared + Cramér's V
  E. Data Quality / Distribution — skewness, outliers, missing, IDs
  F. Fallback                 — distribution histograms so UI always has charts
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

_MAX_MISSING = 0.60       # skip columns with >60 % missing
_MIN_ROWS    = 5          # minimum paired rows for any test
_MAX_CATS    = 50         # max unique values for a categorical column


def _usable_num(series: pd.Series) -> bool:
    if series.isna().sum() / max(len(series), 1) > _MAX_MISSING:
        return False
    clean = pd.to_numeric(series, errors="coerce").dropna()
    return len(clean) >= _MIN_ROWS and clean.nunique() > 1


def _usable_cat(series: pd.Series) -> bool:
    if series.isna().sum() / max(len(series), 1) > _MAX_MISSING:
        return False
    n = len(series.dropna())
    u = series.nunique()
    return n >= _MIN_ROWS and 2 <= u <= _MAX_CATS


def _top_by_variance(df: pd.DataFrame, cols: list[str], n: int = 12) -> list[str]:
    scored = []
    for c in cols:
        s    = pd.to_numeric(df[c], errors="coerce").dropna()
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


def _num_series(df: pd.DataFrame, col: str) -> pd.Series:
    """Return a clean numeric series for a column, coercing strings if needed."""
    return pd.to_numeric(df[col], errors="coerce")


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

            a_vals = _num_series(df, col_a)
            b_vals = _num_series(df, col_b)
            pair   = pd.DataFrame({"a": a_vals, "b": b_vals}).dropna()
            n      = len(pair)
            if n < _MIN_ROWS:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    corr, p_val = stats.pearsonr(pair["a"], pair["b"])
                except Exception:
                    continue

            if math.isnan(corr):
                continue

            abs_c = abs(corr)

            # Always include the pair — classify strength, never skip entirely
            if abs_c >= 0.70:
                strength = "strong"
                conf     = min(0.99, 0.85 + (abs_c - 0.70) * 0.47)
            elif abs_c >= 0.40:
                strength = "moderate"
                conf     = 0.55 + (abs_c - 0.40) * 1.10
            elif abs_c >= 0.20:
                strength = "weak"
                conf     = 0.25 + (abs_c - 0.20) * 1.50
            else:
                strength = "very weak"
                conf     = 0.10 + abs_c * 0.50

            direction = "positive" if corr > 0 else "negative"
            if p_val > 0.05:
                conf *= 0.70
            conf = round(min(conf, 0.99), 3)

            if abs_c >= 0.40:
                summary = (
                    f"{col_a} and {col_b} show a {strength} {direction} relationship "
                    f"(r = {corr:.2f})."
                )
            elif abs_c >= 0.20:
                summary = (
                    f"{col_a} and {col_b} have a weak {direction} relationship "
                    f"(r = {corr:.2f}). The pattern exists but is not strong."
                )
            else:
                summary = (
                    f"No meaningful linear relationship found between {col_a} and {col_b} "
                    f"(r = {corr:.2f})."
                )

            insights.append({
                "type":    "numerical_numerical",
                "columns": [col_a, col_b],
                "title":   f"{col_a} vs {col_b}",
                "summary": summary,
                "strength":   strength,
                "confidence": conf,
                "recommended_chart": {
                    "chart_type": "scatter",
                    "x": col_a, "y": col_b,
                    "aggregation": None, "group_by": None,
                    "title": f"{col_a} vs {col_b}",
                },
                "metadata": {
                    "correlation": round(corr, 3),
                    "p_value":     round(p_val, 4),
                    "sample_size": n,
                    "direction":   direction,
                },
            })

    return insights


# ---------------------------------------------------------------------------
# B. Categorical vs Numerical (always generate insight)
# ---------------------------------------------------------------------------

def _detect_cat_num(
    df: pd.DataFrame, cat_cols: list[str], num_cols: list[str]
) -> list[dict]:
    insights = []

    for cat_col in cat_cols:
        for num_col in num_cols:
            num_vals = _num_series(df, num_col)
            sub      = pd.DataFrame({"cat": df[cat_col], "num": num_vals}).dropna()
            if len(sub) < _MIN_ROWS:
                continue

            grp       = sub.groupby("cat")["num"]
            means     = grp.mean().sort_values(ascending=False)
            overall_m = sub["num"].mean()
            overall_s = sub["num"].std()

            if len(means) < 2:
                continue

            top_cat  = str(means.index[0])
            top_val  = float(means.iloc[0])
            low_cat  = str(means.index[-1])
            low_val  = float(means.iloc[-1])

            # Always generate this insight regardless of spread
            if overall_s > 0 and not math.isnan(overall_s):
                diff_z = (top_val - overall_m) / overall_s
            else:
                diff_z = 0.0

            if abs(diff_z) >= 1.0:
                strength, conf = "strong",   0.82
            elif abs(diff_z) >= 0.50:
                strength, conf = "moderate", 0.62
            elif abs(diff_z) >= 0.20:
                strength, conf = "weak",     0.40
            else:
                strength, conf = "very weak", 0.20

            # ANOVA
            groups_data = [g.values for _, g in grp if len(g) >= 3]
            p_val = None
            if len(groups_data) >= 2:
                try:
                    _, p_val = stats.f_oneway(*groups_data)
                    if not math.isnan(p_val) and p_val > 0.10:
                        conf *= 0.65
                except Exception:
                    pass

            conf = round(min(conf, 0.99), 3)

            diff_pct = abs(top_val - low_val) / max(abs(overall_m), 1e-9) * 100
            summary = (
                f"'{top_cat}' has the highest average {num_col} ({top_val:.1f}) "
                f"while '{low_cat}' has the lowest ({low_val:.1f}). "
                f"Overall mean: {overall_m:.1f}."
            )

            insights.append({
                "type":    "categorical_numerical",
                "columns": [cat_col, num_col],
                "title":   f"{num_col} by {cat_col}",
                "summary": summary,
                "strength":   strength,
                "confidence": conf,
                "recommended_chart": {
                    "chart_type": "bar",
                    "x": cat_col, "y": num_col,
                    "aggregation": "mean", "group_by": None,
                    "title": f"Average {num_col} by {cat_col}",
                },
                "metadata": {
                    "top_category":  top_cat,
                    "top_value":     round(top_val, 2),
                    "low_category":  low_cat,
                    "low_value":     round(low_val, 2),
                    "overall_mean":  round(overall_m, 2),
                    "diff_z":        round(diff_z, 3),
                    "diff_pct":      round(diff_pct, 1),
                    "anova_p_value": round(p_val, 4) if p_val is not None else None,
                    "group_count":   len(means),
                    "sample_size":   len(sub),
                    "group_means":   {str(k): round(float(v), 2) for k, v in means.items()},
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
            num_vals = _num_series(df, num_col)
            sub      = pd.DataFrame({"_date": dates, "val": num_vals})
            sub      = sub[valid_mask & sub["val"].notna()].sort_values("_date")

            if len(sub) < _MIN_ROWS:
                continue

            x      = sub["_date"].map(lambda d: d.toordinal()).values.astype(float)
            y      = sub["val"].values.astype(float)
            x_norm = (x - x.mean()) / max(x.std(), 1e-9)

            try:
                slope, _, r_val, p_val, _ = stats.linregress(x_norm, y)
            except Exception:
                continue

            r_sq  = r_val ** 2
            trend = "increasing" if slope > 0 else "decreasing"

            if r_sq >= 0.50:
                strength, conf = "strong",   0.82
            elif r_sq >= 0.20:
                strength, conf = "moderate", 0.60
            elif r_sq >= 0.05:
                strength, conf = "weak",     0.35
            else:
                strength, conf = "very weak", 0.18

            if p_val > 0.05:
                conf *= 0.60
            conf = round(min(conf, 0.99), 3)

            insights.append({
                "type":    "date_numerical",
                "columns": [date_col, num_col],
                "title":   f"{num_col} trend over time",
                "summary": (
                    f"{num_col} shows a {strength} {trend} trend over {date_col} "
                    f"(R² = {r_sq:.2f}). Time explains {r_sq*100:.0f}% of the variation."
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
            if len(sub) < 10:
                continue

            try:
                ct = pd.crosstab(sub[col_a], sub[col_b])
                if ct.shape[0] < 2 or ct.shape[1] < 2:
                    continue
                chi2, p_val, dof, _ = stats.chi2_contingency(ct)
                n        = int(ct.values.sum())
                min_dim  = min(ct.shape) - 1
                cramer_v = math.sqrt(chi2 / (n * min_dim)) if n > 0 and min_dim > 0 else 0.0
            except Exception:
                continue

            if cramer_v >= 0.50:
                strength, conf = "strong",   0.85
            elif cramer_v >= 0.30:
                strength, conf = "moderate", 0.65
            elif cramer_v >= 0.10:
                strength, conf = "weak",     0.38
            else:
                continue   # essentially no association

            if p_val > 0.05:
                conf *= 0.60
            conf = round(min(conf, 0.99), 3)

            top_idx  = ct.stack().idxmax()
            top_a, top_b = str(top_idx[0]), str(top_idx[1])

            insights.append({
                "type":    "categorical_categorical",
                "columns": [col_a, col_b],
                "title":   f"{col_a} vs {col_b}",
                "summary": (
                    f"{col_a} and {col_b} show a {strength} association "
                    f"(Cramér's V = {cramer_v:.2f}). "
                    f"The most common combination is '{top_a}' with '{top_b}'."
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
        s           = df[col]
        missing_pct = s.isna().sum() / max(n, 1) * 100

        if missing_pct > 30:
            insights.append({
                "type":    "data_quality",
                "columns": [col],
                "title":   f"High missing values in {col}",
                "summary": (
                    f"{col} has {missing_pct:.1f}% missing values — "
                    "analyses involving this column may be unreliable."
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
            clean = _num_series(df, col).dropna()
            if len(clean) < 5:
                continue

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
                        f"{'A few very high values are pulling the distribution up.' if direction == 'right' else 'A few very low values are pulling the distribution down.'}"
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

            q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
            iqr    = q3 - q1
            if iqr > 0:
                n_out   = int(((clean < q1 - 1.5 * iqr) | (clean > q3 + 1.5 * iqr)).sum())
                out_pct = n_out / len(clean) * 100
                if out_pct >= 5:
                    insights.append({
                        "type":    "distribution",
                        "columns": [col],
                        "title":   f"Outliers detected in {col}",
                        "summary": (
                            f"{col} contains {n_out} outliers ({out_pct:.1f}% of values) "
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
                            "q1":  round(float(q1), 2),
                            "q3":  round(float(q3), 2),
                            "iqr": round(float(iqr), 2),
                        },
                    })

        if ctype == "id":
            insights.append({
                "type":    "data_quality",
                "columns": [col],
                "title":   f"{col} looks like an identifier",
                "summary": (
                    f"'{col}' is a unique identifier column and should not be used "
                    "in relationship analysis."
                ),
                "strength":   "info",
                "confidence": 0.85,
                "recommended_chart": None,
                "metadata": {"col_type": "id", "unique_count": int(s.nunique())},
            })

    return insights


# ---------------------------------------------------------------------------
# F. Fallback — ensure at least 3 insights always exist
# ---------------------------------------------------------------------------

def _fallback_distribution_insights(
    df: pd.DataFrame,
    col_types: dict[str, str],
    existing: list[dict],
    target: int = 3,
) -> list[dict]:
    """
    If fewer than `target` insights were detected, add histogram insights for
    the first numerical columns so the UI always has something to show.
    """
    if len(existing) >= target:
        return []

    already = {tuple(i["columns"]) for i in existing}
    extra   = []

    for col, ctype in col_types.items():
        if len(existing) + len(extra) >= target:
            break
        if ctype != "numerical":
            continue
        if (col,) in already:
            continue
        clean = _num_series(df, col).dropna()
        if len(clean) < _MIN_ROWS:
            continue

        mean_v = float(clean.mean())
        std_v  = float(clean.std())
        min_v  = float(clean.min())
        max_v  = float(clean.max())

        extra.append({
            "type":    "distribution",
            "columns": [col],
            "title":   f"Distribution of {col}",
            "summary": (
                f"{col} ranges from {min_v:.1f} to {max_v:.1f} "
                f"with mean {mean_v:.1f} and std {std_v:.1f}."
            ),
            "strength":   "moderate",
            "confidence": 0.80,
            "recommended_chart": {
                "chart_type": "histogram",
                "x": col, "y": "count",
                "aggregation": "count", "group_by": None,
                "title": f"Distribution of {col}",
            },
            "metadata": {
                "mean":  round(mean_v, 2),
                "std":   round(std_v, 2),
                "min":   round(min_v, 2),
                "max":   round(max_v, 2),
                "col_type": "numerical",
            },
        })

    return extra


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def detect_all(df: pd.DataFrame, col_types: dict[str, str]) -> list[dict]:
    """Run all detectors and return a flat list of raw (unranked) insights."""
    num_cols  = [c for c, t in col_types.items() if t == "numerical"   and _usable_num(df[c])]
    cat_cols  = [c for c, t in col_types.items() if t == "categorical" and _usable_cat(df[c])]
    date_cols = [c for c, t in col_types.items() if t == "date"]

    # Cap to avoid O(n²) blowup on very wide tables
    num_cols  = _top_by_variance(df, num_cols, 12)
    cat_cols  = cat_cols[:8]
    date_cols = date_cols[:4]

    insights: list[dict] = []
    insights += _detect_num_num(df, num_cols)
    insights += _detect_cat_num(df, cat_cols, num_cols)
    insights += _detect_date_num(df, date_cols, num_cols)
    insights += _detect_cat_cat(df, cat_cols)
    insights += _detect_quality(df, col_types)
    insights += _fallback_distribution_insights(df, col_types, insights, target=3)

    return insights
