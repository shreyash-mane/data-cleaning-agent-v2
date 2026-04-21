"""
Statistical Analyzer
====================
Produces per-column statistical profiles plus dataset-level summaries.

Separated into pure functions so they can be tested independently and
composed easily into larger pipelines.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe(v: Any) -> Any:
    """Convert numpy scalars / NaN / Inf to JSON-safe Python types."""
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _is_numerical(series: pd.Series, threshold: float = 0.80) -> bool:
    """Return True if ≥ threshold fraction of non-null values are numeric."""
    if pd.api.types.is_numeric_dtype(series):
        return True
    non_null = series.dropna()
    if len(non_null) == 0:
        return False
    converted = pd.to_numeric(non_null, errors="coerce")
    return float(converted.notna().sum()) / len(non_null) >= threshold


def _skew_label(skewness: float) -> str:
    if skewness is None:
        return "unknown"
    abs_sk = abs(skewness)
    direction = "right" if skewness > 0 else "left"
    if abs_sk < 0.5:
        return "approximately symmetric"
    if abs_sk < 1.0:
        return f"moderately skewed {direction}"
    return f"highly skewed {direction}"


# ---------------------------------------------------------------------------
# Numerical column analysis
# ---------------------------------------------------------------------------

def analyze_numerical(series: pd.Series) -> dict:
    numeric = pd.to_numeric(series, errors="coerce")
    clean = numeric.dropna()
    missing = int(series.isna().sum())
    total = len(series)

    if len(clean) == 0:
        return {
            "type": "numerical",
            "count": 0,
            "missing": missing,
            "missing_pct": _safe(round(missing / total * 100, 2)) if total else 0,
            "note": "No valid numeric values found",
        }

    q1 = float(clean.quantile(0.25))
    q2 = float(clean.quantile(0.50))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1
    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outlier_count = int(((clean < lower_fence) | (clean > upper_fence)).sum())

    skewness = _safe(round(float(clean.skew()), 4)) if len(clean) >= 4 else None
    kurtosis = _safe(round(float(clean.kurtosis()), 4)) if len(clean) >= 4 else None

    # Normality test (Shapiro-Wilk for ≤5000 samples, else skip)
    normality = None
    if 8 <= len(clean) <= 5000:
        try:
            stat, p = scipy_stats.shapiro(clean.sample(min(len(clean), 5000), random_state=42))
            normality = {"statistic": _safe(round(float(stat), 4)), "p_value": _safe(round(float(p), 6)), "is_normal": bool(p > 0.05)}
        except Exception:
            normality = None

    return {
        "type":          "numerical",
        "count":         int(len(clean)),
        "missing":       missing,
        "missing_pct":   _safe(round(missing / total * 100, 2)),
        "mean":          _safe(round(float(clean.mean()), 4)),
        "median":        _safe(round(float(clean.median()), 4)),
        "std":           _safe(round(float(clean.std()), 4)),
        "variance":      _safe(round(float(clean.var()), 4)),
        "min":           _safe(round(float(clean.min()), 4)),
        "max":           _safe(round(float(clean.max()), 4)),
        "range":         _safe(round(float(clean.max() - clean.min()), 4)),
        "q1":            _safe(round(q1, 4)),
        "q2":            _safe(round(q2, 4)),
        "q3":            _safe(round(q3, 4)),
        "iqr":           _safe(round(iqr, 4)),
        "lower_fence":   _safe(round(lower_fence, 4)),
        "upper_fence":   _safe(round(upper_fence, 4)),
        "skewness":      skewness,
        "skewness_label": _skew_label(skewness),
        "kurtosis":      kurtosis,
        "outlier_count": outlier_count,
        "outlier_pct":   _safe(round(outlier_count / len(clean) * 100, 2)),
        "zeros":         int((clean == 0).sum()),
        "negative_count": int((clean < 0).sum()),
        "normality":     normality,
    }


# ---------------------------------------------------------------------------
# Categorical column analysis
# ---------------------------------------------------------------------------

def analyze_categorical(series: pd.Series) -> dict:
    clean = series.dropna().astype(str).str.strip()
    missing = int(series.isna().sum())
    total = len(series)

    if len(clean) == 0:
        return {
            "type": "categorical",
            "count": 0,
            "missing": missing,
            "missing_pct": _safe(round(missing / total * 100, 2)) if total else 0,
            "unique_values": 0,
            "top_values": {},
            "note": "No non-null values",
        }

    vc = clean.value_counts()
    top5 = {str(k): int(v) for k, v in vc.head(5).items()}
    top5_pct = {str(k): _safe(round(int(v) / len(clean) * 100, 2)) for k, v in vc.head(5).items()}
    mode_val = str(vc.index[0]) if len(vc) else None
    mode_freq = _safe(round(float(vc.iloc[0] / len(clean) * 100), 2)) if len(vc) else 0

    # Shannon entropy (diversity measure)
    probs = vc / len(clean)
    entropy = _safe(round(float((-probs * np.log2(probs + 1e-12)).sum()), 4))

    return {
        "type":           "categorical",
        "count":          int(len(clean)),
        "missing":        missing,
        "missing_pct":    _safe(round(missing / total * 100, 2)),
        "unique_values":  int(series.nunique()),
        "top_values":     top5,
        "top_values_pct": top5_pct,
        "mode":           mode_val,
        "mode_frequency": mode_freq,
        "entropy":        entropy,
        "cardinality_ratio": _safe(round(int(series.nunique()) / len(clean) * 100, 2)),
    }


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------

def compute_correlation(df: pd.DataFrame) -> dict:
    num_cols = [c for c in df.columns if _is_numerical(df[c])]
    if len(num_cols) < 2:
        return {}
    num_df = df[num_cols].apply(pd.to_numeric, errors="coerce")
    corr = num_df.corr().round(4)
    result = {}
    for col in corr.columns:
        result[col] = {
            other: (_safe(corr.loc[col, other]) if pd.notna(corr.loc[col, other]) else None)
            for other in corr.columns
        }
    return result


# ---------------------------------------------------------------------------
# Dataset-level summary
# ---------------------------------------------------------------------------

def compute_summary(df: pd.DataFrame, columns_meta: dict) -> dict:
    missing_per_col = {col: int(df[col].isna().sum()) for col in df.columns}
    total_cells = len(df) * len(df.columns)
    total_missing = sum(missing_per_col.values())
    num_count = sum(1 for v in columns_meta.values() if v.get("type") == "numerical")
    cat_count = sum(1 for v in columns_meta.values() if v.get("type") == "categorical")

    return {
        "total_rows":          int(len(df)),
        "total_columns":       int(len(df.columns)),
        "numerical_columns":   num_count,
        "categorical_columns": cat_count,
        "duplicate_rows":      int(df.duplicated().sum()),
        "complete_rows":       int((~df.isnull().any(axis=1)).sum()),
        "total_missing_cells": total_missing,
        "missing_rate_pct":    _safe(round(total_missing / max(total_cells, 1) * 100, 2)),
        "missing_per_column":  missing_per_col,
        "memory_usage_kb":     _safe(round(float(df.memory_usage(deep=True).sum()) / 1024, 2)),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze_dataset(df: pd.DataFrame) -> dict:
    """
    Full statistical analysis of a DataFrame.

    Returns
    -------
    {
        "columns":           { col_name: { type, mean/unique_values, ... } },
        "dataset_summary":   { total_rows, duplicates, missing_rate, ... },
        "correlation_matrix":{ col: { col2: value, ... } }
    }
    """
    if df.empty:
        return {
            "columns": {},
            "dataset_summary": {"total_rows": 0, "total_columns": 0},
            "correlation_matrix": {},
            "error": "Dataset is empty",
        }

    columns_meta = {}
    for col in df.columns:
        try:
            if _is_numerical(df[col]):
                columns_meta[col] = analyze_numerical(df[col])
            else:
                columns_meta[col] = analyze_categorical(df[col])
        except Exception as exc:
            columns_meta[col] = {"type": "error", "error": str(exc)}

    return {
        "columns":            columns_meta,
        "dataset_summary":    compute_summary(df, columns_meta),
        "correlation_matrix": compute_correlation(df),
    }
