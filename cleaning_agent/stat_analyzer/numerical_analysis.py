"""
numerical_analysis.py
=====================
Full statistical profile for a single numerical column.

Computed metrics
----------------
count, missing, missing_pct
mean, median, mode, std, variance
min, max, range
Q1, Q2 (median), Q3, IQR
lower_fence, upper_fence  (IQR × 1.5)
skewness + human-readable label
kurtosis
outlier_count, outlier_pct
Shapiro-Wilk normality test  (only when 3 ≤ n ≤ 5 000)
warnings  (missing > 30 %, |skewness| > 2, outliers > 10 %)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _safe(v: Any) -> Any:
    """Convert numpy scalars / NaN / Inf to JSON-serialisable Python types."""
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


def _skew_label(sk: float | None) -> str:
    if sk is None:
        return "unknown"
    a = abs(sk)
    direction = "right" if sk > 0 else "left"
    if a < 0.5:
        return "approximately symmetric"
    if a < 1.0:
        return f"moderately skewed {direction}"
    return f"highly skewed {direction}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(series: pd.Series, col_name: str) -> dict:
    """
    Return a full statistical profile for *series* (expected to be numerical).

    The function coerces non-numeric values to NaN so it degrades gracefully
    when a column contains a mix of numbers and stray strings.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    clean   = numeric.dropna()

    total       = len(series)
    missing     = int(series.isna().sum())
    missing_pct = round(missing / total * 100, 2) if total else 0.0

    # ── Edge case: no valid numbers ──────────────────────────────────────────
    if len(clean) == 0:
        return {
            "column":       col_name,
            "type":         "numerical",
            "count":        0,
            "missing":      missing,
            "missing_pct":  missing_pct,
            "warnings":     ["No valid numeric values — all entries are missing or non-numeric."],
            "error":        "No valid numeric values found.",
        }

    # ── Core statistics ──────────────────────────────────────────────────────
    q1 = float(clean.quantile(0.25))
    q2 = float(clean.quantile(0.50))
    q3 = float(clean.quantile(0.75))
    iqr = q3 - q1

    lower_fence = q1 - 1.5 * iqr
    upper_fence = q3 + 1.5 * iqr
    outlier_mask  = (clean < lower_fence) | (clean > upper_fence)
    outlier_count = int(outlier_mask.sum())
    outlier_pct   = round(outlier_count / len(clean) * 100, 2)

    skewness = _safe(round(float(clean.skew()), 4))     if len(clean) >= 4 else None
    kurtosis = _safe(round(float(clean.kurtosis()), 4)) if len(clean) >= 4 else None

    mode_series = clean.mode()
    mode_val    = _safe(float(mode_series.iloc[0])) if len(mode_series) else None

    # ── Shapiro-Wilk normality test ──────────────────────────────────────────
    normality: dict | None = None
    if 3 <= len(clean) <= 5000:
        try:
            sample = clean if len(clean) <= 5000 else clean.sample(5000, random_state=42)
            stat, p = scipy_stats.shapiro(sample)
            normality = {
                "statistic":      _safe(round(float(stat), 4)),
                "p_value":        _safe(round(float(p), 6)),
                "is_normal":      bool(p > 0.05),
                "interpretation": (
                    "Normal distribution (p > 0.05)"
                    if p > 0.05
                    else "Not normally distributed (p ≤ 0.05)"
                ),
            }
        except Exception:
            normality = None

    # ── Data-quality warnings ────────────────────────────────────────────────
    warnings: list[str] = []
    if missing_pct > 30:
        warnings.append(
            f"High missing rate: {missing_pct}% of values are missing."
        )
    if skewness is not None and abs(skewness) > 2:
        warnings.append(
            f"High skewness ({skewness:.2f}): distribution is heavily skewed. "
            "Consider log-transform or robust statistics."
        )
    if outlier_pct > 10:
        warnings.append(
            f"Many outliers: {outlier_count} outlier{'' if outlier_count == 1 else 's'} "
            f"({outlier_pct}% of valid values) detected via IQR method."
        )
    if len(clean) < 30:
        warnings.append(
            f"Small sample size ({len(clean)} values): statistical tests may be unreliable."
        )

    return {
        "column":        col_name,
        "type":          "numerical",
        # Counts
        "count":         int(len(clean)),
        "missing":       missing,
        "missing_pct":   missing_pct,
        # Central tendency
        "mean":          _safe(round(float(clean.mean()),   4)),
        "median":        _safe(round(float(clean.median()), 4)),
        "mode":          mode_val,
        # Spread
        "std":           _safe(round(float(clean.std()),    4)),
        "variance":      _safe(round(float(clean.var()),    4)),
        "min":           _safe(round(float(clean.min()),    4)),
        "max":           _safe(round(float(clean.max()),    4)),
        "range":         _safe(round(float(clean.max() - clean.min()), 4)),
        # Quartiles
        "q1":            _safe(round(q1, 4)),
        "q2":            _safe(round(q2, 4)),
        "q3":            _safe(round(q3, 4)),
        "iqr":           _safe(round(iqr, 4)),
        "lower_fence":   _safe(round(lower_fence, 4)),
        "upper_fence":   _safe(round(upper_fence, 4)),
        # Shape
        "skewness":      skewness,
        "skewness_label": _skew_label(skewness),
        "kurtosis":      kurtosis,
        # Quality
        "outlier_count": outlier_count,
        "outlier_pct":   outlier_pct,
        "normality":     normality,
        "warnings":      warnings,
    }
