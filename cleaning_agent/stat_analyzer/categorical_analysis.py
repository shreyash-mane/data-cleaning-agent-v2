"""
categorical_analysis.py
=======================
Statistical profile for a single categorical (or date) column.

Computed metrics
----------------
total, count (non-null), missing, missing_pct
unique_count, cardinality_pct
mode, mode_count, mode_pct
top_10 most frequent values with counts + percentages
warnings  (missing > 30 %, very high cardinality)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helper
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def analyze(series: pd.Series, col_name: str, col_type: str = "categorical") -> dict:
    """
    Return a categorical (or date) statistical profile for *series*.

    *col_type* is passed through to the response so the frontend can
    render the date variant differently if needed.
    """
    total   = len(series)
    missing = int(series.isna().sum())
    missing_pct = round(missing / total * 100, 2) if total else 0.0

    clean = series.dropna().astype(str).str.strip()

    # ── Edge case: all null ──────────────────────────────────────────────────
    if len(clean) == 0:
        return {
            "column":      col_name,
            "type":        col_type,
            "total":       total,
            "count":       0,
            "missing":     missing,
            "missing_pct": missing_pct,
            "warnings":    ["All values are missing — nothing to analyse."],
            "error":       "Column is entirely empty.",
        }

    vc = clean.value_counts()

    # Top 10
    top10:     dict[str, int]   = {str(k): int(v)   for k, v in vc.head(10).items()}
    top10_pct: dict[str, float] = {
        str(k): round(int(v) / len(clean) * 100, 2)
        for k, v in vc.head(10).items()
    }

    mode_val   = str(vc.index[0]) if len(vc) else None
    mode_count = int(vc.iloc[0])  if len(vc) else 0
    mode_pct   = round(mode_count / len(clean) * 100, 2) if len(clean) else 0.0

    unique_count     = int(series.nunique())
    cardinality_pct  = round(unique_count / len(clean) * 100, 2) if len(clean) else 0.0

    # ── Data-quality warnings ────────────────────────────────────────────────
    warnings: list[str] = []
    if missing_pct > 30:
        warnings.append(
            f"High missing rate: {missing_pct}% of values are missing."
        )
    if cardinality_pct > 90 and unique_count > 10:
        warnings.append(
            f"Very high cardinality: {unique_count} unique values "
            f"({cardinality_pct}% of non-null rows). "
            "This column may be an ID/free-text field and unsuitable for grouping."
        )
    if mode_pct > 80 and unique_count > 1:
        warnings.append(
            f"Dominant value: '{mode_val}' accounts for {mode_pct}% of non-null rows. "
            "Low variance may limit analytical usefulness."
        )

    return {
        "column":          col_name,
        "type":            col_type,
        "total":           total,
        "count":           int(len(clean)),
        "missing":         missing,
        "missing_pct":     missing_pct,
        "unique_count":    unique_count,
        "cardinality_pct": cardinality_pct,
        "mode":            mode_val,
        "mode_count":      mode_count,
        "mode_pct":        mode_pct,
        "top_values":      top10,
        "top_values_pct":  top10_pct,
        "warnings":        warnings,
    }
