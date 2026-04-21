"""
column_detector.py
==================
Infers the semantic type of every column in a DataFrame.

Types returned
--------------
"numerical"   – integers or floats (or strings that parse ≥80 % numeric)
"date"        – datetime columns or strings that look like dates ≥60 % of the time
"categorical" – everything else (text, booleans, mixed)
"""

from __future__ import annotations

import re

import pandas as pd

# ---------------------------------------------------------------------------
# Date-pattern heuristic
# ---------------------------------------------------------------------------

_DATE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^\d{4}-\d{2}-\d{2}"),           # ISO: 2024-01-31
    re.compile(r"^\d{2}/\d{2}/\d{4}"),            # DD/MM/YYYY or MM/DD/YYYY
    re.compile(r"^\d{2}-\d{2}-\d{4}"),            # DD-MM-YYYY
    re.compile(r"^\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}"),  # 12 Jan 2024
    re.compile(r"^[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}"), # Jan 12, 2024
    re.compile(r"^\d{4}/\d{2}/\d{2}"),            # YYYY/MM/DD
]


def _looks_like_dates(sample: pd.Series) -> bool:
    """Return True if ≥60 % of *sample* strings match a known date pattern."""
    if len(sample) == 0:
        return False
    hits = sum(
        1 for v in sample
        if any(pat.match(str(v).strip()) for pat in _DATE_PATTERNS)
    )
    return hits / len(sample) >= 0.60


# ---------------------------------------------------------------------------
# Single-column type detection
# ---------------------------------------------------------------------------

def detect_type(series: pd.Series) -> str:
    """Return 'numerical', 'date', or 'categorical' for *series*."""

    # Already a proper datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"

    # Boolean → treat as categorical (True/False groups)
    if pd.api.types.is_bool_dtype(series):
        return "categorical"

    # Native numeric dtype
    if pd.api.types.is_numeric_dtype(series):
        return "numerical"

    # Object / string columns — probe further
    non_null = series.dropna()
    if len(non_null) == 0:
        return "categorical"

    # Try date heuristic on a sample (max 100 values to keep it fast)
    sample = non_null.astype(str).head(100)
    if _looks_like_dates(sample):
        return "date"

    # Try numeric conversion: if ≥80 % succeed, call it numerical
    converted = pd.to_numeric(non_null, errors="coerce")
    if converted.notna().sum() / len(non_null) >= 0.80:
        return "numerical"

    return "categorical"


# ---------------------------------------------------------------------------
# Dataset-level column list
# ---------------------------------------------------------------------------

def get_column_info(df: pd.DataFrame) -> list[dict]:
    """
    Return a list of ``{"name": col, "type": type_str}`` dicts for every
    column in *df*, ordered as they appear in the DataFrame.
    """
    return [{"name": col, "type": detect_type(df[col])} for col in df.columns]
