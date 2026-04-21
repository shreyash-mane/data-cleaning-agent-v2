"""
column_inference.py
===================
Infer column types from a pandas DataFrame.

Types returned
--------------
numerical   – numeric column suitable for quantitative analysis
categorical – low-cardinality string / boolean column
date        – date or datetime column
id          – high-cardinality integer / identifier column (skip for analysis)
text        – very-high-cardinality free-text column (skip for analysis)
"""

from __future__ import annotations

import re
import warnings

import pandas as pd

_DATE_PATTERNS = [
    re.compile(r"^\d{4}-\d{1,2}-\d{1,2}"),           # 2020-01-15
    re.compile(r"^\d{1,2}/\d{1,2}/\d{4}"),            # 01/15/2020
    re.compile(r"^\d{1,2}-\d{1,2}-\d{4}"),            # 15-01-2020
    re.compile(r"^\d{4}/\d{1,2}/\d{1,2}"),            # 2020/01/15
    re.compile(r"^\w{3}\s+\d{1,2},?\s+\d{4}"),        # Jan 15, 2020
    re.compile(r"^\d{1,2}\s+\w{3}\s+\d{4}"),          # 15 Jan 2020
    re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}"),   # ISO 8601
]

_ID_PAT = re.compile(
    r"\bid\b|_id$|^id_|\buid\b|\bguid\b|\buuid\b"
    r"|^(row|record|entry|seq|serial|index|idx)$"
    r"|_(no|num|nbr|number|code|key)$",
    re.I,
)


def infer_type(series: pd.Series) -> str:
    col_name = str(series.name or "")

    # Already-parsed datetime
    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"

    if pd.api.types.is_bool_dtype(series):
        return "categorical"

    if pd.api.types.is_numeric_dtype(series):
        return "id" if _is_id_column(series, col_name) else "numerical"

    # String column — probe the first 200 non-null values
    sample = series.dropna().astype(str).str.strip().head(200)
    if len(sample) == 0:
        return "categorical"

    # Date pattern match
    date_hits = sum(1 for v in sample if any(p.match(v) for p in _DATE_PATTERNS))
    if date_hits / len(sample) >= 0.60:
        return "date"

    # Pandas datetime parse attempt
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            parsed = pd.to_datetime(sample, infer_datetime_format=True, errors="coerce")
            if parsed.notna().sum() / len(sample) >= 0.75:
                return "date"
        except Exception:
            pass

    # Numeric string
    if pd.to_numeric(sample, errors="coerce").notna().sum() / len(sample) >= 0.80:
        return "numerical"

    # High-cardinality free text
    unique_ratio = series.nunique() / max(len(series.dropna()), 1)
    if unique_ratio > 0.90 and series.nunique() > 100:
        return "text"

    return "categorical"


def _is_id_column(series: pd.Series, col_name: str) -> bool:
    if _ID_PAT.search(col_name):
        return True
    clean = series.dropna()
    if len(clean) < 5:
        return False
    if pd.api.types.is_integer_dtype(clean) and clean.nunique() / len(clean) > 0.95:
        return True
    return False


def get_column_types(df: pd.DataFrame) -> dict[str, str]:
    """Return {column_name: type_string} for every column."""
    return {col: infer_type(df[col]) for col in df.columns}


def get_column_info(df: pd.DataFrame, col_types: dict[str, str]) -> list[dict]:
    """Return list of column metadata dicts for the upload response."""
    result = []
    for col in df.columns:
        s = df[col]
        result.append(
            {
                "name": col,
                "type": col_types[col],
                "missing_pct": round(s.isna().sum() / max(len(s), 1) * 100, 1),
                "unique_count": int(s.nunique()),
            }
        )
    return result
