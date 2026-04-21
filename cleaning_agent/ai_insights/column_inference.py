"""
column_inference.py
===================
Infer column types from a pandas DataFrame.

Types returned
--------------
numerical   – numeric column suitable for quantitative analysis
categorical – low-cardinality string / boolean column
date        – date or datetime column
id          – identifier column (name contains "id" AND unique ratio > 95%)
text        – very-high-cardinality free-text column
"""

from __future__ import annotations

import re
import warnings

import pandas as pd

_DATE_PATTERNS = [
    re.compile(r"^\d{4}-\d{1,2}-\d{1,2}"),
    re.compile(r"^\d{1,2}/\d{1,2}/\d{4}"),
    re.compile(r"^\d{1,2}-\d{1,2}-\d{4}"),
    re.compile(r"^\d{4}/\d{1,2}/\d{1,2}"),
    re.compile(r"^\w{3}\s+\d{1,2},?\s+\d{4}"),
    re.compile(r"^\d{1,2}\s+\w{3}\s+\d{4}"),
    re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}"),
]

# STRICT: only match when the column name clearly signals an identifier.
# Must contain the word "id" as a word boundary, or end with _id / start with id_.
# This intentionally does NOT match: age, salary, score, credit_score, etc.
_ID_NAME_PAT = re.compile(r"\bid\b|_id$|^id_|\buid\b|\bguid\b|\buuid\b", re.I)


def infer_type(series: pd.Series) -> str:
    col_name = str(series.name or "")

    if pd.api.types.is_datetime64_any_dtype(series):
        return "date"

    if pd.api.types.is_bool_dtype(series):
        return "categorical"

    if pd.api.types.is_numeric_dtype(series):
        # Rule: only an identifier if name contains "id" AND >95% unique values.
        # Columns like age, salary, credit_score are ALWAYS numerical.
        if _is_id_column(series, col_name):
            return "id"
        return "numerical"

    # String column — probe first 200 non-null values
    sample = series.dropna().astype(str).str.strip().head(200)
    if len(sample) == 0:
        return "categorical"

    # Date pattern check
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

    # Numeric strings
    if pd.to_numeric(sample, errors="coerce").notna().sum() / len(sample) >= 0.80:
        return "numerical"

    # Very high cardinality free-text
    unique_ratio = series.nunique() / max(len(series.dropna()), 1)
    if unique_ratio > 0.90 and series.nunique() > 100:
        return "text"

    return "categorical"


def _is_id_column(series: pd.Series, col_name: str) -> bool:
    """
    A column is an identifier ONLY when BOTH conditions hold:
      1. The column NAME matches the ID pattern (contains "id", "_id", "id_", etc.)
      2. The unique value ratio exceeds 95%

    This prevents columns like 'age', 'salary', 'credit_score' from being
    misclassified as identifiers just because they have many distinct values.
    """
    if not _ID_NAME_PAT.search(col_name):
        return False   # name doesn't look like an ID → always numerical
    clean = series.dropna()
    if len(clean) < 5:
        return True    # tiny sample + ID name → treat as ID
    return clean.nunique() / len(clean) > 0.95


def get_column_types(df: pd.DataFrame) -> dict[str, str]:
    """Return {column_name: type_string} for every column."""
    return {col: infer_type(df[col]) for col in df.columns}


def get_column_info(df: pd.DataFrame, col_types: dict[str, str]) -> list[dict]:
    result = []
    for col in df.columns:
        s = df[col]
        result.append({
            "name":         col,
            "type":         col_types[col],
            "missing_pct":  round(s.isna().sum() / max(len(s), 1) * 100, 1),
            "unique_count": int(s.nunique()),
        })
    return result
