"""
Column Profiler
===============
Computes the 8 features used by the ML predictor for every column:
  col_type, missing_rate, unique_ratio, invalid_numeric_count,
  outlier_count_iqr, invalid_date_count, invalid_email_count, skewness

Column type is detected from (1) name patterns first, then (2) data patterns,
so it works on ANY column name — not just a hard-coded business list.
"""

from __future__ import annotations

import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Date formats tried during validation
# ---------------------------------------------------------------------------
_DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%Y%m%d",
    "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
    "%d-%m-%y", "%d/%m/%y",
    "%m-%d-%Y", "%m/%d/%Y",
    "%B %d %Y", "%b %d %Y", "%B %d, %Y", "%b %d, %Y",
    "%d %B %Y", "%d %b %Y",
]

# ---------------------------------------------------------------------------
# Type detection
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

_NAME_PATTERNS = {
    "email":    re.compile(r"e.?mail", re.I),
    "date":     re.compile(r"date|_dt$|_at$|time|login|created|updated|dob|birth", re.I),
    "age":      re.compile(r"^age$|_age$|^age_", re.I),
    "salary":   re.compile(r"salary|wage|income|pay|compensation", re.I),
    # credit_score must come before "score" — FICO range is 300-850, not 0-100
    "credit_score": re.compile(r"credit.?score|fico|creditworthiness", re.I),
    "score":    re.compile(r"score|rating|grade|mark|rank", re.I),
    "boolean":  re.compile(r"^is_|^has_|^was_|^can_|^allow|verified|active|enabled|subscribed|opted|flag(ged)?$|approved|deleted|visible|published", re.I),
    "category": re.compile(r"department|dept|category|cat|type|status|gender|country|region|city|state", re.I),
    "id":       re.compile(r"^id$|_id$|^id_|uuid|guid", re.I),
}

_BOOL_TRUE  = {"true", "yes", "y", "1", "t", "on"}
_BOOL_FALSE = {"false", "no", "n", "0", "f", "off"}


def detect_column_type(col_name: str, series: pd.Series) -> str:
    """
    Detect column semantic type.
    Priority: name pattern → data-driven inference → fallback.
    """
    name = col_name.strip()

    # 1. Name-pattern matching (order matters — check id last among named types)
    for type_label, pattern in _NAME_PATTERNS.items():
        if pattern.search(name):
            return type_label

    # 2. Data-driven inference on non-null values
    non_null = series.dropna()
    if len(non_null) == 0:
        return "text"

    # Boolean dtype → boolean
    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    # Numeric dtype → numeric (check after bool since bool is a subtype of numeric in pandas)
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"

    # String values that are overwhelmingly boolean-like → boolean
    sample = non_null.astype(str).str.strip().str.lower().head(200)
    bool_hits = sample.isin(_BOOL_TRUE | _BOOL_FALSE).mean()
    if bool_hits >= 0.90:
        return "boolean"

    sample = non_null.astype(str).str.strip().head(200)
    sample_lower = sample.str.lower()

    # Email: >50% match email pattern
    email_hits = sample.str.match(_EMAIL_RE, na=False).mean()
    if email_hits > 0.5:
        return "email"

    # Date: >50% parseable as date
    date_hits = _count_parseable_dates(sample) / max(len(sample), 1)
    if date_hits > 0.5:
        return "date"

    # Numeric-like strings (after stripping symbols)
    numeric_like = (
        sample
        .str.replace(r"[£$€,\s]", "", regex=True)
        .pipe(pd.to_numeric, errors="coerce")
        .notna()
        .mean()
    )
    if numeric_like > 0.7:
        return "numeric"

    # Low cardinality → category
    unique_ratio = series.nunique(dropna=True) / max(len(non_null), 1)
    if unique_ratio < 0.15:
        return "category"

    return "text"


def _count_parseable_dates(sample: pd.Series) -> int:
    count = 0
    for val in sample:
        if _try_parse_date(val) is not None:
            count += 1
    return count


def _try_parse_date(val: str):
    for fmt in _DATE_FORMATS:
        try:
            return pd.to_datetime(val, format=fmt)
        except Exception:
            pass
    try:
        result = pd.to_datetime(val, errors="coerce", dayfirst=True)
        if pd.notna(result):
            return result
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Per-column profiling
# ---------------------------------------------------------------------------

def _try_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"[£$€,\s]", "", regex=True)
        .str.replace(r"(?i)(gbp|usd|eur)", "", regex=True)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _iqr_outlier_count(series_numeric: pd.Series) -> int:
    s = series_numeric.dropna()
    if len(s) < 4:
        return 0
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return int(((s < lower) | (s > upper)).sum())


def _invalid_date_count(series: pd.Series) -> int:
    """Count values that look like they should be dates but can't be parsed."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0
    count = 0
    for val in non_null.astype(str).str.strip():
        if _try_parse_date(val) is None:
            count += 1
    return count


def _invalid_email_count(series: pd.Series) -> int:
    non_null = series.dropna()
    cleaned = non_null.astype(str).str.strip().str.lower()
    return int((~cleaned.str.match(_EMAIL_RE, na=False)).sum())


def profile_column(series: pd.Series, col_name: str) -> dict:
    """
    Compute the full profile for one column.
    Returns a dict with all features needed by the ML predictor.
    """
    total = len(series)
    missing_count = int(series.isna().sum())
    missing_rate = round(missing_count / total, 4) if total else 0.0
    non_null = series.dropna()
    unique_count = int(series.nunique(dropna=True))
    unique_ratio = round(unique_count / total, 4) if total else 0.0

    col_type = detect_column_type(col_name, series)

    profile: dict = {
        "column_name": col_name,
        "col_type": col_type,
        "total_rows": total,
        "missing_count": missing_count,
        "missing_rate": missing_rate,
        "unique_count": unique_count,
        "unique_ratio": unique_ratio,
        # ML features (always present, 0 when not applicable)
        "invalid_numeric_count": 0,
        "outlier_count_iqr": 0,
        "invalid_date_count": 0,
        "invalid_email_count": 0,
        "skewness": 0.0,
    }

    if col_type in ("numeric", "age", "score", "credit_score", "salary"):
        numeric = _try_numeric_series(series)
        valid = numeric.dropna()
        original_non_null = series.dropna()

        profile["invalid_numeric_count"] = int(
            numeric.isna().sum() - missing_count
        )
        profile["outlier_count_iqr"] = _iqr_outlier_count(valid)
        profile["skewness"] = round(float(valid.skew()), 4) if len(valid) > 2 else 0.0
        profile["min"] = float(valid.min()) if len(valid) else None
        profile["max"] = float(valid.max()) if len(valid) else None
        profile["mean"] = float(valid.mean()) if len(valid) else None
        profile["median"] = float(valid.median()) if len(valid) else None
        profile["std"] = float(valid.std()) if len(valid) > 1 else None

    elif col_type == "date":
        profile["invalid_date_count"] = _invalid_date_count(series)
        profile["missing_rate"] = missing_rate  # already set

    elif col_type == "email":
        profile["invalid_email_count"] = _invalid_email_count(non_null) if len(non_null) else 0

    elif col_type == "boolean":
        normalized = non_null.astype(str).str.strip().str.lower()
        true_count  = int(normalized.isin(_BOOL_TRUE).sum())
        false_count = int(normalized.isin(_BOOL_FALSE).sum())
        invalid_count = int((~normalized.isin(_BOOL_TRUE | _BOOL_FALSE)).sum())
        profile["bool_true_count"]    = true_count
        profile["bool_false_count"]   = false_count
        profile["bool_invalid_count"] = invalid_count
        profile["invalid_numeric_count"] = invalid_count  # reuse this feature for ML
        profile["most_common"] = "true" if true_count >= false_count else "false"

    elif col_type in ("text", "category"):
        if len(non_null) > 0:
            vc = non_null.astype(str).str.strip().value_counts()
            profile["top_values"] = vc.head(5).to_dict()
            profile["most_common"] = vc.index[0] if len(vc) else None

    return profile


def profile_dataset(df: pd.DataFrame) -> dict:
    """Profile every column in the dataframe."""
    profiles = []
    for col in df.columns:
        profiles.append(profile_column(df[col], col))
    return {"profiles": profiles, "column_count": len(profiles), "row_count": len(df)}
