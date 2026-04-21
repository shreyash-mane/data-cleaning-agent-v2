"""
Action Executor
===============
Applies the chosen cleaning action to a single column.

Key fixes over v1:
- Works on ANY column name (no BUSINESS_COLUMNS hardcoding)
- All dates output as ISO YYYY-MM-DD (not DD/MM/YYYY)
- No duplicate function definitions
- No reference to undefined safe_ml_action()
- Adds review_required / review_notes only when there is actually a problem
- Returns a structured change_log entry per column
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_WORD_NUMBERS: dict[str, int] = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "twenty one": 21, "twenty two": 22, "twenty three": 23,
    "twenty four": 24, "twenty five": 25, "twenty six": 26, "twenty seven": 27,
    "twenty eight": 28, "twenty nine": 29, "thirty": 30, "thirty one": 31,
    "thirty two": 32, "thirty three": 33, "thirty four": 34, "thirty five": 35,
    "thirty six": 36, "thirty seven": 37, "thirty eight": 38, "thirty nine": 39,
    "forty": 40, "forty one": 41, "forty two": 42, "forty three": 43,
    "forty four": 44, "forty five": 45, "forty six": 46, "forty seven": 47,
    "forty eight": 48, "forty nine": 49, "fifty": 50, "fifty five": 55,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100,
}

_DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%Y%m%d",
    "%d-%m-%Y", "%d/%m/%Y", "%d.%m.%Y",
    "%d-%m-%y", "%d/%m/%y",
    "%m-%d-%Y", "%m/%d/%Y",
    "%B %d %Y", "%b %d %Y", "%B %d, %Y", "%b %d, %Y",
    "%d %B %Y", "%d %b %Y",
]

_EMAIL_RE = re.compile(r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$")

_CURRENCY_RE = re.compile(r"[£$€]|(gbp|usd|eur|inr)", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def _is_integer_like(series: pd.Series) -> bool:
    """True if every non-null numeric value in *series* is a whole number."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) == 0:
        return False
    return bool((numeric % 1 == 0).all())


def _word_to_number(val: Any) -> Any:
    if pd.isna(val):
        return pd.NA
    text = str(val).strip().lower()
    text = re.sub(r"[-_]", " ", text)
    text = re.sub(r"\s+", " ", text)
    if text in _WORD_NUMBERS:
        return _WORD_NUMBERS[text]
    m = re.search(r"(\d+\.?\d*)", text)
    if m:
        return float(m.group(1))
    return val


def _strip_currency(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.replace(_CURRENCY_RE, "", regex=True)
        .str.replace(",", "", regex=False)
        .str.strip()
    )


def _parse_date_value(val: str) -> pd.Timestamp | None:
    """Try every known format then fall back to pandas inference."""
    for fmt in _DATE_FORMATS:
        try:
            return pd.to_datetime(val, format=fmt)
        except Exception:
            pass
    # pandas inference with both dayfirst settings
    for dayfirst in (True, False):
        try:
            result = pd.to_datetime(val, errors="coerce", dayfirst=dayfirst)
            if pd.notna(result):
                return result
        except Exception:
            pass
    return None


def _parse_date_series(series: pd.Series) -> pd.Series:
    result = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    for idx, val in series.items():
        if pd.isna(val):
            continue
        parsed = _parse_date_value(str(val).strip())
        if parsed is not None:
            result.loc[idx] = parsed
    return result


def _clean_email(val: Any) -> Any:
    if pd.isna(val):
        return pd.NA
    email = str(val).strip().lower().replace(" ", "").rstrip(".")
    return email


def _compute_mode(series: pd.Series) -> Any:
    mode_vals = series.dropna().mode()
    return mode_vals.iloc[0] if len(mode_vals) else "Unknown"


# ---------------------------------------------------------------------------
# Action implementations
# ---------------------------------------------------------------------------

def _action_impute_mean(df: pd.DataFrame, col: str, log: dict) -> pd.DataFrame:
    original = df[col].copy()
    numeric = pd.to_numeric(_strip_currency(df[col]), errors="coerce")
    mean_val = numeric.mean()
    null_mask = df[col].isna()
    filled = numeric.fillna(mean_val)
    # Preserve integer type when original data had only whole numbers
    if _is_integer_like(original):
        filled = filled.round(0).astype("Int64")
        log["fill_value"] = int(round(float(mean_val))) if pd.notna(mean_val) else None
    else:
        log["fill_value"] = round(float(mean_val), 4) if pd.notna(mean_val) else None
    df[col] = filled
    log["cells_changed"] = int(null_mask.sum())
    return df


def _action_impute_median(df: pd.DataFrame, col: str, log: dict) -> pd.DataFrame:
    original = df[col].copy()
    numeric = pd.to_numeric(_strip_currency(df[col]), errors="coerce")
    median_val = numeric.median()
    null_mask = df[col].isna()
    filled = numeric.fillna(median_val)
    # Preserve integer type when original data had only whole numbers
    if _is_integer_like(original):
        filled = filled.round(0).astype("Int64")
        log["fill_value"] = int(round(float(median_val))) if pd.notna(median_val) else None
    else:
        log["fill_value"] = round(float(median_val), 4) if pd.notna(median_val) else None
    df[col] = filled
    log["cells_changed"] = int(null_mask.sum())
    return df


def _action_convert_and_flag(df: pd.DataFrame, col: str, log: dict) -> pd.DataFrame:
    """Convert numeric-like strings, flag rows that cannot be converted."""
    if "review_required" not in df.columns:
        df["review_required"] = False
    if "review_notes" not in df.columns:
        df["review_notes"] = ""

    original = df[col].copy()

    # Step 1: word-to-number for age-style columns
    col_name_lower = col.lower()
    if re.search(r"age|count|qty|quantity|num", col_name_lower):
        df[col] = df[col].apply(_word_to_number)

    # Step 2: strip currency symbols and convert
    cleaned = pd.to_numeric(_strip_currency(df[col]), errors="coerce")
    invalid_mask = cleaned.isna() & original.notna()

    df[col] = cleaned
    df.loc[invalid_mask, "review_required"] = True
    df.loc[invalid_mask, "review_notes"] = (
        df.loc[invalid_mask, "review_notes"].astype(str)
        + f"; {col}: could not convert to numeric"
    ).str.lstrip("; ")

    # Domain rules
    if re.search(r"^age$|_age$", col_name_lower):
        neg_mask = df[col] < 0
        df.loc[neg_mask, col] = pd.NA
        df.loc[neg_mask, "review_required"] = True
        df.loc[neg_mask, "review_notes"] += f"; {col}: negative age removed"

        out_mask = df[col] > 120
        df.loc[out_mask, "review_required"] = True
        df.loc[out_mask, "review_notes"] += f"; {col}: age > 120 flagged"

    if re.search(r"score|rating|grade|mark", col_name_lower):
        out_mask = (df[col] < 0) | (df[col] > 100)
        df.loc[out_mask, col] = pd.NA
        df.loc[out_mask, "review_required"] = True
        df.loc[out_mask, "review_notes"] += f"; {col}: score out of 0-100 range"

    # Salary / monetary: negative values are impossible
    if re.search(r"salary|wage|income|pay|compensation|ctc|bonus|price|revenue|cost|amount|earning", col_name_lower):
        neg_mask = cleaned < 0
        if neg_mask.any():
            df.loc[neg_mask, col] = pd.NA
            df.loc[neg_mask, "review_required"] = True
            df.loc[neg_mask, "review_notes"] = (
                df.loc[neg_mask, "review_notes"].astype(str)
                + f"; {col}: negative value removed (impossible for this column)"
            ).str.lstrip("; ")

        # Extreme upper outliers (>3×IQR) — removes sentinel placeholders like 999999
        current_numeric = pd.to_numeric(df[col], errors="coerce")
        valid_after_clean = current_numeric.dropna()
        if len(valid_after_clean) >= 4:
            q1_v, q3_v = valid_after_clean.quantile(0.25), valid_after_clean.quantile(0.75)
            iqr_v = q3_v - q1_v
            if iqr_v > 0:
                extreme_upper = q3_v + 3 * iqr_v
                extreme_mask = current_numeric > extreme_upper
                if extreme_mask.any():
                    df.loc[extreme_mask, col] = pd.NA
                    df.loc[extreme_mask, "review_required"] = True
                    df.loc[extreme_mask, "review_notes"] = (
                        df.loc[extreme_mask, "review_notes"].astype(str)
                        + f"; {col}: extreme outlier removed (> {extreme_upper:.0f})"
                    ).str.lstrip("; ")

    # Step 3: impute any remaining NaN (original missing + converted invalids)
    #         using median (robust to outliers after conversion)
    still_null = df[col].isna()
    if still_null.any():
        valid_vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(valid_vals) > 0:
            skew = float(valid_vals.skew()) if len(valid_vals) >= 4 else 0.0
            fill_val = valid_vals.mean() if abs(skew) < 0.5 else valid_vals.median()
            fill_method = "mean" if abs(skew) < 0.5 else "median"
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(fill_val)
            log["imputed_nulls"] = int(still_null.sum())
            log["impute_method"] = fill_method
            log["impute_fill_value"] = round(float(fill_val), 4)

    log["cells_changed"] = int(invalid_mask.sum())
    log["cells_flagged"] = int(df["review_required"].sum())

    # Preserve integer type if original valid values were all whole numbers
    if _is_integer_like(original[~invalid_mask]):
        df[col] = df[col].round(0).astype("Int64") if df[col].notna().any() else df[col]

    return df


def _action_parse_and_flag(df: pd.DataFrame, col: str, log: dict) -> pd.DataFrame:
    """Parse date strings to ISO YYYY-MM-DD, flag unparseable values."""
    if "review_required" not in df.columns:
        df["review_required"] = False
    if "review_notes" not in df.columns:
        df["review_notes"] = ""

    original = df[col].copy()
    parsed = _parse_date_series(df[col].astype(str).str.strip())
    invalid_mask = parsed.isna() & original.notna()

    # Output as ISO string (YYYY-MM-DD), keep null where unparseable
    df[col] = parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), other=pd.NA)
    df.loc[invalid_mask, "review_required"] = True
    df.loc[invalid_mask, "review_notes"] = (
        df.loc[invalid_mask, "review_notes"].astype(str)
        + f"; {col}: unparseable date"
    ).str.lstrip("; ")

    log["cells_changed"] = int(parsed.notna().sum())
    log["cells_flagged"] = int(invalid_mask.sum())
    return df


def _action_normalize_and_flag(df: pd.DataFrame, col: str, log: dict) -> pd.DataFrame:
    """Normalize email formatting, null-out invalid addresses."""
    if "review_required" not in df.columns:
        df["review_required"] = False
    if "review_notes" not in df.columns:
        df["review_notes"] = ""

    original = df[col].copy()
    normalized = df[col].apply(_clean_email)
    valid_mask = normalized.astype(str).str.match(_EMAIL_RE, na=False)
    invalid_mask = ~valid_mask & original.notna()

    df[col] = normalized.where(valid_mask, other=pd.NA)
    df.loc[invalid_mask, "review_required"] = True
    df.loc[invalid_mask, "review_notes"] = (
        df.loc[invalid_mask, "review_notes"].astype(str)
        + f"; {col}: invalid email"
    ).str.lstrip("; ")

    log["cells_changed"] = int(valid_mask.sum())
    log["cells_flagged"] = int(invalid_mask.sum())
    return df


def _action_flag_outliers(df: pd.DataFrame, col: str, log: dict) -> pd.DataFrame:
    """Flag IQR outliers without removing them."""
    if "review_required" not in df.columns:
        df["review_required"] = False
    if "review_notes" not in df.columns:
        df["review_notes"] = ""

    numeric = pd.to_numeric(_strip_currency(df[col]), errors="coerce")
    valid = numeric.dropna()
    if len(valid) < 4:
        log["cells_flagged"] = 0
        return df

    q1, q3 = valid.quantile(0.25), valid.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        log["cells_flagged"] = 0
        return df

    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    out_mask = (numeric < lower) | (numeric > upper)
    df.loc[out_mask, "review_required"] = True
    df.loc[out_mask, "review_notes"] = (
        df.loc[out_mask, "review_notes"].astype(str)
        + f"; {col}: IQR outlier [bounds {lower:.2f}–{upper:.2f}]"
    ).str.lstrip("; ")

    log["cells_flagged"] = int(out_mask.sum())
    log["iqr_lower"] = round(float(lower), 4)
    log["iqr_upper"] = round(float(upper), 4)
    return df


def _action_fill_mode(df: pd.DataFrame, col: str, log: dict) -> pd.DataFrame:
    mode_val = _compute_mode(df[col])
    null_mask = df[col].isna()
    df[col] = df[col].fillna(mode_val)
    log["cells_changed"] = int(null_mask.sum())
    log["fill_value"] = str(mode_val)
    return df


def _action_fill_unknown(df: pd.DataFrame, col: str, log: dict) -> pd.DataFrame:
    null_mask = df[col].isna()
    df[col] = df[col].fillna("Unknown")
    df[col] = df[col].replace("nan", "Unknown")
    log["cells_changed"] = int(null_mask.sum())
    log["fill_value"] = "Unknown"
    return df


def _action_normalize_boolean(df: pd.DataFrame, col: str, log: dict) -> pd.DataFrame:
    """
    Normalise a boolean column to integer 1 / 0.
    Accepts: True/False, yes/no, y/n, t/f, on/off, 1/0 (case-insensitive).
    Fills NaN with mode (most common value). Flags truly invalid values.
    """
    if "review_required" not in df.columns:
        df["review_required"] = False
    if "review_notes" not in df.columns:
        df["review_notes"] = ""

    _TRUE_VALS  = {"true", "yes", "y", "1", "t", "on"}
    _FALSE_VALS = {"false", "no", "n", "0", "f", "off"}

    original = df[col].copy()

    def _to_int_bool(val):
        if pd.isna(val):
            return pd.NA
        # Collapse float representation: 1.0 → "1", 0.0 → "0"
        try:
            f = float(str(val).strip())
            if f == int(f):
                s = str(int(f))
            else:
                s = str(val).strip().lower()
        except (ValueError, TypeError):
            s = str(val).strip().lower()
        if s in _TRUE_VALS:
            return 1
        if s in _FALSE_VALS:
            return 0
        return pd.NA  # truly invalid

    converted = df[col].apply(_to_int_bool)
    invalid_mask = converted.isna() & original.notna()

    # Fill NaN (both original missing + converted invalids) with mode
    valid_vals = converted.dropna()
    mode_val   = int(valid_vals.mode().iloc[0]) if len(valid_vals) else 0
    filled     = converted.fillna(mode_val).astype("Int64")

    df[col] = filled

    # Flag rows that had truly invalid values
    df.loc[invalid_mask, "review_required"] = True
    df.loc[invalid_mask, "review_notes"] = (
        df.loc[invalid_mask, "review_notes"].astype(str)
        + f"; {col}: unrecognised boolean value"
    ).str.lstrip("; ")

    null_filled = int(original.isna().sum())
    log["cells_changed"] = int(converted.notna().sum())
    log["cells_flagged"] = int(invalid_mask.sum())
    log["fill_value"]    = mode_val
    log["null_filled"]   = null_filled
    return df


def _action_title_case(df: pd.DataFrame, col: str, log: dict) -> pd.DataFrame:
    """Title-case a text column (e.g. Name → 'alice' → 'Alice', City → 'london' → 'London')."""
    original = df[col].copy()
    df[col] = df[col].apply(
        lambda v: str(v).strip().title() if pd.notna(v) and str(v).strip() != "" else v
    )
    changed_mask = original.notna() & (original.astype(str).str.strip() != df[col].astype(str).str.strip())
    log["cells_changed"] = int(changed_mask.sum())
    return df


def _action_no_change(df: pd.DataFrame, col: str, log: dict) -> pd.DataFrame:
    log["cells_changed"] = 0
    return df


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_ACTION_MAP = {
    "impute_mean":          _action_impute_mean,
    "impute_median":        _action_impute_median,
    "convert_and_flag":     _action_convert_and_flag,
    "parse_and_flag":       _action_parse_and_flag,
    "normalize_and_flag":   _action_normalize_and_flag,
    "normalize_boolean":    _action_normalize_boolean,
    "flag_outliers":        _action_flag_outliers,
    "fill_mode":            _action_fill_mode,
    "fill_unknown":         _action_fill_unknown,
    "title_case":           _action_title_case,
    "no_change":            _action_no_change,
    "date_manual_review":   _action_no_change,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_action(
    df: pd.DataFrame,
    col_name: str,
    action: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Apply *action* to *col_name* in *df*.

    Returns
    -------
    (modified_df, change_log_dict)
    """
    log: dict = {
        "column": col_name,
        "action": action,
        "cells_changed": 0,
        "cells_flagged": 0,
    }

    if col_name not in df.columns:
        log["error"] = f"Column '{col_name}' not found"
        return df, log

    fn = _ACTION_MAP.get(action, _action_no_change)
    if action not in _ACTION_MAP:
        log["warning"] = f"Unknown action '{action}', no change applied"

    df = fn(df, col_name, log)
    return df, log
