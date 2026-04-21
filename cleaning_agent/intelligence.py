"""
Intelligence Layer
==================
After the ML model predicts an action, this layer inspects the *actual*
column data to confirm or refine that decision.

This is the smart bridge between prediction and execution:
- impute_mean vs impute_median  => re-checked against real skewness + outliers
- fill_mode vs fill_unknown     => re-checked against real mode dominance
- score/age range violations    => escalated to convert_and_flag
- integer-type columns          => flagged so executor rounds to whole number
"""

from __future__ import annotations

import re
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_integer_like(series: pd.Series) -> bool:
    """True if every non-null numeric value is a whole number."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) == 0:
        return False
    return bool((numeric % 1 == 0).all())


def _actual_skewness(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 4:
        return 0.0
    try:
        return float(numeric.skew())
    except Exception:
        return 0.0


def _actual_outlier_count(series: pd.Series) -> int:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if len(numeric) < 4:
        return 0
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return 0
    return int(((numeric < q1 - 1.5 * iqr) | (numeric > q3 + 1.5 * iqr)).sum())


def _mode_dominance(series: pd.Series) -> float:
    """Fraction of non-null values that equal the most frequent value."""
    non_null = series.dropna()
    if len(non_null) == 0:
        return 0.0
    return float(non_null.value_counts().iloc[0] / len(non_null))


def _range_violations(series: pd.Series, lo: float, hi: float) -> int:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    return int(((numeric < lo) | (numeric > hi)).sum())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def refine_action(
    predicted_action: str,
    col_series: pd.Series,
    profile: dict,
) -> tuple[str, dict]:
    """
    Examine actual column data and confirm or refine the predicted action.

    Parameters
    ----------
    predicted_action : action string returned by the ML predictor
    col_series       : the raw column Series (before any cleaning)
    profile          : column profile dict (col_type, missing_rate, …)

    Returns
    -------
    (refined_action, reasoning_dict)
    """
    col_type = profile.get("col_type", "text")
    col_name = profile.get("column_name", col_series.name or "")
    col_name_lower = col_name.lower()

    reasoning: dict = {
        "original_action": predicted_action,
        "refined": False,
        "is_integer_column": False,
    }

    # ── -1. Name / city / region columns → title case ───────────────────────
    #    If the column name suggests a person name or location, ensure values
    #    are properly title-cased (e.g. "alice" → "Alice", "london" → "London").
    _NAME_CITY_PAT = re.compile(
        r"\bname\b|\bfirst[\s_]?name\b|\blast[\s_]?name\b|\bfull[\s_]?name\b|"
        r"\bcity\b|\btown\b|\bstate\b|\bcountry\b|\bregion\b|\bdistrict\b",
        re.I,
    )
    if _NAME_CITY_PAT.search(col_name_lower):
        non_null = col_series.dropna().astype(str).str.strip()
        if len(non_null) > 0 and (non_null != non_null.str.title()).any():
            reasoning["refined"] = predicted_action != "title_case"
            reasoning["reason"] = (
                f"'{col_name}' contains inconsistent casing — applying title case"
            )
            return "title_case", reasoning

    # ── 0. Boolean columns — normalise to 1/0 regardless of ML ─────────────
    if col_type == "boolean":
        import pandas as _pd
        _BOOL_TRUE  = {"true", "yes", "y", "1", "t", "on"}
        _BOOL_FALSE = {"false", "no", "n", "0", "f", "off"}
        missing_count = int(col_series.isna().sum())

        # Native pandas bool dtype with no missing values → already clean
        if _pd.api.types.is_bool_dtype(col_series) and missing_count == 0:
            reasoning["reason"] = "Boolean column is clean (native bool dtype) — no change"
            return "no_change", reasoning

        # Anything else: string booleans, float 0.0/1.0, mixed formats, missing → normalise
        def _norm_str(v):
            try:
                f = float(str(v).strip())
                return str(int(f)) if f == int(f) else str(v).strip().lower()
            except (ValueError, TypeError):
                return str(v).strip().lower()

        normalized    = col_series.dropna().apply(_norm_str)
        invalid_count = int((~normalized.isin(_BOOL_TRUE | _BOOL_FALSE)).sum())

        reasoning["bool_invalid_count"] = invalid_count
        reasoning["bool_missing_count"] = missing_count
        reasoning["refined"] = predicted_action != "normalize_boolean"
        reasoning["reason"]  = (
            f"Boolean column: {invalid_count} unrecognised value(s), "
            f"{missing_count} missing — normalising to 1/0"
        )
        return "normalize_boolean", reasoning

    # ── 1a. Salary / monetary negative-value check ──────────────────────────
    #    Negative salary, price, revenue, or bonus values are impossible in
    #    most real-world datasets.  Flag and null them out.
    _SALARY_PAT = re.compile(
        r"salary|wage|income|pay|compensation|ctc|bonus|"
        r"price|revenue|cost|amount|earning",
        re.I,
    )
    if _SALARY_PAT.search(col_name_lower):
        numeric_vals = pd.to_numeric(col_series, errors="coerce").dropna()
        neg_count = int((numeric_vals < 0).sum())
        if neg_count > 0:
            reasoning["refined"] = True
            reasoning["negative_values"] = neg_count
            reasoning["reason"] = (
                f"{neg_count} negative value(s) in '{col_name}' — "
                f"impossible for this column type; escalated to convert_and_flag"
            )
            return "convert_and_flag", reasoning

        # Extreme upper outliers (> 3×IQR above Q3) — catches sentinel values
        # like 999999 that are clearly placeholder rather than real salary.
        pos_vals = numeric_vals[numeric_vals >= 0]
        if len(pos_vals) >= 4:
            q1_s, q3_s = pos_vals.quantile(0.25), pos_vals.quantile(0.75)
            iqr_s = q3_s - q1_s
            if iqr_s > 0:
                extreme_upper = q3_s + 3 * iqr_s
                extreme_count = int((pos_vals > extreme_upper).sum())
                if extreme_count > 0:
                    reasoning["refined"] = True
                    reasoning["extreme_outliers"] = extreme_count
                    reasoning["extreme_upper"] = round(float(extreme_upper), 2)
                    reasoning["reason"] = (
                        f"{extreme_count} extreme value(s) above {extreme_upper:.0f} "
                        f"(3×IQR threshold) in '{col_name}' — escalated to convert_and_flag"
                    )
                    return "convert_and_flag", reasoning

    # ── 1. Score / age out-of-range check (before imputation decisions) ──────
    #    If a numeric column has values outside a known valid range, those
    #    need to be flagged/converted first — not just imputed over.
    if col_type == "score" or re.search(r"\bscore\b|\brating\b|\bgrade\b", col_name_lower):
        violations = _range_violations(col_series, 0, 100)
        if violations > 0 and predicted_action not in ("convert_and_flag", "flag_outliers"):
            reasoning["refined"] = True
            reasoning["range_violations"] = violations
            reasoning["reason"] = (
                f"{violations} value(s) outside valid score range [0-100] — "
                f"escalated from '{predicted_action}' to 'convert_and_flag'"
            )
            return "convert_and_flag", reasoning

    if col_type == "age" or re.search(r"\bage\b", col_name_lower):
        violations = _range_violations(col_series, 0, 120)
        if violations > 0 and predicted_action not in ("convert_and_flag",):
            reasoning["refined"] = True
            reasoning["range_violations"] = violations
            reasoning["reason"] = (
                f"{violations} value(s) outside valid age range [0-120] — "
                f"escalated to 'convert_and_flag'"
            )
            return "convert_and_flag", reasoning

    # ── 2. Numeric imputation: mean vs median ────────────────────────────────
    #    Re-derive the decision from actual data, not just the profile snapshot.
    if predicted_action in ("impute_mean", "impute_median"):
        actual_skew = _actual_skewness(col_series)
        outlier_count = _actual_outlier_count(col_series)
        is_int = _is_integer_like(col_series)

        reasoning["actual_skewness"] = round(actual_skew, 4)
        reasoning["actual_outliers"] = outlier_count
        reasoning["is_integer_column"] = is_int

        # Decision rule: outliers or high skew => median is safer
        if outlier_count > 1 or abs(actual_skew) >= 0.5:
            refined = "impute_median"
        else:
            refined = "impute_mean"

        if refined != predicted_action:
            reasoning["refined"] = True
            reasoning["reason"] = (
                f"Switched '{predicted_action}' => '{refined}' "
                f"(actual skew={actual_skew:.2f}, outliers={outlier_count})"
            )
        return refined, reasoning

    # ── 3. Categorical fill: mode vs unknown ─────────────────────────────────
    #    If one value dominates ≥30 % of non-null rows, fill with that mode.
    #    Otherwise use "Unknown" to avoid introducing a misleading dominant value.
    if predicted_action in ("fill_mode", "fill_unknown"):
        dominance = _mode_dominance(col_series)
        reasoning["mode_dominance"] = round(dominance, 4)

        refined = "fill_mode" if dominance >= 0.30 else "fill_unknown"

        if refined != predicted_action:
            reasoning["refined"] = True
            reasoning["reason"] = (
                f"Switched '{predicted_action}' => '{refined}' "
                f"(mode covers {dominance*100:.1f}% of non-null values)"
            )
        return refined, reasoning

    # ── 4. For any numeric action: still mark integer columns ────────────────
    if col_type in ("numeric", "age", "score", "salary") or \
       re.search(r"\bage\b|\bcount\b|\bqty\b|\bnum\b", col_name_lower):
        is_int = _is_integer_like(col_series)
        reasoning["is_integer_column"] = is_int

    return predicted_action, reasoning
