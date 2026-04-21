"""
explanation_engine.py
=====================
Generate natural-language explanations for detected insights.

Mode A: Rule-based templates (always available, zero dependencies)
Mode B: LLM/API layer  (implement generate_llm_explanation to activate)

The public function build_explanation() tries Mode B first and falls back to
Mode A automatically if the LLM returns None or raises an exception.
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Mode B — LLM placeholder
# ---------------------------------------------------------------------------

def generate_llm_explanation(insight_payload: dict) -> str | None:
    """
    Optional LLM explanation layer.

    Implement this function to call any LLM API (OpenAI, Anthropic, Gemini …).
    The insight_payload dict contains all structured statistics needed to build
    a prompt.  Return None to fall back to rule-based templates.

    Example skeleton
    ----------------
    prompt = _build_prompt(insight_payload)
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content
    """
    return None   # LLM not configured — use rule-based fallback


def _build_prompt(ins: dict) -> str:
    """Structured prompt scaffold for LLM mode.  Not called unless LLM is wired up."""
    return (
        f"You are a data analyst assistant. "
        f"Explain the following statistical insight to a non-technical user in 3–4 sentences:\n\n"
        f"Insight type: {ins.get('type')}\n"
        f"Title: {ins.get('title')}\n"
        f"Summary: {ins.get('summary')}\n"
        f"Strength: {ins.get('strength')}\n"
        f"Key metadata: {ins.get('metadata')}\n\n"
        "Focus on what it means, how strong the pattern is, and one practical implication."
    )


# ---------------------------------------------------------------------------
# Mode A — Rule-based templates
# ---------------------------------------------------------------------------

def _explain_num_num(ins: dict) -> str:
    meta      = ins.get("metadata") or {}
    cols      = ins.get("columns", ["A", "B"])
    col_a, col_b = cols[0], cols[1]
    corr      = meta.get("correlation", 0)
    p_val     = meta.get("p_value", 1)
    n         = meta.get("sample_size", 0)
    strength  = ins.get("strength", "weak")
    direction = meta.get("direction", "positive")

    verb = "increase" if direction == "positive" else "decrease"
    body = {
        "strong":   (
            f"There is a **strong {direction} relationship** between {col_a} and {col_b} "
            f"(Pearson r = {corr:.2f}). As {col_a} increases, {col_b} tends to {verb} substantially."
        ),
        "moderate": (
            f"There is a **moderate {direction} relationship** between {col_a} and {col_b} "
            f"(r = {corr:.2f}). The pattern is meaningful but not perfectly consistent — "
            f"other factors also influence {col_b}."
        ),
        "weak":     (
            f"There is a **weak {direction} tendency** between {col_a} and {col_b} "
            f"(r = {corr:.2f}). The pattern exists but explains only a small fraction of the variation."
        ),
    }.get(strength, ins.get("summary", ""))

    sig = (
        f"This result is **statistically significant** (p = {p_val:.4f}, n = {n})."
        if p_val < 0.05
        else f"⚠️ This result is **not statistically significant** (p = {p_val:.4f}). Treat with caution."
    )
    caveat = "Note: correlation does not imply causation. Other variables may drive this relationship."
    return f"{body}\n\n{sig}\n\n{caveat}"


def _explain_cat_num(ins: dict) -> str:
    meta       = ins.get("metadata") or {}
    cols       = ins.get("columns", ["Cat", "Num"])
    cat_col, num_col = cols[0], cols[1]
    top_cat    = meta.get("top_category", "?")
    top_val    = meta.get("top_value", 0)
    overall    = meta.get("overall_mean", 0)
    n_groups   = meta.get("group_count", 0)
    p_val      = meta.get("anova_p_value")
    strength   = ins.get("strength", "weak")

    diff_pct   = abs(top_val - overall) / max(abs(overall), 1e-9) * 100
    hi_lo      = "highest" if top_val >= overall else "lowest"

    body = (
        f"When the data is grouped by **{cat_col}**, the '{top_cat}' group shows "
        f"the {hi_lo} average **{num_col}** ({top_val:.2f} vs overall mean {overall:.2f} — "
        f"a {diff_pct:.1f}% difference)."
    )

    comment = {
        "strong":   f"The differences across the {n_groups} groups are **substantial** — this is likely a meaningful segmentation.",
        "moderate": f"Group differences are **moderate** across {n_groups} categories.",
        "weak":     f"Group differences are **small** across {n_groups} categories.",
    }.get(strength, "")

    sig = (
        f"ANOVA test confirms group differences are statistically significant (p = {p_val:.4f})."
        if p_val is not None and p_val < 0.05
        else (
            f"⚠️ ANOVA test suggests group differences may not be statistically significant (p = {p_val:.4f})."
            if p_val is not None
            else ""
        )
    )
    parts = [body, comment]
    if sig:
        parts.append(sig)
    return "\n\n".join(p for p in parts if p)


def _explain_date_num(ins: dict) -> str:
    meta      = ins.get("metadata") or {}
    cols      = ins.get("columns", ["Date", "Num"])
    date_col, num_col = cols[0], cols[1]
    trend     = meta.get("trend_direction", "stable")
    r_sq      = meta.get("r_squared", 0)
    p_val     = meta.get("p_value", 1)
    n         = meta.get("sample_size", 0)
    strength  = ins.get("strength", "weak")

    body = {
        "strong":   (
            f"**{num_col}** shows a **strong {trend} trend** over {date_col}. "
            f"Time explains {r_sq * 100:.1f}% of its variation (R² = {r_sq:.2f})."
        ),
        "moderate": (
            f"**{num_col}** shows a **moderate {trend} trend** over {date_col} "
            f"(R² = {r_sq:.2f}). The direction is clear but with notable fluctuations."
        ),
        "weak":     (
            f"**{num_col}** has a **weak {trend} tendency** over {date_col} "
            f"(R² = {r_sq:.2f}). The trend is mild and may not be reliable."
        ),
    }.get(strength, ins.get("summary", ""))

    sig = (
        f"Trend is statistically significant (p = {p_val:.4f}, n = {n})."
        if p_val < 0.05
        else f"⚠️ Trend is not statistically significant (p = {p_val:.4f}). Treat with caution."
    )
    return f"{body}\n\n{sig}"


def _explain_cat_cat(ins: dict) -> str:
    meta      = ins.get("metadata") or {}
    cols      = ins.get("columns", ["A", "B"])
    col_a, col_b = cols[0], cols[1]
    v         = meta.get("cramers_v", 0)
    p_val     = meta.get("p_value", 1)
    n         = meta.get("sample_size", 0)
    top       = meta.get("top_combination") or {}
    strength  = ins.get("strength", "weak")

    body = (
        f"**{col_a}** and **{col_b}** show a {strength} association (Cramér's V = {v:.2f}). "
        f"Cramér's V ranges from 0 (no association) to 1 (perfect association)."
    )
    combo = (
        f"The most common combination is '{top.get('col_a', '?')}' in {col_a} "
        f"paired with '{top.get('col_b', '?')}' in {col_b}."
        if top else ""
    )
    sig = (
        f"Chi-squared test confirms this association is statistically significant "
        f"(p = {p_val:.4f}, n = {n})."
        if p_val < 0.05
        else f"⚠️ Chi-squared test does not confirm significance (p = {p_val:.4f})."
    )
    return "\n\n".join(p for p in [body, combo, sig] if p)


def _explain_quality(ins: dict) -> str:
    meta    = ins.get("metadata") or {}
    cols    = ins.get("columns", ["col"])
    col     = cols[0]
    subtype = ins.get("type", "")

    if subtype == "data_quality":
        missing = meta.get("missing_pct")
        if missing:
            level = "severely" if missing > 60 else "moderately"
            return (
                f"**{col}** is {level} affected by missing data ({missing}% of values are null). "
                f"Any analysis involving this column may produce biased or unreliable results. "
                f"Consider imputing missing values before using this column in models or aggregations."
            )
        return (
            f"**{col}** appears to be a unique identifier column. "
            "ID columns have near-100% unique values and should not be used in correlation "
            "or grouping analyses — they won't yield meaningful patterns."
        )

    skew     = meta.get("skewness")
    out_pct  = meta.get("outlier_pct")

    if skew is not None:
        direction = "right" if skew > 0 else "left"
        cause = (
            "a small number of very high values (e.g. extreme salaries, prices, or ages) pulling the tail up."
            if direction == "right"
            else "a small number of very low or negative values pulling the tail down."
        )
        return (
            f"**{col}** has a skewness of {skew:.2f} — a **{direction}-skewed distribution** caused by {cause} "
            f"The mean is higher than the median for right-skewed data, which can distort averages. "
            f"Consider log-transformation if using this column in statistical models."
        )

    if out_pct is not None:
        q1, q3, iqr = meta.get("q1", 0), meta.get("q3", 0), meta.get("iqr", 0)
        return (
            f"**{col}** contains {out_pct:.1f}% outliers (values outside "
            f"[{q1 - 1.5 * iqr:.1f}, {q3 + 1.5 * iqr:.1f}] by the IQR method). "
            "Outliers can distort means and correlations. "
            "Verify whether they are genuine extremes or data entry errors before modelling."
        )

    return ins.get("summary", "No additional details available.")


_DISPATCH = {
    "numerical_numerical":    _explain_num_num,
    "categorical_numerical":  _explain_cat_num,
    "date_numerical":         _explain_date_num,
    "categorical_categorical": _explain_cat_cat,
    "distribution":           _explain_quality,
    "data_quality":           _explain_quality,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_explanation(ins: dict) -> str:
    """Return explanation text for an insight (LLM if configured, else rule-based)."""
    try:
        llm_result = generate_llm_explanation(ins)
        if llm_result:
            return llm_result
    except Exception:
        pass

    fn = _DISPATCH.get(ins.get("type", ""))
    if fn:
        try:
            return fn(ins)
        except Exception:
            pass
    return ins.get("summary", "No explanation available.")


def get_reliability_notes(ins: dict) -> list[str]:
    """Return bullet-point reliability caveats for the frontend."""
    notes: list[str] = []
    meta  = ins.get("metadata") or {}

    n = meta.get("sample_size", 0)
    if n and n < 30:
        notes.append(f"Small sample ({n} rows) — results may not generalise.")
    elif n and n < 100:
        notes.append(f"Moderate sample ({n} rows) — interpret with some caution.")

    for key in ("p_value", "anova_p_value"):
        p = meta.get(key)
        if p is not None and p > 0.05:
            notes.append(f"Not statistically significant (p = {p:.4f}).")
            break

    missing = meta.get("missing_pct", 0) or 0
    if missing > 20:
        notes.append(f"High missing rate ({missing}%) may affect accuracy.")

    if ins.get("type") == "numerical_numerical":
        notes.append("Pearson r assumes linearity. Consider Spearman for non-linear patterns.")

    if not notes:
        notes.append("No major reliability concerns detected.")
    return notes
