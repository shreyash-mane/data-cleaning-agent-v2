"""
insight_ranker.py
=================
Score, rank, deduplicate, and label insights.

Scoring factors
---------------
• confidence   (0–1)        — statistical confidence from the detector
• strength     (strong / moderate / weak / high / info)
• type priority (relationship insights rank above quality alerts)
• sample size  (log-scaled bonus)
• missing value penalty
"""

from __future__ import annotations

import math

_STRENGTH_SCORE: dict[str, float] = {
    "strong":    1.00,
    "high":      0.85,
    "moderate":  0.65,
    "weak":      0.35,
    "very weak": 0.15,
    "info":      0.10,
}

_TYPE_PRIORITY: dict[str, float] = {
    "numerical_numerical":   1.00,
    "categorical_numerical": 0.95,
    "date_numerical":        0.90,
    "categorical_categorical": 0.75,
    "distribution":          0.55,
    "data_quality":          0.35,
}


def _score(ins: dict) -> float:
    conf     = float(ins.get("confidence", 0.5))
    strength = _STRENGTH_SCORE.get(str(ins.get("strength", "weak")), 0.30)
    priority = _TYPE_PRIORITY.get(str(ins.get("type", "")), 0.50)

    meta         = ins.get("metadata") or {}
    missing_pct  = float(meta.get("missing_pct") or 0)
    sample_size  = float(meta.get("sample_size") or 50)

    missing_pen  = 1.0 - (missing_pct / 100) * 0.50
    sample_bonus = min(1.0, math.log(max(sample_size, 2)) / math.log(500))

    score = (
        conf         * 0.35
        + strength   * 0.28
        + priority   * 0.22
        + sample_bonus * 0.15
    ) * missing_pen

    return round(score, 4)


def rank_insights(insights: list[dict], top_n: int = 10) -> list[dict]:
    """
    Score every insight, sort descending, deduplicate by (type, column-pair),
    keep top_n, and assign insight_id.
    """
    scored = [{**ins, "score": _score(ins)} for ins in insights]
    scored.sort(key=lambda x: -x["score"])

    # Deduplicate: same relationship type + same column set is counted once
    seen: set[tuple] = set()
    unique: list[dict] = []
    for ins in scored:
        key = (ins["type"], tuple(sorted(ins["columns"])))
        if key in seen:
            continue
        seen.add(key)
        unique.append(ins)

    top = unique[:top_n]
    for i, ins in enumerate(top):
        ins["insight_id"] = f"ins_{i + 1:03d}"
    return top
