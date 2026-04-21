"""
ML Predictor
============
Loads the trained Decision Tree and predicts the cleaning action for a
column given its profile dict.

Falls back to the rule-based recommender when:
  - the model file is not found (first-run before training)
  - the top predicted class probability < CONFIDENCE_THRESHOLD
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "model.joblib")
CONFIDENCE_THRESHOLD = 0.65

# Lazily loaded so the package can be imported even before training
_bundle: dict | None = None


def _load_model() -> dict | None:
    global _bundle
    if _bundle is not None:
        return _bundle
    path = os.path.normpath(MODEL_PATH)
    if not os.path.exists(path):
        return None
    try:
        import joblib
        _bundle = joblib.load(path)
        return _bundle
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

def _rule_based_action(profile: dict) -> str:
    """
    Deterministic rules that mirror the training label logic.
    Used when the model is unavailable or not confident enough.
    """
    col_type = profile.get("col_type", "text")
    missing_rate = profile.get("missing_rate", 0.0)
    invalid_numeric = profile.get("invalid_numeric_count", 0)
    outlier_count = profile.get("outlier_count_iqr", 0)
    invalid_date = profile.get("invalid_date_count", 0)
    invalid_email = profile.get("invalid_email_count", 0)
    skewness = profile.get("skewness", 0.0)
    unique_ratio = profile.get("unique_ratio", 1.0)

    # Invalids take priority
    if invalid_numeric > 0 and col_type in ("numeric", "age", "score", "salary"):
        return "convert_and_flag"
    if invalid_date > 0 and col_type == "date":
        return "parse_and_flag"
    if invalid_email > 0 and col_type == "email":
        return "normalize_and_flag"

    # Outliers (numeric, no invalids)
    if outlier_count > 0 and col_type in ("numeric", "age", "score", "salary") and invalid_numeric == 0:
        return "flag_outliers"

    # Missing values
    if missing_rate > 0:
        if col_type in ("numeric", "age", "score", "salary"):
            return "impute_mean" if abs(skewness) < 0.5 else "impute_median"
        if col_type in ("text", "category"):
            return "fill_mode" if unique_ratio < 0.3 else "fill_unknown"
        if col_type == "date":
            return "date_manual_review"
        if col_type == "email":
            return "normalize_and_flag"

    return "no_change"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def predict_action(profile: dict) -> dict[str, Any]:
    """
    Predict the best cleaning action for a column given its profile.

    Returns
    -------
    {
        "predicted_action": str,
        "confidence": float,          # top class probability (0–1)
        "probabilities": dict,        # all class probs
        "method": "ml" | "rule_based" | "fallback_low_confidence",
        "rule_action": str,           # what rule-based would have chosen
    }
    """
    rule_action = _rule_based_action(profile)
    bundle = _load_model()

    if bundle is None:
        return {
            "predicted_action": rule_action,
            "confidence": 1.0,
            "probabilities": {rule_action: 1.0},
            "method": "rule_based",
            "rule_action": rule_action,
        }

    model = bundle["model"]
    le = bundle["label_encoder"]

    col_type = profile.get("col_type", "text")

    # Encode col_type — handle unseen labels gracefully
    try:
        col_type_encoded = int(le.transform([col_type])[0])
    except ValueError:
        col_type_encoded = int(le.transform(["text"])[0])

    features = pd.DataFrame([{
        "col_type_encoded": col_type_encoded,
        "missing_rate": profile.get("missing_rate", 0.0),
        "unique_ratio": profile.get("unique_ratio", 0.0),
        "invalid_numeric_count": profile.get("invalid_numeric_count", 0),
        "outlier_count_iqr": profile.get("outlier_count_iqr", 0),
        "invalid_date_count": profile.get("invalid_date_count", 0),
        "invalid_email_count": profile.get("invalid_email_count", 0),
        "skewness": profile.get("skewness", 0.0),
    }])

    predicted = model.predict(features)[0]
    probs_raw = {}
    confidence = 1.0

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0]
        probs_raw = {
            str(cls): round(float(p), 4)
            for cls, p in zip(model.classes_, proba)
        }
        confidence = round(float(max(proba)), 4)

    if confidence >= CONFIDENCE_THRESHOLD:
        return {
            "predicted_action": predicted,
            "confidence": confidence,
            "probabilities": probs_raw,
            "method": "ml",
            "rule_action": rule_action,
        }

    # Low confidence → trust the rule-based decision
    return {
        "predicted_action": rule_action,
        "confidence": confidence,
        "probabilities": probs_raw,
        "method": "fallback_low_confidence",
        "rule_action": rule_action,
    }
