"""
Dataset Type Detector
=====================
Analyses column names (and optionally column profiles) to identify the
domain of a dataset and return a cleaning template with tailored
per-column action recommendations.
"""

from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------------------
# Signature registry
# Each entry defines keyword signals, domain-specific warnings, and a set of
# column-pattern → recommended-action rules that override the generic ML
# suggestion when confidence is high enough.
# ---------------------------------------------------------------------------

DATASET_SIGNATURES = [
    {
        "type": "ecommerce",
        "name": "E-commerce / Sales",
        "icon": "🛒",
        "color": "#f59e0b",
        "keywords": [
            "order", "product", "sku", "cart", "purchase", "item",
            "price", "quantity", "qty", "revenue", "discount", "coupon",
            "shipping", "customer", "invoice", "checkout", "basket",
        ],
        "warnings": [
            "Check for duplicate order IDs before cleaning",
            "Negative quantities may indicate returns — do not drop blindly",
            "Verify price > 0; zero-price rows may be free items or data errors",
        ],
        "recommendations": [
            {"pattern": r"price|amount|revenue|cost|discount|total|subtotal|tax", "action": "impute_median", "reason": "Monetary values are right-skewed — median is safer than mean"},
            {"pattern": r"qty|quantity|count|units|volume",                        "action": "impute_median", "reason": "Count columns: median avoids inflating averages"},
            {"pattern": r"date|time|created|updated|ordered|shipped|delivered",    "action": "parse_and_flag", "reason": "Standardise all date formats and flag invalid"},
            {"pattern": r"category|status|type|region|country|city|state",         "action": "fill_mode",    "reason": "Fill missing categoricals with most frequent value"},
            {"pattern": r"customer_id|order_id|product_id|sku|invoice_id",         "action": "no_change",    "reason": "ID columns must not be imputed"},
        ],
    },
    {
        "type": "hr",
        "name": "HR / People Analytics",
        "icon": "👥",
        "color": "#8b5cf6",
        "keywords": [
            "employee", "staff", "hire", "department", "salary", "payroll",
            "leave", "attendance", "performance", "manager", "headcount",
            "tenure", "role", "position", "job", "workforce",
        ],
        "warnings": [
            "Salary outliers often represent executives vs IC — investigate before imputing",
            "Negative tenure values are impossible — flag as convert_and_flag",
            "Gender / ethnicity fields require special handling (do not impute with mode)",
        ],
        "recommendations": [
            {"pattern": r"salary|compensation|pay|wage|bonus|ctc",        "action": "impute_median", "reason": "Salary data is typically right-skewed — use median"},
            {"pattern": r"age|tenure|years_exp|experience",               "action": "impute_median", "reason": "Outliers common in age/tenure — median is safer"},
            {"pattern": r"hire_date|start_date|end_date|dob|birth",       "action": "parse_and_flag", "reason": "Parse date columns and flag any invalid dates"},
            {"pattern": r"department|role|position|level|grade|band",     "action": "fill_mode",    "reason": "Categorical HR fields: fill with most frequent"},
            {"pattern": r"rating|score|performance|kpi|review",           "action": "normalize_and_flag", "reason": "Performance scores: normalise and flag out-of-range"},
            {"pattern": r"employee_id|emp_id|staff_id",                   "action": "no_change",    "reason": "Employee IDs must not be imputed"},
        ],
    },
    {
        "type": "healthcare",
        "name": "Healthcare / Clinical",
        "icon": "🏥",
        "color": "#ef4444",
        "keywords": [
            "patient", "diagnosis", "icd", "medication", "drug", "dose",
            "clinical", "hospital", "visit", "admission", "discharge",
            "bmi", "blood", "pressure", "cholesterol", "glucose", "vitals",
        ],
        "warnings": [
            "Dataset likely contains PII — ensure compliance before any cleaning",
            "Missing values may be clinically significant, not random — MCAR assumption may not hold",
            "Flag implausible vitals (BMI < 10 or > 80, age > 120) rather than imputing",
        ],
        "recommendations": [
            {"pattern": r"bmi|weight|height",                                      "action": "impute_median", "reason": "Use median; extreme values indicate entry errors"},
            {"pattern": r"age",                                                    "action": "impute_median", "reason": "Patient age: median safer with skewed populations"},
            {"pattern": r"blood_pressure|bp|systolic|diastolic|glucose|cholesterol|pulse|spo2", "action": "flag_outliers", "reason": "Flag clinical values outside physiological range"},
            {"pattern": r"diagnosis|icd|condition|disease|medication|drug|allergy","action": "fill_unknown",  "reason": "Clinical categories: flag missing as Unknown"},
            {"pattern": r"date|admission|discharge|visit|dob|birth",               "action": "parse_and_flag", "reason": "Parse all date columns and flag invalid"},
            {"pattern": r"patient_id|record_id|mrn",                              "action": "no_change",    "reason": "Patient identifiers must not be imputed"},
        ],
    },
    {
        "type": "finance",
        "name": "Finance / Transactions",
        "icon": "💰",
        "color": "#10b981",
        "keywords": [
            "transaction", "account", "balance", "debit", "credit", "ledger",
            "payment", "invoice", "bank", "fund", "portfolio", "asset",
            "liability", "equity", "return", "interest", "trade",
        ],
        "warnings": [
            "Negative balances may be legitimate (overdrafts) — do not impute away",
            "Verify transaction amounts are non-zero before aggregating",
            "Watch for duplicate transaction IDs — dedup before any analysis",
        ],
        "recommendations": [
            {"pattern": r"amount|balance|debit|credit|payment|value|net|gross",   "action": "impute_median", "reason": "Financial amounts: use median due to extreme skew"},
            {"pattern": r"date|timestamp|period|quarter|month|year|settlement",   "action": "parse_and_flag", "reason": "Standardise all date/period columns"},
            {"pattern": r"account|category|type|status|channel|currency|country", "action": "fill_mode",    "reason": "Fill categorical fields with most frequent"},
            {"pattern": r"rate|return|yield|interest|percentage|pct|ratio",       "action": "normalize_and_flag", "reason": "Rate columns: normalise and flag impossible values"},
            {"pattern": r"transaction_id|account_id|reference|txn_id",            "action": "no_change",    "reason": "Transaction IDs must not be imputed"},
        ],
    },
    {
        "type": "marketing",
        "name": "Marketing / Campaign",
        "icon": "📣",
        "color": "#ec4899",
        "keywords": [
            "campaign", "impression", "click", "ctr", "conversion", "cpc",
            "cpm", "roas", "ad", "utm", "channel", "audience", "lead",
            "funnel", "bounce", "open_rate", "unsubscribe",
        ],
        "warnings": [
            "CTR > 1.0 or negative values are data errors — flag with convert_and_flag",
            "Zero impressions with non-zero clicks need investigation before cleaning",
            "ROAS and CPC can be null for organic channels — fill_unknown may be more appropriate than mode",
        ],
        "recommendations": [
            {"pattern": r"impression|reach|view|exposure",                        "action": "impute_median", "reason": "Impression counts are highly skewed — use median"},
            {"pattern": r"click|conversion|lead|signup",                          "action": "impute_median", "reason": "Event counts: use median"},
            {"pattern": r"ctr|cpc|cpm|roas|rate|ratio|pct|open_rate|bounce",      "action": "normalize_and_flag", "reason": "Rate metrics: normalise and flag impossible values"},
            {"pattern": r"date|period|week|month|start|end|day",                  "action": "parse_and_flag", "reason": "Parse date columns for time-series analysis"},
            {"pattern": r"channel|source|medium|campaign|audience|platform|ad_group", "action": "fill_mode", "reason": "Categorical marketing dims: fill with mode"},
        ],
    },
    {
        "type": "survey",
        "name": "Survey / Research",
        "icon": "📋",
        "color": "#6366f1",
        "keywords": [
            "respondent", "survey", "response", "question", "answer",
            "likert", "rating", "score", "agree", "disagree",
            "satisfaction", "nps", "feedback",
        ],
        "warnings": [
            "Watch for straight-liners (respondent gave the same answer across all questions)",
            "Likert scale values should be integers — float values indicate data entry errors",
            "Extreme NPS scores (0 or 10 only) may indicate satisficing bias",
        ],
        "recommendations": [
            {"pattern": r"score|rating|q\d+|question|satisfaction|nps|likert|agree", "action": "impute_median", "reason": "Survey responses: median avoids scale distortion"},
            {"pattern": r"age|gender|income|education|region|occupation",            "action": "fill_mode",    "reason": "Demographic fields: fill with most frequent"},
            {"pattern": r"date|timestamp|completed|submitted|start|end",             "action": "parse_and_flag", "reason": "Parse all submission date columns"},
            {"pattern": r"comment|feedback|text|open|other|reason",                 "action": "fill_unknown",  "reason": "Open-ended text: mark missing as Unknown"},
            {"pattern": r"respondent_id|response_id|participant_id",                "action": "no_change",    "reason": "Respondent IDs must not be imputed"},
        ],
    },
    {
        "type": "timeseries",
        "name": "Time-series / IoT",
        "icon": "📡",
        "color": "#14b8a6",
        "keywords": [
            "timestamp", "sensor", "temperature", "humidity", "pressure",
            "vibration", "voltage", "current", "reading", "measurement",
            "telemetry", "signal", "metric", "interval", "frequency",
        ],
        "warnings": [
            "Missing timestamps may represent true gaps — do not interpolate without domain knowledge",
            "Check that timestamps are monotonically increasing within each device/sensor ID",
            "Outlier spikes may be genuine anomalies — flag rather than impute",
        ],
        "recommendations": [
            {"pattern": r"timestamp|datetime|time|date",                           "action": "parse_and_flag", "reason": "Parse and validate all time columns"},
            {"pattern": r"temperature|humidity|pressure|voltage|current|power|reading|value|measurement|speed|flow", "action": "impute_median", "reason": "Sensor readings: median preserves signal shape"},
            {"pattern": r"status|state|mode|flag|alarm|alert",                    "action": "fill_mode",    "reason": "State columns: fill with most frequent"},
            {"pattern": r"device_id|sensor_id|unit_id|machine_id",               "action": "no_change",    "reason": "Device/sensor IDs must not be imputed"},
        ],
    },
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_dataset_type(columns: list[str], profiles: list[dict] | None = None) -> dict[str, Any]:
    """
    Score all signatures against column names and return the best match
    with a cleaning template.

    Parameters
    ----------
    columns  : list of column name strings
    profiles : optional list of per-column profile dicts (used for richer signals)

    Returns
    -------
    {
        dataset_type, name, icon, color, confidence,
        detected_signals, warnings, column_recommendations
    }
    """
    col_text = " ".join(c.lower() for c in columns)

    scores: list[tuple[float, int, dict, list]] = []
    for sig in DATASET_SIGNATURES:
        hits = [kw for kw in sig["keywords"] if re.search(r"\b" + re.escape(kw) + r"\b", col_text)]
        ratio = len(hits) / max(len(sig["keywords"]), 1)
        scores.append((ratio, len(hits), sig, hits))

    scores.sort(reverse=True, key=lambda x: (x[0], x[1]))
    best_ratio, best_hit_count, best_sig, detected_signals = scores[0]

    # Need at least 2 keyword hits to commit to a domain
    if best_hit_count < 2:
        return {
            "dataset_type": "generic",
            "name": "General Dataset",
            "icon": "📄",
            "color": "#6b7280",
            "confidence": 0.0,
            "detected_signals": [],
            "warnings": [],
            "column_recommendations": {},
        }

    # Build per-column template recommendations
    column_recommendations: dict[str, dict] = {}
    for col in columns:
        col_l = col.lower()
        for rec in best_sig["recommendations"]:
            if re.search(rec["pattern"], col_l):
                column_recommendations[col] = {
                    "suggested_action": rec["action"],
                    "reason": rec["reason"],
                }
                break

    confidence = round(min(0.97, best_ratio * 2.5), 2)

    return {
        "dataset_type": best_sig["type"],
        "name": best_sig["name"],
        "icon": best_sig["icon"],
        "color": best_sig["color"],
        "confidence": confidence,
        "detected_signals": detected_signals[:8],
        "warnings": best_sig["warnings"],
        "column_recommendations": column_recommendations,
    }
