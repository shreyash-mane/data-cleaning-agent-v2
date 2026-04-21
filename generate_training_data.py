"""
Generate synthetic training data for the cleaning action classifier.
=====================================================================
Produces ~3600 rows (400 per class × 9 classes) with deterministic,
rule-correct labels — so the model learns the exact same logic as the
rule-based fallback but can also generalise to borderline cases.

Run:
    python generate_training_data.py
"""

import os
import numpy as np
import pandas as pd

np.random.seed(42)
rng = np.random.default_rng(42)

ROWS_PER_CLASS = 4000

ALL_TYPES = ["numeric", "age", "score", "salary", "date", "email", "text", "category"]

rows = []


def row(col_type, missing_rate, unique_ratio, inv_num, outliers,
        inv_date, inv_email, skew, label):
    return {
        "col_type":              col_type,
        "missing_rate":          round(float(missing_rate), 4),
        "unique_ratio":          round(float(unique_ratio), 4),
        "invalid_numeric_count": int(inv_num),
        "outlier_count_iqr":     int(outliers),
        "invalid_date_count":    int(inv_date),
        "invalid_email_count":   int(inv_email),
        "skewness":              round(float(skew), 4),
        "recommended_action":    label,
    }


# ── 1. no_change ─────────────────────────────────────────────────────────────
# Clean columns of every type — no missing, no invalids, no outliers
for _ in range(ROWS_PER_CLASS):
    ct = rng.choice(ALL_TYPES)
    rows.append(row(
        ct,
        missing_rate=0.0,
        unique_ratio=rng.uniform(0.05, 1.0),
        inv_num=0,
        outliers=0,
        inv_date=0,
        inv_email=0,
        skew=rng.uniform(-0.49, 0.49),
        label="no_change",
    ))

# ── 2. impute_mean ────────────────────────────────────────────────────────────
# Numeric with missing values and low skew — mean is appropriate
for _ in range(ROWS_PER_CLASS):
    ct = rng.choice(["numeric", "score", "salary"])
    rows.append(row(
        ct,
        missing_rate=rng.uniform(0.01, 0.35),
        unique_ratio=rng.uniform(0.1, 1.0),
        inv_num=0,
        outliers=rng.integers(0, 3),
        inv_date=0,
        inv_email=0,
        skew=rng.uniform(-0.49, 0.49),
        label="impute_mean",
    ))

# ── 3. impute_median ──────────────────────────────────────────────────────────
# Numeric with missing values and high skew — median is safer
for _ in range(ROWS_PER_CLASS):
    ct = rng.choice(["numeric", "age", "score", "salary"])
    # Skew magnitude > 0.5 on either side
    skew = float(rng.choice([
        rng.uniform(0.5, 5.0),
        rng.uniform(-5.0, -0.5),
    ]))
    rows.append(row(
        ct,
        missing_rate=rng.uniform(0.01, 0.40),
        unique_ratio=rng.uniform(0.05, 1.0),
        inv_num=0,
        outliers=rng.integers(0, 8),
        inv_date=0,
        inv_email=0,
        skew=skew,
        label="impute_median",
    ))

# ── 4. convert_and_flag ───────────────────────────────────────────────────────
# Numeric-type column with invalid (non-numeric) values present
for _ in range(ROWS_PER_CLASS):
    ct = rng.choice(["numeric", "age", "score", "salary"])
    rows.append(row(
        ct,
        missing_rate=rng.uniform(0.0, 0.3),
        unique_ratio=rng.uniform(0.05, 1.0),
        inv_num=rng.integers(1, 25),
        outliers=rng.integers(0, 10),
        inv_date=0,
        inv_email=0,
        skew=rng.uniform(-4.0, 5.0),
        label="convert_and_flag",
    ))

# ── 5. parse_and_flag ─────────────────────────────────────────────────────────
# Date column with invalid / mixed format values
for _ in range(ROWS_PER_CLASS):
    rows.append(row(
        "date",
        missing_rate=rng.uniform(0.0, 0.3),
        unique_ratio=rng.uniform(0.1, 1.0),
        inv_num=0,
        outliers=0,
        inv_date=rng.integers(1, 30),
        inv_email=0,
        skew=0.0,
        label="parse_and_flag",
    ))

# ── 6. normalize_and_flag ─────────────────────────────────────────────────────
# Email column with invalid addresses
for _ in range(ROWS_PER_CLASS):
    rows.append(row(
        "email",
        missing_rate=rng.uniform(0.0, 0.25),
        unique_ratio=rng.uniform(0.2, 1.0),
        inv_num=0,
        outliers=0,
        inv_date=0,
        inv_email=rng.integers(1, 20),
        skew=0.0,
        label="normalize_and_flag",
    ))

# ── 7. flag_outliers ──────────────────────────────────────────────────────────
# Numeric with IQR outliers but no invalid values and no missing
for _ in range(ROWS_PER_CLASS):
    ct = rng.choice(["numeric", "age", "score", "salary"])
    rows.append(row(
        ct,
        missing_rate=0.0,
        unique_ratio=rng.uniform(0.05, 1.0),
        inv_num=0,
        outliers=rng.integers(5, 60),
        inv_date=0,
        inv_email=0,
        skew=rng.uniform(0.5, 5.0),   # high skew correlates with outliers
        label="flag_outliers",
    ))

# ── 8. fill_mode ─────────────────────────────────────────────────────────────
# Text/category column with missing values and low cardinality → fill with mode
for _ in range(ROWS_PER_CLASS):
    ct = rng.choice(["text", "category"])
    rows.append(row(
        ct,
        missing_rate=rng.uniform(0.01, 0.45),
        unique_ratio=rng.uniform(0.0, 0.29),  # low cardinality
        inv_num=0,
        outliers=0,
        inv_date=0,
        inv_email=0,
        skew=0.0,
        label="fill_mode",
    ))

# ── 9. fill_unknown ───────────────────────────────────────────────────────────
# Text/category column with missing values and high cardinality → "Unknown"
for _ in range(ROWS_PER_CLASS):
    ct = rng.choice(["text", "category"])
    rows.append(row(
        ct,
        missing_rate=rng.uniform(0.01, 0.45),
        unique_ratio=rng.uniform(0.30, 1.0),  # high cardinality
        inv_num=0,
        outliers=0,
        inv_date=0,
        inv_email=0,
        skew=0.0,
        label="fill_unknown",
    ))

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)
df = pd.DataFrame(rows)

# Add small Gaussian noise to continuous features to improve generalisation
noise_cols = ["missing_rate", "unique_ratio", "skewness"]
for nc in noise_cols:
    df[nc] = (df[nc] + rng.normal(0, 0.01, size=len(df))).clip(-10, 10).round(4)

df["missing_rate"] = df["missing_rate"].clip(0, 1)
df["unique_ratio"] = df["unique_ratio"].clip(0, 1)

output_path = "data/training_data.csv"
df.to_csv(output_path, index=False)

print(f"Generated {len(df)} training rows -> {output_path}")
print("\nClass distribution:")
print(df["recommended_action"].value_counts().to_string())
print(f"\nColumn types in training set:")
print(df["col_type"].value_counts().to_string())
