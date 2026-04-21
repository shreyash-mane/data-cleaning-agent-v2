"""
Production model test on fresh raw data.
Prints a full structured report of what the model detected and fixed.
"""
import pandas as pd
from cleaning_agent.pipeline import CleaningPipeline

RAW_PATH    = "data/raw_production_test.csv"
OUTPUT_PATH = "data/cleaned_production_v2.csv"

df_raw = pd.read_csv(RAW_PATH)
pipeline = CleaningPipeline()
result   = pipeline.run(df_raw)
cleaned  = result["cleaned_df"]
report   = result["report"]
summary  = result["summary"]

SEP  = "=" * 68
SEP2 = "-" * 68

# ─────────────────────────────────────────────────────────────────────────────
# 1. RAW DATA SNAPSHOT
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  RAW DATA SNAPSHOT")
print(SEP)
print(f"  Rows        : {len(df_raw)}")
print(f"  Columns     : {len(df_raw.columns)}")
print(f"\n  {'Column':<22} {'Type (raw)':<12} {'Missing':>8}   {'Sample issues'}")
print(f"  {SEP2}")
for col in df_raw.columns:
    missing = df_raw[col].isna().sum()
    pct     = missing / len(df_raw) * 100
    # collect non-null unique looking issues
    issues = []
    for v in df_raw[col].dropna().astype(str).unique():
        if v.lower() in ("n/a","na","none","null","abc","invalid","invalid-date","not-a-date"):
            issues.append(v)
    issue_str = ", ".join(issues[:3]) if issues else ""
    miss_str  = f"{missing} ({pct:.0f}%)" if missing else "-"
    print(f"  {col:<22} {str(df_raw[col].dtype):<12} {miss_str:>8}   {issue_str}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. PIPELINE SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  PIPELINE SUMMARY")
print(SEP)
labels = {
    "columns_processed":       "Columns processed",
    "columns_skipped":         "Columns skipped (ID type)",
    "total_cells_changed":     "Total cells changed",
    "total_cells_flagged":     "Total cells flagged for review",
    "rows_before":             "Rows before",
    "rows_after":              "Rows after",
    "duplicate_rows_dropped":  "Duplicate rows dropped",
    "ml_used_count":           "Decisions by ML model",
    "rule_used_count":         "Decisions by rule fallback",
}
for k, label in labels.items():
    print(f"  {label:<35} {summary.get(k, 0)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PER-COLUMN MODEL DECISIONS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  MODEL DECISIONS  (per column)")
print(SEP)
print(f"  {'Column':<22} {'Det. Type':<12} {'Action':<22} {'Conf':>6}  {'Chg':>4}  {'Flg':>4}  {'Note'}")
print(f"  {SEP2}")
for r in report:
    conf    = r.get("confidence", 0)
    ml_act  = r.get("ml_action", r["action"])
    final   = r["action"]
    refined = r.get("intelligence_refined", False)
    note    = ""
    if r["action"] == "skipped":
        note = "auto-skip ID col"
    elif refined:
        note = f"ML={ml_act} => refined"
    elif r.get("fill_value") is not None:
        note = f"fill={r['fill_value']}"
    elif r.get("imputed_nulls"):
        note = f"imputed {r['imputed_nulls']} nulls via {r.get('impute_method')} ({r.get('impute_fill_value')})"
    intel = r.get("intelligence", {})
    if intel.get("is_integer_column") and r["action"] in ("impute_mean","impute_median","convert_and_flag"):
        note += "  [int]"
    print(f"  {r['column']:<22} {r['col_type']:<12} {final:<22} {conf:>6.2f}  {r.get('cells_changed',0):>4}  {r.get('cells_flagged',0):>4}  {note}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. COLUMN-BY-COLUMN DETAIL
# ─────────────────────────────────────────────────────────────────────────────

def diff_table(col, id_col="customer_id", name_col="full_name", show_all=False):
    """Print raw vs cleaned for a column, only changed/missing rows by default."""
    if col not in df_raw.columns or col not in cleaned.columns:
        return
    raw_s   = df_raw[col].reset_index(drop=True)
    clean_s = cleaned[col].reset_index(drop=True)
    ids     = df_raw[id_col].reset_index(drop=True)
    names   = df_raw[name_col].reset_index(drop=True)

    rows_out = []
    for i in range(len(raw_s)):
        rv = raw_s.iloc[i]
        cv = clean_s.iloc[i]
        rv_str = str(rv) if pd.notna(rv) else "(missing)"
        cv_str = str(cv) if pd.notna(cv) else "NaN"

        if rv_str == "(missing)" and cv_str != "NaN":
            status = "FILLED"
        elif rv_str in ("nan","(missing)") and cv_str == "NaN":
            status = "STILL NULL"
        elif cv_str == "NaN" and rv_str not in ("nan","(missing)"):
            status = "FLAGGED/NULLED"
        elif rv_str != cv_str:
            status = "FIXED"
        elif show_all:
            status = "OK"
        else:
            continue
        rows_out.append((str(ids.iloc[i]), str(names.iloc[i])[:18], rv_str[:28], cv_str[:16], status))

    if not rows_out:
        print(f"  All values OK — no changes needed.")
        return

    print(f"  {'ID':<6} {'Name':<20} {'Raw':<30} {'Cleaned':<18} Status")
    print(f"  {'-'*6} {'-'*20} {'-'*30} {'-'*18} {'-'*14}")
    for r in rows_out:
        print(f"  {r[0]:<6} {r[1]:<20} {r[2]:<30} {r[3]:<18} {r[4]}")


print(f"\n{SEP}")
print("  AGE  — missing values filled by model")
print(SEP)
diff_table("age")

print(f"\n{SEP}")
print("  ANNUAL_INCOME  — invalid & missing values handled")
print(SEP)
diff_table("annual_income")

print(f"\n{SEP}")
print("  LOYALTY_SCORE  — missing values filled by model")
print(SEP)
diff_table("loyalty_score")

print(f"\n{SEP}")
print("  ACCOUNT_CREATED  — date formats normalised to ISO")
print(SEP)
diff_table("account_created")

print(f"\n{SEP}")
print("  LAST_PURCHASE_DATE  — date formats normalised to ISO")
print(SEP)
diff_table("last_purchase_date")

print(f"\n{SEP}")
print("  EMAIL  — invalid addresses nulled & flagged")
print(SEP)
diff_table("email")

# ─────────────────────────────────────────────────────────────────────────────
# 5. FLAGGED ROWS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  ROWS FLAGGED FOR HUMAN REVIEW")
print(SEP)
if "review_required" in cleaned.columns:
    flagged = cleaned[cleaned["review_required"] == True][["customer_id","full_name","review_notes"]]
    if len(flagged):
        print(f"  {len(flagged)} row(s) require attention:\n")
        print(f"  {'ID':<8} {'Name':<22} Review Notes")
        print(f"  {'-'*8} {'-'*22} {'-'*40}")
        for _, row in flagged.iterrows():
            print(f"  {row['customer_id']:<8} {str(row['full_name'])[:21]:<22} {row['review_notes']}")
    else:
        print("  No rows flagged.")

# ─────────────────────────────────────────────────────────────────────────────
# 6. BEFORE vs AFTER MISSING VALUES
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  MISSING VALUES  —  BEFORE vs AFTER")
print(SEP)
before = df_raw.isnull().sum()
after  = cleaned[[c for c in df_raw.columns if c in cleaned.columns]].isnull().sum()
print(f"  {'Column':<22} {'Before':>8}  {'After':>8}  {'Resolved':>10}")
print(f"  {SEP2}")
any_remaining = False
for col in df_raw.columns:
    b = before.get(col, 0)
    a = after.get(col, 0)
    resolved = b - a
    tag = ""
    if a > 0:
        tag = f"  <- {a} still null (unparseable)"
        any_remaining = True
    print(f"  {col:<22} {b:>8}  {a:>8}  {resolved:>8} fixed{tag}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. SAVE
# ─────────────────────────────────────────────────────────────────────────────
cleaned.to_csv(OUTPUT_PATH, index=False)
print(f"\n{SEP}")
print(f"  Cleaned file saved -> {OUTPUT_PATH}")
print(f"{SEP}\n")
