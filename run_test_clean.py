"""
Test the cleaning pipeline on raw_test_data.csv and print a detailed report.
"""
import pandas as pd
from cleaning_agent.pipeline import CleaningPipeline

RAW_PATH    = "data/raw_test_data.csv"
OUTPUT_PATH = "data/cleaned_test_output.csv"

df_raw = pd.read_csv(RAW_PATH)
print(f"\n{'='*60}")
print("RAW DATA OVERVIEW")
print(f"{'='*60}")
print(f"Rows: {len(df_raw)}  |  Columns: {len(df_raw.columns)}")
print(f"\nMissing values per column (before cleaning):")
print(df_raw.isnull().sum().to_string())

pipeline = CleaningPipeline()
result   = pipeline.run(df_raw)

cleaned_df = result["cleaned_df"]
summary    = result["summary"]
report     = result["report"]

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("CLEANING SUMMARY")
print(f"{'='*60}")
for k, v in summary.items():
    print(f"  {k:<35} {v}")

# ── Per-column report ─────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("PER-COLUMN REPORT")
print(f"{'='*60}")
for r in report:
    conf = r.get("confidence")
    conf_str = f"{conf:.2f}" if conf is not None else "N/A"
    ml_act = r.get("ml_action", r["action"])
    final_act = r["action"]
    refined = r.get("intelligence_refined", False)
    reason = r.get("intelligence_reason", "")
    intel = r.get("intelligence", {})

    print(f"\n  Column : {r['column']}")
    print(f"  Type   : {r['col_type']}")
    if refined:
        print(f"  ML said: {ml_act}  =>  Intelligence refined to: {final_act}")
        print(f"  Reason : {reason}")
    else:
        print(f"  Action : {final_act}  (ML={ml_act}, method={r.get('method','?')}, confidence={conf_str})")

    if intel.get("is_integer_column"):
        print(f"  IntCol : Yes — fill value rounded to integer")
    if "actual_skewness" in intel:
        print(f"  DataStats: skew={intel['actual_skewness']}, outliers={intel.get('actual_outliers',0)}")
    if "mode_dominance" in intel:
        print(f"  DataStats: mode_dominance={intel['mode_dominance']*100:.1f}%")
    print(f"  Changed: {r.get('cells_changed',0)} cells  |  Flagged: {r.get('cells_flagged',0)} cells")

# ── Missing values after ──────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("MISSING VALUES AFTER CLEANING")
print(f"{'='*60}")
print(cleaned_df.isnull().sum().to_string())

# ── Side-by-side diff for key columns ────────────────────────────────────────
print(f"\n{'='*60}")
print("SIDE-BY-SIDE: RAW vs CLEANED (first 40 rows, key columns)")
print(f"{'='*60}")

key_cols = ["age", "salary", "signup_date", "email", "score"]
for col in key_cols:
    if col not in df_raw.columns:
        continue
    raw_series     = df_raw[col].reset_index(drop=True)
    cleaned_series = cleaned_df[col].reset_index(drop=True) if col in cleaned_df.columns else None
    # check for a __flag__ column
    flag_col = f"{col}__flag__"
    flag_series = cleaned_df[flag_col].reset_index(drop=True) if flag_col in cleaned_df.columns else None

    changes = []
    for i in range(min(len(raw_series), len(cleaned_series) if cleaned_series is not None else 0)):
        rv = raw_series.iloc[i]
        cv = cleaned_series.iloc[i] if cleaned_series is not None else rv
        flag = flag_series.iloc[i] if flag_series is not None else ""
        if str(rv) != str(cv) or flag:
            changes.append((i + 1, rv, cv, flag))

    if changes:
        print(f"\n  [{col}]")
        print(f"  {'Row':<5} {'Raw':<25} {'Cleaned':<25} {'Flag'}")
        print(f"  {'-'*5} {'-'*25} {'-'*25} {'-'*20}")
        for row_num, rv, cv, flag in changes:
            print(f"  {row_num:<5} {str(rv):<25} {str(cv):<25} {str(flag)}")

# ── Save ──────────────────────────────────────────────────────────────────────
cleaned_df.to_csv(OUTPUT_PATH, index=False)
print(f"\n{'='*60}")
print(f"Cleaned data saved -> {OUTPUT_PATH}")
print(f"{'='*60}\n")
