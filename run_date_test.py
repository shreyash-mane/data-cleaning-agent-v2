"""
Date-format stress test for the cleaning pipeline.
Shows: raw formats detected, corrections made, unparseable values flagged.
"""
import pandas as pd
from cleaning_agent.pipeline import CleaningPipeline

RAW_PATH    = "data/raw_date_test.csv"
OUTPUT_PATH = "data/cleaned_date_test.csv"

df_raw = pd.read_csv(RAW_PATH)

print(f"\n{'='*65}")
print("RAW DATA — DATE COLUMNS (all formats as-is)")
print(f"{'='*65}")
date_cols = ["order_date", "delivery_date", "registration_date", "last_login"]
print(df_raw[["order_id"] + date_cols].to_string(index=False))

pipeline = CleaningPipeline()
result   = pipeline.run(df_raw)
cleaned  = result["cleaned_df"]
report   = result["report"]
summary  = result["summary"]

print(f"\n{'='*65}")
print("CLEANING SUMMARY")
print(f"{'='*65}")
for k, v in summary.items():
    print(f"  {k:<35} {v}")

print(f"\n{'='*65}")
print("PER-COLUMN INTELLIGENCE REPORT")
print(f"{'='*65}")
for r in report:
    conf = r.get("confidence")
    conf_str = f"{conf:.2f}" if conf is not None else "N/A"
    ml_act   = r.get("ml_action", r["action"])
    final    = r["action"]
    refined  = r.get("intelligence_refined", False)
    reason   = r.get("intelligence_reason", "")
    intel    = r.get("intelligence", {})

    print(f"\n  [{r['column']}]  type={r['col_type']}")
    if refined:
        print(f"  ML => {ml_act}  |  Intelligence => {final}")
        print(f"  Reason : {reason}")
    else:
        print(f"  Action : {final}  (method={r.get('method','?')}, confidence={conf_str})")
    if intel.get("is_integer_column"):
        print(f"  IntCol : Yes")
    if "actual_skewness" in intel:
        print(f"  Stats  : skew={intel['actual_skewness']}, outliers={intel.get('actual_outliers',0)}")
    if "mode_dominance" in intel:
        print(f"  Stats  : mode_dominance={intel['mode_dominance']*100:.1f}%")
    if r.get("imputed_nulls"):
        print(f"  Imputed: {r['imputed_nulls']} missing values via {r.get('impute_method')} = {r.get('impute_fill_value')}")
    print(f"  Changed: {r.get('cells_changed',0)} cells  |  Flagged: {r.get('cells_flagged',0)} cells")

print(f"\n{'='*65}")
print("DATE COLUMNS — RAW vs CLEANED (side by side)")
print(f"{'='*65}")
for col in date_cols:
    if col not in df_raw.columns or col not in cleaned.columns:
        continue
    raw_s     = df_raw[col].reset_index(drop=True)
    clean_s   = cleaned[col].reset_index(drop=True)
    print(f"\n  [{col}]")
    print(f"  {'Row':<4} {'Raw Format':<28} {'Cleaned (ISO)':<16} Status")
    print(f"  {'-'*4} {'-'*28} {'-'*16} {'-'*15}")
    for i in range(len(raw_s)):
        rv = str(raw_s.iloc[i]) if pd.notna(raw_s.iloc[i]) else "(missing)"
        cv = str(clean_s.iloc[i]) if pd.notna(clean_s.iloc[i]) else "NaN (flagged)"
        status = "OK" if pd.notna(clean_s.iloc[i]) else "FLAGGED"
        if str(raw_s.iloc[i]) != str(clean_s.iloc[i]):
            status = "FIXED" if pd.notna(clean_s.iloc[i]) else "FLAGGED"
        print(f"  {i+1:<4} {rv:<28} {cv:<16} {status}")

print(f"\n{'='*65}")
print("FLAGGED ROWS (review_required = True)")
print(f"{'='*65}")
if "review_required" in cleaned.columns:
    flagged = cleaned[cleaned["review_required"] == True]
    if len(flagged):
        print(flagged[["order_id","customer_name","review_notes"]].to_string(index=False))
    else:
        print("  None — all rows clean!")

print(f"\n{'='*65}")
print("MISSING VALUES AFTER CLEANING")
print(f"{'='*65}")
print(cleaned.isnull().sum().to_string())

cleaned.to_csv(OUTPUT_PATH, index=False)
print(f"\nCleaned data saved -> {OUTPUT_PATH}\n")
