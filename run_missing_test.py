"""
Test the cleaning pipeline on raw_missing_test.csv
Focus: missing age, salary, and date columns — verify model decisions.
"""
import pandas as pd
from cleaning_agent.pipeline import CleaningPipeline

RAW_PATH    = "data/raw_missing_test.csv"
OUTPUT_PATH = "data/cleaned_missing_test.csv"

df_raw = pd.read_csv(RAW_PATH)

# ── Raw overview ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("RAW DATA OVERVIEW")
print("=" * 65)
print(f"  Rows     : {len(df_raw)}  |  Columns: {len(df_raw.columns)}")

print(f"\n  Missing values per column (BEFORE cleaning):")
missing = df_raw.isnull().sum()
for col, cnt in missing.items():
    pct = cnt / len(df_raw) * 100
    bar = "#" * int(pct / 4)
    print(f"    {col:<22} {cnt:>3} missing  ({pct:5.1f}%)  {bar}")

# Also show N/A strings that pandas didn't catch as NaN
print(f"\n  String 'N/A' or 'abc' in salary (raw strings):")
print(f"    {df_raw['salary'].tolist()}")

# ── Run pipeline ──────────────────────────────────────────────────────────────
pipeline = CleaningPipeline()
result   = pipeline.run(df_raw)
cleaned  = result["cleaned_df"]
report   = result["report"]
summary  = result["summary"]

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("PIPELINE SUMMARY")
print("=" * 65)
for k, v in summary.items():
    print(f"  {k:<35} {v}")

# ── Per-column intelligence report ───────────────────────────────────────────
print("\n" + "=" * 65)
print("PER-COLUMN DECISIONS  (ML + Intelligence Layer)")
print("=" * 65)
for r in report:
    conf     = r.get("confidence")
    conf_str = f"{conf:.2f}" if conf is not None else "N/A"
    ml_act   = r.get("ml_action", r["action"])
    final    = r["action"]
    refined  = r.get("intelligence_refined", False)
    reason   = r.get("intelligence_reason", "")
    intel    = r.get("intelligence", {})

    print(f"\n  [{r['column']}]  detected-type={r['col_type']}")
    if refined:
        print(f"  ML predicted : {ml_act}")
        print(f"  Refined to   : {final}  <-- intelligence layer")
        print(f"  Reason       : {reason}")
    else:
        print(f"  Action       : {final}  (method={r.get('method','?')}, confidence={conf_str})")

    if intel.get("is_integer_column"):
        print(f"  Integer col  : Yes — values will be whole numbers")
    if "actual_skewness" in intel:
        print(f"  Data stats   : skew={intel['actual_skewness']}, outliers={intel.get('actual_outliers',0)}")
    if "mode_dominance" in intel:
        print(f"  Data stats   : mode dominance={intel['mode_dominance']*100:.1f}%")
    if r.get("fill_value") is not None:
        print(f"  Fill value   : {r['fill_value']}")
    if r.get("imputed_nulls"):
        print(f"  Imputed      : {r['imputed_nulls']} missing via {r.get('impute_method')} (fill={r.get('impute_fill_value')})")
    print(f"  Changed      : {r.get('cells_changed', 0)} cells   Flagged: {r.get('cells_flagged', 0)} cells")

# ── Focus: age ────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("AGE — RAW vs CLEANED")
print("=" * 65)
raw_age   = df_raw["age"].reset_index(drop=True)
clean_age = cleaned["age"].reset_index(drop=True)
print(f"  {'ID':<5} {'Name':<20} {'Raw Age':<12} {'Cleaned Age':<12} Status")
print(f"  {'-'*5} {'-'*20} {'-'*12} {'-'*12} {'-'*10}")
for i in range(len(df_raw)):
    name = df_raw["name"].iloc[i]
    rv   = str(raw_age.iloc[i]) if pd.notna(raw_age.iloc[i]) else "(missing)"
    cv   = str(clean_age.iloc[i])
    status = "FILLED" if rv == "(missing)" else ("CHANGED" if str(rv) != cv else "OK")
    print(f"  {i+1:<5} {name:<20} {rv:<12} {cv:<12} {status}")

# ── Focus: salary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("SALARY — RAW vs CLEANED")
print("=" * 65)
raw_sal   = df_raw["salary"].reset_index(drop=True)
clean_sal = cleaned["salary"].reset_index(drop=True)
print(f"  {'ID':<5} {'Name':<20} {'Raw Salary':<14} {'Cleaned':<14} Status")
print(f"  {'-'*5} {'-'*20} {'-'*14} {'-'*14} {'-'*10}")
for i in range(len(df_raw)):
    name = df_raw["name"].iloc[i]
    rv   = str(raw_sal.iloc[i]) if pd.notna(raw_sal.iloc[i]) else "(missing)"
    cv   = str(clean_sal.iloc[i]) if pd.notna(clean_sal.iloc[i]) else "NaN"
    status = "OK"
    if rv in ("(missing)", "nan"):
        status = "FILLED"
    elif rv in ("abc", "N/A"):
        status = "CONVERTED+FILLED"
    elif rv != cv:
        status = "CHANGED"
    print(f"  {i+1:<5} {name:<20} {rv:<14} {cv:<14} {status}")

# ── Focus: dates ──────────────────────────────────────────────────────────────
for date_col in ["join_date", "last_review_date"]:
    if date_col not in df_raw.columns:
        continue
    print(f"\n" + "=" * 65)
    print(f"{date_col.upper()} — RAW vs CLEANED")
    print("=" * 65)
    raw_d   = df_raw[date_col].reset_index(drop=True)
    clean_d = cleaned[date_col].reset_index(drop=True)
    print(f"  {'ID':<5} {'Name':<20} {'Raw':<25} {'Cleaned':<14} Status")
    print(f"  {'-'*5} {'-'*20} {'-'*25} {'-'*14} {'-'*10}")
    for i in range(len(df_raw)):
        name = df_raw["name"].iloc[i]
        rv   = str(raw_d.iloc[i]) if pd.notna(raw_d.iloc[i]) else "(missing)"
        cv   = str(clean_d.iloc[i]) if pd.notna(clean_d.iloc[i]) else "NaN (flagged)"
        if rv == "(missing)":
            status = "MISSING"
        elif cv == "NaN (flagged)":
            status = "FLAGGED"
        elif rv != cv:
            status = "FIXED"
        else:
            status = "OK"
        print(f"  {i+1:<5} {name:<20} {rv:<25} {cv:<14} {status}")

# ── Flagged rows ──────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("ROWS FLAGGED FOR REVIEW")
print("=" * 65)
if "review_required" in cleaned.columns:
    flagged = cleaned[cleaned["review_required"] == True]
    if len(flagged):
        print(f"  {len(flagged)} rows flagged:\n")
        print(f"  {'ID':<5} {'Name':<20} Notes")
        print(f"  {'-'*5} {'-'*20} {'-'*40}")
        for _, row in flagged.iterrows():
            print(f"  {row['employee_id']:<5} {row['name']:<20} {row['review_notes']}")
    else:
        print("  None.")

# ── After cleaning missing values ─────────────────────────────────────────────
print("\n" + "=" * 65)
print("MISSING VALUES AFTER CLEANING")
print("=" * 65)
after = cleaned.isnull().sum()
for col, cnt in after.items():
    status = "OK" if cnt == 0 else f"{cnt} remaining"
    print(f"  {col:<25} {status}")

cleaned.to_csv(OUTPUT_PATH, index=False)
print(f"\nCleaned file saved -> {OUTPUT_PATH}\n")
