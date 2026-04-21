"""
Side-by-side RAW vs CLEANED comparison for every column.
Makes it easy to visually verify the model's corrections.
"""
import pandas as pd

RAW_PATH     = "data/raw_production_test.csv"
CLEANED_PATH = "data/cleaned_production_test.csv"

raw     = pd.read_csv(RAW_PATH)
cleaned = pd.read_csv(CLEANED_PATH)

COMPARE_COLS = [
    "full_name", "age", "gender", "email",
    "annual_income", "account_created", "last_purchase_date",
    "loyalty_score", "total_orders", "notes",
]

SEP  = "=" * 110
SEP2 = "-" * 110

def cell(v, width=22):
    s = "" if (isinstance(v, float) and pd.isna(v)) else str(v)
    s = s[:width]
    return f"{s:<{width}}"

def status_tag(raw_val, clean_val):
    r = "" if (isinstance(raw_val, float) and pd.isna(raw_val)) else str(raw_val).strip()
    c = "" if (isinstance(clean_val, float) and pd.isna(clean_val)) else str(clean_val).strip()
    if r == "" and c == "":        return "MISSING"
    if r == "" and c != "":        return "FILLED  <--"
    if c == "" and r != "":        return "FLAGGED <--"
    if r.lower() in ("n/a","abc","invalid-date","not-a-date","invalid") and r != c:
        return "FIXED   <--"
    if r != c:                     return "FIXED   <--"
    return "OK"

# ─────────────────────────────────────────────────────────────────────────────
# FULL SIDE-BY-SIDE TABLE (all columns at once, wide format)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  FULL RAW  vs  CLEANED  —  ALL COLUMNS")
print(SEP)

# Print in two passes to keep width manageable: split into groups of 4 cols
col_groups = [COMPARE_COLS[i:i+4] for i in range(0, len(COMPARE_COLS), 4)]

for group in col_groups:
    # Header
    hdr = f"  {'ID':<7} {'Name':<18}"
    for col in group:
        hdr += f"  {'RAW: '+col:<22}  {'CLEANED: '+col:<22}"
    print(f"\n{hdr}")
    print(f"  {'-'*7} {'-'*18}" + ("  " + "-"*22 + "  " + "-"*22) * len(group))
    for i, row in raw.iterrows():
        cid  = str(row["customer_id"])
        name = str(row["full_name"])[:17] if pd.notna(row["full_name"]) else "(missing)"
        line = f"  {cid:<7} {name:<18}"
        for col in group:
            rv = row[col]
            cv = cleaned.loc[i, col]
            line += f"  {cell(rv)}  {cell(cv)}"
        print(line)

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE-ONLY TABLE  (rows where something changed, with status tag)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n\n{SEP}")
print("  CHANGES ONLY  —  what the model corrected  (one row per change)")
print(SEP)
print(f"  {'ID':<7} {'Name':<18} {'Column':<22} {'Raw Value':<28} {'Cleaned Value':<28} Status")
print(f"  {'-'*7} {'-'*18} {'-'*22} {'-'*28} {'-'*28} {'-'*14}")

change_count  = 0
fill_count    = 0
fix_count     = 0
flag_count    = 0

for i, row in raw.iterrows():
    cid  = str(row["customer_id"])
    name = str(row["full_name"])[:17] if pd.notna(row["full_name"]) else "(missing)"
    for col in COMPARE_COLS:
        rv = row[col]
        cv = cleaned.loc[i, col]
        tag = status_tag(rv, cv)
        if tag == "OK":
            continue
        rv_str = "(missing)" if (isinstance(rv, float) and pd.isna(rv)) else str(rv)[:27]
        cv_str = "(null)"    if (isinstance(cv, float) and pd.isna(cv)) else str(cv)[:27]
        print(f"  {cid:<7} {name:<18} {col:<22} {rv_str:<28} {cv_str:<28} {tag}")
        change_count += 1
        if "FILLED"  in tag: fill_count += 1
        if "FIXED"   in tag: fix_count  += 1
        if "FLAGGED" in tag: flag_count += 1

print(f"\n  Total changes : {change_count}  |  Filled: {fill_count}  |  Fixed/Normalised: {fix_count}  |  Flagged/Nulled: {flag_count}")

# ─────────────────────────────────────────────────────────────────────────────
# FLAGGED ROWS
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  FLAGGED ROWS  —  need human review")
print(SEP)
flagged = cleaned[
    cleaned["review_notes"].notna() &
    cleaned["review_notes"].astype(str).str.contains("email|income|date|could not", case=False)
][["customer_id","full_name","review_notes"]]

if len(flagged):
    print(f"  {'ID':<8} {'Name':<22} Review Notes")
    print(f"  {'-'*8} {'-'*22} {'-'*50}")
    for _, r in flagged.iterrows():
        print(f"  {r['customer_id']:<8} {str(r['full_name'])[:21]:<22} {r['review_notes']}")
else:
    print("  None.")

# ─────────────────────────────────────────────────────────────────────────────
# MISSING VALUE SCORECARD
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  MISSING VALUE SCORECARD")
print(SEP)
print(f"  {'Column':<22} {'Before':>8}  {'After':>8}  {'Fixed':>8}  Result")
print(f"  {'-'*22} {'-'*8}  {'-'*8}  {'-'*8}  {'-'*20}")
for col in COMPARE_COLS:
    before = raw[col].isna().sum()
    after  = cleaned[col].isna().sum() if col in cleaned.columns else 0
    fixed  = before - after
    if after == 0 and before > 0:
        result = f"All {before} filled"
    elif after > 0 and before == 0:
        result = f"{after} nulled (invalid data)"
    elif after > 0:
        result = f"{fixed} filled, {after} unparseable remain"
    else:
        result = "No missing"
    print(f"  {col:<22} {before:>8}  {after:>8}  {fixed:>8}  {result}")
