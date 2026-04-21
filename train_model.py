"""
Train the Cleaning Action Classifier  —  Live Progress Edition
==============================================================
Shows real-time per-fold CV, learning curve, and full metrics.
"""

import os, sys, time
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.preprocessing import LabelEncoder
import joblib

# ── helpers ───────────────────────────────────────────────────────────────────
def bar(val, width=40, fill="#", empty="-"):
    filled = int(round(val * width))
    return fill * filled + empty * (width - filled)

def pbar(current, total, prefix="", suffix="", width=35):
    pct   = current / total
    done  = int(pct * width)
    b     = "=" * done + ">" + " " * (width - done - 1) if done < width else "=" * width
    sys.stdout.write(f"\r  {prefix} [{b}] {current:>5}/{total}  {pct*100:5.1f}%  {suffix}")
    sys.stdout.flush()

SEP  = "=" * 68
SEP2 = "-" * 68

# ── Load data ─────────────────────────────────────────────────────────────────
DATA_PATH = "data/training_data.csv"
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"{DATA_PATH} not found. Run generate_training_data.py first.")

print(f"\n{SEP}")
print("  STEP 1 — Loading training data")
print(SEP)

sys.stdout.write("  Reading CSV ..."); sys.stdout.flush()
df = pd.read_csv(DATA_PATH)
print(f"\r  Loaded {len(df):,} rows  x  {len(df.columns)} columns  ({len(df['recommended_action'].unique())} classes)    ")

print(f"\n  Class distribution:")
for action, cnt in df["recommended_action"].value_counts().items():
    print(f"    {action:<25}  {cnt:>5}  {bar(cnt/len(df), 25)}")

# ── Encode ────────────────────────────────────────────────────────────────────
le = LabelEncoder()
df["col_type_encoded"] = le.fit_transform(df["col_type"])
print(f"\n  col_type classes : {list(le.classes_)}")

FEATURES = [
    "col_type_encoded", "missing_rate", "unique_ratio",
    "invalid_numeric_count", "outlier_count_iqr",
    "invalid_date_count", "invalid_email_count", "skewness",
]
X = df[FEATURES].fillna(0)
y = df["recommended_action"]

# ── Train / Test split ────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 2 — Train / Test split  (80% / 20%, stratified)")
print(SEP)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Total rows    : {len(df):,}")
print(f"  Training rows : {len(X_train):,}  ({len(X_train)/len(df)*100:.0f}%)")
print(f"  Test rows     : {len(X_test):,}  ({len(X_test)/len(df)*100:.0f}%)")
print(f"\n  Per-class breakdown:")
print(f"  {'Class':<25} {'Train':>7}  {'Test':>6}")
print(f"  {SEP2}")
for cls in sorted(y.unique()):
    tr = (y_train == cls).sum()
    te = (y_test  == cls).sum()
    print(f"  {cls:<25} {tr:>7}  {te:>6}")

# ── Learning curve ────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 3 — Learning curve  (how accuracy grows with more data)")
print(SEP)
print(f"  {'Data %':>7}  {'Train rows':>11}  {'Train Acc':>10}  {'Test Acc':>9}  {'Gap':>6}  Curve")
print(f"  {SEP2}")

STEPS = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
lc_model = DecisionTreeClassifier(
    max_depth=10, min_samples_leaf=4,
    min_samples_split=8, class_weight="balanced", random_state=42
)
for step in STEPS:
    n = max(int(len(X_train) * step), 9)
    Xs = X_train.iloc[:n]; ys = y_train.iloc[:n]
    lc_model.fit(Xs, ys)
    tr_acc = accuracy_score(ys,      lc_model.predict(Xs))
    te_acc = accuracy_score(y_test,  lc_model.predict(X_test))
    gap    = tr_acc - te_acc
    curve  = bar(te_acc, 30)
    print(f"  {step*100:>6.0f}%  {n:>11,}  {tr_acc*100:>9.2f}%  {te_acc*100:>8.2f}%  {gap*100:>5.2f}%  {curve}")
    time.sleep(0.05)

# ── Train final model ─────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 4 — Training final model on full training set")
print(SEP)

model = DecisionTreeClassifier(
    max_depth=10, min_samples_leaf=4,
    min_samples_split=8, class_weight="balanced", random_state=42
)
sys.stdout.write("  Fitting ..."); sys.stdout.flush()
t0 = time.time()
model.fit(X_train, y_train)
elapsed = time.time() - t0
print(f"\r  Done in {elapsed:.2f}s  |  depth={model.get_depth()}  leaves={model.get_n_leaves()}    ")

y_train_pred = model.predict(X_train)
y_test_pred  = model.predict(X_test)
train_acc    = accuracy_score(y_train, y_train_pred)
test_acc     = accuracy_score(y_test,  y_test_pred)
gap          = train_acc - test_acc

print(f"\n  Train accuracy : {train_acc*100:.2f}%  {bar(train_acc, 30)}")
print(f"  Test  accuracy : {test_acc*100:.2f}%  {bar(test_acc, 30)}")
print(f"  Gap            : {gap*100:.2f}%  {'GOOD (< 3%)' if gap < 0.03 else 'OK (3-7%)' if gap < 0.07 else 'OVERFIT?'}")

# ── Live 10-fold CV ───────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 5 — 10-fold Cross-Validation  (live per-fold results)")
print(SEP)
print(f"  {'Fold':>5}  {'F1-macro':>9}  {'Accuracy':>9}  Progress")
print(f"  {SEP2}")

cv         = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
f1_scores  = []
acc_scores = []

for fold_idx, (tr_idx, val_idx) in enumerate(cv.split(X, y), 1):
    Xf_tr, Xf_val = X.iloc[tr_idx], X.iloc[val_idx]
    yf_tr, yf_val = y.iloc[tr_idx], y.iloc[val_idx]

    fold_model = DecisionTreeClassifier(
        max_depth=10, min_samples_leaf=4,
        min_samples_split=8, class_weight="balanced", random_state=42
    )
    fold_model.fit(Xf_tr, yf_tr)
    pred   = fold_model.predict(Xf_val)
    f1     = f1_score(yf_val, pred, average="macro")
    acc    = accuracy_score(yf_val, pred)
    f1_scores.append(f1)
    acc_scores.append(acc)

    running_f1  = np.mean(f1_scores)
    running_acc = np.mean(acc_scores)
    prog_bar    = bar(fold_idx / 10, 20)

    print(f"  {fold_idx:>5}  {f1*100:>8.2f}%  {acc*100:>8.2f}%  [{prog_bar}]  "
          f"running avg F1={running_f1*100:.2f}%")
    time.sleep(0.05)

print(f"  {SEP2}")
print(f"  {'Mean':>5}  {np.mean(f1_scores)*100:>8.2f}%  {np.mean(acc_scores)*100:>8.2f}%")
print(f"  {'Std':>5}  {np.std(f1_scores)*100:>8.2f}%  {np.std(acc_scores)*100:>8.2f}%")
print(f"  {'Min':>5}  {np.min(f1_scores)*100:>8.2f}%  {np.min(acc_scores)*100:>8.2f}%")
print(f"  {'Max':>5}  {np.max(f1_scores)*100:>8.2f}%  {np.max(acc_scores)*100:>8.2f}%")

# ── Training set report ───────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 6 — Full evaluation reports")
print(SEP)
print("  [ TRAINING SET ]")
print(f"  Accuracy : {train_acc*100:.2f}%")
print(classification_report(y_train, y_train_pred, zero_division=0))

print("  [ TEST SET ]")
print(f"  Accuracy : {test_acc*100:.2f}%")
print(classification_report(y_test, y_test_pred, zero_division=0))

# ── Confusion matrix ──────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 7 — Confusion matrix  (test set,  rows=actual  cols=predicted)")
print(SEP)
labels = sorted(y.unique())
cm     = confusion_matrix(y_test, y_test_pred, labels=labels)
short  = [l[:8] for l in labels]
header = f"  {'':25}" + "".join(f"{s:>10}" for s in short)
print(header)
print("  " + "-" * (25 + 10 * len(labels)))
for i, actual in enumerate(labels):
    row = f"  {actual[:24]:<25}" + "".join(
        f"\033[1m{cm[i][j]:>10}\033[0m" if i == j else f"{cm[i][j]:>10}"
        for j in range(len(labels))
    )
    print(row)

# ── Feature importances ───────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 8 — Feature importances")
print(SEP)
importances = sorted(zip(FEATURES, model.feature_importances_), key=lambda x: x[1], reverse=True)
for rank, (feat, imp) in enumerate(importances, 1):
    print(f"  #{rank}  {feat:<30}  {imp:.4f}  {bar(imp, 35)}")

# ── Overfitting check ─────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 9 — Overfitting check")
print(SEP)
print(f"  Train accuracy    : {train_acc*100:.2f}%")
print(f"  Test  accuracy    : {test_acc*100:.2f}%")
print(f"  10-fold CV F1     : {np.mean(f1_scores)*100:.2f}% +/- {np.std(f1_scores)*100:.2f}%")
print(f"  Train-Test gap    : {gap*100:.2f}%")
if gap < 0.03:
    verdict = "GOOD  — generalises well  (gap < 3%)"
elif gap < 0.07:
    verdict = "OK    — slight overfit, acceptable  (gap 3-7%)"
else:
    verdict = "WARN  — possible overfit  (gap > 7%)"
print(f"  Verdict           : {verdict}")

# ── Save ──────────────────────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  STEP 10 — Saving model")
print(SEP)

os.makedirs("data", exist_ok=True)
bundle = {"model": model, "label_encoder": le, "features": FEATURES}

sys.stdout.write("  Saving model.joblib ..."); sys.stdout.flush()
joblib.dump(bundle, "data/model.joblib")
print("\r  Saved  -> data/model.joblib               ")

tree_text = export_text(model, feature_names=FEATURES, max_depth=4)
with open("data/tree_summary.txt", "w") as f:
    f.write(tree_text)
print("  Saved  -> data/tree_summary.txt")

print(f"\n{SEP}")
print("  TRAINING COMPLETE")
print(SEP)
print(f"  Rows trained on   : {len(X_train):,}")
print(f"  Test accuracy     : {test_acc*100:.2f}%")
print(f"  CV macro-F1       : {np.mean(f1_scores)*100:.2f}% +/- {np.std(f1_scores)*100:.2f}%")
print(f"  Train-test gap    : {gap*100:.2f}%  ({verdict})")
print()
