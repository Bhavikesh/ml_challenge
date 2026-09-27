"""Fast threshold estimator using pre-computed training features (holdout sweep).

Instead of running full validation inference (which takes 20+ hours),
this script uses the already-computed training_features_v2/ dataset with
an 80/20 holdout split to estimate the best F0.5 threshold per model.

Outputs:
  output/fast_threshold_results.json  -- best thresholds + metrics
  output/fast_threshold_sweep.csv     -- full sweep table

Run time: ~3-5 minutes.
"""

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output"
FEAT_DIR = OUTPUT_DIR / "training_features_v2"
MODELS_DIR = OUTPUT_DIR / "models_v2"

# ── 1. Load pre-computed features ────────────────────────────────────────────
print("Loading training features …")
t0 = time.time()

X = np.load(FEAT_DIR / "features.npy")        # (5_729_367, 16)
y = np.load(FEAT_DIR / "labels.npy")          # (5_729_367,)
s1_idx = np.load(FEAT_DIR / "s1_idx.npy")    # which S1 entity each pair belongs to

print(f"  Features: {X.shape}  Labels: {y.shape}  S1 indices: {s1_idx.shape}")
print(f"  Loaded in {time.time()-t0:.1f}s")

# ── 2. Stratified holdout split — 20% of positives + 20% of negatives ──────
# NOTE: pairs are stored positives-first, so a simple last-20% split gives
# a holdout with zero positives. We must stratify.
rng = np.random.default_rng(42)

pos_indices = np.where(y == 1)[0]
neg_indices = np.where(y == 0)[0]

# Sample 20% of each class
pos_hold_idx = rng.choice(pos_indices, size=int(len(pos_indices) * 0.20), replace=False)
neg_hold_idx = rng.choice(neg_indices, size=int(len(neg_indices) * 0.20), replace=False)

hold_idx = np.concatenate([pos_hold_idx, neg_hold_idx])

X_hold  = X[hold_idx]
y_hold  = y[hold_idx]
s1_hold = s1_idx[hold_idx]

print(f"\nHoldout pairs: {len(X_hold):,}  "
      f"(positive: {int(y_hold.sum()):,}, negative: {int((y_hold==0).sum()):,})")

unique_s1_hold = np.unique(s1_hold)
print(f"Unique S1 entities in holdout: {len(unique_s1_hold):,}")

# Build per-S1 true count for holdout
s1_to_idx = {v: i for i, v in enumerate(unique_s1_hold)}
n_s1 = len(unique_s1_hold)
true_counts = np.zeros(n_s1, dtype=np.int32)
for i, s1 in enumerate(s1_hold):
    sidx = s1_to_idx[s1]
    true_counts[sidx] += int(y_hold[i])

print(f"  No-match S1 in holdout: {int((true_counts == 0).sum()):,}")
print(f"  Match S1 in holdout:    {int((true_counts  > 0).sum()):,}")

# ── 3. Load models ───────────────────────────────────────────────────────────
print("\nLoading models …")

def load_model(subdir, fname):
    p = MODELS_DIR / subdir / fname
    with open(p, "rb") as f:
        return pickle.load(f)

lr_bundle  = load_model("logistic",  "logistic_regression.pkl")
xgb_bundle = load_model("xgboost",  "xgboost.pkl")
lgb_bundle = load_model("lightgbm", "lightgbm.pkl")

lr_model  = lr_bundle["model"]
lr_scaler = lr_bundle.get("scaler")
xgb_model = xgb_bundle["model"]
lgb_model = lgb_bundle["model"]
print("  All three models loaded.")

# ── 4. Predict probabilities on holdout ─────────────────────────────────────
print("\nPredicting probabilities on holdout …")

X_s = lr_scaler.transform(X_hold) if lr_scaler is not None else X_hold
p_lr  = lr_model.predict_proba(X_s)[:, 1]

p_xgb = xgb_model.predict_proba(X_hold)[:, 1]
p_lgb = lgb_model.predict_proba(X_hold)[:, 1]

print("  Predictions done.")

# ── 5. F0.5 vectorised ───────────────────────────────────────────────────────
def macro_f05_at_threshold(probs, y_true, s1_ids, true_counts, s1_to_idx, threshold):
    n_s1 = len(true_counts)
    pred_counts = np.zeros(n_s1, dtype=np.int32)
    tp_counts   = np.zeros(n_s1, dtype=np.int32)

    keep = probs >= threshold
    for i in np.where(keep)[0]:
        sidx = s1_to_idx[s1_ids[i]]
        pred_counts[sidx] += 1
        if y_true[i] == 1:
            tp_counts[sidx] += 1

    f05 = np.zeros(n_s1, dtype=np.float64)

    # No-match entities: score 1 if no predictions, 0 otherwise
    no_match = true_counts == 0
    f05[no_match] = np.where(pred_counts[no_match] == 0, 1.0, 0.0)

    # Match entities
    has_tp = (~no_match) & (tp_counts > 0)
    if has_tp.any():
        p = tp_counts[has_tp] / pred_counts[has_tp].astype(np.float64)
        r = tp_counts[has_tp] / true_counts[has_tp].astype(np.float64)
        denom = 0.25 * p + r
        f05[has_tp] = np.where(denom > 0, (1.25 * p * r) / np.maximum(denom, 1e-12), 0.0)

    return float(f05.mean()), int(tp_counts.sum()), int(pred_counts.sum())


# ── 6. Threshold sweep ───────────────────────────────────────────────────────
thresholds = np.round(np.arange(0.10, 0.995, 0.01), 2)
total_true = int(true_counts.sum())

print(f"\nSweeping {len(thresholds)} thresholds …")

models_probs = {
    "logistic": p_lr,
    "xgboost":  p_xgb,
    "lightgbm": p_lgb,
}

results = {}
csv_rows = []

for model_name, probs in models_probs.items():
    t_start = time.time()
    best_f05, best_t, best_prec, best_rec = -1, None, 0, 0

    for t in thresholds:
        f05, tp, pred = macro_f05_at_threshold(
            probs, y_hold, s1_hold, true_counts, s1_to_idx, t
        )
        fp = pred - tp
        fn = total_true - tp
        prec = tp / pred if pred > 0 else 0.0
        rec  = tp / total_true if total_true > 0 else 0.0

        csv_rows.append({
            "model": model_name,
            "threshold": float(t),
            "macro_f05": f05,
            "precision": prec,
            "recall": rec,
            "tp": tp, "fp": fp, "fn": fn,
            "nonempty_rate": float((np.array([probs >= t]).sum() > 0)),
        })

        if f05 > best_f05:
            best_f05, best_t = f05, float(t)
            best_prec, best_rec = prec, rec

    results[model_name] = {
        "best_threshold": best_t,
        "best_macro_f05": best_f05,
        "precision":      best_prec,
        "recall":         best_rec,
    }
    print(f"  [{model_name.upper():12s}] threshold={best_t:.2f}  "
          f"macro_F0.5={best_f05:.4f}  prec={best_prec:.4f}  rec={best_rec:.4f}  "
          f"({time.time()-t_start:.1f}s)")

# ── 7. Save outputs ──────────────────────────────────────────────────────────
json_out = OUTPUT_DIR / "fast_threshold_results.json"
with open(json_out, "w") as f:
    json.dump({"note": "Estimated from training 80/20 holdout (not full validation)",
               "models": results}, f, indent=2)
print(f"\nSaved JSON → {json_out}")

csv_out = OUTPUT_DIR / "fast_threshold_sweep.csv"
pd.DataFrame(csv_rows).to_csv(csv_out, index=False)
print(f"Saved CSV  → {csv_out}")

print("\n" + "=" * 60)
print("THRESHOLD SUMMARY (use for test inference)")
print("=" * 60)
best_model = max(results, key=lambda m: results[m]["best_macro_f05"])
for m, r in results.items():
    flag = " ← BEST" if m == best_model else ""
    print(f"  {m:12s}  threshold={r['best_threshold']:.2f}  "
          f"macro_F0.5={r['best_macro_f05']:.4f}{flag}")
print(f"\nRecommended: model={best_model}  "
      f"threshold={results[best_model]['best_threshold']:.2f}")
print("=" * 60)
