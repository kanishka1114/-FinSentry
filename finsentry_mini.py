"""FinSentry-Mini: verify the saved FinSentry experiment by re-scoring its test period.

No model is trained here and nothing is written. The script reads the saved models in
models/ and the saved results in outputs/metrics.json, rebuilds the test period from the
PaySim CSV, and checks that every test figure matches the saved results.

The PaySim CSV (about 471 MB) is NOT included in this repository. Download it from
https://www.kaggle.com/datasets/ealaxi/paysim1 and pass its path, or place it in data/:

    python finsentry_mini.py [path/to/PS_20174392719_1491204439457_log.csv]
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parent
CSV_NAME = "PS_20174392719_1491204439457_log.csv"
METRICS_PATH = ROOT / "outputs" / "metrics.json"
MODEL_DIR = ROOT / "models"
MODEL_FILES = {"Logistic Regression": "logistic_regression.joblib",
               "Random Forest": "random_forest.joblib"}

# Settings of the saved experiment (exp-0781f6ac1d); they must not be changed here.
SAMPLE_ROWS = 1_000_000
SEED = 42
TRAIN_SHARE = 0.70
VALIDATION_SHARE = 0.15
TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
FEATURES = ["amount", "log_amount", "hour_sin", "hour_cos", "type"]


def load_paysim(csv_path):
    """Read the four columns the models need and drop rows with invalid values."""
    df = pd.read_csv(csv_path, usecols=["step", "type", "amount", "isFraud"])
    valid = (df["type"].isin(TRANSACTION_TYPES) & (df["amount"] >= 0)
             & (df["step"] >= 1) & df["isFraud"].isin([0, 1]))
    return df[valid].reset_index(drop=True)


def uniform_sample(df, n_rows, seed):
    """Pick n_rows at random without looking at the fraud label, then sort by time."""
    chosen = np.sort(np.random.default_rng(seed).choice(len(df), n_rows, replace=False))
    return df.iloc[chosen].sort_values("step", kind="stable").reset_index(drop=True)


def chronological_split(steps, train_share, validation_share):
    """Split by whole hourly steps so train < validation < test in time.

    Each boundary is the step whose cumulative row share is closest to the target share.
    """
    unique_steps, counts = np.unique(steps, return_counts=True)
    cumulative_share = np.cumsum(counts) / counts.sum()

    def last_step_index(target):
        i = int(np.searchsorted(cumulative_share, target, side="left"))
        if i > 0 and (i >= len(cumulative_share)
                      or target - cumulative_share[i - 1] <= cumulative_share[i] - target):
            i -= 1
        return i

    train_end = min(last_step_index(train_share), len(unique_steps) - 3)
    validation_end = min(max(last_step_index(train_share + validation_share), train_end + 1),
                         len(unique_steps) - 2)
    train_last, validation_last = unique_steps[train_end], unique_steps[validation_end]
    return {"train": steps <= train_last,
            "validation": (steps > train_last) & (steps <= validation_last),
            "test": steps > validation_last}


def build_features(df):
    """Amount, log amount, hour of day as a circle, and transaction type. Raw step is not used."""
    hour = df["step"] % 24
    return pd.DataFrame({
        "amount": df["amount"],
        "log_amount": np.log1p(df["amount"]),
        "hour_sin": np.sin(2 * np.pi * hour / 24),
        "hour_cos": np.cos(2 * np.pi * hour / 24),
        "type": df["type"].astype(str),
    })[FEATURES]


def test_metrics(y_true, scores, threshold):
    """Metrics at the saved threshold, plus threshold-free PR-AUC and ROC-AUC."""
    alerts = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, alerts, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {
        "true_positive": int(tp), "false_positive": int(fp),
        "false_negative": int(fn), "true_negative": int(tn),
        "precision": precision, "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if tp + fn else None,
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
    }


def differences(calculated, saved_model):
    """Compare with outputs/metrics.json; counts must match exactly, rates to 1e-9."""
    saved = {**saved_model["strategies"]["max_f1"]["test"], **saved_model["test_ranking"]}
    out = []
    for key, value in calculated.items():
        expected = saved[key]
        same = value == expected if isinstance(value, int) else abs(value - expected) <= 1e-9
        if not same:
            out.append(f"{key}: re-scored {value}, saved {expected}")
    return out


def period_line(name, steps, y):
    return (f"  {name:<11} steps {steps.min():>3}-{steps.max():<3}  {len(y):>9,} transactions  "
            f"{int(y.sum()):>5,} fraud  ({y.mean():.2%} fraud)")


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / CSV_NAME
    if not csv_path.exists():
        sys.exit(f"PaySim CSV not found at {csv_path}. It is not included in this repository; "
                 "download it (see the README) and pass its path.")
    for path in [METRICS_PATH, *(MODEL_DIR / f for f in MODEL_FILES.values())]:
        if not path.exists():
            sys.exit(f"Missing file: {path}")

    saved = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    print("FinSentry-Mini: results of the SAVED experiment "
          f"{saved['run']['experiment_id']} (run {saved['run']['run_id']})")
    print("No model is trained. The saved models are re-scored on the same test period.")
    print("PaySim is simulated data; these results do not show real-world performance.\n")

    print(f"Loading {csv_path.name} ...")
    full = load_paysim(csv_path)
    sample = uniform_sample(full, SAMPLE_ROWS, SEED)
    periods = chronological_split(sample["step"].to_numpy(), TRAIN_SHARE, VALIDATION_SHARE)
    print(f"\nDATASET: {len(full):,} transactions, {int(full['isFraud'].sum()):,} fraud "
          f"({full['isFraud'].mean():.2%})")
    print(f"SAMPLE:  {len(sample):,} random transactions (seed {SEED}), "
          f"{int(sample['isFraud'].sum()):,} fraud")
    print("SPLIT (by time; models were fitted on train, thresholds chosen on validation):")
    for name, mask in periods.items():
        print(period_line(name, sample["step"][mask], sample["isFraud"][mask]))

    test = sample[periods["test"]]
    X_test, y_test = build_features(test), test["isFraud"].to_numpy()
    print(f"\nTEST RESULTS (saved models at their saved thresholds; random-guess PR-AUC "
          f"= test fraud rate = {y_test.mean():.4f})")

    problems = []
    for name, file_name in MODEL_FILES.items():
        bundle = joblib.load(MODEL_DIR / file_name)
        if name == "Random Forest":
            # Saved with n_jobs=2; parallel summing can move a score sitting exactly on the
            # threshold by one bit. Serial scoring (in memory only) reproduces the saved result.
            bundle["pipeline"].set_params(model__n_jobs=1)
        scores = bundle["pipeline"].predict_proba(X_test)[:, 1]
        m = test_metrics(y_test, scores, bundle["threshold"])
        print(f"\n  {name} (alert if score >= {bundle['threshold']:.3f})")
        print(f"    Fraud caught:       {m['true_positive']:>7,}   Missed fraud:   {m['false_negative']:>7,}")
        print(f"    False alerts:       {m['false_positive']:>7,}   Legitimate, not alerted: {m['true_negative']:>7,}")
        print(f"    Precision {m['precision']:.3f} (share of alerts that were fraud)")
        print(f"    Recall    {m['recall']:.3f} (share of fraud that was alerted)")
        print(f"    F1        {m['f1']:.3f}   PR-AUC {m['pr_auc']:.3f}   ROC-AUC {m['roc_auc']:.3f}")
        problems += [f"{name} {d}" for d in differences(m, saved["models"][name])]

    if problems:
        print("\nMISMATCH with outputs/metrics.json; these are NOT the saved results:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("\nCheck: every figure above matches the saved outputs/metrics.json.")


if __name__ == "__main__":
    main()
