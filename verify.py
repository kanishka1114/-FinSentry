"""Check that the saved model reproduces every saved test result. Trains nothing, writes nothing.

    python verify.py [path/to/PS_20174392719_1491204439457_log.csv]

Rebuilds the same sample and time split from the PaySim CSV (not included in this
repository), scores the test period with models/random_forest.joblib at the saved
threshold, and compares the result with outputs/results.json.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import average_precision_score

import finsentry as fs

ROOT = Path(__file__).resolve().parent


def compare(label, calculated, saved, problems):
    for key, expected in saved.items():
        value = calculated[key]
        same = value == expected if isinstance(expected, int) or expected is None \
            else abs(value - expected) <= 1e-9
        if not same:
            problems.append(f"{label} {key}: re-scored {value}, saved {expected}")


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / fs.CSV_NAME
    if not csv_path.exists():
        sys.exit(f"PaySim CSV not found at {csv_path}. It is not included in this repository; "
                 "download it (see the README) and pass its path.")
    saved = json.loads((ROOT / "outputs" / "results.json").read_text(encoding="utf-8"))
    bundle = joblib.load(ROOT / "models" / "random_forest.joblib")

    full = fs.load_paysim(csv_path)
    sample = fs.uniform_sample(full, fs.SAMPLE_ROWS, fs.SEED)
    periods = fs.chronological_split(sample["step"].to_numpy(), fs.TRAIN_SHARE, fs.VALIDATION_SHARE)
    test = sample[periods["test"]].reset_index(drop=True)
    y_test = test["isFraud"].to_numpy()

    problems = []
    if bundle["threshold"] != saved["model"]["threshold"]:
        problems.append("model file threshold differs from results.json")
    scores = bundle["pipeline"].predict_proba(fs.features_from_paysim(test))[:, 1]
    rf = {**fs.alert_metrics(y_test, scores >= bundle["threshold"]),
          "pr_auc": float(average_precision_score(y_test, scores))}
    compare("Random Forest", rf, saved["random_forest"]["test"], problems)
    compare("Rule", fs.alert_metrics(y_test, fs.rule_alerts(test)), saved["rule"]["test"], problems)
    for name, ex in saved["examples"].items():
        score = float(bundle["pipeline"].predict_proba(fs.build_features(
            ex["amount"], ex["type"], ex["hour"], ex["sender_balance_before"],
            ex["sender_balance_after"]))[:, 1][0])
        if abs(score - ex["score"]) > 1e-12 or (score >= bundle["threshold"]) != ex["alert"]:
            problems.append(f"example {name}: score {score}, saved {ex['score']}")

    print(f"Test period: {len(test):,} transactions, {int(y_test.sum())} fraud.")
    print(f"Random Forest at threshold {bundle['threshold']:.6f}: caught {rf['fraud_caught']}, "
          f"false alerts {rf['false_alerts']}, missed {rf['missed_fraud']}, "
          f"precision {rf['precision']:.3f}, recall {rf['recall']:.3f}, F1 {rf['f1']:.3f}")
    if problems:
        print("\nMISMATCH with outputs/results.json:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("Check passed: every saved test figure and both examples match.")


if __name__ == "__main__":
    main()
