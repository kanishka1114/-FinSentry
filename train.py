"""Train the FinSentry Random Forest and save the model and its results.

    python train.py [path/to/PS_20174392719_1491204439457_log.csv]

Steps: take a 1,000,000-row random sample of PaySim, split it by time into train,
validation and test, fit the model on train, choose the alert threshold on validation,
then score the test period once. Writes models/random_forest.joblib and
outputs/results.json. The PaySim CSV (about 471 MB) is not included in this repository.
"""
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import finsentry as fs

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "random_forest.joblib"
RESULTS_PATH = ROOT / "outputs" / "results.json"
RF_SETTINGS = {"n_estimators": 100, "max_depth": 12, "min_samples_leaf": 20, "max_samples": 0.5}

# Test results of the earlier version (experiment exp-0781f6ac1d), whose Random Forest used
# only amount, transaction type and hour, on the same sample and time split.
PREVIOUS_VERSION = {"inputs": "amount, transaction type, hour of day",
                    "precision": 0.652, "recall": 0.260, "f1": 0.372,
                    "fraud_caught": 163, "false_alerts": 87, "missed_fraud": 464}


def build_model():
    preprocess = ColumnTransformer([
        ("numbers", StandardScaler(), fs.NUMERIC_FEATURES),
        ("type", OneHotEncoder(handle_unknown="ignore"), ["type"]),
    ])
    forest = RandomForestClassifier(class_weight="balanced_subsample", random_state=fs.SEED,
                                    n_jobs=-1, **RF_SETTINGS)
    return Pipeline([("preprocess", preprocess), ("model", forest)])


def max_f1_threshold(y_true, scores):
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))])


def period_summary(sample, mask):
    part = sample[mask]
    return {"first_step": int(part["step"].min()), "last_step": int(part["step"].max()),
            "rows": int(len(part)), "fraud": int(part["isFraud"].sum()),
            "fraud_rate": float(part["isFraud"].mean())}


def example(row, score, threshold):
    return {"amount": float(row["amount"]), "type": str(row["type"]), "hour": int(row["step"] % 24),
            "sender_balance_before": float(row["oldbalanceOrg"]),
            "sender_balance_after": float(row["newbalanceOrig"]),
            "actually_fraud": bool(row["isFraud"]), "score": float(score),
            "alert": bool(score >= threshold)}


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / fs.CSV_NAME
    if not csv_path.exists():
        sys.exit(f"PaySim CSV not found at {csv_path}. Download it (see the README) and pass its path.")

    full = fs.load_paysim(csv_path)
    sample = fs.uniform_sample(full, fs.SAMPLE_ROWS, fs.SEED)
    periods = fs.chronological_split(sample["step"].to_numpy(), fs.TRAIN_SHARE, fs.VALIDATION_SHARE)
    X, y = fs.features_from_paysim(sample), sample["isFraud"].to_numpy()
    print(f"{len(full):,} transactions in the file; sample of {len(sample):,}.")
    for name, mask in periods.items():
        s = period_summary(sample, mask)
        print(f"  {name:<10} steps {s['first_step']}-{s['last_step']}: {s['rows']:,} rows, {s['fraud']} fraud")

    model = build_model()
    model.fit(X[periods["train"]], y[periods["train"]])
    model.set_params(model__n_jobs=1)  # serial scoring keeps scores repeatable

    val_scores = model.predict_proba(X[periods["validation"]])[:, 1]
    threshold = max_f1_threshold(y[periods["validation"]], val_scores)
    test_scores = model.predict_proba(X[periods["test"]])[:, 1]
    y_val, y_test = y[periods["validation"]], y[periods["test"]]
    test = sample[periods["test"]].reset_index(drop=True)

    results = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "scikit-learn": sklearn.__version__},
        "data": {"rows_in_file": int(len(full)), "fraud_in_file": int(full["isFraud"].sum()),
                 "sample_rows": int(len(sample)), "sample_fraud": int(sample["isFraud"].sum()),
                 "seed": fs.SEED},
        "split": {name: period_summary(sample, mask) for name, mask in periods.items()},
        "inputs": fs.FEATURES,
        "model": {"name": "Random Forest", "settings": RF_SETTINGS,
                  "class_weight": "balanced_subsample", "threshold": threshold},
        "rule": {"description": fs.RULE_DESCRIPTION,
                 "validation": fs.alert_metrics(y_val, fs.rule_alerts(sample[periods["validation"]])),
                 "test": fs.alert_metrics(y_test, fs.rule_alerts(test))},
        "random_forest": {
            "validation": fs.alert_metrics(y_val, val_scores >= threshold),
            "test": {**fs.alert_metrics(y_test, test_scores >= threshold),
                     "pr_auc": float(average_precision_score(y_test, test_scores))},
        },
        "previous_version": PREVIOUS_VERSION,
    }

    alerted_fraud = np.flatnonzero((y_test == 1) & (test_scores >= threshold))
    quiet_normal = np.flatnonzero((y_test == 0) & (test_scores < threshold)
                                  & (test["type"] == "TRANSFER").to_numpy())
    i, j = alerted_fraud[0], quiet_normal[0]
    results["examples"] = {
        "alert": example(test.iloc[i], test_scores[i], threshold),
        "no_alert": example(test.iloc[j], test_scores[j], threshold),
    }

    MODEL_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({"pipeline": model, "threshold": threshold, "features": fs.FEATURES}, MODEL_PATH)
    RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    rf, rule = results["random_forest"]["test"], results["rule"]["test"]
    print(f"\nThreshold chosen on validation: {threshold:.6f}")
    print("Test period:           precision  recall    F1   caught  false alerts  missed")
    for name, m in (("Simple rule", rule), ("Random Forest", rf)):
        print(f"  {name:<20} {m['precision']:>8.3f} {m['recall']:>7.3f} {m['f1']:>6.3f} "
              f"{m['fraud_caught']:>7} {m['false_alerts']:>12,} {m['missed_fraud']:>7}")
    print(f"\nSaved {MODEL_PATH.relative_to(ROOT)} and {RESULTS_PATH.relative_to(ROOT)}.")


if __name__ == "__main__":
    main()
