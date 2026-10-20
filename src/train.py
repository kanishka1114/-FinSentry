import argparse
import json
import platform
import sys
import time
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.config import (MODEL_DIR, OUTPUT_DIR, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
                        FEATURES, EXCLUDED_FIELDS, SEED, TRAIN_FRAC, VAL_FRAC, ALERT_BUDGET,
                        RF_DEFAULTS, LR_DEFAULTS, FULL_DATA_GUARD_ROWS, EVAL_DEFAULTS,
                        FINAL_FIT_CHOICES, THRESHOLD_TRANSFER_TOLERANCE, SCHEMA_VERSION,
                        DEFAULT_SAMPLING)
from src.data_loader import resolve_path, file_fingerprint, load_data, SAMPLERS, prepare_features
from src.splits import chronological_split, random_split
from src.metrics import split_summary
from src.evaluation import eval_config, evaluate_model_scores, evaluate_rule, select_thresholds
from src.protocol import transfer_threshold, PROTOCOL_STEPS, TEST_USAGE
from src.explain import transformed_mean, EXPLANATION_NOTE
from src.drift import DRIFT_NOTE
from src.calibration import CALIBRATION_NOTE
from src.uncertainty import BOOTSTRAP_NOTE, BLOCK_BOOTSTRAP_NOTE
from src.report import generate_report
from src.rules import rule_based_score, RULE_DESCRIPTION

SAMPLING_NOTES = {
    "stratified": "Class-stratified sample drawn once before splitting; keeps the natural fraud "
                  "rate and rare fraud, but uses global label counts (all periods) to size each class.",
    "uniform": "Label-free uniform sample; fraud count is random. Strictest option for leakage audits.",
}

THRESHOLD_NOTE = (
    "No threshold is universally best. 'Max F1' weights missed fraud and false alerts "
    "equally; an alert budget fixes reviewer workload instead. The right choice depends on "
    "the relative cost of a missed fraud versus a false alert and on review capacity. "
    "All thresholds are chosen on validation data; test labels are never used for tuning.")

MODEL_FILES = {"Logistic Regression": "logistic_regression.joblib",
               "Random Forest": "random_forest.joblib"}
PREVIEW_RANDOM_ROWS = 300
PREVIEW_FRAUD_ROWS = 50
PREVIEW_TOP_ALERTS = 50


def make_pipeline(model, scale_numeric=False):
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([
        ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocess), ("model", model)])


def build_models(seed, rf_settings, lr_settings):
    return {
        "Logistic Regression": make_pipeline(
            LogisticRegression(class_weight="balanced", random_state=seed, **lr_settings),
            scale_numeric=True),
        "Random Forest": make_pipeline(
            RandomForestClassifier(class_weight="balanced_subsample", random_state=seed,
                                   **rf_settings)),
    }


def library_versions():
    versions = {"python": platform.python_version(), "platform": platform.platform()}
    for pkg in ("numpy", "pandas", "scikit-learn", "joblib", "streamlit", "plotly"):
        try:
            versions[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            versions[pkg] = None
    return versions


def build_preview(test_df, y_test, scores, alerts, rule_pred, seed):
    """Natural random test rows plus labelled extra fraud examples and top alerts."""
    base = pd.DataFrame({
        "step": test_df["step"].to_numpy(),
        "hour": (test_df["step"].to_numpy() % 24),
        "type": test_df["type"].astype(str).to_numpy(),
        "amount": test_df["amount"].to_numpy(),
        "actual_isFraud": y_test.to_numpy(),
        "rule_flag": rule_pred.to_numpy(),
    })
    for name, col in (("Logistic Regression", "lr"), ("Random Forest", "rf")):
        base[f"{col}_score"] = np.round(scores[name], 5)
        base[f"{col}_alert"] = alerts[name]
    rng = np.random.default_rng(seed)
    n = len(base)
    random_pos = rng.choice(n, min(PREVIEW_RANDOM_ROWS, n), replace=False)
    fraud_pos = np.flatnonzero(base["actual_isFraud"].to_numpy() == 1)
    fraud_pos = rng.choice(fraud_pos, min(PREVIEW_FRAUD_ROWS, len(fraud_pos)), replace=False)
    top_pos = np.argsort(-scores["Random Forest"], kind="stable")[:PREVIEW_TOP_ALERTS]

    reason = pd.Series("", index=range(n), dtype=object)
    for pos, label in ((top_pos, "top_rf_alert"), (fraud_pos, "fraud_example"),
                       (random_pos, "random_test_sample")):
        reason.iloc[pos] = label
    keep = np.unique(np.concatenate([random_pos, fraud_pos, top_pos])).astype(int)
    preview = base.iloc[keep].copy()
    preview.insert(0, "preview_reason", reason.iloc[keep].to_numpy())
    return preview.sort_values(["preview_reason", "step"]).reset_index(drop=True)


def load_experiment_data(data_path=None, sample_rows=None, max_step=None, seed=SEED,
                         allow_full=False, sampling_method=DEFAULT_SAMPLING):
    """Load, guard against accidental full-data runs, and sample. Returns (df, dataset, sampling)."""
    t0 = time.time()
    path = resolve_path(data_path)
    fingerprint = file_fingerprint(path)
    df, load_info = load_data(path, max_step=max_step)
    if sample_rows is None and len(df) > FULL_DATA_GUARD_ROWS and not allow_full:
        raise SystemExit(
            f"{len(df):,} rows would be used. Pass --sample-rows N (e.g. 300000) for a "
            "development run, or --full to train on all rows deliberately.")
    df = SAMPLERS[sampling_method](df, sample_rows, seed=seed)
    sampling = {
        "mode": "full" if sample_rows is None else f"{sampling_method}_random_sample",
        "method": sampling_method, "note": SAMPLING_NOTES[sampling_method],
        "requested_rows": sample_rows, "max_step": max_step,
        "rows_used": int(len(df)), "fraud_used": int(df["isFraud"].sum()),
        "prevalence_used": float(df["isFraud"].mean()),
        "load_seconds": round(time.time() - t0, 2),
    }
    return df, {**fingerprint, **load_info}, sampling


def run_metadata(seed, experiment_id=None):
    return {
        "run_id": uuid.uuid4().hex[:12],
        "experiment_id": experiment_id,
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "seed": seed, "versions": library_versions(),
    }


def model_settings(name, rf_settings, lr_settings):
    return rf_settings if name == "Random Forest" else {**lr_settings, "class_weight": "balanced"}


def fit_and_select(name, X_train, y_train, X_val, y_val, X_test, final_fit, cfg, seed,
                   rf_settings, lr_settings):
    """Run the documented protocol for one model; test data is only scored at the end."""
    fit_start = time.time()
    pipe0 = build_models(seed, rf_settings, lr_settings)[name]
    pipe0.fit(X_train, y_train)
    val0 = pipe0.predict_proba(X_val)[:, 1]
    selected = select_thresholds(y_val, val0, cfg)
    final_pipe, final_selected, final_X = pipe0, selected, X_train
    info = {"requested": final_fit, "used": "train_only"}

    if final_fit == "refit":
        X_tv, y_tv = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])
        pipe1 = build_models(seed, rf_settings, lr_settings)[name]
        pipe1.fit(X_tv, y_tv)
        train0 = pipe0.predict_proba(X_train)[:, 1]
        insample1 = pipe1.predict_proba(X_tv)[:, 1]
        transfers, ok = {}, True
        for key, sel in selected.items():
            new_t, t_info = transfer_threshold(train0, val0, sel["threshold"], insample1,
                                               THRESHOLD_TRANSFER_TOLERANCE)
            transfers[key] = t_info
            ok = ok and t_info["valid"]
        info["threshold_transfer"] = transfers
        if ok:
            final_pipe, final_X, info["used"] = pipe1, X_tv, "refit"
            final_selected = {k: {**v, "threshold": transfers[k]["new_threshold"],
                                  "threshold_train_only_model": v["threshold"],
                                  "source": v["source"] + " -> transferred to refit model"}
                              for k, v in selected.items()}
        else:
            info["rejected_reason"] = "; ".join(f"{k}: {t['reason']}" for k, t in transfers.items()
                                                if not t["valid"])
    fit_seconds = time.time() - fit_start
    # Test scoring happens only after every choice above is fixed.
    test_final = final_pipe.predict_proba(X_test)[:, 1]
    test0 = test_final if final_pipe is pipe0 else pipe0.predict_proba(X_test)[:, 1]
    return {"final_pipe": final_pipe, "final_X": final_X, "selected": final_selected,
            "val0": val0, "test0": test0, "test_final": test_final, "info": info,
            "fit_seconds": fit_seconds}


def run_training(data_path=None, sample_rows=None, max_step=None, split="chronological",
                 seed=SEED, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC, alert_budget=ALERT_BUDGET,
                 rf_settings=None, lr_settings=None, allow_full=False,
                 model_dir=MODEL_DIR, output_dir=OUTPUT_DIR, eval_settings=None,
                 final_fit="train_only", sampling_method=DEFAULT_SAMPLING, write_report=True,
                 experiment_id=None):
    if final_fit not in FINAL_FIT_CHOICES:
        raise ValueError(f"final_fit must be one of {FINAL_FIT_CHOICES}")
    rf_settings = {**RF_DEFAULTS, **(rf_settings or {})}
    lr_settings = {**LR_DEFAULTS, **(lr_settings or {})}
    cfg = eval_config(**{"seed": seed, **(eval_settings or {})})
    if alert_budget not in cfg["top_k"]:
        cfg["top_k"] = [alert_budget] + cfg["top_k"]
    else:
        cfg["top_k"] = [alert_budget] + [k for k in cfg["top_k"] if k != alert_budget]
    model_dir, output_dir = Path(model_dir), Path(output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(seed)

    df, dataset_info, sampling = load_experiment_data(data_path, sample_rows, max_step, seed,
                                                      allow_full, sampling_method)
    load_seconds = sampling.pop("load_seconds")

    X, y = prepare_features(df)
    steps = df["step"].to_numpy()
    if y.nunique() < 2:
        raise ValueError("Data must contain both fraud and legitimate rows.")
    if split == "chronological":
        train_m, val_m, test_m, bounds = chronological_split(df["step"].to_numpy(),
                                                             train_frac, val_frac)
    elif split == "random":
        train_m, val_m, test_m, bounds = random_split(y.to_numpy(), train_frac, val_frac, seed)
    else:
        raise ValueError("split must be 'chronological' or 'random'")
    X_train, y_train = X[train_m], y[train_m]
    X_val, y_val = X[val_m], y[val_m]
    X_test, y_test = X[test_m], y[test_m]
    if y_train.nunique() < 2:
        raise ValueError("Training period has only one class; use more rows or a different split.")

    warnings = []
    if sample_rows is not None:
        warnings.append("Results come from a sample, not the full dataset.")
        if sampling_method == "stratified":
            warnings.append("Stratified sampling used global label counts (all periods) to pick "
                            "rows; use --sampling uniform for final chronological results.")
    if split == "random":
        warnings.append("Random split ignores time order; it does not test future performance.")
    if y_test.sum() < 30:
        warnings.append(f"Only {int(y_test.sum())} fraud rows in test; metrics are very noisy.")
    if y_val.sum() < 10:
        warnings.append(f"Only {int(y_val.sum())} fraud rows in validation; threshold is unstable.")
    prev = {s: float(v.mean()) for s, v in (("train", y_train), ("test", y_test))}
    if prev["train"] > 0 and abs(prev["test"] / prev["train"] - 1) > 0.5:
        warnings.append("Fraud prevalence differs strongly between train and test periods; "
                        "precision depends heavily on prevalence.")

    run = run_metadata(seed, experiment_id)
    run_id = run["run_id"]
    results = {
        "run": run,
        "dataset": dataset_info,
        "sampling": sampling,
        "split": {
            "method": split, "train_frac": train_frac, "val_frac": val_frac,
            "step_boundaries": bounds,
            "train": split_summary(y_train), "validation": split_summary(y_val),
            "test": split_summary(y_test),
        },
        "protocol": {
            "final_fit_requested": final_fit,
            "steps": PROTOCOL_STEPS, "test_usage": TEST_USAGE,
            "split_method": split, "step_boundaries": bounds,
            "sampling_method": sampling_method,
            "threshold_transfer_tolerance": THRESHOLD_TRANSFER_TOLERANCE,
        },
        "features": FEATURES, "excluded_fields": EXCLUDED_FIELDS,
        "alert_budget": alert_budget,
        "evaluation_settings": cfg,
        "models": {},
        "warnings": warnings,
        "interpretation": THRESHOLD_NOTE,
        "notes": {"drift": DRIFT_NOTE, "calibration": CALIBRATION_NOTE,
                  "explanations": EXPLANATION_NOTE, "row_bootstrap": BOOTSTRAP_NOTE,
                  "block_bootstrap": BLOCK_BOOTSTRAP_NOTE},
        "disclaimer": "Synthetic PaySim data; not evidence of real-world performance.",
    }

    steps_test = steps[test_m]
    rule_test = rule_based_score(X_test)
    rule_eval = evaluate_rule(y_val, rule_based_score(X_val), y_test, rule_test, cfg, steps_test)
    rule_point = rule_eval["strategies"]["rule"]
    results["models"]["Rule-based"] = {
        "kind": "rule", "description": RULE_DESCRIPTION, "threshold": None,
        "validation": rule_point["validation"],
        "test": {**rule_point["test"], **rule_eval["test_ranking"],
                 "alert_budget": None, "precision_at_budget": None,
                 "recall_at_budget": None, "fraud_in_budget": None},
        **rule_eval,
    }

    t1 = time.time()
    scores, alerts = {}, {}
    for name in MODEL_FILES:
        fs = fit_and_select(name, X_train, y_train, X_val, y_val, X_test, final_fit, cfg, seed,
                            rf_settings, lr_settings)
        test_score = fs["test_final"]
        ev = evaluate_model_scores(y_val, fs["val0"], y_test, test_score, cfg,
                                   steps_test=steps_test, selected=fs["selected"],
                                   diag_scores=(fs["val0"], fs["test0"]))
        primary = ev["strategies"]["max_f1"]
        threshold = primary["threshold"]
        scores[name], alerts[name] = test_score, (test_score >= threshold).astype(int)
        results["models"][name] = {
            "kind": "model",
            "settings": model_settings(name, rf_settings, lr_settings),
            "final_fit": fs["info"],
            "threshold": threshold, "threshold_source": primary["source"],
            "fit_seconds": round(fs["fit_seconds"], 2),
            "validation": primary["validation"],
            "test": {**primary["test"], **ev["test_ranking"]},
            "model_file": MODEL_FILES[name],
            **ev,
        }
        if fs["info"]["requested"] == "refit" and fs["info"]["used"] != "refit":
            warnings.append(f"{name}: refit rejected, train-only model kept "
                            f"({fs['info']['rejected_reason']}).")
        joblib.dump({"pipeline": fs["final_pipe"], "run_id": run_id, "threshold": threshold,
                     "thresholds": {k: v["threshold"] for k, v in ev["strategies"].items()},
                     "final_fit": fs["info"]["used"],
                     "background_mean": transformed_mean(fs["final_pipe"], fs["final_X"]),
                     "features": FEATURES}, model_dir / MODEL_FILES[name])

    preview = build_preview(df[test_m], y_test, scores, alerts, rule_test, seed)
    preview.insert(0, "run_id", run_id)
    preview.to_csv(output_dir / "test_preview.csv", index=False)
    results["preview"] = {
        "rows": int(len(preview)),
        "composition": preview["preview_reason"].value_counts().to_dict(),
        "note": "Enriched with extra fraud examples and top alerts; use metrics for rates.",
    }
    results["run"]["load_seconds"] = round(load_seconds, 2)
    results["run"]["training_seconds"] = round(time.time() - t1, 2)
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if write_report:
        generate_report(output_dir)
    return results


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train FinSentry models on PaySim.")
    p.add_argument("--data", default=None, help="Optional CSV path")
    p.add_argument("--sample-rows", type=int, default=None,
                   help="Random sample size across the (filtered) dataset (method set by --sampling)")
    p.add_argument("--full", action="store_true",
                   help=f"Allow training on more than {FULL_DATA_GUARD_ROWS:,} rows")
    p.add_argument("--max-step", type=int, default=None, help="Keep only rows with step <= N")
    p.add_argument("--split", choices=["chronological", "random"], default="chronological")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--train-frac", type=float, default=TRAIN_FRAC)
    p.add_argument("--val-frac", type=float, default=VAL_FRAC)
    p.add_argument("--alert-budget", type=int, default=ALERT_BUDGET,
                   help="Headline K for Precision@K / Recall@K")
    p.add_argument("--final-fit", choices=FINAL_FIT_CHOICES, default="train_only",
                   help="train_only (default) or refit on train+validation with checked threshold transfer")
    add_sampling_arg(p)
    add_eval_args(p, EVAL_DEFAULTS["bootstrap"])
    add_model_args(p)
    args = p.parse_args(argv)
    if not 0 < args.train_frac < 1 or not 0 < args.val_frac < 1 \
            or args.train_frac + args.val_frac >= 1:
        p.error("--train-frac and --val-frac must be in (0, 1) and sum to less than 1")
    return args


def add_eval_args(p, default_bootstrap):
    p.add_argument("--alerts-per-1000", type=float, default=EVAL_DEFAULTS["alerts_per_1000"],
                   help="Alert-budget strategy: target alerts per 1,000 transactions (set on validation)")
    p.add_argument("--top-k", type=int, nargs="+", default=EVAL_DEFAULTS["top_k"])
    p.add_argument("--top-pct", type=float, nargs="*", default=EVAL_DEFAULTS["top_pct"])
    p.add_argument("--bootstrap", type=int, default=default_bootstrap,
                   help="Row-bootstrap replicates for confidence intervals (0 disables)")
    p.add_argument("--block-bootstrap", type=int, default=None,
                   help="Block-bootstrap replicates (default: same as --bootstrap; 0 disables)")
    p.add_argument("--block-steps", type=int, default=None,
                   help="Steps per bootstrap block (default: cube root of test steps)")


def add_sampling_arg(p):
    p.add_argument("--sampling", choices=sorted(SAMPLERS), default=DEFAULT_SAMPLING,
                   help="uniform (default) never looks at labels; stratified uses global label counts (development only)")


def add_model_args(p):
    p.add_argument("--rf-trees", type=int, default=RF_DEFAULTS["n_estimators"])
    p.add_argument("--rf-max-depth", type=int, default=RF_DEFAULTS["max_depth"])
    p.add_argument("--rf-min-leaf", type=int, default=RF_DEFAULTS["min_samples_leaf"])
    p.add_argument("--rf-max-samples", type=float, default=RF_DEFAULTS["max_samples"],
                   help="Fraction of training rows drawn per tree")
    p.add_argument("--n-jobs", type=int, default=RF_DEFAULTS["n_jobs"])


def rf_from_args(args):
    return {"n_estimators": args.rf_trees, "max_depth": args.rf_max_depth,
            "min_samples_leaf": args.rf_min_leaf, "max_samples": args.rf_max_samples,
            "n_jobs": args.n_jobs}


def eval_from_args(args):
    return {"alerts_per_1000": args.alerts_per_1000, "top_k": args.top_k,
            "top_pct": args.top_pct, "bootstrap": args.bootstrap,
            "block_bootstrap": args.block_bootstrap, "block_steps": args.block_steps}


def main(argv=None):
    args = parse_args(argv)
    results = run_training(args.data, args.sample_rows, args.max_step, args.split, args.seed,
                           args.train_frac, args.val_frac, args.alert_budget, rf_from_args(args),
                           allow_full=args.full, eval_settings=eval_from_args(args),
                           final_fit=args.final_fit, sampling_method=args.sampling)
    summary = {name: {k: m["test"].get(k) for k in
                      ("precision", "recall", "f1", "pr_auc", "precision_at_budget")}
               for name, m in results["models"].items()}
    print(json.dumps({"sampling": results["sampling"], "split": results["split"],
                      "test_metrics": summary, "warnings": results["warnings"]}, indent=2))


if __name__ == "__main__":
    main()
