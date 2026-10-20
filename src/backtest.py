"""Walk-forward (expanding or rolling) chronological backtest.

Each window trains on earlier steps, picks thresholds on the validation steps right
before the test window, and evaluates on the test window. Whole steps stay together.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from src.config import (OUTPUT_DIR, SEED, RF_DEFAULTS, LR_DEFAULTS, BACKTEST_DEFAULTS,
                        SCHEMA_VERSION, DEFAULT_SAMPLING,
                        FEATURES)
from src.data_loader import prepare_features
from src.splits import closest_index
from src.metrics import split_summary
from src.evaluation import eval_config, evaluate_model_scores, evaluate_rule, select_thresholds
from src.rules import rule_based_score, RULE_DESCRIPTION
from src.train import (build_models, load_experiment_data, run_metadata, model_settings,
                       add_eval_args, add_model_args, add_sampling_arg, rf_from_args,
                       eval_from_args, THRESHOLD_NOTE)
from src.uncertainty import BOOTSTRAP_NOTE, BLOCK_BOOTSTRAP_NOTE
from src.drift import DRIFT_NOTE
from src.calibration import CALIBRATION_NOTE
from src.report import generate_report

SUMMARY_METRICS = [
    ("test_ranking", "pr_auc"), ("test_ranking", "pr_auc_lift"), ("test_ranking", "roc_auc"),
    ("max_f1", "precision"), ("max_f1", "recall"), ("max_f1", "f1"),
    ("max_f1", "alerts_per_1000"),
    ("alert_budget", "precision"), ("alert_budget", "recall"),
    ("alert_budget", "alerts_per_1000"), ("test_ranking", "precision_at_budget"),
    ("calibration", "brier_raw"), ("calibration", "brier_calibrated"),
    ("calibration", "ece_raw"), ("drift", "psi"),
]


def _metric_source(ev, section):
    if section in ("test_ranking", "calibration", "drift"):
        return ev.get(section) or {}
    return ev["strategies"].get(section, {}).get("test", {})


def make_windows(steps, y=None, n_windows=5, scheme="expanding", initial_frac=0.5,
                 val_frac=0.10, train_window_frac=0.30, window_by="rows",
                 min_test_fraud=0, merge_sparse=False):
    """Return (windows, notes). Each window holds inclusive step ranges for train/validation/test.

    Guarantees: train steps < validation steps < test steps; test windows are successive
    and non-overlapping. If merge_sparse, adjacent test windows are merged until each has
    at least `min_test_fraud` fraud rows (label counts size the evaluation periods only;
    they never influence training or thresholds).
    """
    if scheme not in ("expanding", "rolling"):
        raise ValueError("scheme must be 'expanding' or 'rolling'")
    if window_by not in ("rows", "steps"):
        raise ValueError("window_by must be 'rows' or 'steps'")
    if n_windows < 1:
        raise ValueError("n_windows must be >= 1")
    steps = np.asarray(steps)
    u, counts = np.unique(steps, return_counts=True)
    m = len(u)
    if m < n_windows + 2:
        raise ValueError(f"Need at least {n_windows + 2} distinct steps for {n_windows} windows.")
    cum = np.cumsum(counts) / counts.sum()

    first = min(max(closest_index(cum, initial_frac), 1), m - n_windows - 1)
    if window_by == "rows":
        targets = cum[first] + (1 - cum[first]) * np.arange(1, n_windows) / n_windows
        ends = [closest_index(cum, t) for t in targets]
    else:
        targets = u[first] + (u[-1] - u[first]) * np.arange(1, n_windows) / n_windows
        ends = [int(np.searchsorted(u, t, side="right")) - 1 for t in targets]
    cuts = [first]
    for e in ends + [m - 1]:
        if cuts[-1] < e <= m - 1:
            cuts.append(e)
    if cuts[-1] != m - 1:
        cuts.append(m - 1)
    notes = []
    if len(cuts) - 1 < n_windows:
        notes.append(f"Only {len(cuts) - 1} windows possible with the available steps.")

    if merge_sparse and y is not None and min_test_fraud > 0:
        fraud_by_step = np.bincount(np.searchsorted(u, steps), weights=np.asarray(y), minlength=m)
        merged, start, acc = [cuts[0]], cuts[0], 0
        for end in cuts[1:]:
            acc += fraud_by_step[start + 1:end + 1].sum()
            start = end
            if acc >= min_test_fraud:
                merged.append(end)
                acc = 0
        if merged[-1] != m - 1:
            if len(merged) > 1:
                merged[-1] = m - 1
            else:
                merged.append(m - 1)
        if len(merged) < len(cuts):
            notes.append(f"Merged sparse test windows to reach >= {min_test_fraud} fraud rows: "
                         f"{len(cuts) - 1} -> {len(merged) - 1} windows.")
        cuts = merged

    windows = []
    for i in range(len(cuts) - 1):
        test_start, test_end = cuts[i] + 1, cuts[i + 1]
        val_end = test_start - 1
        val_start = min(max(closest_index(cum, cum[val_end] - val_frac) + 1, 1), val_end)
        train_end = val_start - 1
        if scheme == "expanding":
            train_start = 0
        else:
            train_start = min(max(closest_index(cum, cum[train_end] - train_window_frac) + 1, 0),
                              train_end)
        windows.append({
            "window": i + 1,
            "train_steps": [int(u[train_start]), int(u[train_end])],
            "validation_steps": [int(u[val_start]), int(u[val_end])],
            "test_steps": [int(u[test_start]), int(u[test_end])],
        })
    return windows, notes


def window_masks(steps, window):
    steps = np.asarray(steps)
    return tuple((steps >= window[k][0]) & (steps <= window[k][1])
                 for k in ("train_steps", "validation_steps", "test_steps"))


def summarize(windows, model_names):
    summary = {}
    for name in model_names:
        rows = {}
        for section, metric in SUMMARY_METRICS:
            vals = []
            for w in windows:
                ev = w.get("models", {}).get(name)
                if not ev:
                    continue
                src = _metric_source(ev, section)
                if src.get(metric) is not None:
                    vals.append(src[metric])
            key = metric if section in ("test_ranking", "calibration", "drift") \
                else f"{section}.{metric}"
            arr = np.asarray(vals, dtype=float)
            rows[key] = {"n_windows": int(len(arr)),
                         "mean": float(arr.mean()) if len(arr) else None,
                         "std": float(arr.std(ddof=1)) if len(arr) > 1 else None,
                         "min": float(arr.min()) if len(arr) else None,
                         "max": float(arr.max()) if len(arr) else None}
        summary[name] = rows
    return summary


def run_backtest(data_path=None, sample_rows=None, max_step=None, seed=SEED,
                 backtest_settings=None, rf_settings=None, lr_settings=None,
                 eval_settings=None, allow_full=False, output_dir=OUTPUT_DIR,
                 sampling_method=DEFAULT_SAMPLING, write_report=True, experiment_id=None):
    bt = {**BACKTEST_DEFAULTS, **{k: v for k, v in (backtest_settings or {}).items()
                                  if v is not None}}
    rf_settings = {**RF_DEFAULTS, **(rf_settings or {})}
    lr_settings = {**LR_DEFAULTS, **(lr_settings or {})}
    cfg = eval_config(**{"seed": seed, "bootstrap": bt["bootstrap"], **(eval_settings or {})})
    np.random.seed(seed)
    t0 = time.time()

    df, dataset_info, sampling = load_experiment_data(data_path, sample_rows, max_step, seed,
                                                      allow_full, sampling_method)
    X, y = prepare_features(df)
    steps, y_arr = df["step"].to_numpy(), y.to_numpy()
    windows, notes = make_windows(
        steps, y_arr, bt["windows"], bt["scheme"], bt["initial_frac"], bt["val_frac"],
        bt["train_window_frac"], bt["window_by"], bt["min_test_fraud"], bt["merge_sparse"])

    warnings = list(notes)
    if bt["merge_sparse"]:
        warnings.append("merge_sparse is on: test-period label counts moved window boundaries, "
                        "so later windows' training data depends on earlier test labels.")
    if sample_rows is not None:
        warnings.append("Backtest uses a sample, not the full dataset.")
        if sampling_method == "stratified":
            warnings.append("Stratified sampling used global label counts (all periods) to pick "
                            "rows; use --sampling uniform for final chronological results.")
    model_names = ["Rule-based", "Logistic Regression", "Random Forest"]
    for w in windows:
        train_m, val_m, test_m = window_masks(steps, w)
        for period, mask in (("train", train_m), ("validation", val_m), ("test", test_m)):
            w[period] = split_summary(y_arr[mask])
        w["warnings"], w["models"] = [], {}
        test_fraud, val_fraud = w["test"]["fraud"], w["validation"]["fraud"]
        w["reliable"] = test_fraud >= bt["min_test_fraud"]
        if test_fraud == 0:
            w["warnings"].append("No fraud in test window: recall, PR-AUC and ROC-AUC are undefined.")
        elif not w["reliable"]:
            w["warnings"].append(f"Only {test_fraud} test fraud rows (< {bt['min_test_fraud']}): "
                                 "metrics are very noisy.")
        if val_fraud < bt["min_val_fraud"]:
            w["warnings"].append(f"Only {val_fraud} validation fraud rows: max-F1 threshold is unstable.")
        if w["train"]["fraud"] == 0 or w["train"]["fraud"] == w["train"]["rows"]:
            w["warnings"].append("Training period has one class; models skipped.")
            continue

        X_val, X_test = X[val_m], X[test_m]
        y_val, y_test, steps_test = y_arr[val_m], y_arr[test_m], steps[test_m]
        w["models"]["Rule-based"] = evaluate_rule(
            y_val, rule_based_score(X_val), y_test, rule_based_score(X_test), cfg, steps_test)
        for name, pipe in build_models(seed, rf_settings, lr_settings).items():
            pipe.fit(X[train_m], y_arr[train_m])
            val_score = pipe.predict_proba(X_val)[:, 1]
            selected = select_thresholds(y_val, val_score, cfg)
            ev = evaluate_model_scores(y_val, val_score, y_test, pipe.predict_proba(X_test)[:, 1],
                                       cfg, curves=False, steps_test=steps_test,
                                       selected=selected)
            r = ev["test_ranking"]
            r["pr_auc_lift"] = (r["pr_auc"] / r["pr_auc_baseline"]
                                if r["pr_auc"] is not None and r["pr_auc_baseline"] else None)
            w["models"][name] = ev
        warnings.extend(f"Window {w['window']}: {msg}" for msg in w["warnings"])

    results = {
        "run": {**run_metadata(seed, experiment_id), "seconds": round(time.time() - t0, 2)},
        "dataset": dataset_info,
        "sampling": sampling,
        "backtest_settings": bt,
        "evaluation_settings": cfg,
        "model_settings": {n: model_settings(n, rf_settings, lr_settings)
                           for n in ("Logistic Regression", "Random Forest")},
        "rule_description": RULE_DESCRIPTION,
        "features": FEATURES,
        "windows": windows,
        "summary": summarize(windows, model_names),
        "warnings": warnings,
        "notes": {
            "thresholds": THRESHOLD_NOTE,
            "bootstrap": BOOTSTRAP_NOTE,
            "block_bootstrap": BLOCK_BOOTSTRAP_NOTE,
            "drift": DRIFT_NOTE,
            "calibration": CALIBRATION_NOTE,
            "windows": "Train steps < validation steps < test steps in every window. Variation "
                       "across windows shows time-related instability that a single split hides. "
                       "Each window's model is fit on its training period only (no refit).",
        },
        "disclaimer": "Synthetic PaySim data; not evidence of real-world performance.",
    }
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "backtest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if write_report:
        generate_report(output_dir)
    return results


def parse_args(argv=None):
    d = BACKTEST_DEFAULTS
    p = argparse.ArgumentParser(description="Walk-forward chronological backtest for FinSentry.")
    p.add_argument("--data", default=None)
    p.add_argument("--sample-rows", type=int, default=None)
    p.add_argument("--full", action="store_true", help="Allow more than 1,000,000 rows")
    p.add_argument("--max-step", type=int, default=None)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--windows", type=int, default=d["windows"])
    p.add_argument("--scheme", choices=["expanding", "rolling"], default=d["scheme"])
    p.add_argument("--initial-frac", type=float, default=d["initial_frac"],
                   help="Share of rows before the first test window")
    p.add_argument("--val-frac", type=float, default=d["val_frac"],
                   help="Share of rows used for validation before each test window")
    p.add_argument("--train-window-frac", type=float, default=d["train_window_frac"],
                   help="Rolling scheme: share of rows in each training window")
    p.add_argument("--window-by", choices=["rows", "steps"], default=d["window_by"])
    p.add_argument("--min-test-fraud", type=int, default=d["min_test_fraud"])
    p.add_argument("--min-val-fraud", type=int, default=d["min_val_fraud"])
    p.add_argument("--merge-sparse", action="store_true",
                   help="Merge test windows with too few fraud rows (uses test label counts to "
                        "set boundaries; off by default for leakage safety)")
    add_sampling_arg(p)
    add_eval_args(p, d["bootstrap"])
    add_model_args(p)
    args = p.parse_args(argv)
    for name in ("initial_frac", "val_frac", "train_window_frac"):
        if not 0 < getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be in (0, 1)")
    return args


def main(argv=None):
    args = parse_args(argv)
    bt = {"windows": args.windows, "scheme": args.scheme, "initial_frac": args.initial_frac,
          "val_frac": args.val_frac, "train_window_frac": args.train_window_frac,
          "window_by": args.window_by, "min_test_fraud": args.min_test_fraud,
          "min_val_fraud": args.min_val_fraud, "merge_sparse": args.merge_sparse,
          "bootstrap": args.bootstrap}
    ev = eval_from_args(args)
    ev.pop("bootstrap")
    res = run_backtest(args.data, args.sample_rows, args.max_step, args.seed, bt,
                       rf_from_args(args), eval_settings=ev, allow_full=args.full,
                       sampling_method=args.sampling)
    table = []
    for w in res["windows"]:
        row = {"window": w["window"], "test_steps": w["test_steps"],
               "test_rows": w["test"]["rows"], "test_fraud": w["test"]["fraud"],
               "prevalence": round(w["test"]["prevalence"], 5)}
        for name in ("Logistic Regression", "Random Forest"):
            ev_m = w["models"].get(name)
            pr = ev_m["test_ranking"]["pr_auc"] if ev_m else None
            row[f"{name} PR-AUC"] = round(pr, 4) if pr is not None else None
        table.append(row)
    print(json.dumps({"sampling": res["sampling"], "windows": table,
                      "warnings": res["warnings"], "seconds": res["run"]["seconds"]}, indent=2))


if __name__ == "__main__":
    main()
