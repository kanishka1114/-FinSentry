"""Shared evaluation used by the single-split run (train.py) and the backtest.

Selection (thresholds) uses validation data only; test data is only scored afterwards.
"""
import numpy as np
from src.config import EVAL_DEFAULTS
from src.metrics import (point_metrics, ranking_metrics, top_k_metrics, tune_threshold,
                         budget_threshold, tradeoff_curve, pr_curve_points)
from src.uncertainty import stratified_bootstrap_ci, block_bootstrap_ci, interval_comparison
from src.calibration import calibration_report
from src.drift import drift_report

STRATEGY_LABELS = {
    "max_f1": "Max F1 on validation",
    "alert_budget": "Alert budget on validation",
    "rule": "Fixed rule",
}


def eval_config(**overrides):
    cfg = {**EVAL_DEFAULTS, **{k: v for k, v in overrides.items() if v is not None}}
    cfg["top_k"] = [int(k) for k in cfg["top_k"]]
    cfg["top_pct"] = [float(p) for p in cfg["top_pct"]]
    if cfg.get("block_bootstrap") is None:
        cfg["block_bootstrap"] = cfg["bootstrap"]
    return cfg


def select_thresholds(y_val, val_score, cfg):
    """Validation-only threshold choice. Never pass test data here."""
    y_val, val_score = np.asarray(y_val), np.asarray(val_score)
    f1_t, f1_src, val_best_f1 = tune_threshold(y_val, val_score)
    budget_t, budget_src = budget_threshold(val_score, cfg["alerts_per_1000"])
    selected = {}
    for key, t, src in (("max_f1", f1_t, f1_src), ("alert_budget", budget_t, budget_src)):
        selected[key] = {"threshold": t, "source": src,
                         "validation": point_metrics(y_val, val_score >= t)}
    selected["max_f1"]["validation"]["best_f1"] = val_best_f1
    selected["alert_budget"]["requested_alerts_per_1000"] = cfg["alerts_per_1000"]
    return selected


def evaluate_model_scores(y_val, val_score, y_test, test_score, cfg, curves=True,
                          steps_test=None, selected=None, diag_scores=None):
    """Score the test period once at already-selected thresholds.

    selected: output of select_thresholds (computed here from validation if omitted).
    diag_scores: (validation, test) scores of the train-only model for calibration, drift and
    trade-off diagnostics; defaults to (val_score, test_score).
    """
    y_val, y_test = np.asarray(y_val), np.asarray(y_test)
    test_score = np.asarray(test_score, dtype=float)
    selected = selected or select_thresholds(y_val, val_score, cfg)
    strategies = {k: {**v, "test": point_metrics(y_test, test_score >= v["threshold"])}
                  for k, v in selected.items()}
    thresholds = {k: v["threshold"] for k, v in selected.items()}
    dv, dt = diag_scores if diag_scores is not None else (val_score, test_score)

    primary_k = cfg["top_k"][0] if cfg["top_k"] else 100
    out = {
        "strategies": strategies,
        "test_ranking": ranking_metrics(y_test, test_score, primary_k),
        "validation_ranking": ranking_metrics(y_val, val_score, primary_k),
        "top_k": top_k_metrics(y_test, test_score, cfg["top_k"], cfg["top_pct"]),
        "ci": stratified_bootstrap_ci(y_test, test_score, thresholds, n_boot=cfg["bootstrap"],
                                      seed=cfg["seed"], top_k=primary_k,
                                      min_fraud=cfg["min_ci_fraud"]),
        "calibration": calibration_report(y_val, dv, y_test, dt),
        "drift": drift_report(dv, dt),
    }
    if steps_test is not None:
        out["block_ci"] = block_bootstrap_ci(
            y_test, test_score, steps_test, thresholds, n_boot=cfg["block_bootstrap"],
            seed=cfg["seed"], block_steps=cfg.get("block_steps"), top_k=primary_k,
            min_fraud=cfg["min_ci_fraud"], min_blocks=cfg["min_blocks"])
        out["interval_comparison"] = interval_comparison(out["ci"], out["block_ci"])
    if curves:
        out["tradeoff"] = tradeoff_curve(y_val, dv, y_test, dt)
        out["pr_curve_test"] = pr_curve_points(y_test, test_score)
    return out


def evaluate_rule(y_val, val_pred, y_test, test_pred, cfg, steps_test=None):
    y_test = np.asarray(y_test)
    pred = np.asarray(test_pred, dtype=float)
    out = {
        "strategies": {"rule": {"threshold": None, "source": "fixed rule (no tuning)",
                                "validation": point_metrics(y_val, val_pred),
                                "test": point_metrics(y_test, test_pred)}},
        "test_ranking": {"pr_auc": None, "roc_auc": None,
                         "pr_auc_baseline": float(y_test.mean()) if len(y_test) else None},
        "top_k": None,
        "ci": stratified_bootstrap_ci(y_test, pred, {"rule": 0.5}, n_boot=cfg["bootstrap"],
                                      seed=cfg["seed"], ranking=False,
                                      min_fraud=cfg["min_ci_fraud"]),
    }
    if steps_test is not None:
        out["block_ci"] = block_bootstrap_ci(
            y_test, pred, steps_test, {"rule": 0.5}, n_boot=cfg["block_bootstrap"],
            seed=cfg["seed"], block_steps=cfg.get("block_steps"), ranking=False,
            min_fraud=cfg["min_ci_fraud"], min_blocks=cfg["min_blocks"])
        out["interval_comparison"] = interval_comparison(out["ci"], out["block_ci"])
    return out
