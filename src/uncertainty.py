import math
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

BOOTSTRAP_NOTE = (
    "Stratified row bootstrap of the test period: fraud and legitimate rows are resampled "
    "separately so every replicate keeps the observed fraud count. It reflects sampling noise "
    "in which test rows were observed, for a fixed trained model and threshold. It does NOT "
    "capture time dependence, drift, threshold-selection or retraining variability; see the "
    "temporal backtest for that."
)
BLOCK_BOOTSTRAP_NOTE = (
    "Moving-block bootstrap over simulation steps: contiguous blocks of whole steps are "
    "resampled, so transactions from the same hour (and neighbouring hours) stay together. "
    "This keeps short-range time dependence that the row bootstrap destroys, and the fraud "
    "count varies between replicates. Like the row bootstrap it is conditional on the fitted "
    "model and the already-chosen thresholds. Intervals are estimates, not guarantees: they assume "
    "the test period is roughly stationary, cannot reflect drift beyond the test period, and become "
    "unreliable with few blocks or few fraud cases (some replicates may contain no fraud, making "
    "recall/PR-AUC undefined for them; see n_valid)."
)


def _replicate_metrics(yb, sb, thresholds, ranking, k):
    """Metric values for one replicate; None where undefined."""
    out = {}
    n_pos = int(yb.sum())
    for name, t in thresholds.items():
        pred = sb >= t
        tp = int((pred & (yb == 1)).sum())
        fp = int((pred & (yb == 0)).sum())
        fn = n_pos - tp
        out[f"{name}.precision"] = tp / (tp + fp) if tp + fp else None
        out[f"{name}.recall"] = tp / n_pos if n_pos else None
        out[f"{name}.f1"] = 2 * tp / (2 * tp + fp + fn) if n_pos else None
    if ranking:
        both = 0 < n_pos < len(yb)
        out["pr_auc"] = float(average_precision_score(yb, sb)) if both else None
        out["roc_auc"] = float(roc_auc_score(yb, sb)) if both else None
        kk = min(k, len(yb))
        top = np.argsort(-sb, kind="stable")[:kk]
        out[f"precision_at_{k}"] = float(yb[top].sum() / kk) if kk else None
    return out


def _intervals(samples, level):
    alpha = (1 - level) / 2
    result = {}
    for name, vals in samples.items():
        arr = np.asarray([np.nan if v is None else v for v in vals], dtype=float)
        valid = arr[~np.isnan(arr)]
        result[name] = ({"low": float(np.quantile(valid, alpha)),
                         "high": float(np.quantile(valid, 1 - alpha)),
                         "n_valid": int(len(valid))} if len(valid) else
                        {"low": None, "high": None, "n_valid": 0})
    return result


def stratified_bootstrap_ci(y, score, thresholds, n_boot=300, seed=42, top_k=100,
                            ranking=True, min_fraud=10, level=0.95):
    """Percentile intervals for test metrics at fixed thresholds (row-level, class-stratified).

    thresholds: {"name": t}; alerts are score >= t. Skipped (computed=False) when
    there are too few fraud rows or no legitimate rows.
    """
    y, score = np.asarray(y).astype(int), np.asarray(score, dtype=float)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    base = {"method": "stratified_row_bootstrap", "n_boot": int(n_boot), "level": level,
            "seed": seed, "fraud_rows": int(len(pos)), "note": BOOTSTRAP_NOTE}
    if n_boot <= 0:
        return {**base, "computed": False, "reason": "bootstrap disabled"}
    if len(pos) < min_fraud or len(neg) == 0:
        return {**base, "computed": False,
                "reason": f"needs >= {min_fraud} fraud rows and some legitimate rows "
                          f"(have {len(pos)} fraud, {len(neg)} legitimate)"}
    rng = np.random.default_rng(seed)
    samples = {}
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for key, v in _replicate_metrics(y[idx], score[idx], thresholds, ranking, top_k).items():
            samples.setdefault(key, []).append(v)
    return {**base, "computed": True, "intervals": _intervals(samples, level)}


def default_block_length(n_steps):
    """Rule-of-thumb block length ~ n^(1/3) steps, at least 1."""
    return max(1, int(round(n_steps ** (1 / 3))))


def block_bootstrap_indices(steps, block_steps, rng):
    """Row indices for one moving-block replicate; whole steps are always kept together."""
    steps = np.asarray(steps)
    order = np.argsort(steps, kind="stable")
    _, starts, counts = np.unique(steps[order], return_index=True, return_counts=True)
    n_steps = len(starts)
    n_draw = math.ceil(n_steps / block_steps)
    first = rng.integers(0, n_steps - block_steps + 1, n_draw)
    chosen = (first[:, None] + np.arange(block_steps)).ravel()[:n_steps]
    return np.concatenate([order[starts[s]:starts[s] + counts[s]] for s in chosen])


def block_bootstrap_ci(y, score, steps, thresholds, n_boot=300, seed=42, block_steps=None,
                       top_k=100, ranking=True, min_fraud=10, min_blocks=8, level=0.95):
    """Moving-block bootstrap over contiguous simulation steps (time-aware intervals)."""
    y, score, steps = (np.asarray(y).astype(int), np.asarray(score, dtype=float),
                       np.asarray(steps))
    n_steps = len(np.unique(steps)) if len(steps) else 0
    L = int(block_steps) if block_steps else default_block_length(max(n_steps, 1))
    base = {"method": "moving_block_bootstrap_over_steps", "n_boot": int(n_boot), "level": level,
            "seed": seed, "fraud_rows": int(y.sum()), "n_steps": int(n_steps),
            "block_steps": L, "n_blocks": int(n_steps // L) if L else 0,
            "note": BLOCK_BOOTSTRAP_NOTE}
    if n_boot <= 0:
        return {**base, "computed": False, "reason": "bootstrap disabled"}
    if y.sum() < min_fraud or y.sum() == len(y):
        return {**base, "computed": False,
                "reason": f"needs >= {min_fraud} fraud rows and some legitimate rows "
                          f"(have {int(y.sum())} fraud of {len(y)})"}
    if L > n_steps or n_steps // L < min_blocks:
        return {**base, "computed": False,
                "reason": f"needs >= {min_blocks} blocks of {L} steps (have {n_steps} steps)"}
    rng = np.random.default_rng(seed)
    samples, fraud_counts = {}, []
    for _ in range(n_boot):
        idx = block_bootstrap_indices(steps, L, rng)
        yb = y[idx]
        fraud_counts.append(int(yb.sum()))
        for key, v in _replicate_metrics(yb, score[idx], thresholds, ranking, top_k).items():
            samples.setdefault(key, []).append(v)
    fc = np.asarray(fraud_counts)
    return {**base, "computed": True, "intervals": _intervals(samples, level),
            "replicate_fraud": {"min": int(fc.min()), "median": float(np.median(fc)),
                                "max": int(fc.max()), "zero_fraud_share": float((fc == 0).mean())}}


def interval_comparison(row_ci, block_ci):
    """Side-by-side widths; block/row width ratio > 1 means time dependence widens the interval."""
    if not (row_ci or {}).get("computed") or not (block_ci or {}).get("computed"):
        return []
    rows = []
    for key, r in row_ci["intervals"].items():
        b = block_ci["intervals"].get(key)
        if not b or r["low"] is None or b["low"] is None:
            continue
        rw, bw = r["high"] - r["low"], b["high"] - b["low"]
        rows.append({"metric": key, "row_low": r["low"], "row_high": r["high"],
                     "block_low": b["low"], "block_high": b["high"],
                     "width_ratio_block_vs_row": bw / rw if rw > 0 else None})
    return rows
