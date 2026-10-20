import math
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve


def split_summary(y):
    y = np.asarray(y)
    return {"rows": int(len(y)), "fraud": int(y.sum()),
            "prevalence": float(y.mean()) if len(y) else None}


def point_metrics(y_true, pred):
    """Confusion-matrix metrics. Undefined values are None, never a silent 0.

    precision is None with zero alerts; recall and F1 are None with zero fraud (F1 needs
    recall). With fraud present but no alerts, recall and F1 are a genuine 0.
    """
    y_true, pred = np.asarray(y_true).astype(int), np.asarray(pred).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    n, alerts, fraud = len(y_true), tp + fp, tp + fn
    return {
        "precision": tp / alerts if alerts else None,
        "recall": tp / fraud if fraud else None,
        "f1": 2 * tp / (2 * tp + fp + fn) if fraud else None,
        "true_negative": tn, "false_positive": fp,
        "false_negative": fn, "true_positive": tp,
        "alerts": alerts, "false_alerts": fp,
        "alerts_per_1000": 1000 * alerts / n if n else None,
    }


def ranking_metrics(y_true, score, alert_budget):
    """Threshold-free metrics; each is None when it is undefined for this data."""
    y_true, score = np.asarray(y_true), np.asarray(score)
    has_pos, has_both = y_true.sum() > 0, len(np.unique(y_true)) == 2
    k = int(min(alert_budget, len(y_true)))
    top = np.argsort(-score, kind="stable")[:k]
    hits = int(y_true[top].sum())
    return {
        "pr_auc": float(average_precision_score(y_true, score)) if has_pos and has_both else None,
        "pr_auc_baseline": float(y_true.mean()) if len(y_true) else None,
        "roc_auc": float(roc_auc_score(y_true, score)) if has_both else None,
        "alert_budget": k,
        "precision_at_budget": hits / k if k else None,
        "recall_at_budget": hits / int(y_true.sum()) if has_pos else None,
        "fraud_in_budget": hits,
    }


def top_k_metrics(y_true, score, ks=(100,), pcts=()):
    """Precision/recall among the K highest scores (fixed K or top X% of rows)."""
    y_true, score = np.asarray(y_true), np.asarray(score)
    n, total = len(y_true), int(y_true.sum())
    order = np.argsort(-score, kind="stable")
    rows = []
    specs = [(f"top {k}", int(k)) for k in ks] + \
            [(f"top {p:g}%", max(1, math.ceil(n * p / 100))) for p in pcts]
    for label, k in specs:
        k = min(k, n)
        hits = int(y_true[order[:k]].sum()) if k else 0
        rows.append({"label": label, "k": k, "fraud_found": hits,
                     "precision": hits / k if k else None,
                     "recall": hits / total if total else None})
    return rows


def tune_threshold(y_val, score_val, default=0.5):
    """Pick the threshold that maximises F1 on validation data only."""
    y_val = np.asarray(y_val)
    if y_val.sum() == 0 or y_val.sum() == len(y_val):
        return default, "default_0.5 (validation lacks one class)", None
    precision, recall, thresholds = precision_recall_curve(y_val, score_val)
    p, r = precision[:-1], recall[:-1]
    f1 = np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)
    best = int(np.argmax(f1))
    return float(thresholds[best]), "max_f1_on_validation", float(f1[best])


def budget_threshold(score_val, alerts_per_1000):
    """Threshold giving about `alerts_per_1000` alerts on validation scores (labels unused).

    Tied scores can make the achieved alert rate higher than requested.
    """
    score_val = np.asarray(score_val)
    if len(score_val) == 0:
        return 0.5, "default_0.5 (empty validation)"
    rate = min(max(alerts_per_1000 / 1000, 0.0), 1.0)
    if rate == 0:
        return float(np.nextafter(score_val.max(), np.inf)), "alert_budget_on_validation"
    return float(np.quantile(score_val, 1 - rate, method="higher")), "alert_budget_on_validation"


def counts_at_thresholds(y_true, score, thresholds):
    """Vectorised alert counts for score >= t at each threshold."""
    y_true, score = np.asarray(y_true), np.asarray(score)
    pos, neg = np.sort(score[y_true == 1]), np.sort(score[y_true == 0])
    t = np.asarray(thresholds)
    tp = len(pos) - np.searchsorted(pos, t, side="left")
    fp = len(neg) - np.searchsorted(neg, t, side="left")
    return tp, fp, len(pos) - tp, len(neg) - fp


def tradeoff_curve(y_val, score_val, y_test, score_test, n_points=40):
    """Threshold grid built from validation scores only, evaluated on both periods."""
    score_val = np.asarray(score_val)
    if len(score_val) == 0:
        return []
    rates = np.geomspace(1e-4, 0.5, n_points)
    thresholds = np.unique(np.quantile(score_val, 1 - rates, method="higher"))[::-1]
    out = []
    for period, y, s in (("validation", y_val, score_val), ("test", y_test, score_test)):
        tp, fp, fn, _ = counts_at_thresholds(y, s, thresholds)
        n = len(np.asarray(y))
        for i, t in enumerate(thresholds):
            alerts = int(tp[i] + fp[i])
            out.append({"period": period, "threshold": float(t),
                        "alerts_per_1000": 1000 * alerts / n if n else None,
                        "precision": int(tp[i]) / alerts if alerts else None,
                        "recall": int(tp[i]) / int(tp[i] + fn[i]) if tp[i] + fn[i] else None,
                        "true_positive": int(tp[i]), "false_positive": int(fp[i]),
                        "false_negative": int(fn[i]), "rows": n})
    return out


def pr_curve_points(y_true, score, max_points=200):
    y_true = np.asarray(y_true)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return None
    precision, recall, _ = precision_recall_curve(y_true, score)
    keep = np.unique(np.linspace(0, len(precision) - 1, min(max_points, len(precision))).astype(int))
    return {"precision": [round(float(v), 5) for v in precision[keep]],
            "recall": [round(float(v), 5) for v in recall[keep]]}
