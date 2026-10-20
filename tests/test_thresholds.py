import numpy as np
from src.evaluation import eval_config, evaluate_model_scores
from src.metrics import budget_threshold, top_k_metrics, tradeoff_curve, point_metrics


def scored_data(seed=0, n=5_000, rate=0.02):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    score = np.clip(rng.normal(0.2 + 0.5 * y, 0.15), 0, 1)
    return y, score


def test_thresholds_ignore_test_labels():
    cfg = eval_config(bootstrap=0)
    y_val, s_val = scored_data(1)
    y_test, s_test = scored_data(2)
    a = evaluate_model_scores(y_val, s_val, y_test, s_test, cfg)
    b = evaluate_model_scores(y_val, s_val, 1 - y_test, s_test, cfg)  # flip test labels
    for key in ("max_f1", "alert_budget"):
        assert a["strategies"][key]["threshold"] == b["strategies"][key]["threshold"]
        assert a["strategies"][key]["validation"] == b["strategies"][key]["validation"]


def test_thresholds_ignore_test_scores():
    cfg = eval_config(bootstrap=0)
    y_val, s_val = scored_data(1)
    y_test, s_test = scored_data(2)
    a = evaluate_model_scores(y_val, s_val, y_test, s_test, cfg)
    b = evaluate_model_scores(y_val, s_val, y_test, s_test[::-1].copy(), cfg)
    assert a["strategies"]["max_f1"]["threshold"] == b["strategies"]["max_f1"]["threshold"]
    ta = sorted({p["threshold"] for p in a["tradeoff"]})
    tb = sorted({p["threshold"] for p in b["tradeoff"]})
    assert ta == tb


def test_budget_threshold_hits_requested_rate_without_labels():
    _, s = scored_data(3, n=20_000)
    t, src = budget_threshold(s, alerts_per_1000=10)
    rate = 1000 * (s >= t).mean()
    assert 9 <= rate <= 11 and src == "alert_budget_on_validation"
    t0, _ = budget_threshold(s, 0)
    assert (s >= t0).sum() == 0


def test_alert_budget_strategy_reports_rate():
    cfg = eval_config(bootstrap=0, alerts_per_1000=20)
    y, s = scored_data(4, n=10_000)
    ev = evaluate_model_scores(y, s, y, s, cfg)
    val = ev["strategies"]["alert_budget"]["validation"]
    assert 18 <= val["alerts_per_1000"] <= 22
    assert ev["strategies"]["alert_budget"]["requested_alerts_per_1000"] == 20


def test_top_k_and_top_pct():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.05])
    rows = {r["label"]: r for r in top_k_metrics(y, s, ks=[2, 100], pcts=[30])}
    assert rows["top 2"]["precision"] == 1.0 and rows["top 2"]["recall"] == 2 / 3
    assert rows["top 100"]["k"] == 10  # capped at number of rows
    assert rows["top 30%"]["k"] == 3 and rows["top 30%"]["fraud_found"] == 2


def test_tradeoff_counts_match_point_metrics():
    y_val, s_val = scored_data(5)
    y_test, s_test = scored_data(6)
    for p in tradeoff_curve(y_val, s_val, y_test, s_test, n_points=10):
        y, s = (y_val, s_val) if p["period"] == "validation" else (y_test, s_test)
        m = point_metrics(y, s >= p["threshold"])
        assert (p["true_positive"], p["false_positive"], p["false_negative"]) == \
               (m["true_positive"], m["false_positive"], m["false_negative"])
