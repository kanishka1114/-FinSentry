import numpy as np
from src.metrics import point_metrics, ranking_metrics
from src.uncertainty import stratified_bootstrap_ci
from src.evaluation import eval_config, evaluate_model_scores, evaluate_rule


def data(seed=0, n=4_000, rate=0.02):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < rate).astype(int)
    return y, np.clip(rng.normal(0.3 + 0.4 * y, 0.15), 0, 1)


def test_ci_is_reproducible_and_brackets_point_estimate():
    y, s = data()
    a = stratified_bootstrap_ci(y, s, {"t": 0.5}, n_boot=200, seed=7)
    b = stratified_bootstrap_ci(y, s, {"t": 0.5}, n_boot=200, seed=7)
    assert a == b and a["computed"]
    point_pr = ranking_metrics(y, s, 100)["pr_auc"]
    iv = a["intervals"]["pr_auc"]
    assert iv["low"] <= point_pr <= iv["high"]
    rec = point_metrics(y, s >= 0.5)["recall"]
    assert a["intervals"]["t.recall"]["low"] <= rec <= a["intervals"]["t.recall"]["high"]
    assert "time dependence" in a["note"]


def test_ci_skipped_for_rare_or_single_class():
    y = np.zeros(1_000, dtype=int)
    s = np.linspace(0, 1, 1_000)
    assert stratified_bootstrap_ci(y, s, {"t": 0.5})["computed"] is False
    y[:3] = 1
    out = stratified_bootstrap_ci(y, s, {"t": 0.5}, min_fraud=10)
    assert out["computed"] is False and "3 fraud" in out["reason"]
    assert stratified_bootstrap_ci(np.ones(50, dtype=int), s[:50], {"t": 0.5})["computed"] is False
    assert stratified_bootstrap_ci(*data(), {"t": 0.5}, n_boot=0)["computed"] is False


def test_ci_handles_threshold_with_no_alerts():
    y, s = data()
    out = stratified_bootstrap_ci(y, s, {"never": 2.0}, n_boot=50, seed=1)
    assert out["intervals"]["never.precision"]["n_valid"] == 0
    assert out["intervals"]["never.recall"]["high"] == 0.0


def test_point_metrics_undefined_values_are_none():
    no_alerts = point_metrics([0, 1, 0], [0, 0, 0])
    assert no_alerts["precision"] is None and no_alerts["recall"] == 0.0 and no_alerts["f1"] == 0.0
    no_fraud = point_metrics([0, 0, 0], [0, 1, 0])
    assert no_fraud["recall"] is None and no_fraud["precision"] == 0.0 and no_fraud["f1"] is None
    nothing = point_metrics([0, 0], [0, 0])
    assert nothing["precision"] is None and nothing["recall"] is None and nothing["f1"] is None
    empty = point_metrics([], [])
    assert empty["alerts_per_1000"] is None


def test_evaluation_with_single_class_validation_and_test():
    cfg = eval_config(bootstrap=20)
    s = np.linspace(0, 1, 200)
    zeros = np.zeros(200, dtype=int)
    ev = evaluate_model_scores(zeros, s, zeros, s, cfg)
    assert ev["strategies"]["max_f1"]["threshold"] == 0.5
    assert "lacks one class" in ev["strategies"]["max_f1"]["source"]
    assert ev["test_ranking"]["pr_auc"] is None and ev["test_ranking"]["roc_auc"] is None
    assert ev["pr_curve_test"] is None and ev["ci"]["computed"] is False


def test_rule_evaluation_has_no_ranking_metrics():
    y, _ = data()
    pred = (np.arange(len(y)) % 10 == 0).astype(int)
    ev = evaluate_rule(y, pred, y, pred, eval_config(bootstrap=30))
    assert ev["test_ranking"]["pr_auc"] is None and ev["top_k"] is None
    assert ev["ci"]["computed"] and "pr_auc" not in ev["ci"]["intervals"]
