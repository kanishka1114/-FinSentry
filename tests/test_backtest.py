import json
import numpy as np
import pandas as pd
import pytest
from src.backtest import make_windows, window_masks, run_backtest
from tests.conftest import make_paysim_like

FAST_RF = {"n_estimators": 8, "max_depth": 5, "min_samples_leaf": 5, "n_jobs": 1}


def front_loaded_steps(seed=0):
    rng = np.random.default_rng(seed)
    return np.sort(np.concatenate([rng.integers(1, 200, 40_000), rng.integers(200, 744, 5_000)]))


def check_ordering(steps, windows):
    prev_test_end = None
    for w in windows:
        tr, va, te = window_masks(steps, w)
        assert tr.any() and va.any() and te.any()
        assert not (tr & va).any() and not (va & te).any() and not (tr & te).any()
        assert steps[tr].max() < steps[va].min()
        assert steps[va].max() < steps[te].min()
        if prev_test_end is not None:
            assert w["test_steps"][0] == prev_test_end + 1 or w["test_steps"][0] > prev_test_end
        prev_test_end = w["test_steps"][1]
    assert windows[-1]["test_steps"][1] == steps.max()


@pytest.mark.parametrize("window_by", ["rows", "steps"])
def test_expanding_windows_respect_time(window_by):
    steps = front_loaded_steps()
    windows, _ = make_windows(steps, n_windows=5, window_by=window_by)
    assert len(windows) == 5
    check_ordering(steps, windows)
    assert all(w["train_steps"][0] == steps.min() for w in windows)
    # Expanding: each window trains on at least as much history as the previous one.
    ends = [w["train_steps"][1] for w in windows]
    assert ends == sorted(ends)


def test_rolling_windows_move_forward():
    steps = front_loaded_steps()
    windows, _ = make_windows(steps, n_windows=4, scheme="rolling", train_window_frac=0.2)
    check_ordering(steps, windows)
    starts = [w["train_steps"][0] for w in windows]
    assert starts == sorted(starts) and starts[-1] > starts[0]


def test_test_windows_do_not_overlap_and_cover_tail():
    steps = np.repeat(np.arange(1, 101), 10)
    windows, _ = make_windows(steps, n_windows=5, initial_frac=0.5)
    covered = []
    for w in windows:
        covered.extend(range(w["test_steps"][0], w["test_steps"][1] + 1))
    assert covered == list(range(covered[0], 101))


def test_merge_sparse_windows_reaches_min_fraud():
    steps = np.repeat(np.arange(1, 101), 100)
    y = np.zeros_like(steps)
    y[np.flatnonzero(steps > 90)[:40]] = 1   # fraud only in the last 10 steps
    y[np.flatnonzero(steps <= 50)[:40]] = 1  # and some in training history
    windows, notes = make_windows(steps, y, n_windows=5, min_test_fraud=20, merge_sparse=True)
    check_ordering(steps, windows)
    for w in windows:
        _, _, te = window_masks(steps, w)
        assert y[te].sum() >= 20
    assert len(windows) < 5 and any("Merged" in n for n in notes)


def test_too_few_steps_raises():
    with pytest.raises(ValueError):
        make_windows(np.array([1, 2, 3, 4]), n_windows=5)


def test_run_backtest_end_to_end(paysim_csv, tmp_path):
    res = run_backtest(paysim_csv, sample_rows=10_000, rf_settings=FAST_RF,
                       backtest_settings={"windows": 3, "bootstrap": 20, "min_test_fraud": 5},
                       output_dir=tmp_path)
    saved = json.loads((tmp_path / "backtest.json").read_text())
    assert saved["sampling"]["mode"] == "uniform_random_sample"
    assert len(saved["windows"]) >= 1
    for w in saved["windows"]:
        assert w["train_steps"][1] < w["validation_steps"][0] <= w["validation_steps"][1] < w["test_steps"][0]
        for name in ("Logistic Regression", "Random Forest"):
            r = w["models"][name]["test_ranking"]
            assert r["pr_auc_baseline"] == pytest.approx(w["test"]["prevalence"])
            assert set(w["models"][name]["strategies"]) == {"max_f1", "alert_budget"}
    assert "Random Forest" in saved["summary"]
    assert res["summary"]["Random Forest"]["pr_auc"]["n_windows"] == len(saved["windows"])


def test_backtest_survives_windows_without_fraud(tmp_path):
    df = make_paysim_like(n=12_000, steps=120, fraud_rate=0.02, seed=3)
    df.loc[df["step"] > 100, "isFraud"] = 0  # last test windows have zero fraud
    path = tmp_path / "nofraud_tail.csv"
    df.to_csv(path, index=False)
    res = run_backtest(path, rf_settings=FAST_RF, output_dir=tmp_path,
                       backtest_settings={"windows": 4, "bootstrap": 10, "merge_sparse": False,
                                          "initial_frac": 0.6})
    last = res["windows"][-1]
    assert last["test"]["fraud"] == 0
    ev = last["models"]["Random Forest"]
    assert ev["test_ranking"]["pr_auc"] is None and ev["test_ranking"]["roc_auc"] is None
    assert ev["strategies"]["max_f1"]["test"]["recall"] is None
    assert ev["ci"]["computed"] is False
    assert any("No fraud" in m for m in last["warnings"])
