"""Changing test-period labels must not change any training choice."""
import json
import joblib
import numpy as np
import pandas as pd
import pytest
from src.train import run_training
from src.backtest import run_backtest
from src.protocol import transfer_threshold
from tests.conftest import make_paysim_like

FAST_RF = {"n_estimators": 10, "max_depth": 6, "min_samples_leaf": 5, "n_jobs": 1}


def write_csv(df, path):
    df.to_csv(path, index=False)
    return path


def flip_from_step(df, first_step):
    out = df.copy()
    mask = out["step"] >= first_step
    out.loc[mask, "isFraud"] = 1 - out.loc[mask, "isFraud"]
    return out


def choices(res, model_dir, X_probe):
    """Everything that training/selection decided, excluding test-period metrics."""
    out = {}
    for name, meta in res["models"].items():
        entry = {"validation": {k: v["validation"] for k, v in meta["strategies"].items()}}
        if meta["kind"] == "model":
            entry["thresholds"] = {k: v["threshold"] for k, v in meta["strategies"].items()}
            entry["calibrator"] = meta["calibration"]["calibrator"]
            entry["final_fit"] = meta["final_fit"]
            bundle = joblib.load(model_dir / meta["model_file"])
            entry["probe_scores"] = np.round(bundle["pipeline"].predict_proba(X_probe)[:, 1], 12).tolist()
            entry["background"] = np.round(bundle["background_mean"], 12).tolist()
        out[name] = entry
    return out


@pytest.mark.parametrize("final_fit", ["train_only", "refit"])
@pytest.mark.parametrize("sampling", [("stratified", None), ("uniform", 15_000)])
def test_flipping_test_labels_changes_no_training_choice(tmp_path, final_fit, sampling):
    method, rows = sampling
    df = make_paysim_like(n=20_000, steps=200, fraud_rate=0.01, seed=11)
    probe = pd.DataFrame({"amount": [10.0, 5e5, 2e6], "log_amount": np.log1p([10.0, 5e5, 2e6]),
                          "hour_sin": [0.0, 1.0, -1.0], "hour_cos": [1.0, 0.0, 0.0],
                          "type": ["PAYMENT", "TRANSFER", "CASH_OUT"]})
    kwargs = dict(sample_rows=rows, sampling_method=method, rf_settings=FAST_RF,
                  final_fit=final_fit, eval_settings={"bootstrap": 0})
    a_dir = tmp_path / "a"
    a = run_training(write_csv(df, tmp_path / "a.csv"), model_dir=a_dir, output_dir=a_dir, **kwargs)
    test_start = a["split"]["step_boundaries"]["test"][0]

    b_dir = tmp_path / "b"
    b = run_training(write_csv(flip_from_step(df, test_start), tmp_path / "b.csv"),
                     model_dir=b_dir, output_dir=b_dir, **kwargs)
    assert a["split"]["step_boundaries"] == b["split"]["step_boundaries"]
    assert a["split"]["test"]["fraud"] != b["split"]["test"]["fraud"]   # the flip really happened
    assert choices(a, a_dir, probe) == choices(b, b_dir, probe)


def test_backtest_last_window_labels_do_not_change_earlier_windows(tmp_path):
    df = make_paysim_like(n=20_000, steps=200, fraud_rate=0.01, seed=5)
    settings = {"windows": 3, "bootstrap": 0, "min_test_fraud": 1}
    a = run_backtest(write_csv(df, tmp_path / "a.csv"), rf_settings=FAST_RF,
                     backtest_settings=settings, output_dir=tmp_path / "a", write_report=False)
    last_start = a["windows"][-1]["test_steps"][0]
    b = run_backtest(write_csv(flip_from_step(df, last_start), tmp_path / "b.csv"),
                     rf_settings=FAST_RF, backtest_settings=settings,
                     output_dir=tmp_path / "b", write_report=False)
    strip = lambda w: {k: w[k] for k in ("train_steps", "validation_steps", "test_steps")}
    assert [strip(w) for w in a["windows"]] == [strip(w) for w in b["windows"]]
    for wa, wb in zip(a["windows"][:-1], b["windows"][:-1]):
        assert json.dumps(wa["models"], sort_keys=True) == json.dumps(wb["models"], sort_keys=True)
    la, lb = a["windows"][-1]["models"], b["windows"][-1]["models"]
    for name in ("Logistic Regression", "Random Forest"):
        for key in ("max_f1", "alert_budget"):
            assert la[name]["strategies"][key]["threshold"] == lb[name]["strategies"][key]["threshold"]
            assert la[name]["strategies"][key]["validation"] == lb[name]["strategies"][key]["validation"]
        assert la[name]["calibration"]["calibrator"] == lb[name]["calibration"]["calibrator"]


def test_threshold_transfer_accepts_consistent_rates_and_rejects_others():
    rng = np.random.default_rng(0)
    train, val, new = rng.random(10_000), rng.random(5_000), rng.random(15_000) ** 2
    t, info = transfer_threshold(train, val, 0.9, new)
    assert info["valid"] and info["new_insample_alert_rate"] == pytest.approx(0.1, abs=0.01)
    assert np.mean(new >= t) == pytest.approx(info["old_insample_alert_rate"], abs=0.01)
    overfit_train = np.clip(train + 0.5, 0, 1)    # in-sample scores far higher than validation
    t2, info2 = transfer_threshold(overfit_train, val, 0.9, new)
    assert t2 is None and not info2["valid"] and "ratio" in info2["reason"]
    t3, info3 = transfer_threshold(train, val, 2.0, new)
    assert t3 is None and "zero alerts" in info3["reason"]


def test_protocol_recorded_in_outputs(paysim_csv, tmp_path):
    res = run_training(paysim_csv, sample_rows=8_000, rf_settings=FAST_RF, final_fit="refit",
                       model_dir=tmp_path, output_dir=tmp_path, eval_settings={"bootstrap": 20})
    p = res["protocol"]
    assert p["final_fit_requested"] == "refit" and p["step_boundaries"]["test"]
    assert any("TRAIN+VALIDATION" in s for s in p["steps"])
    for name in ("Logistic Regression", "Random Forest"):
        ff = res["models"][name]["final_fit"]
        assert ff["used"] in ("refit", "train_only") and "threshold_transfer" in ff
        if ff["used"] == "train_only":
            assert ff["rejected_reason"]
    assert (tmp_path / "report_summary.md").exists()
