import json
import joblib
import pytest
from src.train import run_training

FAST_RF = {"n_estimators": 10, "max_depth": 6, "min_samples_leaf": 5, "n_jobs": 1}


def test_end_to_end_chronological(paysim_csv, tmp_path):
    out, models = tmp_path / "outputs", tmp_path / "models"
    res = run_training(paysim_csv, sample_rows=8_000, rf_settings=FAST_RF,
                       model_dir=models, output_dir=out)
    saved = json.loads((out / "metrics.json").read_text())
    assert saved["run"]["run_id"] == res["run"]["run_id"]
    assert saved["sampling"]["mode"] == "uniform_random_sample"
    assert saved["sampling"]["rows_used"] == 8_000
    assert len(saved["dataset"]["sha256"]) == 64
    b = saved["split"]["step_boundaries"]
    assert b["train"][1] < b["validation"][0] and b["validation"][1] < b["test"][0]
    assert "step" not in saved["features"]
    rule = saved["models"]["Rule-based"]["test"]
    assert rule["pr_auc"] is None and rule["precision_at_budget"] is None
    for name in ("Logistic Regression", "Random Forest"):
        m = saved["models"][name]
        assert m["threshold_source"] in ("max_f1_on_validation",) or m["threshold"] == 0.5
        assert m["test"]["pr_auc_baseline"] == saved["split"]["test"]["prevalence"]
        bundle = joblib.load(models / m["model_file"])
        assert bundle["run_id"] == saved["run"]["run_id"]
    preview = (out / "test_preview.csv").read_text()
    assert "fraud_example" in preview and "random_test_sample" in preview


def test_full_data_guard(paysim_csv, tmp_path, monkeypatch):
    monkeypatch.setattr("src.train.FULL_DATA_GUARD_ROWS", 1_000)
    with pytest.raises(SystemExit):
        run_training(paysim_csv, rf_settings=FAST_RF, model_dir=tmp_path, output_dir=tmp_path)


def test_random_split_is_labelled(paysim_csv, tmp_path):
    res = run_training(paysim_csv, sample_rows=5_000, split="random", rf_settings=FAST_RF,
                       model_dir=tmp_path, output_dir=tmp_path)
    assert res["split"]["method"] == "random" and res["split"]["step_boundaries"] is None
    assert any("Random split" in w for w in res["warnings"])
