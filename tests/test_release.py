"""Release checks: the experiment workflow, metric consistency, stale-output detection and a
dashboard smoke test on a tiny synthetic run."""
import json
from pathlib import Path
import pandas as pd
import pytest
from src.experiment import run_experiment
from src.report import build_report, consistency_checks, load_outputs
from src.train import run_training

FAST_RF = {"n_estimators": 10, "max_depth": 6, "min_samples_leaf": 5, "n_jobs": 1}
FAST_EVAL = {"bootstrap": 20, "block_bootstrap": 20}
FAST_BT = {"windows": 3, "bootstrap": 10, "min_test_fraud": 5}
PAGES = ["Overview", "Transaction Explorer", "Model Comparison", "Thresholds & Alerts",
         "Temporal Backtest", "Robustness Study", "Reliability & Calibration", "Explainability",
         "Try a Transaction", "Report Summary"]


@pytest.fixture(scope="module")
def experiment_dirs(tmp_path_factory):
    from tests.conftest import make_paysim_like
    root = tmp_path_factory.mktemp("exp")
    csv = root / "paysim_small.csv"
    make_paysim_like().to_csv(csv, index=False)
    out, models = root / "outputs", root / "models"
    rec = run_experiment(csv, sample_rows=10_000, rf_settings=FAST_RF, eval_overrides=FAST_EVAL,
                         backtest_overrides=FAST_BT, model_dir=models, output_dir=out)
    return {"csv": csv, "out": out, "models": models, "record": rec}


def test_experiment_records_one_consistent_run(experiment_dirs):
    out, rec = experiment_dirs["out"], experiment_dirs["record"]
    metrics, backtest, experiment = load_outputs(out)
    assert experiment["status"] == "complete" and experiment["config"]["sampling"] == "uniform"
    assert metrics["sampling"]["method"] == backtest["sampling"]["method"] == "uniform"
    assert metrics["run"]["experiment_id"] == backtest["run"]["experiment_id"] == rec["experiment_id"]
    assert experiment["train"]["run_id"] == metrics["run"]["run_id"]
    assert experiment["backtest"]["run_id"] == backtest["run"]["run_id"]
    assert len(experiment["dataset"]["sha256"]) == 64 and "scikit-learn" in experiment["versions"]
    assert consistency_checks(metrics, backtest, experiment) == []
    text = (out / "report_summary.md").read_text()
    assert rec["experiment_id"] in text and "synthetic" in text.lower()


def test_reported_metrics_match_predictions(experiment_dirs):
    out = experiment_dirs["out"]
    metrics, _, _ = load_outputs(out)
    n_test = metrics["split"]["test"]["rows"]
    preview = pd.read_csv(out / "test_preview.csv")
    assert (preview["run_id"] == metrics["run"]["run_id"]).all()
    for name, col in (("Logistic Regression", "lr"), ("Random Forest", "rf")):
        m = metrics["models"][name]
        for strat in m["strategies"].values():
            t = strat["test"]
            assert t["true_positive"] + t["false_positive"] + t["true_negative"] + t["false_negative"] == n_test
            assert t["alerts"] == t["true_positive"] + t["false_positive"]
            if t["alerts"]:
                assert t["precision"] == pytest.approx(t["true_positive"] / t["alerts"])
            fraud = t["true_positive"] + t["false_negative"]
            assert fraud == metrics["split"]["test"]["fraud"]
            assert t["recall"] == pytest.approx(t["true_positive"] / fraud)
        assert m["test"]["f1"] == m["strategies"]["max_f1"]["test"]["f1"]
        top = next(r for r in m["top_k"] if r["label"] == "top 100")
        assert top["precision"] == pytest.approx(m["test"]["precision_at_budget"])
        # preview scores are rounded to 5 decimals; alerts must agree away from that rounding
        s, a, thr = preview[f"{col}_score"], preview[f"{col}_alert"], m["threshold"]
        assert (s[a == 1] >= thr - 1e-5).all() and (s[a == 0] <= thr + 1e-5).all()


def test_stale_and_mixed_outputs_are_flagged(experiment_dirs):
    metrics, backtest, experiment = load_outputs(experiment_dirs["out"])
    other = json.loads(json.dumps(backtest))
    other["run"]["experiment_id"], other["sampling"]["mode"] = "exp-other", "stratified_random_sample"
    other["sampling"]["method"] = "stratified"
    issues = consistency_checks(metrics, other, None)
    assert any("different configurations" in i for i in issues)
    assert any("different runs" in i for i in issues)
    assert any("stratified" in i for i in issues)
    old = json.loads(json.dumps(metrics))
    old["run"]["schema_version"] = 1
    assert any("older FinSentry" in i for i in consistency_checks(old, None, None))
    unfinished = {**experiment, "status": "running"}
    assert any("did not finish" in i for i in consistency_checks(metrics, backtest, unfinished))
    replaced = json.loads(json.dumps(metrics))
    replaced["run"]["run_id"] = "someotherrun"
    assert any("not the training run" in i for i in consistency_checks(replaced, backtest, experiment))
    assert "consistency" in build_report(old, None, None).lower()


def test_skip_backtest_removes_previous_backtest(tmp_path):
    from tests.conftest import make_paysim_like
    csv = tmp_path / "d.csv"
    make_paysim_like(n=8_000).to_csv(csv, index=False)
    (tmp_path / "backtest.json").write_text("{}")
    rec = run_experiment(csv, sample_rows=6_000, skip_backtest=True, rf_settings=FAST_RF,
                         eval_overrides=FAST_EVAL, model_dir=tmp_path, output_dir=tmp_path)
    assert not (tmp_path / "backtest.json").exists() and "backtest" not in rec
    metrics, backtest, experiment = load_outputs(tmp_path)
    assert backtest is None and consistency_checks(metrics, backtest, experiment) == []


def test_f1_undefined_without_fraud_in_training_outputs(tmp_path):
    from tests.conftest import make_paysim_like
    df = make_paysim_like(n=10_000, steps=100, seed=5)
    df.loc[df["step"] > 85, "isFraud"] = 0
    csv = tmp_path / "nofraud_test.csv"
    df.to_csv(csv, index=False)
    res = run_training(csv, rf_settings=FAST_RF, eval_settings=FAST_EVAL,
                       model_dir=tmp_path, output_dir=tmp_path)
    assert res["split"]["test"]["fraud"] == 0
    for name in ("Logistic Regression", "Random Forest", "Rule-based"):
        t = res["models"][name]["test"]
        assert t["recall"] is None and t["f1"] is None
        assert t["pr_auc"] is None


def _app(monkeypatch, out, models):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("FINSENTRY_OUTPUT_DIR", str(out))
    monkeypatch.setenv("FINSENTRY_MODEL_DIR", str(models))
    app = Path(__file__).resolve().parents[1] / "app.py"
    return AppTest.from_file(str(app), default_timeout=120)


def test_dashboard_without_outputs_shows_instructions(monkeypatch, tmp_path):
    at = _app(monkeypatch, tmp_path / "none", tmp_path / "none").run()
    assert not at.exception
    assert any("python -m src" in c.value for c in at.code)


def test_dashboard_all_pages_render(monkeypatch, experiment_dirs):
    at = _app(monkeypatch, experiment_dirs["out"], experiment_dirs["models"]).run()
    assert not at.exception, at.exception
    assert not at.error, [e.value for e in at.error]
    for page in PAGES:
        at.sidebar.radio[0].set_value(page).run()
        assert not at.exception, (page, at.exception)
        assert not at.error, (page, [e.value for e in at.error])


def test_dashboard_scores_manual_transaction(monkeypatch, experiment_dirs):
    at = _app(monkeypatch, experiment_dirs["out"], experiment_dirs["models"]).run()
    at.sidebar.radio[0].set_value("Try a Transaction").run()
    buttons = [b for b in at.button if "analyze transaction" in b.label.lower()]
    assert len(buttons) == 1
    buttons[0].click().run()
    assert not at.exception, at.exception
    assert any("contribut" in m.value.lower() or "log-odds" in m.value.lower()
               for m in list(at.markdown) + list(at.caption))


def test_dashboard_ignores_preview_from_other_run(monkeypatch, experiment_dirs, tmp_path):
    import shutil
    out = tmp_path / "out"
    shutil.copytree(experiment_dirs["out"], out)
    preview = pd.read_csv(out / "test_preview.csv")
    preview["run_id"] = "stale"
    preview.to_csv(out / "test_preview.csv", index=False)
    at = _app(monkeypatch, out, experiment_dirs["models"]).run()
    assert not at.exception
    assert any("different training run" in e.value for e in at.error)
