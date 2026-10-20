import json
import pytest
from src.robustness import (aggregate, build_study, build_summary, describe, run_seed,
                            run_study, single_split_values, window_values, seed_dir)
from src.report import load_outputs
from tests.conftest import make_paysim_like

FAST = {"rf_settings": {"n_estimators": 10, "max_depth": 6, "min_samples_leaf": 5, "n_jobs": 1},
        "eval_overrides": {"bootstrap": 20, "block_bootstrap": 20},
        "backtest_overrides": {"windows": 3, "bootstrap": 10, "min_test_fraud": 5}}
SEEDS = (1, 2, 3)


@pytest.fixture(scope="module")
def study_dirs(tmp_path_factory):
    root = tmp_path_factory.mktemp("rob")
    csv = root / "d.csv"
    make_paysim_like().to_csv(csv, index=False)
    primary = root / "primary"
    primary.mkdir()
    (primary / "metrics.json").write_text('{"primary": true}')
    study = run_study(csv, 10_000, SEEDS, root=primary / "robustness",
                      summary_path=primary / "robustness_summary.md", **FAST)
    return {"csv": csv, "primary": primary, "rob": primary / "robustness", "study": study}


def _strip(values):
    return json.loads(json.dumps(values))


def test_same_seed_reproduces_metrics(study_dirs, tmp_path):
    entry = run_seed(study_dirs["csv"], 10_000, 1, tmp_path, **FAST)
    assert entry["status"] == "complete"
    m_a, b_a, _ = load_outputs(seed_dir(study_dirs["rob"], 1))
    m_b, b_b, _ = load_outputs(seed_dir(tmp_path, 1))
    assert m_a["run"]["run_id"] != m_b["run"]["run_id"]
    assert _strip(single_split_values(m_a)) == _strip(single_split_values(m_b))
    assert _strip(window_values(b_a)) == _strip(window_values(b_b))
    m_2, _, _ = load_outputs(seed_dir(study_dirs["rob"], 2))
    assert single_split_values(m_a) != single_split_values(m_2)


def test_runs_are_isolated(study_dirs):
    rob, study = study_dirs["rob"], study_dirs["study"]
    assert (study_dirs["primary"] / "metrics.json").read_text() == '{"primary": true}'
    assert not (study_dirs["primary"] / "backtest.json").exists()
    ids = set()
    for seed in SEEDS:
        d = seed_dir(rob, seed)
        metrics, backtest, experiment = load_outputs(d)
        assert experiment["config"]["seed"] == metrics["run"]["seed"] == seed
        assert metrics["sampling"]["method"] == "uniform"
        assert any((d / "models").glob("*.joblib"))
        ids.add(experiment["experiment_id"])
    assert len(ids) == len(SEEDS)
    assert study["status"] == "complete" and study["n_aggregated"] == 3
    saved = json.loads((rob / "robustness.json").read_text())
    assert saved["study_id"] == study["study_id"]
    assert {r["seed"] for r in saved["runs"]} == set(SEEDS)
    assert all(len(r["sha256"]) == 64 and r["split_boundaries"] for r in saved["runs"])


def test_incomplete_and_failed_runs_are_excluded(study_dirs, tmp_path):
    import shutil
    rob = tmp_path / "rob"
    shutil.copytree(study_dirs["rob"], rob)
    exp_path = seed_dir(rob, 2) / "experiment.json"
    exp = json.loads(exp_path.read_text())
    exp["status"] = "running"
    exp_path.write_text(json.dumps(exp))
    study = build_study(rob, SEEDS, 10_000)
    assert study["status"] == "incomplete" and study["n_aggregated"] == 2
    assert study["aggregate"]["seeds"] == [1, 3]
    bad = next(r for r in study["runs"] if r["seed"] == 2)
    assert not bad["usable"] and any("did not finish" in p for p in bad["problems"])
    assert "NO" in build_summary(study)

    failed = run_seed(tmp_path / "missing.csv", 10_000, 9, tmp_path / "f", **FAST)
    assert failed["status"] == "failed" and failed["error"]
    study = build_study(tmp_path / "f", (9,), 10_000, run_entries=[failed])
    assert study["status"] == "incomplete" and study["aggregate"] is None
    assert "No completed" in build_summary(study)


def test_wrong_sample_size_is_not_aggregated(study_dirs):
    study = build_study(study_dirs["rob"], SEEDS, 20_000)
    assert study["n_aggregated"] == 0 and study["status"] == "incomplete"


def test_describe_and_aggregation_arithmetic():
    d = describe([0.1, 0.2, None, 0.3])
    assert d["n"] == 3 and d["n_undefined"] == 1
    assert d["mean"] == pytest.approx(0.2) and d["sd"] == pytest.approx(0.1)
    assert d["min"] == 0.1 and d["max"] == 0.3
    assert describe([None])["mean"] is None and describe([5])["sd"] is None
    counts = describe([3, 7, 5])
    assert counts["min"] == 3 and isinstance(counts["max"], int) and counts["sd"] == pytest.approx(2.0)

    def window(w, prev, lr, rf):
        return {"window": w, "test_steps": [w, w], "test_rows": 100, "test_fraud": 5,
                "prevalence": prev, "reliable": True,
                "models": {"Logistic Regression": {"pr_auc": lr, "pr_auc_lift": lr / prev},
                           "Random Forest": {"pr_auc": rf, "pr_auc_lift": rf / prev}}}

    def single(lr, rf):
        mk = lambda p: {"pr_auc": p, "max_f1.recall": 0.1, "alert_budget.recall": 0.2,
                        "max_f1.precision": 0.5, "alert_budget.precision": 0.3}
        return {"test_prevalence": 0.01, "models": {"Rule-based": {"rule.recall": 0.5},
                                                    "Logistic Regression": mk(lr),
                                                    "Random Forest": mk(rf)}}

    per_seed = {
        1: {"single": single(0.1, 0.3), "windows": [window(1, 0.01, 0.1, 0.2), window(2, 0.02, 0.3, 0.4)]},
        2: {"single": single(0.2, 0.4), "windows": [window(1, 0.01, 0.2, 0.1), window(2, 0.02, 0.4, 0.6)]},
    }
    agg = aggregate(per_seed)
    rf = agg["single_split"]["Random Forest"]["pr_auc"]
    assert rf["values"] == {1: 0.3, 2: 0.4} and rf["between_seed"]["mean"] == pytest.approx(0.35)
    t = agg["temporal"]["Random Forest"]["pr_auc"]
    assert t["per_seed"][1]["across_windows"]["mean"] == pytest.approx(0.3)
    assert t["per_seed"][2]["across_windows"]["mean"] == pytest.approx(0.35)
    # between-seed SD of window means (0.3, 0.35) vs mean of temporal SDs: kept separate
    assert t["between_seed_of_window_mean"]["sd"] == pytest.approx(0.0353553, rel=1e-4)
    assert t["mean_within_seed_temporal_sd"]["mean"] == pytest.approx(
        (0.1414214 + 0.3535534) / 2, rel=1e-4)
    assert agg["rankings"]["single_split"] == {1: "Random Forest", 2: "Random Forest"}
    assert agg["rankings"]["windows"][2] == {"Random Forest": 1, "Logistic Regression": 1,
                                             "tie": 0, "undefined": 0}


def test_summary_states_scope_and_limits(study_dirs):
    text = (study_dirs["primary"] / "robustness_summary.md").read_text()
    study = study_dirs["study"]
    assert study["study_id"] in text and "synthetic" in text.lower()
    assert "not a confidence interval" in text and "never the full dataset" in text
    for seed in SEEDS:
        assert f"seed {seed}" in text
    assert "python -m src.robustness --sample-rows 10000 --seeds 1 2 3" in text
    assert "Temporal" in text and "Between-seed" in text
