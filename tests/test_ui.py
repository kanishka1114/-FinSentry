"""Presentation helpers (src/ui.py) and the dashboard pages that read a saved robustness study."""
import json
import math
import shutil
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import pytest
from src import ui
from src.experiment import run_experiment
from src.robustness import run_study

FAST_RF = {"n_estimators": 10, "max_depth": 6, "min_samples_leaf": 5, "n_jobs": 1}
FAST_EVAL = {"bootstrap": 20, "block_bootstrap": 20}
FAST_BT = {"windows": 3, "bootstrap": 10, "min_test_fraud": 5}
SEEDS = (1, 2)


@pytest.fixture(scope="module")
def dirs(tmp_path_factory):
    from tests.conftest import make_paysim_like
    root = tmp_path_factory.mktemp("ui")
    csv = root / "d.csv"
    make_paysim_like().to_csv(csv, index=False)
    out, models = root / "outputs", root / "models"
    run_experiment(csv, sample_rows=10_000, rf_settings=FAST_RF, eval_overrides=FAST_EVAL,
                   backtest_overrides=FAST_BT, model_dir=models, output_dir=out)
    study = run_study(csv, 10_000, SEEDS, root=out / "robustness",
                      summary_path=out / "robustness_summary.md", rf_settings=FAST_RF,
                      eval_overrides=FAST_EVAL, backtest_overrides=FAST_BT)
    return {"out": out, "models": models, "study": study}


def test_formatters_never_show_undefined_as_zero():
    for fn in (ui.fmt_pct, ui.fmt_num, ui.fmt_count, ui.fmt_rate):
        assert fn(None) == "undefined" and fn(float("nan")) == "undefined"
    assert ui.fmt_pct(0.6524) == "65.2%" and ui.fmt_pct(0.0042, 2) == "0.42%"
    assert ui.fmt_num(0.34987) == "0.350" and ui.fmt_count(26120) == "26,120"
    assert ui.fmt_rate(1.6617) == "1.66" and ui.fmt_pct(0.0) == "0.0%"
    assert ui.fmt_range(0.564, 0.775, ui.fmt_pct) == "56.4% – 77.5%"
    assert ui.fmt_range(None, 0.5, ui.fmt_pct) == "undefined"
    df = pd.DataFrame({"p": [0.5, None, "n/a (no score)"], "n": [1234, None, 5]})
    shown = ui.format_frame(df, {"p": "pct", "n": "int", "missing": "pct"})
    assert list(shown["p"]) == ["50.0%", "undefined", "n/a (no score)"]
    assert list(shown["n"]) == ["1,234", "undefined", "5"]
    assert df["p"].iloc[0] == 0.5


def test_html_snippets_escape_text():
    evil = "<script>alert('x')</script>"
    cards = ui.kpi_cards([{"label": evil, "value": evil, "sub": evil, "help": "a 'quoted' <tip>"}])
    head = ui.brand_header(evil, [(evil, "warn")])
    for markup in (cards, head, ui.section(evil, evil), ui.note(evil, "warn")):
        assert "<script>" not in markup and "&lt;script&gt;" in markup
    assert "title='a &#x27;quoted&#x27; &lt;tip&gt;'" in cards
    assert "fs-badge warn" in head


def test_style_fig_adds_subtitle_and_clean_log_axis():
    ui.register_plotly_template()
    fig = go.Figure(go.Scatter(x=[1, 2], y=[0.001, 0.1]))
    fig.update_yaxes(type="log")
    out = ui.style_fig(fig, "Title", "Sub <b>", 300)
    assert "Title" in out.layout.title.text and "Sub &lt;b&gt;" in out.layout.title.text
    assert out.layout.height == 300 and out.layout.yaxis.dtick == 1


def test_load_robustness_flags_problems(dirs, tmp_path):
    assert ui.load_robustness(tmp_path) == (None, [])
    (tmp_path / "robustness").mkdir()
    (tmp_path / "robustness" / "robustness.json").write_text("{not json")
    study, problems = ui.load_robustness(tmp_path)
    assert study is None and "could not be read" in problems[0]

    study, problems = ui.load_robustness(dirs["out"])
    assert problems == [] and study["status"] == "complete"
    broken = tmp_path / "b"
    shutil.copytree(dirs["out"] / "robustness", broken / "robustness")
    path = broken / "robustness" / "robustness.json"
    data = json.loads(path.read_text())
    data["status"], data["n_aggregated"] = "incomplete", 1
    data["runs"][1]["usable"], data["runs"][1]["problems"] = False, ["did not finish"]
    path.write_text(json.dumps(data))
    _, problems = ui.load_robustness(broken)
    text = " ".join(problems)
    assert "incomplete" in text and "Only 1 of 2" in text and "did not finish" in text


def test_robustness_tables_match_saved_values(dirs):
    study, _ = ui.load_robustness(dirs["out"])
    entry = study["aggregate"]["single_split"]["Random Forest"]
    table = ui.robustness_summary_table(study, "Random Forest",
                                        [("pr_auc", "PR-AUC", "dec3"),
                                         ("max_f1.false_positive", "False alerts", "int"),
                                         ("not_a_metric", "skipped", "dec3")])
    assert list(table.columns) == ["Metric", "Seed 1", "Seed 2", "Mean", "SD", "Range"]
    assert list(table["Metric"]) == ["PR-AUC", "False alerts"]
    pr = table.iloc[0]
    for seed in ("1", "2"):
        assert pr[f"Seed {seed}"] == ui.fmt_num(entry["pr_auc"]["values"][seed])
    assert pr["Mean"] == ui.fmt_num(entry["pr_auc"]["between_seed"]["mean"])
    fa = entry["max_f1.false_positive"]["between_seed"]
    assert table.iloc[1]["Range"] == f"{fa['min']:,} – {fa['max']:,}"

    pts = ui.strategy_points(study)
    assert len(pts) == 2 * 2 * len(SEEDS)
    row = pts[(pts["Model"] == "Random Forest") & (pts["Strategy"] == "Max F1") & (pts["Seed"] == "1")]
    assert row["Precision"].iloc[0] == entry["max_f1.precision"]["values"]["1"]

    var = ui.robustness_variation_frame(study)
    t = study["aggregate"]["temporal"]["Random Forest"]["pr_auc"]
    rf = var[var["Model"] == "Random Forest"].iloc[0]
    assert rf["Between seeds (SD of window mean)"] == t["between_seed_of_window_mean"]["sd"]
    windows = ui.robustness_window_frame(study)
    assert set(windows["Seed"]) == {"1", "2"} and not windows["PR-AUC"].dropna().map(math.isnan).any()


def _app(monkeypatch, out, models):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("FINSENTRY_OUTPUT_DIR", str(out))
    monkeypatch.setenv("FINSENTRY_MODEL_DIR", str(models))
    return AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=120)


def _texts(at):
    return " ".join(m.value for m in list(at.markdown) + list(at.caption))


def test_dashboard_shows_saved_robustness_study(monkeypatch, dirs):
    study = dirs["study"]
    at = _app(monkeypatch, dirs["out"], dirs["models"]).run()
    assert not at.exception and not at.error
    assert "2-seed robustness study" in _texts(at) and study["study_id"] in _texts(at)
    at.sidebar.radio[0].set_value("Robustness Study").run()
    assert not at.exception, at.exception
    assert not at.error and not at.warning
    text = _texts(at)
    assert study["study_id"] in text and "Synthetic PaySim" in text
    assert "not the primary experiment" in text
    assert any("Seed 1" in df.value.columns for df in at.dataframe)
    at.sidebar.radio[0].set_value("Thresholds & Alerts").run()
    assert not at.exception and "Max-F1 thresholds are unstable" in _texts(at)


def test_dashboard_robustness_page_without_study(monkeypatch, dirs, tmp_path):
    out = tmp_path / "out"
    shutil.copytree(dirs["out"], out, ignore=shutil.ignore_patterns("robustness*"))
    at = _app(monkeypatch, out, dirs["models"]).run()
    at.sidebar.radio[0].set_value("Robustness Study").run()
    assert not at.exception and not at.error
    assert any("python -m src.robustness" in c.value for c in at.code)
    assert "Max-F1 thresholds are unstable" not in _texts(at)
