"""Report-ready Markdown summary built only from saved outputs (no recomputation).

Section A contains measured values copied from outputs/*.json. Section B is explanatory text.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from src.config import OUTPUT_DIR, SCHEMA_VERSION

SYNTHETIC_STATEMENT = (
    "PaySim is a synthetic simulation. These results describe models on simulated data and do "
    "not establish detection performance at any real bank or mobile-money provider.")


def consistency_checks(metrics=None, backtest=None, experiment=None):
    """Problems that would make the outputs stale or mutually inconsistent."""
    issues = []
    for label, d in (("metrics.json", metrics), ("backtest.json", backtest)):
        if d is not None and d.get("run", {}).get("schema_version", 0) < SCHEMA_VERSION:
            issues.append(f"{label} was written by an older FinSentry version; re-run to refresh it.")
    if metrics and backtest:
        m_s, b_s = metrics["sampling"], backtest["sampling"]
        diffs = [k for k in ("mode", "requested_rows", "max_step")
                 if m_s.get(k) != b_s.get(k)]
        if metrics["dataset"].get("sha256") != backtest["dataset"].get("sha256"):
            diffs.append("dataset fingerprint")
        if metrics["run"].get("seed") != backtest["run"].get("seed"):
            diffs.append("seed")
        if diffs:
            issues.append("metrics.json and backtest.json come from different configurations "
                          f"({', '.join(diffs)}); do not read them as one experiment.")
        me, be = metrics["run"].get("experiment_id"), backtest["run"].get("experiment_id")
        if me != be:
            issues.append(f"metrics.json (experiment {me}) and backtest.json (experiment {be}) "
                          "were produced by different runs.")
    if experiment:
        if experiment.get("status") != "complete":
            issues.append(f"experiment.json status is '{experiment.get('status')}' (run did not finish).")
        if metrics and experiment.get("train", {}).get("run_id") not in (None, metrics["run"]["run_id"]):
            issues.append("metrics.json is not the training run recorded in experiment.json.")
        if backtest and experiment.get("backtest", {}).get("run_id") not in (None, backtest["run"]["run_id"]):
            issues.append("backtest.json is not the backtest recorded in experiment.json.")
    for label, d in (("metrics.json", metrics), ("backtest.json", backtest)):
        if d and d["sampling"].get("method") == "stratified" and d["sampling"]["mode"] != "full":
            issues.append(f"{label} used stratified (label-informed) sampling; final chronological "
                          "results should use uniform sampling.")
    return issues


def _ratio(a, b):
    return a / b if a is not None and b else None


def key_facts(metrics=None, backtest=None):
    """Plain statements computed from saved numbers; no interpretation beyond arithmetic."""
    facts = []
    if metrics:
        test = metrics["split"]["test"]
        facts.append(f"Test period: {test['rows']:,} rows, {test['fraud']:,} fraud "
                     f"({fmt(test['prevalence'], pct=True)} prevalence = PR-AUC baseline); train "
                     f"prevalence {fmt(metrics['split']['train']['prevalence'], pct=True)}.")
        for name in ("Logistic Regression", "Random Forest"):
            meta = metrics["models"].get(name)
            if not meta:
                continue
            t = meta["test"]
            lift = _ratio(t.get("pr_auc"), t.get("pr_auc_baseline"))
            facts.append(
                f"{name}: PR-AUC {fmt(t.get('pr_auc'))} ({fmt(lift, digits=1)}× baseline); at the "
                f"max-F1 threshold precision {fmt(t['precision'])}, recall {fmt(t['recall'])}, "
                f"{t['false_positive']:,} false alerts, {t['false_negative']:,} missed fraud, "
                f"{fmt(t['alerts_per_1000'], digits=2)} alerts per 1,000.")
            b = meta.get("strategies", {}).get("alert_budget")
            if b:
                facts.append(
                    f"{name} alert budget: requested {fmt(b.get('requested_alerts_per_1000'), digits=1)}"
                    f"/1,000 on validation, realised {fmt(b['test']['alerts_per_1000'], digits=2)}/1,000 "
                    f"on test; precision {fmt(b['test']['precision'])}, recall {fmt(b['test']['recall'])}.")
            ratios = [r["width_ratio_block_vs_row"] for r in meta.get("interval_comparison", [])
                      if r["width_ratio_block_vs_row"] is not None]
            if ratios:
                facts.append(f"{name}: block-bootstrap intervals are {min(ratios):.1f}–"
                             f"{max(ratios):.1f}× the width of row-bootstrap intervals.")
    if backtest:
        for name in ("Logistic Regression", "Random Forest"):
            vals = [(w["window"], w["models"][name]["test_ranking"]["pr_auc"],
                     w["test"]["prevalence"]) for w in backtest["windows"]
                    if name in w["models"] and w["models"][name]["test_ranking"]["pr_auc"] is not None]
            if vals:
                lo, hi = min(v[1] for v in vals), max(v[1] for v in vals)
                facts.append(f"Backtest {name}: PR-AUC ranged {lo:.3f}–{hi:.3f} across "
                             f"{len(vals)} windows (window baselines "
                             f"{min(v[2] for v in vals):.4f}–{max(v[2] for v in vals):.4f}).")
        sparse = [w["window"] for w in backtest["windows"] if not w["reliable"]]
        if sparse:
            facts.append(f"Backtest windows with too few fraud cases for stable metrics: {sparse}.")
    return facts

MODEL_ORDER = ["Rule-based", "Logistic Regression", "Random Forest"]
LIMITATIONS = [
    "PaySim is synthetic; results are not evidence of real-world fraud-detection performance and "
    "do not establish deployment performance at a real bank.",
    "Intervals are conditional on the fitted model and the chosen thresholds; they omit "
    "threshold-selection and retraining uncertainty.",
    "With rare fraud, ECE is dominated by the lowest score bin; reliability bins are more informative.",
    "The optional refit protocol transfers thresholds under unverifiable assumptions and is "
    "excluded from headline results.",
    "Sample runs contain few fraud cases; many metrics have wide intervals.",
    "Bootstrap intervals are estimates under a stationarity assumption, not guarantees. The row "
    "bootstrap ignores time dependence; the block bootstrap keeps short-range dependence but "
    "cannot represent drift outside the test period.",
    "Thresholds chosen on validation may not transfer when prevalence or score distributions shift.",
    "Feature contributions describe model behaviour, not causes of fraud.",
    "Only amount, transaction type and hour of day are model inputs; balances and identifiers are "
    "excluded by design, which limits achievable performance.",
]


def fmt(v, pct=False, digits=3):
    if v is None:
        return "undefined"
    if isinstance(v, str):
        return v
    return f"{v:.2%}" if pct else f"{v:.{digits}f}" if isinstance(v, float) else f"{v:,}"


def ci_text(ci, key):
    if not ci or not ci.get("computed"):
        return "not computed" + (f" ({ci['reason']})" if ci and ci.get("reason") else "")
    iv = ci["intervals"].get(key)
    if not iv or iv["low"] is None:
        return "undefined"
    return f"{iv['low']:.3f} – {iv['high']:.3f} (n={iv['n_valid']})"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _single_split(m):
    s, samp, run = m["split"], m["sampling"], m["run"]
    lines = ["### A1. Single-split experiment", ""]
    lines.append(table(["Item", "Value"], [
        ["Run id / time (UTC)", f"{run['run_id']} / {run['timestamp_utc']}"],
        ["Command", f"`{run['command']}`"],
        ["Data", "FULL dataset" if samp["mode"] == "full" else
         f"SAMPLE ({samp.get('method', 'stratified')}) of {samp['rows_used']:,} rows"],
        ["Rows / fraud used", f"{samp['rows_used']:,} / {samp['fraud_used']:,}"],
        ["Dataset SHA-256", m["dataset"]["sha256"][:16] + "…"],
        ["Seed", run["seed"]],
        ["Split", s["method"]],
        ["Final fit requested", m.get("protocol", {}).get("final_fit_requested", "train_only")],
    ]))
    b = s.get("step_boundaries") or {}
    lines += ["", "**Periods**", "", table(
        ["Period", "Steps", "Rows", "Fraud", "Fraud rate"],
        [[k.title(), "–".join(map(str, b[k])) if k in b else "random", f"{s[k]['rows']:,}",
          f"{s[k]['fraud']:,}", fmt(s[k]["prevalence"], pct=True)]
         for k in ("train", "validation", "test")])]

    rows = []
    for name in MODEL_ORDER:
        meta = m["models"].get(name)
        if not meta:
            continue
        t = meta["test"]
        is_rule = meta["kind"] == "rule"
        rows.append([name, "rule" if is_rule else meta.get("final_fit", {}).get("used", "n/a"),
                     "n/a (fixed rule)" if is_rule else fmt(meta.get("threshold")),
                     fmt(t["precision"]), fmt(t["recall"]),
                     fmt(t["f1"]), t["false_positive"], t["false_negative"],
                     fmt(t.get("alerts_per_1000"), digits=2),
                     f"{fmt(t.get('pr_auc'))} (baseline {fmt(t.get('pr_auc_baseline'), digits=4)})",
                     fmt(t.get("roc_auc"))])
    lines += ["", "**Final test results (max-F1 threshold for models)**", "", table(
        ["Model", "Final fit", "Threshold", "Precision", "Recall", "F1", "FP", "FN",
         "Alerts/1k", "PR-AUC", "ROC-AUC"], rows)]

    rows = []
    for name in MODEL_ORDER:
        meta = m["models"].get(name, {})
        for key, st in meta.get("strategies", {}).items():
            for period in ("validation", "test"):
                p = st[period]
                thr = "n/a (fixed rule)" if st["threshold"] is None else fmt(st["threshold"])
                rows.append([name, key, period.upper(), thr, fmt(p["precision"]),
                             fmt(p["recall"]), p["false_positive"], p["false_negative"],
                             fmt(p["alerts_per_1000"], digits=2)])
    lines += ["", "**Threshold strategies** (VALIDATION was used to choose; TEST is the result)",
              "", table(["Model", "Strategy", "Period", "Threshold", "Precision", "Recall", "FP",
                         "FN", "Alerts/1k"], rows)]

    rows = []
    for name in MODEL_ORDER:
        meta = m["models"].get(name, {})
        strat = "rule" if meta.get("kind") == "rule" else "max_f1"
        keys = [f"{strat}.precision", f"{strat}.recall"] + (["pr_auc"] if strat != "rule" else [])
        for k in keys:
            rows.append([name, k, ci_text(meta.get("ci"), k), ci_text(meta.get("block_ci"), k)])
    lines += ["", "**95% bootstrap intervals (estimates, not guarantees)**", "",
              table(["Model", "Metric", "Row bootstrap", "Block bootstrap (steps)"], rows)]

    rows = []
    for name in ("Logistic Regression", "Random Forest"):
        meta = m["models"].get(name, {})
        c, d = meta.get("calibration") or {}, meta.get("drift") or {}
        rows.append([name, fmt(c.get("brier_raw"), digits=5), fmt(c.get("brier_calibrated"), digits=5),
                     fmt(c.get("brier_constant_baseline"), digits=5), fmt(c.get("ece_raw"), digits=4),
                     fmt(c.get("ece_calibrated"), digits=4), fmt(d.get("psi")), d.get("psi_band", "")])
    lines += ["", "**Calibration and score drift (train-only model; validation → test)**", "",
              table(["Model", "Brier raw", "Brier calibrated", "Brier constant", "ECE raw",
                     "ECE calibrated", "PSI", "PSI band"], rows)]
    if m.get("warnings"):
        lines += ["", "**Warnings recorded during the run**", ""] + [f"- {w}" for w in m["warnings"]]
    return lines


def _backtest(bt):
    samp, bset = bt["sampling"], bt["backtest_settings"]
    lines = ["", "### A2. Temporal backtest", "",
             f"- Data: {'FULL dataset' if samp['mode'] == 'full' else 'SAMPLE of ' + format(samp['rows_used'], ',') + ' rows'}"
             f"; scheme: {bset['scheme']}; windows: {len(bt['windows'])}; run {bt['run']['run_id']} "
             f"({bt['run']['timestamp_utc']}); command `{bt['run']['command']}`", ""]
    rows = []
    for w in bt["windows"]:
        lr, rf = w["models"].get("Logistic Regression"), w["models"].get("Random Forest")
        pr = lambda ev: fmt(ev["test_ranking"]["pr_auc"]) if ev else "skipped"
        psi = lambda ev: fmt((ev.get("drift") or {}).get("psi")) if ev else "skipped"
        rows.append([w["window"], "–".join(map(str, w["test_steps"])), f"{w['test']['rows']:,}",
                     w["test"]["fraud"], fmt(w["test"]["prevalence"], pct=True), pr(lr), pr(rf),
                     psi(rf), "yes" if w["reliable"] else "NO"])
    lines.append(table(["Window", "Test steps", "Rows", "Fraud", "Fraud rate (PR-AUC baseline)",
                        "LR PR-AUC", "RF PR-AUC", "RF score PSI", "Enough fraud"], rows))
    rows = []
    for name in MODEL_ORDER:
        for key, v in bt["summary"].get(name, {}).items():
            if v["n_windows"]:
                rows.append([name, key, v["n_windows"], fmt(v["mean"]), fmt(v["std"]),
                             fmt(v["min"]), fmt(v["max"])])
    lines += ["", "**Across-window variation**", "",
              table(["Model", "Metric", "Windows", "Mean", "Std", "Min", "Max"], rows)]
    if bt.get("warnings"):
        lines += ["", "**Backtest warnings**", ""] + [f"- {w}" for w in bt["warnings"]]
    return lines


def build_report(metrics=None, backtest=None, experiment=None):
    lines = ["# FinSentry — results summary (auto-generated)", "",
             f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} from saved "
             "outputs._", "", f"> **{SYNTHETIC_STATEMENT}**", ""]
    src_rows = []
    for label, d in (("experiment.json", experiment), ("metrics.json", metrics),
                     ("backtest.json", backtest)):
        if d is None:
            src_rows.append([label, "missing", "", "", ""])
            continue
        run = d.get("run", d)
        src_rows.append([label, run.get("run_id", "—"), run.get("experiment_id", "—"),
                         run.get("timestamp_utc", run.get("started_utc", "—")),
                         run.get("schema_version", "—")])
    lines += ["**Sources**", "", table(["File", "Run id", "Experiment id", "Time (UTC)", "Schema"],
                                        src_rows), ""]
    issues = consistency_checks(metrics, backtest, experiment)
    lines += (["**⚠ Consistency problems — results below may be stale or mixed:**", ""] +
              [f"- {i}" for i in issues] + [""]) if issues else \
             ["**Consistency check:** passed (same configuration, current schema).", ""]
    lines += ["## A. Measured results", "",
              "Every number in this section is copied from `outputs/metrics.json` or "
              "`outputs/backtest.json`. 'undefined' means the metric does not exist for that data "
              "(for example recall with no fraud); it is never replaced by zero.", ""]
    facts = key_facts(metrics, backtest)
    if facts:
        lines += ["### Key measured facts", ""] + [f"- {f}" for f in facts] + [""]
    if metrics:
        lines += _single_split(metrics)
    else:
        lines.append("_No single-split results yet (run `python -m src.experiment`)._")
    if backtest:
        lines += _backtest(backtest)
    else:
        lines += ["", "_No backtest results yet (run `python -m src.backtest --sample-rows 1000000`)._"]

    notes = (metrics or {}).get("notes", {}) or (backtest or {}).get("notes", {})
    lines += ["", "## B. Interpretation guide (explanatory text, not measurements)", ""]
    if metrics and metrics.get("protocol"):
        lines += ["**Evaluation protocol**", ""] + [f"- {s}" for s in metrics["protocol"]["steps"]]
        lines += ["", f"- {metrics['protocol']['test_usage']}", ""]
    for title, key in (("Thresholds", None), ("Row bootstrap", "row_bootstrap"),
                       ("Block bootstrap", "block_bootstrap"), ("Calibration", "calibration"),
                       ("Drift", "drift"), ("Explanations", "explanations")):
        text = (metrics or backtest or {}).get("interpretation") if key is None else notes.get(key)
        if key is None and not text and backtest:
            text = backtest["notes"].get("thresholds")
        if text:
            lines += [f"**{title}.** {text}", ""]
    lines += ["**Limitations**", ""] + [f"- {x}" for x in LIMITATIONS]
    if experiment:
        repro = [experiment["command"].replace("experiment.py", "python -m src.experiment")]
    else:
        repro = [(metrics or {}).get("run", {}).get("command", "python -m src.experiment")
                 .replace("train.py", "python -m src.train"),
                 (backtest or {}).get("run", {}).get("command", "python -m src.backtest --sample-rows 1000000")
                 .replace("backtest.py", "python -m src.backtest")]
    lines += ["", "## C. Reproduce", "", "```bash", *repro, "streamlit run app.py", "```"]
    if metrics:
        v = metrics["run"]["versions"]
        lines += ["", "Library versions: " + ", ".join(f"{k} {val}" for k, val in v.items()
                                                        if k != "platform" and val)]
    return "\n".join(lines) + "\n"


def _load(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def load_outputs(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    return (_load(output_dir / "metrics.json"), _load(output_dir / "backtest.json"),
            _load(output_dir / "experiment.json"))


def generate_report(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    text = build_report(*load_outputs(output_dir))
    path = output_dir / "report_summary.md"
    path.write_text(text, encoding="utf-8")
    return path


if __name__ == "__main__":
    argparse.ArgumentParser(description="Rebuild outputs/report_summary.md from saved outputs.").parse_args()
    print(generate_report())
