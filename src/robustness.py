"""Multi-seed robustness study built on the recommended experiment.

Each seed draws its own uniform sample and runs the unchanged protocol (chronological split,
validation-only thresholds, walk-forward backtest) into outputs/robustness/seed_<seed>/, so runs
never overwrite each other or the primary outputs. The study then summarises only runs that
finished and passed the consistency checks.

Two kinds of variation are kept apart:
- between-seed: the same statistic from different samples/model seeds (n = number of seeds);
- temporal: variation across backtest windows within one seed.
With three seeds the spread is descriptive only; it is not a population confidence interval.
"""
import argparse
import json
import resource
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from src.config import OUTPUT_DIR, SCHEMA_VERSION
from src.experiment import run_experiment, DEFAULT_SAMPLE_ROWS
from src.report import consistency_checks, load_outputs, fmt, table, SYNTHETIC_STATEMENT
from src.train import library_versions

DEFAULT_SEEDS = (42, 123, 2026)
ROBUSTNESS_DIR = OUTPUT_DIR / "robustness"
MODELS = ("Rule-based", "Logistic Regression", "Random Forest")
SCORED = ("Logistic Regression", "Random Forest")
POINT_KEYS = ("precision", "recall", "f1", "false_positive", "false_negative", "alerts_per_1000")
SPREAD_NOTE = ("Spread across seeds is descriptive (n = number of completed seeds). With three "
               "seeds it is not a confidence interval for the population of possible samples, "
               "and per-run bootstrap intervals are not pooled.")


def peak_rss_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / 1e6 if sys.platform == "darwin" else rss / 1e3, 1)


def describe(values):
    """Mean, sample SD, min, max over defined values; undefined values are counted, not zeroed."""
    raw = [v for v in values if v is not None]
    vals = [float(v) for v in raw]
    out = {"n": len(vals), "n_undefined": len(values) - len(vals)}
    if not vals:
        return {**out, "mean": None, "sd": None, "min": None, "max": None}
    return {**out, "mean": float(np.mean(vals)),
            "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "min": min(raw), "max": max(raw)}


def seed_dir(root, seed):
    return Path(root) / f"seed_{seed}"


def run_seed(data_path, sample_rows, seed, root, **experiment_kwargs):
    """Run one experiment into its own directory; failures are recorded, never hidden."""
    out = seed_dir(root, seed)
    t0 = time.time()
    entry = {"seed": seed, "dir": str(out)}
    try:
        rec = run_experiment(data_path, sample_rows=sample_rows, seed=seed,
                             model_dir=out / "models", output_dir=out, **experiment_kwargs)
        entry.update(status=rec["status"], experiment_id=rec["experiment_id"])
    except (Exception, SystemExit) as err:
        entry.update(status="failed", error=f"{type(err).__name__}: {err}",
                     traceback=traceback.format_exc(limit=5))
    entry["seconds"] = round(time.time() - t0, 2)
    entry["process_peak_rss_mb"] = peak_rss_mb()
    return entry


def inspect_run(seed, root, sample_rows):
    """Load one seed's outputs and decide whether it may enter the aggregate."""
    out = seed_dir(root, seed)
    metrics, backtest, experiment = load_outputs(out)
    info = {"seed": seed, "dir": str(out), "usable": False, "problems": []}
    if experiment is None or metrics is None or backtest is None:
        info["problems"].append("missing experiment.json, metrics.json or backtest.json")
        return info, None
    info["problems"] += consistency_checks(metrics, backtest, experiment)
    if experiment["config"]["seed"] != seed or metrics["run"]["seed"] != seed:
        info["problems"].append("seed in outputs does not match the directory")
    if metrics["sampling"].get("requested_rows") != sample_rows:
        info["problems"].append(f"sample size {metrics['sampling'].get('requested_rows')} "
                                f"differs from requested {sample_rows}")
    info.update({
        "status": experiment.get("status"), "experiment_id": experiment["experiment_id"],
        "train_run_id": metrics["run"]["run_id"], "backtest_run_id": backtest["run"]["run_id"],
        "sha256": metrics["dataset"].get("sha256"), "sampling": metrics["sampling"],
        "split_boundaries": metrics["split"]["step_boundaries"],
        "backtest_windows": [{k: w[k] for k in ("window", "train_steps", "validation_steps",
                                                "test_steps")} for w in backtest["windows"]],
        "seconds": experiment.get("seconds"),
        "warnings": metrics.get("warnings", []) + backtest.get("warnings", []),
    })
    info["usable"] = not info["problems"]
    return info, (metrics, backtest)


def single_split_values(metrics):
    """Per-model test values at both validation-chosen thresholds (max F1, alert budget)."""
    test = metrics["split"]["test"]
    out = {"test_prevalence": test["prevalence"], "test_rows": test["rows"],
           "test_fraud": test["fraud"], "models": {}}
    for name in MODELS:
        m = metrics["models"][name]
        vals = {"pr_auc": m["test"].get("pr_auc")}
        for strat, s in m["strategies"].items():
            for k in POINT_KEYS:
                vals[f"{strat}.{k}"] = s["test"].get(k)
        out["models"][name] = vals
    return out


def window_values(backtest):
    rows = []
    for w in backtest["windows"]:
        row = {"window": w["window"], "test_steps": w["test_steps"], "test_rows": w["test"]["rows"],
               "test_fraud": w["test"]["fraud"], "prevalence": w["test"]["prevalence"],
               "reliable": w.get("reliable"), "models": {}}
        for name in MODELS:
            ev = w["models"].get(name)
            if ev is None:
                continue
            r = ev["test_ranking"]
            vals = {"pr_auc": r.get("pr_auc"), "pr_auc_baseline": r.get("pr_auc_baseline"),
                    "pr_auc_lift": (r["pr_auc"] / r["pr_auc_baseline"]
                                    if r.get("pr_auc") is not None and r.get("pr_auc_baseline")
                                    else None)}
            for strat, s in ev["strategies"].items():
                for k in POINT_KEYS:
                    vals[f"{strat}.{k}"] = s["test"].get(k)
            row["models"][name] = vals
        rows.append(row)
    return rows


def aggregate(per_seed):
    """per_seed: {seed: {"single": single_split_values, "windows": window_values}}."""
    seeds = sorted(per_seed)
    single = {"test_prevalence": {"values": {s: per_seed[s]["single"]["test_prevalence"]
                                             for s in seeds}}}
    single["test_prevalence"]["between_seed"] = describe(list(single["test_prevalence"]["values"].values()))
    for name in MODELS:
        keys = per_seed[seeds[0]]["single"]["models"][name].keys() if seeds else []
        single[name] = {}
        for k in keys:
            values = {s: per_seed[s]["single"]["models"][name].get(k) for s in seeds}
            single[name][k] = {"values": values, "between_seed": describe(list(values.values()))}

    temporal = {}
    for name in MODELS:
        temporal[name] = {}
        for k in ("pr_auc", "pr_auc_lift", "max_f1.precision", "max_f1.recall",
                  "max_f1.false_positive", "max_f1.false_negative", "max_f1.alerts_per_1000",
                  "alert_budget.precision", "alert_budget.recall", "alert_budget.alerts_per_1000",
                  "rule.precision", "rule.recall", "rule.false_positive", "rule.false_negative",
                  "rule.alerts_per_1000"):
            per = {}
            for s in seeds:
                vals = [w["models"].get(name, {}).get(k) for w in per_seed[s]["windows"]]
                if all(v is None for v in vals):
                    continue
                per[s] = {"values": vals, "across_windows": describe(vals)}
            if not per:
                continue
            temporal[name][k] = {
                "per_seed": per,
                # between-seed spread of each seed's window mean, and the typical within-seed
                # (temporal) SD: two different questions, never pooled into one number
                "between_seed_of_window_mean": describe([p["across_windows"]["mean"] for p in per.values()]),
                "mean_within_seed_temporal_sd": describe([p["across_windows"]["sd"] for p in per.values()]),
            }
    prevalence = {s: [w["prevalence"] for w in per_seed[s]["windows"]] for s in seeds}

    rankings = {"single_split": {}, "windows": {}}
    for s in seeds:
        m = per_seed[s]["single"]["models"]
        lr, rf = m["Logistic Regression"]["pr_auc"], m["Random Forest"]["pr_auc"]
        rankings["single_split"][s] = (None if lr is None or rf is None else
                                       "Random Forest" if rf > lr else
                                       "Logistic Regression" if lr > rf else "tie")
        wins = {"Random Forest": 0, "Logistic Regression": 0, "tie": 0, "undefined": 0}
        for w in per_seed[s]["windows"]:
            a = w["models"].get("Logistic Regression", {}).get("pr_auc")
            b = w["models"].get("Random Forest", {}).get("pr_auc")
            key = ("undefined" if a is None or b is None else "Random Forest" if b > a
                   else "Logistic Regression" if a > b else "tie")
            wins[key] += 1
        rankings["windows"][s] = wins
    beats_baseline = {name: {s: sum(1 for w in per_seed[s]["windows"]
                                    if (w["models"].get(name, {}).get("pr_auc_lift") or 0) > 1)
                             for s in seeds} for name in SCORED}
    rankings["windows_above_baseline"] = beats_baseline
    rankings["windows_per_seed"] = {s: len(per_seed[s]["windows"]) for s in seeds}
    return {"seeds": seeds, "single_split": single, "temporal": temporal,
            "window_prevalence": prevalence, "rankings": rankings,
            "per_seed_windows": {s: per_seed[s]["windows"] for s in seeds}}


def consistency_findings(agg):
    """Plain statements derived only from the aggregate; 'inconclusive' when evidence is mixed."""
    seeds, find = agg["seeds"], []
    if len(seeds) < 2:
        return ["Fewer than two completed seeds: stability cannot be assessed."]
    ss = agg["rankings"]["single_split"]
    leaders = set(ss.values())
    if len(leaders) == 1 and None not in leaders:
        find.append(f"Single split: {leaders.pop()} had the higher test PR-AUC in all "
                    f"{len(seeds)} seeds.")
    else:
        find.append(f"Single split: the PR-AUC leader changed between seeds ({ss}); ranking is "
                    "inconclusive.")
    total = {k: sum(agg["rankings"]["windows"][s][k] for s in seeds)
             for k in ("Random Forest", "Logistic Regression", "tie", "undefined")}
    n = sum(agg["rankings"]["windows_per_seed"].values())
    find.append(f"Backtest: Random Forest had the higher PR-AUC in {total['Random Forest']} of {n} "
                f"seed-windows, Logistic Regression in {total['Logistic Regression']}"
                + (f", undefined in {total['undefined']}" if total["undefined"] else "") + ".")
    for name in SCORED:
        above = sum(agg["rankings"]["windows_above_baseline"][name].values())
        find.append(f"{name} PR-AUC exceeded its window's fraud-rate baseline in {above} of {n} "
                    "seed-windows.")
    for name in SCORED:
        t = agg["temporal"][name].get("pr_auc")
        if not t:
            continue
        between = t["between_seed_of_window_mean"]["sd"]
        within = t["mean_within_seed_temporal_sd"]["mean"]
        if between is not None and within is not None:
            bigger = "larger" if within > between else "smaller"
            find.append(f"{name}: typical temporal SD of window PR-AUC ({within:.3f}) is {bigger} "
                        f"than the between-seed SD of the window mean ({between:.3f}).")
    for name in SCORED:
        mf = agg["single_split"][name]
        r_f1, r_b = mf["max_f1.recall"]["between_seed"], mf["alert_budget.recall"]["between_seed"]
        p_f1, p_b = mf["max_f1.precision"]["between_seed"], mf["alert_budget.precision"]["between_seed"]
        vals_ok = all(x["n"] == len(seeds) for x in (r_f1, r_b, p_f1, p_b))
        if not vals_ok:
            find.append(f"{name}: threshold trade-off undefined in some seeds; inconclusive.")
            continue
        recall_up = all((mf["alert_budget.recall"]["values"][s] or 0) >= (mf["max_f1.recall"]["values"][s] or 0)
                        for s in seeds)
        prec_down = all((mf["alert_budget.precision"]["values"][s] or 0) <= (mf["max_f1.precision"]["values"][s] or 0)
                        for s in seeds)
        if recall_up and prec_down:
            find.append(f"{name}: in every seed the alert-budget threshold traded precision for "
                        "recall relative to max F1 (consistent operational trade-off).")
        else:
            find.append(f"{name}: the direction of the max-F1 vs alert-budget trade-off was not "
                        "the same in every seed; inconclusive.")
    return find


def build_summary(study):
    agg, L = study.get("aggregate"), []
    L += ["# FinSentry — multi-seed robustness summary (auto-generated)", "",
          f"_Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} from "
          "outputs/robustness/robustness.json._", "",
          f"> **{SYNTHETIC_STATEMENT}** Every run uses an independent uniform sample of "
          f"{study['config']['sample_rows']:,} rows, never the full dataset; full-data results "
          "were not measured.", "",
          f"Study `{study['study_id']}` · status **{study['status']}** · seeds "
          f"{', '.join(map(str, study['config']['seeds']))}", ""]
    rows = []
    for r in study["runs"]:
        rows.append([r["seed"], r.get("status", "?"), "yes" if r.get("usable") else "NO",
                     r.get("experiment_id", "—"), r.get("train_run_id", "—"),
                     r.get("backtest_run_id", "—"), (r.get("sha256") or "—")[:12],
                     fmt(r.get("seconds"), digits=0) if r.get("seconds") else "—",
                     "; ".join(r.get("problems", [])) or r.get("error", "") or "none"])
    L += ["## Runs", "", table(["Seed", "Status", "Aggregated", "Experiment", "Train run",
                                "Backtest run", "Data SHA-256", "Seconds", "Problems"], rows), ""]
    if study.get("resources"):
        res = study["resources"]
        L += [f"Total wall time {res['seconds']:.0f} s; process peak RSS {res['process_peak_rss_mb']} MB "
              "(measured with getrusage for the whole study process).", ""]
    if not agg:
        L += ["_No completed, consistent runs to aggregate._"]
        return "\n".join(L) + "\n"
    seeds = agg["seeds"]
    L += ["Boundaries (single split, steps):", ""]
    L += [table(["Seed", "Train", "Validation", "Test"],
                [[r["seed"], *("–".join(map(str, r["split_boundaries"][p]))
                               for p in ("train", "validation", "test"))]
                 for r in study["runs"] if r.get("usable")]), ""]

    L += ["## A. Measured results", "", "Every number below is copied or computed arithmetically "
          "from the per-seed JSON outputs. SD is the sample SD across seeds (n shown). " + SPREAD_NOTE, ""]
    L += ["### A1. Single chronological split — between-seed variation", ""]
    prev = agg["single_split"]["test_prevalence"]
    L += [f"Test fraud prevalence (= PR-AUC baseline) per seed: "
          + ", ".join(f"{s}: {fmt(v, pct=True)}" for s, v in prev["values"].items())
          + f" (mean {fmt(prev['between_seed']['mean'], pct=True)}).", ""]
    labels = [("pr_auc", "PR-AUC"), ("max_f1.precision", "Precision (max F1)"),
              ("max_f1.recall", "Recall (max F1)"), ("max_f1.false_positive", "False alerts (max F1)"),
              ("max_f1.false_negative", "Missed fraud (max F1)"),
              ("max_f1.alerts_per_1000", "Alerts / 1,000 (max F1)"),
              ("alert_budget.precision", "Precision (alert budget)"),
              ("alert_budget.recall", "Recall (alert budget)"),
              ("alert_budget.false_positive", "False alerts (alert budget)"),
              ("alert_budget.false_negative", "Missed fraud (alert budget)"),
              ("alert_budget.alerts_per_1000", "Alerts / 1,000 (alert budget)"),
              ("rule.precision", "Precision (rule)"), ("rule.recall", "Recall (rule)"),
              ("rule.false_positive", "False alerts (rule)"), ("rule.false_negative", "Missed fraud (rule)"),
              ("rule.alerts_per_1000", "Alerts / 1,000 (rule)")]
    rows = []
    for name in MODELS:
        for key, label in labels:
            e = agg["single_split"][name].get(key)
            if e is None or (name == "Rule-based" and key == "pr_auc"):
                continue
            d = e["between_seed"]
            digits = 1 if "false" in key else 2 if "alerts" in key else 3
            rows.append([name, label, *(fmt(e["values"][s], digits=digits) for s in seeds),
                         fmt(d["mean"], digits=digits), fmt(d["sd"], digits=digits),
                         f"{fmt(d['min'], digits=digits)} – {fmt(d['max'], digits=digits)}"])
    L += [table(["Model", "Metric", *[f"seed {s}" for s in seeds], "Mean", "SD", "Range"], rows), ""]

    L += ["### A2. Walk-forward backtest — window PR-AUC vs that window's fraud-rate baseline", ""]
    rows = []
    for s in seeds:
        for w in agg["per_seed_windows"][s]:
            lr = w["models"].get("Logistic Regression", {})
            rf = w["models"].get("Random Forest", {})
            rows.append([s, w["window"], "–".join(map(str, w["test_steps"])), f"{w['test_rows']:,}",
                         w["test_fraud"], fmt(w["prevalence"], digits=4),
                         fmt(lr.get("pr_auc")), fmt(lr.get("pr_auc_lift"), digits=1),
                         fmt(rf.get("pr_auc")), fmt(rf.get("pr_auc_lift"), digits=1),
                         "yes" if w.get("reliable") else "no"])
    L += [table(["Seed", "Window", "Test steps", "Rows", "Fraud", "Baseline (prevalence)",
                 "LR PR-AUC", "LR × baseline", "RF PR-AUC", "RF × baseline", "Enough fraud"],
                rows), ""]
    L += ["Window boundaries are set by row share within each sample, so window k covers "
          "similar but not identical steps in different seeds.", ""]

    L += ["### A3. Temporal vs between-seed variation (backtest)", "",
          "Temporal: SD across the windows of one seed. Between-seed: SD of each seed's window "
          "mean. They answer different questions and are not combined.", ""]
    rows = []
    for name in MODELS:
        for key, e in agg["temporal"][name].items():
            per = e["per_seed"]
            digits = 2 if "alerts" in key or "lift" in key or "false" in key else 3
            per_txt = [f"{fmt(per[s]['across_windows']['mean'], digits=digits)} ± "
                       f"{fmt(per[s]['across_windows']['sd'], digits=digits)} "
                       f"[{fmt(per[s]['across_windows']['min'], digits=digits)}–"
                       f"{fmt(per[s]['across_windows']['max'], digits=digits)}]"
                       if s in per else "—" for s in seeds]
            rows.append([name, key, *per_txt,
                         fmt(e["mean_within_seed_temporal_sd"]["mean"], digits=digits),
                         fmt(e["between_seed_of_window_mean"]["sd"], digits=digits)])
    L += [table(["Model", "Metric", *[f"seed {s}: window mean ± SD [min–max]" for s in seeds],
                 "Typical temporal SD", "Between-seed SD of mean"], rows), ""]
    L += ["Window fraud prevalence per seed: " + "; ".join(
        f"{s}: " + ", ".join(fmt(p, pct=True) for p in agg["window_prevalence"][s]) for s in seeds), ""]

    L += ["### A4. Ranking and trade-off consistency", ""]
    L += [f"- {f}" for f in study["findings"]] + [""]
    warn = sorted({w for r in study["runs"] if r.get("usable") for w in r.get("warnings", [])})
    if warn:
        L += ["**Warnings recorded in the runs**", ""] + [f"- {w}" for w in warn] + [""]

    L += ["## B. Interpretation (explanatory text, not measurements)", "",
          "- Between-seed variation mixes two sources: which rows the uniform sample drew and the "
          "Random Forest's random seed. It says how much a headline number could move with a "
          "different 1M-row sample of the same synthetic data, not how a model would perform on "
          "other data.",
          "- Temporal variation is usually the larger effect here because the fraud rate and score "
          "distribution change over simulated time; PR-AUC must be read against each window's own "
          "baseline.",
          "- Three seeds give a rough stability check. SDs from n = 3 are themselves very uncertain; "
          "they are not confidence intervals and per-run bootstrap intervals were not pooled.",
          "- Where the findings say 'inconclusive', the evidence does not support a stable ranking "
          "or trade-off claim.",
          "- Re-running a seed reproduces the sample, split, Logistic Regression and PR-AUC exactly. "
          "Random Forest prediction with n_jobs > 1 sums tree probabilities in thread order, so "
          "scores can differ in the last floating-point digit and a transaction scoring exactly at "
          "a threshold may flip (a difference of about one alert). With n_jobs=1 runs are "
          "bit-identical.",
          f"- {SYNTHETIC_STATEMENT}", "",
          "## C. Reproduce", "", "```bash", study["command_recommended"], "```", "",
          "Library versions: " + ", ".join(f"{k} {v}" for k, v in study["versions"].items())]
    return "\n".join(L) + "\n"


def build_study(root, seeds, sample_rows, study_id=None, status="complete", resources=None,
                run_entries=None):
    runs, per_seed = [], {}
    run_entries = {e["seed"]: e for e in (run_entries or [])}
    for seed in seeds:
        info, data = inspect_run(seed, root, sample_rows)
        entry = run_entries.get(seed, {})
        if entry.get("status") == "failed":
            info["usable"] = False
            info["problems"] = [f"run failed: {entry.get('error')}"] + info["problems"]
            info["error"] = entry.get("error")
        info["run_seconds"] = entry.get("seconds")
        runs.append(info)
        if info["usable"]:
            metrics, backtest = data
            per_seed[seed] = {"single": single_split_values(metrics),
                              "windows": window_values(backtest)}
    agg = aggregate(per_seed) if per_seed else None
    study = {
        "study_id": study_id or "rob-" + uuid.uuid4().hex[:10],
        "schema_version": SCHEMA_VERSION,
        "status": status if all(r["usable"] for r in runs) else "incomplete",
        "config": {"seeds": list(seeds), "sample_rows": sample_rows, "sampling": "uniform",
                   "protocol": "src.experiment (chronological split, train_only, "
                               "validation-only thresholds, 5-window expanding backtest)"},
        "command_recommended": "python -m src.robustness --sample-rows "
                               f"{sample_rows} --seeds {' '.join(map(str, seeds))}",
        "versions": library_versions(),
        "runs": runs, "resources": resources,
        "n_aggregated": len(per_seed),
        "spread_note": SPREAD_NOTE,
        "disclaimer": SYNTHETIC_STATEMENT,
        "aggregate": agg,
        "findings": consistency_findings(agg) if agg else [],
    }
    return study


def write_study(study, root, summary_path):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "robustness.json").write_text(json.dumps(study, indent=2, default=str), encoding="utf-8")
    Path(summary_path).write_text(build_summary(study), encoding="utf-8")


def run_study(data_path=None, sample_rows=DEFAULT_SAMPLE_ROWS, seeds=DEFAULT_SEEDS,
              root=ROBUSTNESS_DIR, summary_path=None, reuse=False, **experiment_kwargs):
    """Run each seed into root/seed_<seed>/, then aggregate and write the summary.

    reuse=True keeps a seed whose directory already holds a complete, consistent run with the
    same sample size (useful after an interrupted study); otherwise every seed is re-run.
    """
    root = Path(root)
    summary_path = Path(summary_path or root.parent / "robustness_summary.md")
    study_id = "rob-" + uuid.uuid4().hex[:10]
    root.mkdir(parents=True, exist_ok=True)
    started = time.time()
    (root / "robustness.json").write_text(json.dumps(
        {"study_id": study_id, "status": "running", "config": {"seeds": list(seeds),
         "sample_rows": sample_rows}}, indent=2), encoding="utf-8")
    entries = []
    for seed in seeds:
        if reuse:
            info, _ = inspect_run(seed, root, sample_rows)
            if info["usable"]:
                entries.append({"seed": seed, "status": "complete", "reused": True})
                print(f"seed {seed}: reusing complete run {info['experiment_id']}", flush=True)
                continue
        print(f"seed {seed}: running experiment ({sample_rows:,} rows)", flush=True)
        entry = run_seed(data_path, sample_rows, seed, root, **experiment_kwargs)
        entries.append(entry)
        print(f"seed {seed}: {entry['status']} in {entry['seconds']} s, process peak RSS "
              f"{entry['process_peak_rss_mb']} MB" + (f" — {entry.get('error')}" if
                                                       entry.get("error") else ""), flush=True)
    resources = {"seconds": round(time.time() - started, 2),
                 "process_peak_rss_mb": peak_rss_mb(),
                 "per_seed": [{k: e.get(k) for k in ("seed", "seconds", "process_peak_rss_mb",
                                                     "reused")} for e in entries]}
    study = build_study(root, seeds, sample_rows, study_id, resources=resources,
                        run_entries=entries)
    write_study(study, root, summary_path)
    return study


def main(argv=None):
    p = argparse.ArgumentParser(description="Multi-seed robustness study (uniform samples).")
    p.add_argument("--data", default=None)
    p.add_argument("--sample-rows", type=int, default=DEFAULT_SAMPLE_ROWS)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--reuse", action="store_true",
                   help="Keep seeds that already have a complete, consistent run")
    p.add_argument("--aggregate-only", action="store_true",
                   help="Rebuild robustness.json and the summary from existing seed folders")
    args = p.parse_args(argv)
    if args.aggregate_only:
        prev_path = ROBUSTNESS_DIR / "robustness.json"
        prev = json.loads(prev_path.read_text()) if prev_path.exists() else {}
        same = prev.get("config", {}).get("seeds") == args.seeds and \
            prev.get("config", {}).get("sample_rows") == args.sample_rows
        entries = [{"seed": r["seed"], "seconds": r.get("run_seconds")} for r in prev.get("runs", [])]
        study = build_study(ROBUSTNESS_DIR, args.seeds, args.sample_rows,
                            prev.get("study_id") if same else None,
                            resources=prev.get("resources") if same else None,
                            run_entries=entries if same else None)
        write_study(study, ROBUSTNESS_DIR, OUTPUT_DIR / "robustness_summary.md")
    else:
        study = run_study(args.data, args.sample_rows, args.seeds, reuse=args.reuse)
    print(json.dumps({"study_id": study["study_id"], "status": study["status"],
                      "n_aggregated": study["n_aggregated"], "findings": study["findings"]},
                     indent=2))
    return 0 if study["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
