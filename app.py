import json
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src import config
from src.config import TRANSACTION_TYPES, SCHEMA_VERSION
from src.data_loader import build_features
from src.rules import rule_based_score, rule_points, RULE_DESCRIPTION
from src.explain import global_feature_importance, explain_transaction, EXPLANATION_NOTE
from src.drift import DRIFT_NOTE
from src.report import build_report, load_outputs, consistency_checks, SYNTHETIC_STATEMENT
from src import ui
from src.ui import (MODEL_COLORS, METRIC_HELP, fmt_pct, fmt_num, fmt_count, fmt_rate, fmt_range,
                    format_frame, kpi_cards, section, note, style_fig)

# Read at every script run so tests (and users) can point the app at another output folder.
OUTPUT_DIR = Path(os.environ.get("FINSENTRY_OUTPUT_DIR", config.OUTPUT_DIR))
MODEL_DIR = Path(os.environ.get("FINSENTRY_MODEL_DIR", config.MODEL_DIR))

st.set_page_config(page_title="FinSentry · Fraud-risk analytics", page_icon=":material/shield:",
                   layout="wide")
st.markdown(ui.CSS, unsafe_allow_html=True)
ui.register_plotly_template()
TAGLINE = ("Fraud-risk analytics research on synthetic PaySim mobile-money data: ranking quality, "
           "alert workload and stability over time.")

metrics_path = OUTPUT_DIR / "metrics.json"
preview_path = OUTPUT_DIR / "test_preview.csv"
backtest_path = OUTPUT_DIR / "backtest.json"
TRAIN_CMD = "python -m src.experiment            # recommended (uniform 1M-row sample)\nstreamlit run app.py"
BACKTEST_CMD = "python -m src.experiment            # or: python -m src.backtest --sample-rows 1000000"
ROBUSTNESS_CMD = "python -m src.robustness --sample-rows 1000000 --seeds 42 123 2026   # ~15 min"
STRATEGY_LABELS = {"max_f1": "Max F1 (validation)", "alert_budget": "Alert budget (validation)",
                   "rule": "Fixed rule"}
PAGES = ["Overview", "Transaction Explorer", "Model Comparison", "Thresholds & Alerts",
         "Temporal Backtest", "Robustness Study", "Reliability & Calibration", "Explainability",
         "Try a Transaction", "Report Summary"]


def chart(fig, title, subtitle=None, height=360, container=st):
    container.plotly_chart(style_fig(fig, title, subtitle, height), width="stretch", theme=None)


def html(markup, container=st):
    container.markdown(markup, unsafe_allow_html=True)


if not metrics_path.exists():
    html(ui.brand_header(TAGLINE, [("Synthetic PaySim", "warn"), ("No results yet", "")]))
    st.info("No experiment results were found. Add the PaySim CSV to data/, install the "
            "requirements and run the recommended experiment; the dashboard only reads saved outputs.")
    st.code("pip install -r requirements.txt\n" + TRAIN_CMD, language="bash")
    st.markdown("[PaySim dataset on Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)")
    st.stop()

metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
if "run" not in metrics:
    st.error("outputs/metrics.json is from an older FinSentry version. Please retrain.")
    st.code(TRAIN_CMD, language="bash")
    st.stop()

run, sampling, split = metrics["run"], metrics["sampling"], metrics["split"]
models_meta = metrics["models"]
is_sample = sampling["mode"] != "full"
is_chrono = split["method"] == "chronological"
rows_total = metrics["dataset"]["rows_after_filter"]

badge_items = [("Synthetic PaySim data", "warn"),
               (f"{sampling['rows_used']:,}-row {sampling.get('method', '')} sample" if is_sample
                else "Full data", ""),
               ("Chronological split" if is_chrono else "Random split", "accent" if is_chrono else "warn"),
               (f"Primary experiment {run.get('experiment_id') or run['run_id']}", "")]
html(ui.brand_header(TAGLINE, badge_items))

if run.get("schema_version", 0) < SCHEMA_VERSION:
    st.error("outputs/metrics.json was written by an older FinSentry version; some pages may be "
             "incomplete. Re-run the experiment to refresh all outputs.")
    st.code(TRAIN_CMD, language="bash")
preview = pd.read_csv(preview_path) if preview_path.exists() else None
if preview is not None and ("run_id" not in preview or (preview["run_id"] != run["run_id"]).any()):
    st.error("outputs/test_preview.csv belongs to a different training run and is ignored.")
    preview = None
_, _bt_for_check, _exp_for_check = load_outputs(OUTPUT_DIR)
consistency_issues = consistency_checks(metrics, _bt_for_check, _exp_for_check)


@st.cache_resource
def load_models(run_id, model_dir):
    loaded, problems = {}, []
    for name, meta in models_meta.items():
        if meta.get("kind") != "model":
            continue
        path = Path(model_dir) / meta["model_file"]
        if not path.exists():
            problems.append(f"{name}: model file missing")
            continue
        bundle = joblib.load(path)
        if bundle.get("run_id") != run_id:
            problems.append(f"{name}: model file is from a different training run")
            continue
        loaded[name] = bundle
    return loaded, problems


models, model_problems = load_models(run["run_id"], str(MODEL_DIR))
for problem in model_problems:
    st.error(problem + " — retrain so models and metrics match.")

data_label = (f"Sample: {sampling['rows_used']:,} of {rows_total:,} rows" if is_sample
              else f"Full data: {sampling['rows_used']:,} rows")
if sampling.get("max_step"):
    data_label += f" (steps ≤ {sampling['max_step']})"

with st.sidebar:
    html("<div style='display:flex;align-items:center;gap:10px;margin:4px 0 14px 0'>"
         "<div class='fs-mark' style='width:22px;height:25px'></div>"
         "<div style='font-weight:700;font-size:1.05rem;letter-spacing:-0.01em'>FinSentry</div></div>")
    page = st.radio("Navigate", PAGES, label_visibility="collapsed")
    st.markdown("")
    method = sampling.get("method", "stratified")
    html("<div class='fs-ctx'>"
         f"<b>Primary experiment</b><br>{run.get('experiment_id') or '—'} · run <code>{run['run_id']}</code><br>"
         f"<b>Data</b> {data_label}<br>"
         f"<b>Sampling</b> {method}{' (label-free)' if method == 'uniform' else ' (label-informed)' if is_sample else ''}<br>"
         f"<b>Split</b> {'chronological, by step' if is_chrono else 'random (ignores time)'}<br>"
         f"<b>Seed</b> {run['seed']} · {run['timestamp_utc'][:16].replace('T', ' ')} UTC</div>")
    if consistency_issues:
        st.error(f"{len(consistency_issues)} output consistency problem(s): see Overview.")
    else:
        html("<div class='fs-badges' style='margin-top:8px'><span class='fs-badge accent'>"
             "Outputs consistent</span></div>")
    st.caption("Synthetic PaySim · educational prototype · not for real financial decisions.")


def show_explanation(name, bundle, features, container):
    try:
        ex = explain_transaction(bundle, features)
    except ValueError as err:
        container.info(str(err))
        return
    by_input = ex["by_input"].copy()
    by_input["direction"] = by_input["contribution"].map(
        lambda c: "raises fraud score" if c > 0 else "lowers fraud score")
    fig = px.bar(by_input.sort_values("contribution"), x="contribution", y="input",
                 orientation="h", color="direction",
                 color_discrete_map={"raises fraud score": ui.DANGER, "lowers fraud score": "#38BDF8"})
    fig.update_layout(showlegend=False, yaxis_title=None, xaxis_title=f"contribution ({ex['space']})")
    chart(fig, name, "Red raises the fraud score, blue lowers it", 270, container)
    if ex["space"] == "log-odds":
        sig = lambda z: 1 / (1 + np.exp(-z))
        container.caption(f"Baseline log-odds {ex['base']:.3f} (score {sig(ex['base']):.2%}) → "
                          f"{ex['output']:.3f} (score {sig(ex['output']):.2%}). Exact decomposition.")
    else:
        container.caption(f"Baseline score {ex['base']:.2%} → {ex['output']:.2%}. "
                          "Tree-path contributions; they sum exactly to the score.")
    with container.expander("Per-feature detail"):
        st.dataframe(ex["by_feature"], width="stretch", hide_index=True)


def fmt_ci(ci, key, pct=False):
    if not ci or not ci.get("computed"):
        return "n/a"
    iv = ci["intervals"].get(key)
    if not iv or iv["low"] is None:
        return "undefined"
    f = (lambda v: f"{v:.1%}") if pct else (lambda v: f"{v:.3f}")
    return f"{f(iv['low'])} – {f(iv['high'])}"


def primary_strategy(meta):
    return "rule" if meta["kind"] == "rule" else "max_f1"


def metric_table(section_name="test"):
    rows = []
    for name, meta in models_meta.items():
        m = meta[section_name]
        ci, strat = meta.get("ci"), primary_strategy(meta)
        is_rule = meta["kind"] == "rule"
        na = "n/a (no score)"
        rows.append({
            "Model": name,
            "Threshold": "n/a (fixed rule)" if is_rule else meta.get("threshold"),
            "Precision": m.get("precision"), "Recall": m.get("recall"), "F1": m.get("f1"),
            "Alerts": m.get("alerts"), "False alerts": m.get("false_alerts"),
            "Missed fraud": m.get("false_negative"),
            "PR-AUC": na if is_rule else m.get("pr_auc"), "PR-AUC baseline": m.get("pr_auc_baseline"),
            "ROC-AUC": na if is_rule else m.get("roc_auc"),
            f"Precision@{metrics['alert_budget']}": na if is_rule else m.get("precision_at_budget"),
            f"Recall@{metrics['alert_budget']}": na if is_rule else m.get("recall_at_budget"),
            "Precision 95% CI": fmt_ci(ci, f"{strat}.precision"),
            "Recall 95% CI": fmt_ci(ci, f"{strat}.recall"),
            "PR-AUC 95% CI": fmt_ci(ci, "pr_auc") if meta["kind"] == "model" else "n/a",
        })
    return pd.DataFrame(rows)


TABLE_FORMAT = {"Threshold": "dec3", "Precision": "pct", "Recall": "pct", "F1": "dec3",
                "Alerts": "int", "False alerts": "int", "Missed fraud": "int", "PR-AUC": "dec3",
                "PR-AUC baseline": "dec4", "ROC-AUC": "dec3",
                f"Precision@{metrics['alert_budget']}": "pct", f"Recall@{metrics['alert_budget']}": "pct"}
COLUMN_HELP = {"Precision": METRIC_HELP["precision"], "Recall": METRIC_HELP["recall"],
               "False alerts": METRIC_HELP["false_alerts"], "Missed fraud": METRIC_HELP["missed_fraud"],
               "PR-AUC": METRIC_HELP["pr_auc"], "PR-AUC baseline": METRIC_HELP["baseline"],
               "Threshold": METRIC_HELP["threshold"], "F1": METRIC_HELP["f1"],
               "Alerts per 1,000": METRIC_HELP["alerts_per_1000"]}


def show_table(df, spec=None, height=None, help_map=COLUMN_HELP):
    shown = format_frame(df, spec or {})
    for c in shown.columns:
        if shown[c].dtype == object:
            shown[c] = shown[c].map(lambda v: "undefined" if v is None or (isinstance(v, float) and np.isnan(v))
                                    else str(v))
    cfg = {c: st.column_config.Column(help=h) for c, h in help_map.items() if c in shown}
    kwargs = {"height": height} if height else {}
    st.dataframe(shown, width="stretch", hide_index=True, column_config=cfg, **kwargs)


def confusion_figure(m, name):
    z = [[m["true_negative"], m["false_positive"]], [m["false_negative"], m["true_positive"]]]
    labels = [["Correctly ignored", "False alerts"], ["Missed fraud", "Fraud caught"]]
    text = [[f"{labels[i][j]}<br><b>{z[i][j]:,}</b>" for j in range(2)] for i in range(2)]
    fig = go.Figure(go.Heatmap(z=np.log10(np.array(z) + 1), text=text, texttemplate="%{text}",
                               x=["Predicted legitimate", "Predicted fraud"],
                               y=["Actual legitimate", "Actual fraud"], showscale=False,
                               colorscale=[[0, "#0F1A17"], [1, "#166F55"]], hoverinfo="skip"))
    fig.update_yaxes(autorange="reversed")
    return fig


def model_names_scored():
    return [n for n, m in models_meta.items() if m["kind"] == "model"]


study, study_problems = ui.load_robustness(OUTPUT_DIR)
STUDY_TITLE = (f"{len(study['aggregate']['seeds'])}-seed robustness study"
               if study and study.get("aggregate") else "Multi-seed robustness study")
best = "Random Forest" if "Random Forest" in models_meta else model_names_scored()[0]

# ===================================================================================== pages
if page == "Overview":
    for issue in consistency_issues:
        st.error(issue)
    html(note(SYNTHETIC_STATEMENT + " Results come from a sample, not the full dataset, and the "
              "system is not deployed anywhere.", "warn") if is_sample else
         note(SYNTHETIC_STATEMENT, "warn"))

    test, bm = split["test"], models_meta[best]["test"]
    lift = bm["pr_auc"] / bm["pr_auc_baseline"] if bm.get("pr_auc") and bm.get("pr_auc_baseline") else None
    html(section(f"Primary experiment · held-out TEST period (steps "
                 f"{'–'.join(map(str, split['step_boundaries']['test'])) if split.get('step_boundaries') else 'random'})",
                 f"{best} at its max-F1 threshold chosen on validation. Evaluated once; "
                 "hover a card for a definition."))
    html(kpi_cards([
        {"label": "Fraud prevalence", "value": fmt_pct(test["prevalence"], 2),
         "sub": f"{test['fraud']:,} fraud in {test['rows']:,} rows", "help": METRIC_HELP["baseline"]},
        {"label": f"PR-AUC · {best}", "value": fmt_num(bm.get("pr_auc")), "accent": True,
         "sub": f"{lift:.0f}× the {test['prevalence']:.4f} baseline" if lift else "baseline n/a",
         "help": METRIC_HELP["pr_auc"]},
        {"label": "Precision", "value": fmt_pct(bm.get("precision")),
         "sub": "share of alerts that were fraud", "help": METRIC_HELP["precision"]},
        {"label": "Recall", "value": fmt_pct(bm.get("recall")),
         "sub": "share of fraud alerted", "help": METRIC_HELP["recall"]},
        {"label": "False alerts", "value": fmt_count(bm.get("false_positive")),
         "sub": "legitimate but alerted", "help": METRIC_HELP["false_alerts"]},
        {"label": "Missed fraud", "value": fmt_count(bm.get("false_negative")),
         "sub": "fraud not alerted", "help": METRIC_HELP["missed_fraud"]},
        {"label": "Alert volume", "value": fmt_rate(bm.get("alerts_per_1000")),
         "sub": "alerts per 1,000 transactions", "help": METRIC_HELP["alerts_per_1000"]},
    ]))

    c1, c2 = st.columns([1.15, 1])
    rank = [{"Model": n, "PR-AUC": m["test"]["pr_auc"]} for n, m in models_meta.items()
            if m["test"].get("pr_auc") is not None]
    if rank:
        rdf = pd.DataFrame(rank).sort_values("PR-AUC")
        fig = go.Figure(go.Bar(x=rdf["PR-AUC"], y=rdf["Model"], orientation="h",
                               marker_color=[MODEL_COLORS.get(n, ui.ACCENT) for n in rdf["Model"]],
                               text=[f"{v:.3f}" for v in rdf["PR-AUC"]], textposition="outside"))
        fig.add_vline(x=test["prevalence"], line_dash="dot", line_color=ui.WARN,
                      annotation_text=f"random baseline {test['prevalence']:.4f}",
                      annotation_position="bottom right", annotation_font_color=ui.WARN)
        fig.update_layout(xaxis_range=[0, max(rdf["PR-AUC"]) * 1.25], xaxis_title="PR-AUC")
        chart(fig, "Ranking quality", "Test PR-AUC vs the fraud-rate baseline (higher is better). "
              "The rule outputs 0/1, so it has no PR-AUC.", 300, c1)
    err = []
    for n, m in models_meta.items():
        t = m["test"]
        err += [{"Model": n, "Error": "False alerts", "Count": t["false_positive"]},
                {"Model": n, "Error": "Missed fraud", "Count": t["false_negative"]}]
    fig = px.bar(pd.DataFrame(err), x="Model", y="Count", color="Error", barmode="group", log_y=True,
                 color_discrete_map={"False alerts": ui.WARN, "Missed fraud": ui.DANGER}, text="Count")
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(xaxis_title=None, yaxis_title="count (log scale)")
    chart(fig, "Operational errors", "False alerts and missed fraud on TEST at each model's "
          "threshold (rule: fixed). Log scale.", 300, c2)

    c3, c4 = st.columns([1, 1.15])
    bounds = split.get("step_boundaries") or {}
    periods = pd.DataFrame([{"Period": f"{label}<br>steps {'–'.join(map(str, bounds[key]))}" if key in bounds else label,
                             "Fraud rate": split[key]["prevalence"], "Rows": split[key]["rows"],
                             "Fraud": split[key]["fraud"]}
                            for label, key in (("Train", "train"), ("Validation", "validation"), ("Test", "test"))])
    fig = go.Figure(go.Bar(x=periods["Period"], y=periods["Fraud rate"],
                           marker_color=[ui.MUTED, "#38BDF8", ui.ACCENT],
                           text=[fmt_pct(v, 2) for v in periods["Fraud rate"]], textposition="outside",
                           customdata=periods[["Rows", "Fraud"]],
                           hovertemplate="%{x}<br>rows %{customdata[0]:,}<br>fraud %{customdata[1]:,}<extra></extra>"))
    fig.update_layout(yaxis_tickformat=".2%", yaxis_range=[0, periods["Fraud rate"].max() * 1.3])
    chart(fig, "Prevalence shift", "The fraud rate changes across periods, so precision is not "
          "comparable between them", 300, c3)
    with c4:
        html(section(STUDY_TITLE, "Separate runs in outputs/robustness/, not the "
                     "primary experiment"))
        if study and study.get("aggregate"):
            ss = study["aggregate"]["single_split"]
            rf = ss.get("Random Forest", {})
            pr, prec = rf.get("pr_auc", {}).get("between_seed", {}), rf.get("max_f1.precision", {}).get("between_seed", {})
            fa = rf.get("max_f1.false_positive", {}).get("between_seed", {})
            html(kpi_cards([
                {"label": "RF PR-AUC across seeds", "value": f"{fmt_num(pr.get('mean'))} ± {fmt_num(pr.get('sd'))}",
                 "sub": f"range {fmt_range(pr.get('min'), pr.get('max'), fmt_num)}, n={pr.get('n')}",
                 "help": "Mean ± sample SD over independent 1M-row samples. Descriptive, not a confidence interval."},
                {"label": "RF max-F1 precision", "value": fmt_range(prec.get("min"), prec.get("max"), fmt_pct),
                 "sub": "range across seeds", "help": "Threshold-dependent metrics move more than PR-AUC."},
                {"label": "RF false alerts", "value": fmt_range(fa.get("min"), fa.get("max"), fmt_count),
                 "sub": "range across seeds", "help": METRIC_HELP["false_alerts"]},
            ]))
            for p in study_problems:
                st.warning(p)
            st.caption(f"Study {study['study_id']} · {study['n_aggregated']} of "
                       f"{len(study['config']['seeds'])} seeds aggregated. Details on 'Robustness Study'.")
        else:
            st.info("No robustness study found in outputs/robustness/.")

    html(section("Final TEST performance", "All models on the untouched test period; thresholds were "
                 "chosen on validation only. Hover column headers for definitions."))
    show_table(metric_table(), TABLE_FORMAT)
    any_ci = next((m["ci"] for m in models_meta.values() if m.get("ci")), None)
    if any_ci:
        st.caption(f"95% intervals: row bootstrap, {any_ci['n_boot']} replicates, seed {any_ci['seed']} "
                   "(estimates, not guarantees; time-aware block intervals are on 'Reliability & "
                   "Calibration'). PR-AUC and Precision@budget are undefined for the rule because it "
                   "only outputs 0/1.")
    for w in metrics.get("warnings", []):
        st.warning(w)
    protocol = metrics.get("protocol")
    if protocol:
        with st.expander("Evaluation protocol: how leakage is prevented"):
            used = {n: m.get("final_fit", {}).get("used") for n, m in models_meta.items()
                    if m["kind"] == "model"}
            st.write(f"Final fit requested: **{protocol['final_fit_requested']}** · used: {used}")
            for s in protocol["steps"]:
                st.markdown(f"- {s}")
            st.markdown(f"- {protocol['test_usage']}")
            for n, m in models_meta.items():
                if m.get("final_fit", {}).get("rejected_reason"):
                    st.warning(f"{n}: refit rejected — {m['final_fit']['rejected_reason']}")
    st.caption("Model inputs: " + ", ".join(metrics["features"]) +
               ". Excluded: " + ", ".join(metrics["excluded_fields"]) + ".")

elif page == "Transaction Explorer":
    html(section("Transaction Explorer", "Held-out TEST transactions from the synthetic sample with "
                 "each model's score and alert decision. No live feed: these are saved rows."))
    if preview is None:
        st.info("No preview for this run. It is written by the experiment.")
        st.code(TRAIN_CMD, language="bash")
    else:
        html(note("This table is enriched: random_test_sample rows follow the natural test "
                  "distribution; fraud_example and top_rf_alert rows were added to show interesting "
                  "cases. Do not compute rates from it."))
        c1, c2 = st.columns([1, 1.4])
        reasons = sorted(preview["preview_reason"].unique())
        chosen = c1.multiselect("Rows to show", reasons, default=reasons)
        view = preview[preview["preview_reason"].isin(chosen)]
        filt = c2.radio("Filter", ["All", "Actual fraud", "Any model alert",
                                   "Missed fraud (RF)", "False alert (RF)"], horizontal=True)
        if filt == "Actual fraud":
            view = view[view["actual_isFraud"] == 1]
        elif filt == "Any model alert":
            view = view[(view["lr_alert"] == 1) | (view["rf_alert"] == 1)]
        elif filt == "Missed fraud (RF)":
            view = view[(view["actual_isFraud"] == 1) & (view["rf_alert"] == 0)]
        elif filt == "False alert (RF)":
            view = view[(view["actual_isFraud"] == 0) & (view["rf_alert"] == 1)]
        st.caption(f"{len(view):,} rows")
        st.dataframe(view, width="stretch", height=420, hide_index=True,
                     column_config={"amount": st.column_config.NumberColumn("amount", format="%.2f"),
                                    "lr_score": st.column_config.ProgressColumn(
                                        "LR score", min_value=0.0, max_value=1.0, format="%.3f"),
                                    "rf_score": st.column_config.ProgressColumn(
                                        "RF score", min_value=0.0, max_value=1.0, format="%.3f")})
        st.download_button("Download displayed rows (CSV)", view.to_csv(index=False),
                           "finsentry_test_preview.csv", "text/csv")

        if models and len(view):
            html(section("Explain one transaction", "How each input moved each model's score "
                         "relative to its baseline"))
            pos = st.selectbox("Row (position in the table above)", list(range(len(view))),
                               format_func=lambda i: f"{i}: step {view.iloc[i]['step']}, "
                               f"{view.iloc[i]['type']}, {view.iloc[i]['amount']:,.2f}, "
                               f"fraud={view.iloc[i]['actual_isFraud']}")
            row = view.iloc[[pos]][["step", "type", "amount"]]
            cols = st.columns(len(models))
            for col, (name, bundle) in zip(cols, models.items()):
                show_explanation(name, bundle, build_features(row), col)
            st.caption(EXPLANATION_NOTE)

elif page == "Model Comparison":
    html(section("Model Comparison", "Primary experiment, TEST period. Models use their max-F1 "
                 "threshold from validation; the rule is fixed."))
    table = metric_table()
    metric = st.segmented_control("Metric", ["PR-AUC", "Precision", "Recall", "F1", "False alerts",
                                             "Missed fraud", "ROC-AUC",
                                             f"Precision@{metrics['alert_budget']}"],
                                  default="PR-AUC") or "PR-AUC"
    plotted = table.assign(**{metric: pd.to_numeric(table[metric], errors="coerce")}).dropna(subset=[metric])
    fig = go.Figure(go.Bar(x=plotted["Model"], y=plotted[metric],
                           marker_color=[MODEL_COLORS.get(n, ui.ACCENT) for n in plotted["Model"]],
                           text=[ui.FORMATTERS[TABLE_FORMAT.get(metric, "dec3")](v)
                                 for v in plotted[metric]], textposition="outside"))
    if metric == "PR-AUC":
        fig.add_hline(y=split["test"]["prevalence"], line_dash="dot", line_color=ui.WARN,
                      annotation_text="random baseline (prevalence)", annotation_font_color=ui.WARN)
    if metric in ("Precision", "Recall", f"Precision@{metrics['alert_budget']}"):
        fig.update_layout(yaxis_tickformat=".0%")
    fig.update_layout(xaxis_title=None, yaxis_title=metric,
                      yaxis_range=[0, max(plotted[metric].max() * 1.2, 1e-9)] if len(plotted) else None)
    chart(fig, f"Test {metric} by model", COLUMN_HELP.get(metric, ""), 340)
    if len(plotted) < len(table):
        st.caption(f"{metric} is not defined for: "
                   + ", ".join(sorted(set(table["Model"]) - set(plotted["Model"]))))

    fig = go.Figure()
    for name, meta in models_meta.items():
        curve = meta.get("pr_curve_test")
        if curve:
            fig.add_trace(go.Scatter(x=curve["recall"], y=curve["precision"], mode="lines", name=name,
                                     line={"color": MODEL_COLORS.get(name), "width": 2}))
            t = meta["test"]
            if t.get("recall") is not None and t.get("precision") is not None:
                fig.add_trace(go.Scatter(x=[t["recall"]], y=[t["precision"]], mode="markers",
                                         marker={"size": 10, "color": MODEL_COLORS.get(name),
                                                 "line": {"color": "#0B0F0E", "width": 2}},
                                         name=f"{name} · max-F1 threshold", showlegend=False))
        elif meta["kind"] == "rule":
            m = meta["test"]
            if m.get("recall") is not None:
                fig.add_trace(go.Scatter(x=[m["recall"]], y=[m["precision"]], mode="markers",
                                         marker={"size": 12, "symbol": "diamond",
                                                 "color": MODEL_COLORS["Rule-based"]},
                                         name=f"{name} (single point)"))
    fig.add_hline(y=split["test"]["prevalence"], line_dash="dot", line_color=ui.WARN,
                  annotation_text="random baseline", annotation_font_color=ui.WARN)
    fig.update_layout(xaxis_title="Recall", yaxis_title="Precision", yaxis_range=[0, 1.02],
                      xaxis_range=[0, 1.0])
    chart(fig, "Precision–recall curves (TEST)", "Each point on a curve is one possible threshold; "
          "dots mark the exact max-F1 operating point. Curves are saved as 200 sampled thresholds, "
          "so straight segments are interpolation and a dot can sit slightly off its line", 420)

    html(section("Confusion matrices (TEST)", "Counts at each model's threshold; shading is "
                 "log-scaled so rare fraud cells stay visible"))
    cols = st.columns(len(models_meta))
    for col, (name, meta) in zip(cols, models_meta.items()):
        sub = f"threshold {meta['threshold']:.3f} (max F1)" if meta.get("threshold") is not None else "fixed rule"
        chart(confusion_figure(meta["test"], name), name, sub, 280, col)

    with st.expander("Validation metrics (used only to pick thresholds)"):
        val = pd.DataFrame([{"Model": n, **{k: v for k, v in m["validation"].items()}}
                            for n, m in models_meta.items()])
        show_table(val, {"precision": "pct", "recall": "pct", "f1": "dec3", "alerts_per_1000": "rate",
                         "best_f1": "dec3"})
    st.caption("Rule: " + models_meta["Rule-based"].get("description", ""))

elif page == "Thresholds & Alerts":
    html(section("Thresholds & Alerts", "Two ways to turn scores into alerts, both chosen on the "
                 "VALIDATION period only. TEST shows what happened afterwards."))
    html(note(metrics.get("interpretation", "Thresholds are chosen on validation data only.")))
    if study and study.get("aggregate"):
        rf = study["aggregate"]["single_split"].get("Random Forest", {})
        p = rf.get("max_f1.precision", {}).get("between_seed", {})
        f_ = rf.get("max_f1.false_positive", {}).get("between_seed", {})
        html(note(f"Max-F1 thresholds are unstable: in the {STUDY_TITLE} the Random "
                  f"Forest's max-F1 precision ranged {fmt_range(p.get('min'), p.get('max'), fmt_pct)} "
                  f"and false alerts {fmt_range(f_.get('min'), f_.get('max'), fmt_count)}, because the "
                  "threshold is fit on a validation period with few fraud cases.", "warn"))
    rows = []
    for name, meta in models_meta.items():
        for key, s in meta.get("strategies", {}).items():
            for period in ("validation", "test"):
                m = s[period]
                rows.append({
                    "Model": name, "Strategy": STRATEGY_LABELS.get(key, key),
                    "Period": "TEST (final)" if period == "test" else "VALIDATION (used to choose)",
                    "Threshold": "n/a (fixed rule)" if key == "rule" else s["threshold"],
                    "Precision": m["precision"], "Recall": m["recall"],
                    "F1": m["f1"], "False alerts": m["false_positive"],
                    "Missed fraud": m["false_negative"], "Alerts per 1,000": m["alerts_per_1000"],
                    "Precision 95% CI": fmt_ci(meta.get("ci"), f"{key}.precision") if period == "test" else "",
                    "Recall 95% CI": fmt_ci(meta.get("ci"), f"{key}.recall") if period == "test" else "",
                })
    strat_df = pd.DataFrame(rows)
    if strat_df.empty:
        st.info("This metrics.json has no strategy results. Retrain to generate them.")
        st.code(TRAIN_CMD, language="bash")
        st.stop()
    shown = st.segmented_control("Period", ["TEST (final)", "VALIDATION (used to choose)", "Both"],
                                 default="TEST (final)") or "TEST (final)"
    view = strat_df if shown == "Both" else strat_df[strat_df["Period"] == shown]
    show_table(view, {"Threshold": "dec3", "Precision": "pct", "Recall": "pct", "F1": "dec3",
                      "False alerts": "int", "Missed fraud": "int", "Alerts per 1,000": "rate"})
    requested = metrics.get("evaluation_settings", {}).get("alerts_per_1000")
    if requested is not None:
        st.caption(f"Alert budget: threshold set so the VALIDATION period gets about {requested:g} "
                   "alerts per 1,000 transactions (labels not used). The realised TEST rate can differ "
                   "when score distributions shift over time; compare the 'Alerts per 1,000' column.")

    test_rows = strat_df[strat_df["Period"] == "TEST (final)"].copy()
    test_rows["Model / strategy"] = test_rows["Model"] + " · " + test_rows["Strategy"].str.replace(" (validation)", "")
    counts = test_rows.melt(id_vars="Model / strategy", value_vars=["False alerts", "Missed fraud"],
                            var_name="Error type", value_name="Count")
    fig = px.bar(counts, x="Model / strategy", y="Count", color="Error type", barmode="group",
                 log_y=True, text="Count",
                 color_discrete_map={"False alerts": ui.WARN, "Missed fraud": ui.DANGER})
    fig.update_traces(texttemplate="%{text:,}", textposition="outside")
    fig.update_layout(xaxis_title=None, yaxis_title="count (log scale)")
    chart(fig, "Missed fraud vs false alerts (TEST)", "Every strategy trades one error for the other", 360)

    topk = [{"Model": n, **r} for n, m in models_meta.items() for r in (m.get("top_k") or [])]
    if topk:
        html(section("Top-K review queues (TEST)", "Rank transactions by score and review the top K; "
                     "no threshold is involved. This mirrors a fixed reviewer capacity."))
        tk = pd.DataFrame(topk).rename(columns={"label": "Budget", "fraud_found": "Fraud found",
                                                "precision": "Precision", "recall": "Recall", "k": "K"})
        show_table(tk, {"Precision": "pct", "Recall": "pct", "K": "int", "Fraud found": "int"})

    curve_models = [n for n, m in models_meta.items() if m.get("tradeoff")]
    if curve_models:
        html(section("Threshold trade-off", "Precision and recall as the alert volume grows. "
                     "Thresholds on this grid come from validation scores only."))
        name = st.selectbox("Model", curve_models, index=curve_models.index(best) if best in curve_models else 0)
        tr = pd.DataFrame(models_meta[name]["tradeoff"])
        long = tr.melt(id_vars=["period", "alerts_per_1000"], value_vars=["precision", "recall"],
                       var_name="metric", value_name="value").dropna()
        long["series"] = long["metric"].str.title() + " · " + long["period"]
        fig = px.line(long, x="alerts_per_1000", y="value", color="series", log_x=True, markers=True,
                      labels={"alerts_per_1000": "Alerts per 1,000 transactions (log)", "value": ""},
                      color_discrete_map={"Precision · validation": "#38BDF8", "Precision · test": ui.ACCENT,
                                          "Recall · validation": "#C084FC", "Recall · test": ui.ACCENT_2})
        fig.update_layout(yaxis_tickformat=".0%")
        chart(fig, f"{name}: precision and recall vs alert volume",
              "Test lines show how the same validation thresholds behaved later", 400)

        html(section("Cost explorer", "How much worse is a missed fraud than a false alert? The "
                     "lowest-cost threshold is picked on VALIDATION and then checked on TEST."))
        ratio = st.number_input("Cost of one missed fraud ÷ cost of one false alert",
                                min_value=1.0, max_value=10_000.0, value=50.0, step=10.0)
        val = tr[tr["period"] == "validation"].copy()
        val["cost"] = val["false_negative"] * ratio + val["false_positive"]
        best_row = val.loc[val["cost"].idxmin()]
        test_at = tr[(tr["period"] == "test") & (tr["threshold"] == best_row["threshold"])]
        cards = [{"label": "Chosen threshold", "value": fmt_num(best_row["threshold"], 4),
                  "sub": "lowest cost on validation", "help": METRIC_HELP["threshold"], "accent": True},
                 {"label": "Validation false alerts", "value": fmt_count(best_row["false_positive"]),
                  "sub": f"{fmt_rate(best_row['alerts_per_1000'])} alerts / 1,000"},
                 {"label": "Validation missed fraud", "value": fmt_count(best_row["false_negative"])}]
        if not test_at.empty:
            t = test_at.iloc[0]
            cards += [{"label": "Test false alerts", "value": fmt_count(t["false_positive"]),
                       "sub": f"{fmt_rate(t['alerts_per_1000'])} alerts / 1,000"},
                      {"label": "Test missed fraud", "value": fmt_count(t["false_negative"])},
                      {"label": "Test precision · recall",
                       "value": f"{fmt_pct(None if pd.isna(t['precision']) else t['precision'])} · "
                                f"{fmt_pct(None if pd.isna(t['recall']) else t['recall'])}"}]
        html(kpi_cards(cards))
        st.caption("The lowest-cost threshold changes with the cost ratio; there is no universally "
                   "best threshold.")

elif page == "Temporal Backtest":
    html(section("Temporal Backtest", "Walk-forward evaluation: each window trains on the past, picks "
                 "thresholds on the period just before, and tests on the next block of time."))
    if not backtest_path.exists():
        st.info("No backtest results yet. Run the experiment (it includes the backtest):")
        st.code(BACKTEST_CMD, language="bash")
        st.stop()
    bt = json.loads(backtest_path.read_text(encoding="utf-8"))
    bset, bsamp = bt["backtest_settings"], bt["sampling"]
    bt_label = (f"{bsamp['rows_used']:,}-row sample of {bt['dataset']['rows_after_filter']:,}"
                if bsamp["mode"] != "full" else f"full data ({bsamp['rows_used']:,} rows)")
    html(ui.badges([("Primary experiment backtest", "accent"), (bt_label, ""),
                    (f"{bset['scheme']} windows × {len(bt['windows'])}", ""),
                    (f"validation = {bset['val_frac']:.0%} of rows before each window", ""),
                    (f"run {bt['run']['run_id']}", "")]))
    for w in bt.get("warnings", []):
        st.warning(w)

    model_names = ["Rule-based", "Logistic Regression", "Random Forest"]
    table, long, pr_rows = [], [], []
    for w in bt["windows"]:
        row = {"Window": w["window"],
               "Train steps": "–".join(map(str, w["train_steps"])),
               "Validation steps": "–".join(map(str, w["validation_steps"])),
               "Test steps": "–".join(map(str, w["test_steps"])),
               "Test rows": w["test"]["rows"], "Test fraud": w["test"]["fraud"],
               "Test fraud rate": w["test"]["prevalence"],
               "Validation fraud": w["validation"]["fraud"],
               "Enough fraud": "yes" if w["reliable"] else "NO"}
        pr_rows.append({"Window": w["window"], "Series": "Fraud-rate baseline",
                        "PR-AUC": w["test"]["prevalence"]})
        for name in ("Logistic Regression", "Random Forest"):
            ev = w["models"].get(name)
            row[f"{name} PR-AUC"] = ev["test_ranking"]["pr_auc"] if ev else None
            row[f"{name} lift"] = ev["test_ranking"].get("pr_auc_lift") if ev else None
            if ev and ev["test_ranking"]["pr_auc"] is not None:
                pr_rows.append({"Window": w["window"], "Series": name,
                                "PR-AUC": ev["test_ranking"]["pr_auc"]})
        row["Warnings"] = " | ".join(w["warnings"])
        table.append(row)
        for name in model_names:
            ev = w["models"].get(name)
            if not ev:
                continue
            for key, s in ev["strategies"].items():
                long.append({"Window": w["window"], "Model": name, "Strategy": STRATEGY_LABELS[key],
                             "strategy_key": key, **s["test"], "threshold": s["threshold"]})

    rf_vals = [r["PR-AUC"] for r in pr_rows if r["Series"] == "Random Forest"]
    if rf_vals:
        html(kpi_cards([
            {"label": "Windows", "value": str(len(bt["windows"])), "sub": f"{bset['scheme']} scheme"},
            {"label": "RF PR-AUC range", "value": fmt_range(min(rf_vals), max(rf_vals), fmt_num),
             "sub": "across test windows", "help": METRIC_HELP["pr_auc"], "accent": True},
            {"label": "Window fraud rate", "value": fmt_range(min(r["Test fraud rate"] for r in table),
                                                              max(r["Test fraud rate"] for r in table),
                                                              lambda v: fmt_pct(v, 2)),
             "sub": "prevalence varies over time", "help": METRIC_HELP["baseline"]},
        ]))
    fig = go.Figure()
    pdf = pd.DataFrame(pr_rows)
    for series in ("Fraud-rate baseline", "Logistic Regression", "Random Forest"):
        s = pdf[pdf["Series"] == series]
        fig.add_trace(go.Scatter(x=s["Window"], y=s["PR-AUC"], name=series, mode="lines+markers",
                                 line={"color": MODEL_COLORS[series], "width": 2,
                                       "dash": "dot" if series == "Fraud-rate baseline" else "solid"}))
    fig.update_layout(yaxis_type="log", xaxis_title="Test window (earlier → later)",
                      yaxis_title="PR-AUC (log scale)", xaxis_dtick=1)
    chart(fig, "PR-AUC per window vs that window's fraud rate",
          "A random scorer's PR-AUC equals the window's fraud rate. Missing points mean PR-AUC is "
          "undefined (no fraud).", 380)
    show_table(pd.DataFrame(table), {"Test rows": "int", "Test fraud": "int", "Test fraud rate": "pct2",
                                     "Validation fraud": "int", "Logistic Regression PR-AUC": "dec3",
                                     "Random Forest PR-AUC": "dec3", "Logistic Regression lift": "lift",
                                     "Random Forest lift": "lift"})

    if long:
        ldf = pd.DataFrame(long)
        c1, c2 = st.columns(2)
        strat = c1.selectbox("Strategy", ["Max F1 (validation)", "Alert budget (validation)"])
        metric = c2.selectbox("Metric", ["precision", "recall", "f1", "alerts_per_1000"],
                              format_func={"precision": "Precision", "recall": "Recall", "f1": "F1",
                                           "alerts_per_1000": "Alerts per 1,000"}.get)
        sub = ldf[(ldf["Strategy"] == strat) | (ldf["strategy_key"] == "rule")]
        fig = px.line(sub.dropna(subset=[metric]), x="Window", y=metric, color="Model", markers=True,
                      color_discrete_map=MODEL_COLORS)
        if metric in ("precision", "recall"):
            fig.update_layout(yaxis_tickformat=".0%")
        fig.update_layout(xaxis_dtick=1)
        chart(fig, f"Test {metric.replace('_', ' ')} per window", f"{strat}; the rule is fixed", 360)
        err = sub.melt(id_vars=["Window", "Model"], value_vars=["false_positive", "false_negative"],
                       var_name="Error", value_name="Count")
        err["Error"] = err["Error"].map({"false_positive": "False alerts", "false_negative": "Missed fraud"})
        fig = px.bar(err, x="Window", y="Count", color="Error", barmode="group", facet_col="Model",
                     log_y=True, color_discrete_map={"False alerts": ui.WARN, "Missed fraud": ui.DANGER})
        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
        chart(fig, "False alerts and missed fraud per window", "Log scale", 360)

    html(section("Drift diagnostics across windows", "Label-free score drift and calibration per window"))
    drift_rows = []
    for w in bt["windows"]:
        for name in ("Logistic Regression", "Random Forest"):
            ev = w["models"].get(name)
            if not ev:
                continue
            d, c = ev.get("drift") or {}, ev.get("calibration") or {}
            cur = d.get("current_summary") or {}
            drift_rows.append({
                "Window": w["window"], "Model": name, "Fraud rate": w["test"]["prevalence"],
                "PSI (val→test)": d.get("psi"), "Score p99": cur.get("p99"),
                "Alerts/1k (max F1)": ev["strategies"]["max_f1"]["test"]["alerts_per_1000"],
                "Alerts/1k (budget)": ev["strategies"]["alert_budget"]["test"]["alerts_per_1000"],
                "Brier raw": c.get("brier_raw"), "Brier calibrated": c.get("brier_calibrated"),
                "PR-AUC": ev["test_ranking"]["pr_auc"],
                "PR-AUC baseline": ev["test_ranking"]["pr_auc_baseline"]})
    if drift_rows and any(r["PSI (val→test)"] is not None for r in drift_rows):
        ddf = pd.DataFrame(drift_rows)
        dmetric = st.selectbox("Diagnostic", ["PSI (val→test)", "Score p99", "Alerts/1k (max F1)",
                                              "Alerts/1k (budget)", "Brier raw", "Brier calibrated",
                                              "Fraud rate"])
        fig = px.line(ddf.dropna(subset=[dmetric]), x="Window", y=dmetric, color="Model", markers=True,
                      color_discrete_map=MODEL_COLORS)
        if dmetric == "PSI (val→test)":
            fig.add_hline(y=0.1, line_dash="dot", line_color=ui.WARN, annotation_text="0.1 moderate")
        fig.update_layout(xaxis_dtick=1)
        chart(fig, f"{dmetric} per window", METRIC_HELP["psi"] if dmetric.startswith("PSI") else None, 340)
        show_table(ddf, {"Fraud rate": "pct2", "PSI (val→test)": "dec3", "Score p99": "dec3",
                         "Alerts/1k (max F1)": "rate", "Alerts/1k (budget)": "rate", "Brier raw": "dec4",
                         "Brier calibrated": "dec4", "PR-AUC": "dec3", "PR-AUC baseline": "dec4"})
        st.info(DRIFT_NOTE)
    else:
        st.info("This backtest.json has no drift diagnostics. Re-run the backtest to generate them.")
        st.code(BACKTEST_CMD, language="bash")

    html(section("Variation across windows", "Mean, SD, min and max over windows where the metric is "
                 "defined. A large spread means one aggregate score overstates stability."))
    summ = [{"Model": name, "Metric": metric, **vals}
            for name, rows in bt["summary"].items() for metric, vals in rows.items()
            if vals["n_windows"]]
    show_table(pd.DataFrame(summ), {"mean": "dec3", "std": "dec3", "min": "dec3", "max": "dec3"})

    with st.expander("Per-window 95% bootstrap intervals"):
        ci_rows = []
        for w in bt["windows"]:
            for name in model_names:
                ev = w["models"].get(name)
                if not ev:
                    continue
                strat_key = "rule" if name == "Rule-based" else "max_f1"
                blk = ev.get("block_ci")
                ci_rows.append({"Window": w["window"], "Model": name,
                                "PR-AUC CI (row)": fmt_ci(ev["ci"], "pr_auc") if name != "Rule-based" else "n/a",
                                "PR-AUC CI (block)": fmt_ci(blk, "pr_auc") if name != "Rule-based" else "n/a",
                                "Precision CI (row)": fmt_ci(ev["ci"], f"{strat_key}.precision"),
                                "Precision CI (block)": fmt_ci(blk, f"{strat_key}.precision"),
                                "Recall CI (row)": fmt_ci(ev["ci"], f"{strat_key}.recall"),
                                "Recall CI (block)": fmt_ci(blk, f"{strat_key}.recall"),
                                "Note": "; ".join(x.get("reason", "") for x in (ev["ci"], blk or {})
                                                  if x and not x.get("computed"))})
        show_table(pd.DataFrame(ci_rows))
        st.caption(bt["notes"]["bootstrap"])
        if bt["notes"].get("block_bootstrap"):
            st.caption(bt["notes"]["block_bootstrap"])
    st.caption(bt["notes"]["thresholds"])

elif page == "Robustness Study":
    html(section(STUDY_TITLE, "The same experiment repeated on independent uniform "
                 "samples. These are separate saved runs, not the primary experiment shown elsewhere."))
    if study is None:
        for p in study_problems:
            st.warning(p)
        st.info("No robustness study found in outputs/robustness/. It takes about 15 minutes and "
                "never touches the primary outputs:")
        st.code(ROBUSTNESS_CMD, language="bash")
        st.stop()
    cfg = study["config"]
    html(ui.badges([(f"Study {study['study_id']}", "accent"),
                    (f"status {study['status']}", "accent" if study["status"] == "complete" else "warn"),
                    (f"seeds {', '.join(map(str, cfg['seeds']))}", ""),
                    (f"{cfg['sample_rows']:,}-row {cfg.get('sampling', 'uniform')} samples", ""),
                    ("Synthetic PaySim", "warn")]))
    for p in study_problems:
        st.warning(p)
    if not study.get("aggregate"):
        st.info("The study has no aggregated results to show.")
        st.stop()
    html(note(study.get("spread_note", "") + " Every run is a sample of synthetic data, not "
              "full-data or real-bank performance.", "warn"))

    ss = study["aggregate"]["single_split"]
    prev = ss["test_prevalence"]["between_seed"]
    var = ui.robustness_variation_frame(study)
    rf_pr, lr_pr = ss["Random Forest"]["pr_auc"]["between_seed"], ss["Logistic Regression"]["pr_auc"]["between_seed"]
    rf_prec = ss["Random Forest"]["max_f1.precision"]["between_seed"]
    rf_var = var[var["Model"] == "Random Forest"].iloc[0] if not var.empty else None
    html(kpi_cards([
        {"label": "Test fraud prevalence", "value": fmt_range(prev["min"], prev["max"], lambda v: fmt_pct(v, 2)),
         "sub": "range across seeds (PR-AUC baseline)", "help": METRIC_HELP["baseline"]},
        {"label": "RF test PR-AUC", "value": f"{fmt_num(rf_pr['mean'])} ± {fmt_num(rf_pr['sd'])}",
         "sub": f"range {fmt_range(rf_pr['min'], rf_pr['max'], fmt_num)}", "accent": True,
         "help": "Mean ± sample SD across seeds (n = 3). Descriptive, not a confidence interval."},
        {"label": "LR test PR-AUC", "value": f"{fmt_num(lr_pr['mean'])} ± {fmt_num(lr_pr['sd'])}",
         "sub": f"range {fmt_range(lr_pr['min'], lr_pr['max'], fmt_num)}"},
        {"label": "RF max-F1 precision", "value": fmt_range(rf_prec["min"], rf_prec["max"], fmt_pct),
         "sub": "moves far more than PR-AUC", "help": METRIC_HELP["threshold"]},
        {"label": "RF PR-AUC SD: time vs seed",
         "value": (f"{fmt_num(rf_var['Temporal (typical SD across windows)'])} vs "
                   f"{fmt_num(rf_var['Between seeds (SD of window mean)'])}") if rf_var is not None else "n/a",
         "sub": "across windows vs between seeds",
         "help": "Typical SD of backtest PR-AUC across windows within a seed, vs the SD of each "
                 "seed's window mean. Two different questions, never pooled."},
    ]))

    c1, c2 = st.columns(2)
    fig = go.Figure()
    seeds = [str(s) for s in study["aggregate"]["seeds"]]
    labels = [f"seed {s}" for s in seeds]
    for model in ("Logistic Regression", "Random Forest"):
        vals = [ss[model]["pr_auc"]["values"].get(s) for s in seeds]
        fig.add_trace(go.Bar(x=labels, y=vals, name=model, marker_color=MODEL_COLORS[model],
                             text=[fmt_num(v) for v in vals], textposition="outside"))
    fig.add_trace(go.Scatter(x=labels, y=[ss["test_prevalence"]["values"][s] for s in seeds],
                             mode="markers", name="fraud-rate baseline",
                             marker={"symbol": "line-ew-open", "size": 34, "color": ui.WARN,
                                     "line": {"width": 2, "color": ui.WARN}}))
    fig.update_layout(barmode="group", yaxis_title="Test PR-AUC", yaxis_range=[0, 0.45])
    chart(fig, "Test PR-AUC per seed", "Amber ticks: each seed's fraud-rate baseline. The ranking "
          "held in every seed.", 340, c1)

    pts = ui.strategy_points(study)
    fig = px.scatter(pts, x="Recall", y="Precision", color="Model", symbol="Strategy",
                     hover_data=["Seed", "False alerts", "Alerts per 1,000"],
                     color_discrete_map=MODEL_COLORS, symbol_map={"Max F1": "circle", "Alert budget": "diamond"})
    fig.update_traces(marker={"size": 12, "line": {"color": "#0B0F0E", "width": 1}})
    fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%", xaxis_range=[0, 0.5], yaxis_range=[0, 1])
    chart(fig, "Operating points per seed (TEST)", "Circles: max F1. Diamonds: alert budget. Spread "
          "shows threshold instability.", 340, c2)

    wdf = ui.robustness_window_frame(study)
    model_pick = st.segmented_control("Backtest model", ["Random Forest", "Logistic Regression"],
                                      default="Random Forest") or "Random Forest"
    fig = go.Figure()
    for i, s in enumerate(seeds):
        sub = wdf[(wdf["Seed"] == s) & (wdf["Model"] == model_pick)]
        fig.add_trace(go.Scatter(x=sub["Window"], y=sub["PR-AUC"], mode="lines+markers", name=f"seed {s}",
                                 line={"color": MODEL_COLORS[model_pick], "dash": ui.SEED_DASH[i % 4]}))
        fig.add_trace(go.Scatter(x=sub["Window"], y=sub["Baseline"], mode="lines", showlegend=i == 0,
                                 name="fraud-rate baseline", line={"color": ui.WARN, "dash": "dot", "width": 1}))
    fig.update_layout(yaxis_type="log", xaxis_dtick=1, xaxis_title="Backtest window (earlier → later)",
                      yaxis_title="PR-AUC (log scale)")
    chart(fig, f"{model_pick}: backtest PR-AUC per window, all seeds",
          "Lines move together over time: temporal variation dominates sample-to-sample variation", 380)

    c3, c4 = st.columns([1, 1.3])
    if not var.empty:
        vlong = var.melt(id_vars="Model", var_name="Variation", value_name="SD of PR-AUC")
        fig = px.bar(vlong, x="Model", y="SD of PR-AUC", color="Variation", barmode="group",
                     color_discrete_map={"Temporal (typical SD across windows)": ui.ACCENT,
                                         "Between seeds (SD of window mean)": ui.MUTED},
                     text_auto=".3f")
        fig.update_layout(xaxis_title=None)
        chart(fig, "Where the variation comes from", "Backtest PR-AUC: SD across windows vs SD between "
              "seeds", 340, c3)
    with c4:
        html(section("Findings", "Generated from the aggregate; 'inconclusive' when evidence is mixed"))
        for f in study.get("findings", []):
            st.markdown(f"- {f}")

    html(section("Per-seed values (TEST period)", "Between-seed mean, sample SD and range. n = 3 "
                 "seeds: descriptive only."))
    model_tbl = st.segmented_control("Model", ["Random Forest", "Logistic Regression", "Rule-based"],
                                     default="Random Forest", key="rob_model") or "Random Forest"
    keys = ([("pr_auc", "PR-AUC", "dec3")] if model_tbl != "Rule-based" else [])
    prefix = ["rule"] if model_tbl == "Rule-based" else ["max_f1", "alert_budget"]
    for p_ in prefix:
        lbl = {"max_f1": "max F1", "alert_budget": "alert budget", "rule": "rule"}[p_]
        keys += [(f"{p_}.precision", f"Precision ({lbl})", "pct"), (f"{p_}.recall", f"Recall ({lbl})", "pct"),
                 (f"{p_}.false_positive", f"False alerts ({lbl})", "int"),
                 (f"{p_}.false_negative", f"Missed fraud ({lbl})", "int"),
                 (f"{p_}.alerts_per_1000", f"Alerts / 1,000 ({lbl})", "rate")]
    st.dataframe(ui.robustness_summary_table(study, model_tbl, keys), width="stretch", hide_index=True)

    with st.expander("Runs, data fingerprints and boundaries"):
        run_rows = [{"Seed": r["seed"], "Aggregated": "yes" if r.get("usable") else "NO",
                     "Experiment": r.get("experiment_id"), "Train run": r.get("train_run_id"),
                     "Backtest run": r.get("backtest_run_id"), "SHA-256": (r.get("sha256") or "")[:12],
                     "Test steps": "–".join(map(str, (r.get("split_boundaries") or {}).get("test", []))),
                     "Seconds": r.get("run_seconds")} for r in study["runs"]]
        st.dataframe(pd.DataFrame(run_rows), width="stretch", hide_index=True)
        res = study.get("resources") or {}
        if res:
            st.caption(f"Total wall time {res.get('seconds', 0):.0f} s; peak RSS {res.get('process_peak_rss_mb')} MB.")
        st.code(study.get("command_recommended", ROBUSTNESS_CMD), language="bash")
    summary_md = OUTPUT_DIR / "robustness_summary.md"
    if summary_md.exists():
        st.download_button("Download robustness_summary.md", summary_md.read_text(encoding="utf-8"),
                           "finsentry_robustness_summary.md", "text/markdown")

elif page == "Reliability & Calibration":
    html(section("Reliability & Calibration", "How certain are the primary TEST results, and do the "
                 "scores mean what they say? All from saved outputs."))
    names = [n for n, m in models_meta.items() if m.get("interval_comparison")]
    html(section("Uncertainty: row vs block bootstrap", "95% intervals; block bootstrap keeps "
                 "time clustering. Estimates, not guarantees."))
    if not names:
        st.info("No block-bootstrap results in metrics.json. Retrain to generate them.")
        st.code(TRAIN_CMD, language="bash")
    else:
        rows = []
        for n in names:
            for r in models_meta[n]["interval_comparison"]:
                for method, lo, hi in (("Row bootstrap", r["row_low"], r["row_high"]),
                                       ("Block bootstrap (steps)", r["block_low"], r["block_high"])):
                    rows.append({"Model": n, "Metric": r["metric"], "Method": method,
                                 "low": lo, "high": hi, "mid": (lo + hi) / 2,
                                 "Width ratio block/row": r["width_ratio_block_vs_row"]})
        ci_df = pd.DataFrame(rows)
        metric_pick = st.multiselect("Metrics", sorted(ci_df["Metric"].unique()),
                                     default=[m for m in ("max_f1.precision", "max_f1.recall",
                                                          "pr_auc", "rule.precision")
                                              if m in set(ci_df["Metric"])])
        sub = ci_df[ci_df["Metric"].isin(metric_pick)].copy()
        sub["label"] = sub["Model"] + " · " + sub["Metric"]
        fig = px.scatter(sub, x="mid", y="label", color="Method",
                         error_x=sub["high"] - sub["mid"], error_x_minus=sub["mid"] - sub["low"],
                         labels={"mid": "95% interval", "label": ""},
                         color_discrete_map={"Row bootstrap": "#38BDF8", "Block bootstrap (steps)": ui.ACCENT})
        fig.update_traces(marker={"size": 7})
        fig.for_each_trace(lambda t: t.update(error_x={"color": t.marker.color, "thickness": 2}))
        fig.update_layout(scattermode="group", scattergap=0.55)
        chart(fig, "95% intervals by method", "Wider block intervals mean the row bootstrap "
              "understates uncertainty", max(300, 60 * sub["label"].nunique() + 120))
        show_table(sub.drop(columns=["mid", "label"]), {"low": "dec3", "high": "dec3",
                                                        "Width ratio block/row": "dec3"})
        b = next((models_meta[n]["block_ci"] for n in names
                  if models_meta[n].get("block_ci", {}).get("computed")), None)
        if b:
            st.caption(f"Block bootstrap: {b['n_boot']} replicates, blocks of {b['block_steps']} "
                       f"steps over {b['n_steps']} test steps ({b['n_blocks']} blocks), seed "
                       f"{b['seed']}. Replicate fraud counts {b['replicate_fraud']['min']}–"
                       f"{b['replicate_fraud']['max']}. A width ratio above 1 means time "
                       "clustering makes the row interval too narrow.")
        with st.expander("Method notes and limitations"):
            st.markdown(metrics["notes"]["row_bootstrap"])
            st.markdown(metrics["notes"]["block_bootstrap"])

    html(section("Calibration", "Train-only model; Platt calibrator fit on VALIDATION only"))
    cal_models = [n for n, m in models_meta.items() if m.get("calibration", {}).get("computed")]
    if cal_models:
        show_table(pd.DataFrame([{
            "Model": n,
            "Brier raw": models_meta[n]["calibration"]["brier_raw"],
            "Brier calibrated (Platt)": models_meta[n]["calibration"]["brier_calibrated"],
            "Brier constant (validation fraud rate)": models_meta[n]["calibration"]["brier_constant_baseline"],
            "ECE raw": models_meta[n]["calibration"]["ece_raw"],
            "ECE calibrated": models_meta[n]["calibration"]["ece_calibrated"],
            "Calibrator": models_meta[n]["calibration"]["calibrator"].get("reason") or "fitted",
        } for n in cal_models]), {"Brier raw": "dec4", "Brier calibrated (Platt)": "dec4",
                                  "Brier constant (validation fraud rate)": "dec4", "ECE raw": "dec4",
                                  "ECE calibrated": "dec4"})
        name = st.selectbox("Reliability diagram for", cal_models)
        cal = models_meta[name]["calibration"]
        rel = []
        for kind, key in (("raw", "bins_raw"), ("calibrated", "bins_calibrated")):
            for b_ in cal.get(key, []):
                rel.append({**b_, "scores": kind})
        if rel:
            rdf = pd.DataFrame(rel)
            fig = px.scatter(rdf, x="mean_predicted", y="observed_rate", color="scores",
                             size="rows", hover_data=["rows", "fraud", "enough_rows"],
                             color_discrete_map={"raw": ui.WARN, "calibrated": ui.ACCENT})
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line={"dash": "dot", "color": ui.MUTED})
            fig.update_layout(xaxis_range=[0, 1], yaxis_range=[0, 1],
                              xaxis_title="Mean predicted score", yaxis_title="Observed fraud rate")
            chart(fig, f"{name}: reliability on TEST", "Bubble size = rows; points on the diagonal "
                  "are well calibrated", 400)
            thin = rdf[~rdf["enough_rows"]]
            if len(thin):
                st.warning(f"{len(thin)} bins have fewer than 30 rows; their observed rates are unreliable.")
        for w in cal.get("warnings", []):
            st.warning(w)
        st.caption(cal["note"])

    html(section("Score drift: validation → test", "Label-free comparison of score distributions"))
    drift_rows = []
    for n, m in models_meta.items():
        d = m.get("drift")
        if not d:
            continue
        for period, summ in (("validation", d["reference_summary"]), ("test", d["current_summary"])):
            if summ:
                drift_rows.append({"Model": n, "Period": period, "PSI (val→test)": d["psi"],
                                   "PSI band": d["psi_band"], **summ})
    if drift_rows:
        show_table(pd.DataFrame(drift_rows), {"PSI (val→test)": "dec3", "rows": "int", "mean": "dec4",
                                              "p50": "dec4", "p90": "dec4", "p99": "dec4",
                                              "p999": "dec4", "max": "dec4"})
        prev_df = pd.DataFrame([{"Period": k.title(), "Fraud rate": split[k]["prevalence"]}
                                for k in ("train", "validation", "test")])
        fig = go.Figure(go.Bar(x=prev_df["Period"], y=prev_df["Fraud rate"],
                               marker_color=[ui.MUTED, "#38BDF8", ui.ACCENT],
                               text=[fmt_pct(v, 2) for v in prev_df["Fraud rate"]], textposition="outside"))
        fig.update_layout(yaxis_tickformat=".2%", yaxis_range=[0, prev_df["Fraud rate"].max() * 1.3])
        chart(fig, "Prevalence shift across periods", "A calibrator fit on validation cannot know the "
              "test fraud rate", 300)
    st.info(DRIFT_NOTE)

elif page == "Explainability":
    html(section("Explainability", "Which inputs each model relies on overall. Per-transaction "
                 "explanations are on 'Transaction Explorer' and 'Try a Transaction'."))
    if not models:
        st.info("Train the models to generate feature importance.")
        st.code(TRAIN_CMD, language="bash")
    else:
        name = st.segmented_control("Model", list(models), default=best if best in models else list(models)[0]) \
            or list(models)[0]
        importance = global_feature_importance(models[name]["pipeline"])
        if importance.empty:
            st.info("Feature importance is not available for this model.")
        else:
            label = "importance" if name == "Random Forest" else "importance (|coefficient|)"
            fig = go.Figure(go.Bar(x=importance.sort_values("importance")["importance"],
                                   y=importance.sort_values("importance")["feature"], orientation="h",
                                   marker_color=MODEL_COLORS.get(name, ui.ACCENT)))
            fig.update_layout(xaxis_title=label)
            chart(fig, f"Global feature importance: {name}",
                  "Impurity importance" if name == "Random Forest" else "Absolute coefficients on standardised inputs",
                  max(300, 28 * len(importance) + 100))
            show_table(importance, {"importance": "dec4"})
        st.caption("Importance describes model reliance, not causality or proof of fraud. "
                   "Logistic Regression coefficients are on standardized inputs.")

elif page == "Try a Transaction":
    html(section("Try a Transaction", "Score a hypothetical transaction with the saved primary-"
                 "experiment models. Illustrative only: no real customer data."))
    with st.form("txn_form"):
        c1, c2, c3 = st.columns(3)
        kind = c1.selectbox("Transaction type", TRANSACTION_TYPES,
                            index=TRANSACTION_TYPES.index("TRANSFER"))
        amount = c2.number_input("Amount", min_value=0.0, value=150000.0, step=1000.0)
        step = c3.number_input("Simulation step (hour)", min_value=1, max_value=744, value=300,
                               help="Only the hour of day (step mod 24) is used by the models.")
        submitted = st.form_submit_button("Analyze transaction", type="primary")
    if submitted:
        raw = pd.DataFrame([{"step": int(step), "type": kind, "amount": float(amount)}])
        features = build_features(raw)
        points = int(rule_points(features).iloc[0])
        flagged = int(rule_based_score(features).iloc[0])
        cards = [{"label": "Rule baseline", "value": "Flagged" if flagged else "Not flagged",
                  "sub": f"{points} of 3 points (flag at 2+)", "help": RULE_DESCRIPTION}]
        for name, bundle in models.items():
            score = float(bundle["pipeline"].predict_proba(features)[:, 1][0])
            alert = score >= bundle["threshold"]
            budget_t = bundle.get("thresholds", {}).get("alert_budget")
            budget = (f" · budget {budget_t:.3f}: {'ALERT' if score >= budget_t else 'no alert'}"
                      if budget_t is not None else "")
            cards.append({"label": f"{name} score", "value": f"{score:.2%}", "accent": alert,
                          "sub": f"{'ALERT' if alert else 'no alert'} at max-F1 {bundle['threshold']:.3f}{budget}",
                          "help": "Uncalibrated model score, not a real-world fraud probability."})
        html(kpi_cards(cards))
        st.caption(RULE_DESCRIPTION + " Model scores are not calibrated real-world fraud probabilities.")
        with st.expander("Features passed to the models"):
            st.dataframe(features, width="stretch", hide_index=True)
        if models:
            html(section("Why did each model score it this way?", "Contributions relative to each "
                         "model's baseline"))
            ecols = st.columns(len(models))
            for col, (name, bundle) in zip(ecols, models.items()):
                show_explanation(name, bundle, features, col)
            st.caption(EXPLANATION_NOTE)
    else:
        st.caption("Set the inputs and press 'Analyze transaction'.")

elif page == "Report Summary":
    html(section("Report Summary", "Built from the current primary-experiment JSON every time, so a "
                 "stale file can never be shown."))
    text = build_report(*load_outputs(OUTPUT_DIR))
    report_path = OUTPUT_DIR / "report_summary.md"
    on_disk = report_path.read_text(encoding="utf-8") if report_path.exists() else None
    strip = lambda t: "\n".join(l for l in t.splitlines() if not l.startswith("_Generated"))
    if on_disk is None or strip(on_disk) != strip(text):
        st.info("outputs/report_summary.md is missing or older than the current outputs. The "
                "version below is current; save it to update the file.")
        if st.button("Save current report to outputs/report_summary.md"):
            report_path.write_text(text, encoding="utf-8")
            st.success("Saved.")
    else:
        html(ui.badges([("report file matches current outputs", "accent")]))
    c1, c2 = st.columns(2)
    c1.download_button("Download primary report (Markdown)", text, "finsentry_report_summary.md",
                       "text/markdown", width="stretch")
    rob_md = OUTPUT_DIR / "robustness_summary.md"
    if rob_md.exists():
        c2.download_button("Download robustness summary (Markdown)", rob_md.read_text(encoding="utf-8"),
                           "finsentry_robustness_summary.md", "text/markdown", width="stretch")
    st.caption("Section A holds measured values copied from outputs/*.json; section B is "
               "explanatory text.")
    with st.container(border=True, key="fs-report"):
        st.markdown(text)
