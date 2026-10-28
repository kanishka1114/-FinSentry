"""FinSentry-Mini dashboard: Results page.

Shows the saved results of experiment exp-0781f6ac1d from outputs/metrics.json, an exact
copy of the original project's file. Nothing is trained or scored here.

    streamlit run Results.py
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

METRICS_PATH = Path(__file__).resolve().parent / "outputs" / "metrics.json"
EXPECTED_EXPERIMENT = "exp-0781f6ac1d"
APPROACHES = {"Rule-based": "Rule", "Logistic Regression": "Logistic Regression",
              "Random Forest": "Random Forest"}
COLORS = {"Rule": "#7f7f7f", "Logistic Regression": "#1f77b4", "Random Forest": "#2ca02c"}


@st.cache_data
def load_metrics(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rate(value):
    return "undefined" if value is None else f"{value:.3f}"


def percent(value):
    return f"{value:.2%}"


def decision_rule(model):
    if model["kind"] == "rule":
        return "Fixed rule (no score)"
    return f"Score ≥ {model['threshold']:.3f}"


def comparison_table(models):
    rows = []
    for key, label in APPROACHES.items():
        t = models[key]["test"]
        rows.append({
            "Approach": label,
            "Alert when": decision_rule(models[key]),
            "Precision": rate(t["precision"]),
            "Recall": rate(t["recall"]),
            "F1": rate(t["f1"]),
            "Fraud caught": f"{t['true_positive']:,}",
            "False alerts": f"{t['false_positive']:,}",
            "Missed fraud": f"{t['false_negative']:,}",
        })
    return pd.DataFrame(rows).set_index("Approach")


def confusion_table(test):
    return pd.DataFrame(
        {"Alerted": [f"{test['true_positive']:,}", f"{test['false_positive']:,}"],
         "Not alerted": [f"{test['false_negative']:,}", f"{test['true_negative']:,}"]},
        index=["Actually fraud", "Actually legitimate"])


def pr_chart(models, baseline):
    fig = go.Figure()
    for key in ("Logistic Regression", "Random Forest"):
        curve = models[key].get("pr_curve_test")
        if curve:
            fig.add_trace(go.Scatter(
                x=curve["recall"], y=curve["precision"], mode="lines+markers", name=key,
                line={"color": COLORS[key], "width": 2}, marker={"size": 4}))
    for key, label in APPROACHES.items():
        t = models[key]["test"]
        name = ("Rule: its only result" if models[key]["kind"] == "rule"
                else f"{label}: result at its saved threshold")
        fig.add_trace(go.Scatter(
            x=[t["recall"]], y=[t["precision"]], mode="markers", name=name,
            marker={"color": COLORS[label], "size": 11, "symbol": "diamond",
                    "line": {"color": "black", "width": 1}}))
    fig.add_hline(y=baseline, line_dash="dot", line_color="#555555",
                  annotation_text=f"Random guessing (fraud rate {baseline:.4f})",
                  annotation_position="top left")
    fig.update_layout(
        xaxis_title="Recall (share of fraud that was alerted)",
        yaxis_title="Precision (share of alerts that were fraud)",
        xaxis={"range": [0, 1.02], "dtick": 0.2}, yaxis={"range": [0, 1.02], "dtick": 0.2},
        height=460,
        template="simple_white", legend={"orientation": "h", "y": -0.2},
        margin={"t": 20, "b": 10})
    return fig


st.set_page_config(page_title="FinSentry-Mini: Results", layout="wide")
st.title("FinSentry-Mini: Results")

if not METRICS_PATH.exists():
    st.error(f"Saved results not found at {METRICS_PATH}.")
    st.stop()

m = load_metrics(str(METRICS_PATH))
run = m["run"]
if run["experiment_id"] != EXPECTED_EXPERIMENT:
    st.warning(f"metrics.json belongs to experiment {run['experiment_id']}, not "
               f"{EXPECTED_EXPERIMENT}. The figures below are from that file.")

st.markdown(
    "These are the results on the **test period**: later transactions that the models never "
    "saw while they were trained or while their thresholds were chosen.")
st.info(
    "**Simulated data.** PaySim is made-up mobile-money data, not real bank records. "
    f"Fraud is rare: about 1 in {round(1 / m['sampling']['prevalence_used']):,} transactions. "
    "These results show how the models did on this simulated data, not how they would do "
    "in the real world.")

st.header("Data and time split")
ds, sp, split = m["dataset"], m["sampling"], m["split"]
c1, c2, c3 = st.columns(3)
c1.metric("Transactions in PaySim file", f"{ds['rows_in_file']:,}")
c2.metric("Random sample used", f"{sp['rows_used']:,}", help=f"Seed {run['seed']}; rows chosen "
          "without looking at the fraud label.")
c3.metric("Fraud in sample", f"{sp['fraud_used']:,} ({percent(sp['prevalence_used'])})")

periods = []
for key, label, role in (("train", "Train", "Models fitted"),
                         ("validation", "Validation", "Thresholds chosen"),
                         ("test", "Test", "Results below")):
    first, last = split["step_boundaries"][key]
    s = split[key]
    periods.append({"Period": label, "Hourly steps": f"{first}–{last}",
                    "Transactions": f"{s['rows']:,}", "Fraud": f"{s['fraud']:,}",
                    "Fraud rate": percent(s["prevalence"]), "Used for": role})
st.dataframe(pd.DataFrame(periods).set_index("Period"), width="stretch")
st.caption(
    "The sample is split by time, so every model is tested on transactions that come after "
    "the ones it learned from. Model inputs are amount, log amount, hour of day and "
    "transaction type. The fraud rate differs between periods, which affects precision.")

models = m["models"]
st.header("Test results")
st.dataframe(comparison_table(models), width="stretch")
st.caption(
    "Precision: share of alerts that were fraud. Recall: share of fraud that was alerted. "
    "F1 is the harmonic mean of precision and recall. Each model's threshold is the one that maximised F1 on the "
    "validation period. The rule flags a TRANSFER or CASH_OUT of at least 200,000, or any "
    "transaction of at least 1,000,000.")

st.subheader("Confusion matrices (test period)")
for col, (key, label) in zip(st.columns(3), APPROACHES.items()):
    with col:
        st.markdown(f"**{label}**")
        st.dataframe(confusion_table(models[key]["test"]), width="stretch")

st.header("Precision-recall curve (test period)")
baseline = models["Random Forest"]["test"]["pr_auc_baseline"]
st.plotly_chart(pr_chart(models, baseline), width="stretch")
lr, rf = models["Logistic Regression"]["test"], models["Random Forest"]["test"]
st.caption(
    "Each curve shows the precision and recall a model would reach at every possible "
    "threshold; diamonds mark the results in the table above. The rule has no score, so it "
    "is a single point. PR-AUC summarises the whole curve in one number (saved as average "
    f"precision): Logistic Regression {rate(lr['pr_auc'])}, Random Forest "
    f"{rate(rf['pr_auc'])}; random guessing would score about {baseline:.4f}.")
st.caption(
    "Limitation of the saved curve: `metrics.json` keeps 200 of the curve's points, spaced "
    "by position rather than by recall, and dots mark them. Between dots the line is only a "
    "connector. Logistic Regression has no saved points between recall 0 and 0.22, so its "
    "diamond lies off the drawn line. The point at recall 0 and precision 1 is a "
    "convention of the plotting library, not a measured threshold.")
