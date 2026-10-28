"""FinSentry-Mini dashboard: How the threshold works.

Uses only saved validation trade-off data and saved threshold results from
outputs/metrics.json. Nothing is trained, scored or re-tuned here.
"""
import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

METRICS_PATH = Path(__file__).resolve().parents[1] / "outputs" / "metrics.json"
MODELS = ["Logistic Regression", "Random Forest"]
COLORS = {"Precision": "#1f77b4", "Recall": "#d62728", "F1": "#2ca02c"}


@st.cache_data
def load_metrics(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rate(value):
    return "undefined" if value is None else f"{value:.3f}"


def f1_from_counts(p):
    """F1 for a saved trade-off point, calculated from its saved counts."""
    tp, fp, fn = p["true_positive"], p["false_positive"], p["false_negative"]
    return 2 * tp / (2 * tp + fp + fn) if tp + fn else None


def validation_points(model):
    rows = [p for p in model.get("tradeoff", []) if p["period"] == "validation"]
    return pd.DataFrame([{
        "Threshold": p["threshold"],
        "Alerts per 1,000": p["alerts_per_1000"],
        "Precision": p["precision"],
        "Recall": p["recall"],
        "F1": f1_from_counts(p),
        "Fraud caught": p["true_positive"],
        "False alerts": p["false_positive"],
        "Missed fraud": p["false_negative"],
    } for p in rows])


def tradeoff_chart(points, chosen):
    fig = go.Figure()
    for metric in ("Precision", "Recall", "F1"):
        fig.add_trace(go.Scatter(
            x=points["Alerts per 1,000"], y=points[metric], mode="lines+markers", name=metric,
            line={"color": COLORS[metric], "width": 2}, marker={"size": 5},
            customdata=points["Threshold"],
            hovertemplate=f"{metric} %{{y:.3f}}<br>Threshold %{{customdata:.6f}}<br>"
                          "Alerts per 1,000: %{x:.2f}<extra></extra>"))
    v = chosen["validation"]
    for metric, key in (("Precision", "precision"), ("Recall", "recall"), ("F1", "f1")):
        fig.add_trace(go.Scatter(
            x=[v["alerts_per_1000"]], y=[v[key]], mode="markers", showlegend=metric == "F1",
            name="Saved threshold (validation result)",
            marker={"color": COLORS[metric], "size": 12, "symbol": "diamond",
                    "line": {"color": "black", "width": 1}},
            hovertemplate=f"Saved threshold {chosen['threshold']:.6f}<br>{metric} %{{y:.3f}}"
                          "<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=[v["alerts_per_1000"]] * 2, y=[0, 1.02], mode="lines", showlegend=False,
        hoverinfo="skip", line={"color": "#555555", "dash": "dot", "width": 1.5}))
    fig.update_layout(
        xaxis={"title": "Alerts per 1,000 validation transactions (log scale). "
                        "Lower threshold → more alerts →",
               "type": "log", "tickvals": [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000]},
        yaxis={"title": "Value on the validation period", "range": [0, 1.02], "dtick": 0.2},
        height=440, template="simple_white", legend={"orientation": "h", "y": -0.25},
        margin={"t": 20, "b": 10})
    return fig


st.set_page_config(page_title="FinSentry-Mini: How the threshold works", layout="wide")
st.title("How the threshold works")

if not METRICS_PATH.exists():
    st.error(f"Saved results not found at {METRICS_PATH}.")
    st.stop()

m = load_metrics(str(METRICS_PATH))
models = m["models"]
split = m["split"]
st.markdown(
    f"Saved data from experiment **{m['run']['experiment_id']}**. Nothing on this page is "
    "trained, re-scored or re-tuned.")

st.header("Scores and thresholds")
st.markdown(
    "Each model gives every transaction a **score** between 0 and 1; a higher score means the "
    "model considers the transaction more likely to be fraud. A transaction is **flagged** "
    "(an alert is raised) when its score is **at or above** the model's threshold.")
st.dataframe(pd.DataFrame([{
    "Model": name,
    "Saved threshold": f"{models[name]['threshold']:.6f}",
    "Alert when": f"Score ≥ {models[name]['threshold']:.3f}",
} for name in MODELS]).set_index("Model"), width="stretch")
st.markdown(
    "The rule baseline has no score and no threshold: it flags fixed combinations of "
    "transaction type and amount.")

st.header("How the thresholds were chosen")
v_rows, t_rows = split["validation"], split["test"]
st.markdown(
    f"1. Each model was fitted on the **training** period (steps "
    f"{split['step_boundaries']['train'][0]}–{split['step_boundaries']['train'][1]}).\n"
    f"2. Its threshold was chosen on the **validation** period (steps "
    f"{split['step_boundaries']['validation'][0]}–{split['step_boundaries']['validation'][1]}, "
    f"{v_rows['rows']:,} transactions, {v_rows['fraud']:,} fraud): the threshold with the "
    "highest F1 among all thresholds possible on validation was kept.\n"
    f"3. The model and that threshold were then applied once to the **test** period (steps "
    f"{split['step_boundaries']['test'][0]}–{split['step_boundaries']['test'][1]}, "
    f"{t_rows['rows']:,} transactions, {t_rows['fraud']:,} fraud). The test metrics on the "
    "Results page are for these saved thresholds.\n\n"
    "Test labels were not used to choose the threshold.")

st.header("Validation trade-off")
st.markdown(
    "Lowering the threshold flags more transactions, so it can only keep or increase the "
    "fraud caught and can only keep or increase the false alerts. Raising it does the "
    "opposite. Precision may go up or down. The chart shows how this played out on the "
    "validation period; it does not guarantee the same balance on other data.")

for tab, name in zip(st.tabs(MODELS), MODELS):
    with tab:
        model = models[name]
        chosen = model["strategies"]["max_f1"]
        points = validation_points(model)
        if points.empty:
            st.warning("No validation trade-off data is saved for this model, so no chart is "
                       "shown.")
            continue
        st.plotly_chart(tradeoff_chart(points, chosen), width="stretch")
        st.caption(
            f"Lines join {len(points)} saved validation thresholds, chosen by the original "
            "experiment to target between 0.1 and 500 alerts per 1,000 validation transactions "
            "(tied scores can give more); they are a sample of all possible thresholds. F1 for these points is calculated "
            "from their saved counts. Diamonds and the dotted line mark the saved threshold, "
            f"which is not one of the sampled points; its validation values are saved "
            f"exactly. Validation has only {v_rows['fraud']} fraud cases, so these values "
            "move in large steps.")

        v, t = chosen["validation"], chosen["test"]
        st.markdown(f"**At the saved threshold ({chosen['threshold']:.6f})**")
        st.dataframe(pd.DataFrame([
            {"Period": "Validation (threshold chosen here)", "Precision": rate(v["precision"]),
             "Recall": rate(v["recall"]), "F1": rate(v["f1"]),
             "Fraud caught": f"{v['true_positive']:,}", "False alerts": f"{v['false_positive']:,}",
             "Missed fraud": f"{v['false_negative']:,}",
             "Alerts per 1,000": f"{v['alerts_per_1000']:.2f}"},
            {"Period": "Test (reported on Results page)", "Precision": rate(t["precision"]),
             "Recall": rate(t["recall"]), "F1": rate(t["f1"]),
             "Fraud caught": f"{t['true_positive']:,}", "False alerts": f"{t['false_positive']:,}",
             "Missed fraud": f"{t['false_negative']:,}",
             "Alerts per 1,000": f"{t['alerts_per_1000']:.2f}"},
        ]).set_index("Period"), width="stretch")

        with st.expander("Saved validation trade-off points"):
            st.dataframe(points.style.format({
                "Threshold": "{:.6f}", "Alerts per 1,000": "{:.2f}", "Precision": "{:.3f}",
                "Recall": "{:.3f}", "F1": "{:.3f}", "Fraud caught": "{:,}",
                "False alerts": "{:,}", "Missed fraud": "{:,}"}, na_rep="undefined"),
                width="stretch", hide_index=True)

st.caption(
    f"Validation precision is much lower than test precision partly because fraud was rarer "
    f"in validation ({v_rows['prevalence']:.2%}) than in test ({t_rows['prevalence']:.2%}). "
    "The original experiment also saved the same sampled thresholds evaluated on the test "
    "period; they are not shown here because test results must not be used to choose a "
    "threshold.")
