"""FinSentry-Mini dashboard: Try a transaction.

Scores one transaction with the saved model in models/random_forest.joblib at its saved
threshold. No model is trained and nothing saved is changed.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import finsentry as fs

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "random_forest.joblib"
RESULTS_PATH = ROOT / "outputs" / "results.json"
DEFAULTS = {"amount": 100_000.0, "kind": "TRANSFER", "hour": 12,
            "balance_before": 150_000.0, "balance_after": 50_000.0}
EXAMPLE_LABELS = {"alert": "Load fraud example (alert)",
                  "no_alert": "Load normal example (no alert)"}


@st.cache_resource
def load_model(path):
    bundle = joblib.load(path)
    # Serial scoring, in memory only, so a score on the threshold edge is repeatable.
    bundle["pipeline"].set_params(model__n_jobs=1)
    return bundle


@st.cache_data
def load_examples(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))["examples"]


def load_example(example):
    st.session_state.update({
        "amount": example["amount"], "kind": example["type"], "hour": example["hour"],
        "balance_before": example["sender_balance_before"],
        "balance_after": example["sender_balance_after"], "check": True})


st.set_page_config(page_title="FinSentry-Mini: Try a transaction", layout="wide")
st.title("Try a transaction")
st.info(
    "Classroom demonstration. The model was trained on PaySim, a simulated dataset, and an "
    "alert does not prove that a transaction is fraud.")
st.markdown(
    "Enter a transaction. The Random Forest gives it a score; if the score is at or above "
    "the threshold, it raises an alert. **If the score is below the threshold, there is no "
    "alert.**")

missing = [p.name for p in (MODEL_PATH, RESULTS_PATH) if not p.exists()]
if missing:
    st.error(f"Saved file(s) missing: {', '.join(missing)}.")
    st.stop()

examples = load_examples(str(RESULTS_PATH))
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)

st.markdown("**Load a real test-period example**, or enter your own values below.")
for col, (key, label) in zip(st.columns(2), EXAMPLE_LABELS.items()):
    col.button(label, on_click=load_example, args=(examples[key],), width="stretch")

with st.form("transaction"):
    c1, c2, c3 = st.columns(3)
    amount = c1.number_input("Amount", min_value=0.0, step=1_000.0, format="%.2f",
                             key="amount")
    kind = c2.selectbox("Transaction type", fs.TRANSACTION_TYPES, key="kind")
    hour = c3.number_input("Hour of day (0–23)", min_value=0, max_value=23, step=1, key="hour")
    b1, b2 = st.columns(2)
    balance_before = b1.number_input("Sender's balance before", min_value=0.0, step=1_000.0,
                                     format="%.2f", key="balance_before")
    balance_after = b2.number_input("Sender's balance after", min_value=0.0, step=1_000.0,
                                    format="%.2f", key="balance_after")
    submitted = st.form_submit_button("Check transaction")

if submitted or st.session_state.pop("check", False):
    bundle = load_model(str(MODEL_PATH))
    row = fs.build_features(amount, kind, hour, balance_before, balance_after)
    score = float(bundle["pipeline"].predict_proba(row)[:, 1][0])
    threshold = float(bundle["threshold"])
    alert = score >= threshold

    st.header("Result")
    st.dataframe(pd.DataFrame({
        "Amount": [f"{amount:,.2f}"], "Type": [kind], "Hour": [int(hour)],
        "Sender's balance before": [f"{balance_before:,.2f}"],
        "Sender's balance after": [f"{balance_after:,.2f}"],
    }), hide_index=True, width="stretch")
    r1, r2, r3 = st.columns(3)
    r1.metric("Score", f"{score:.3f}")
    r2.metric("Threshold", f"{threshold:.3f}")
    r3.metric("Alert?", "Yes" if alert else "No")
    if alert:
        st.warning(f"Alert: the score {score:.3f} is at or above the threshold {threshold:.3f}.")
    else:
        st.success(f"No alert: the score {score:.3f} is below the threshold {threshold:.3f}.")
    st.caption(
        "The score is the model's output from 0 to 1, used only to compare with the "
        "threshold. It is not the exact chance that the transaction is fraud.")
