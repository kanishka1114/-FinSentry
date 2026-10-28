"""FinSentry-Mini dashboard: Try a transaction.

Scores one example with a saved model from models/ (copies of the original project's files).
No model is trained and no saved result or threshold is changed.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_FILES = {"Logistic Regression": "logistic_regression.joblib",
               "Random Forest": "random_forest.joblib"}
TRANSACTION_TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
FEATURES = ["amount", "log_amount", "hour_sin", "hour_cos", "type"]


@st.cache_resource
def load_model(file_name):
    bundle = joblib.load(MODEL_DIR / file_name)
    if file_name == MODEL_FILES["Random Forest"]:
        # Serial scoring, in memory only, so a score on the threshold edge is repeatable.
        bundle["pipeline"].set_params(model__n_jobs=1)
    return bundle


def features(amount, kind, hour):
    """The same five inputs the models were trained on (hour = simulation step mod 24)."""
    amount = max(float(amount), 0.0)
    return pd.DataFrame({
        "amount": [amount],
        "log_amount": [np.log1p(amount)],
        "hour_sin": [np.sin(2 * np.pi * hour / 24)],
        "hour_cos": [np.cos(2 * np.pi * hour / 24)],
        "type": [kind],
    })[FEATURES]


st.set_page_config(page_title="FinSentry-Mini: Try a transaction", layout="wide")
st.title("Try a transaction")
st.info(
    "This is a classroom demonstration. The models were trained on PaySim, a simulated "
    "dataset, and an alert does not prove that a transaction is fraud.")
st.markdown(
    "Enter an example transaction and choose a saved model. The model raises an alert when "
    "its score is at or above the threshold. Only amount, transaction type and hour of day "
    "are used; no names, account numbers or other personal details are needed.")

missing = [f for f in MODEL_FILES.values() if not (MODEL_DIR / f).exists()]
if missing:
    st.error(f"Saved model file(s) missing from {MODEL_DIR}: {', '.join(missing)}.")
    st.stop()

with st.form("transaction"):
    model_name = st.radio("Model", list(MODEL_FILES), horizontal=True)
    c1, c2, c3 = st.columns(3)
    amount = c1.number_input("Amount", min_value=0.0, value=100_000.0, step=1_000.0,
                             format="%.2f")
    kind = c2.selectbox("Transaction type", TRANSACTION_TYPES,
                        index=TRANSACTION_TYPES.index("TRANSFER"))
    hour = c3.number_input("Hour of day (0–23)", min_value=0, max_value=23, value=12, step=1,
                           help="In the experiment, hour of day is the PaySim simulation "
                                "step modulo 24.")
    submitted = st.form_submit_button("Check transaction")

if submitted:
    bundle = load_model(MODEL_FILES[model_name])
    score = float(bundle["pipeline"].predict_proba(features(amount, kind, hour))[:, 1][0])
    threshold = float(bundle["threshold"])
    alert = score >= threshold

    st.header("Result")
    st.dataframe(pd.DataFrame({
        "Amount": [f"{amount:,.2f}"], "Transaction type": [kind], "Hour of day": [int(hour)],
        "Model": [model_name],
    }), hide_index=True, width="stretch")
    r1, r2, r3 = st.columns(3)
    r1.metric("Model score", f"{score:.6f}")
    r2.metric("Saved threshold", f"{threshold:.6f}")
    r3.metric("Alert?", "Yes" if alert else "No")
    if alert:
        st.warning(f"Alert: the score is at or above the threshold ({score:.6f} ≥ {threshold:.6f}).")
    else:
        st.success(f"No alert: the score is below the threshold ({score:.6f} < {threshold:.6f}).")
    st.caption(
        "The score is the model's output between 0 and 1, used only for ranking and comparing "
        "with the threshold. It is not a guaranteed probability that the transaction is fraud: "
        "the models were trained with extra weight on the rare fraud cases, which pushes scores "
        "upwards. The threshold is the one saved with the model, chosen on the validation "
        "period of the original experiment.")
